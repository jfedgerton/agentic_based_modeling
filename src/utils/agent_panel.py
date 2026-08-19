"""Streaming writer for per-agent panel data.

A run produces one row per agent per step — on the order of 10^5 rows for a
single Civil Violence run and 10^7 across a batch. Accumulating those in
memory and serialising at the end (the approach :class:`ExperimentLogger`
takes for its own records) does not scale, so this writer flushes fixed-size
batches to Parquet as the simulation proceeds and never holds more than one
batch.

The writer is deliberately independent of :class:`ExperimentLogger`: panel
rows go to their own file and never enter ``ExperimentLogger._records``.

When disabled, ``append`` is a cheap no-op, so gating panel collection on a
config flag costs a single attribute check per row.
"""

import csv
from pathlib import Path
from typing import Any, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except ImportError:  # pragma: no cover - only hit where pyarrow is absent
    _HAVE_PYARROW = False


# --- Panel schemas -------------------------------------------------------
#
# Declared explicitly rather than inferred: a column that happens to be all
# None in one batch would otherwise get a null type and fail to merge with
# later batches.

def _schema(fields: list) -> Optional["pa.Schema"]:
    return pa.schema(fields) if _HAVE_PYARROW else None


def _cv_panel_fields() -> list:
    if not _HAVE_PYARROW:
        return []
    f, i, s, b = pa.float64(), pa.int32(), pa.string(), pa.bool_()
    return [
        pa.field("step", i), pa.field("activation_idx", i),
        pa.field("agent_id", pa.int64()), pa.field("agent_type", s),
        # Position at decision time, i.e. before this step's move.
        pa.field("x", i), pa.field("y", i),
        pa.field("state_before", s), pa.field("action", s),
        # Position after this step's move; equal to (x, y) if it did not move.
        pa.field("moved_to_x", i), pa.field("moved_to_y", i),
        pa.field("hardship", f), pa.field("risk_aversion", f),
        pa.field("legitimacy", f), pa.field("grievance", f),
        pa.field("cops_nearby", i), pa.field("actives_nearby", i),
        pa.field("quiets_nearby", i), pa.field("arrest_prob", f),
        pa.field("jail_term", i), pa.field("confidence", f),
        pa.field("parse_failed", b),
    ]


def _cv_arrest_fields() -> list:
    if not _HAVE_PYARROW:
        return []
    i = pa.int32()
    return [
        pa.field("step", i), pa.field("activation_idx", i),
        pa.field("cop_id", pa.int64()), pa.field("cop_x", i), pa.field("cop_y", i),
        pa.field("target_id", pa.int64()),
        pa.field("target_x", i), pa.field("target_y", i),
        pa.field("jail_term", i),
    ]


def _pd_panel_fields() -> list:
    if not _HAVE_PYARROW:
        return []
    f, i, s, b = pa.float64(), pa.int32(), pa.string(), pa.bool_()
    return [
        pa.field("step", i), pa.field("agent_id", pa.int64()),
        pa.field("agent_type", s),
        pa.field("x", i), pa.field("y", i),
        pa.field("action_before", s), pa.field("action", s),
        pa.field("payoff", f),
        pa.field("num_neighbors", i), pa.field("coop_count", i),
        pa.field("defect_count", i), pa.field("coop_rate", f),
        pa.field("best_neighbor_action", s),
        pa.field("confidence", f), pa.field("parse_failed", b),
    ]


def _ic_dyad_fields() -> list:
    if not _HAVE_PYARROW:
        return []
    f, i, s, b = pa.float64(), pa.int32(), pa.string(), pa.bool_()
    lid = pa.int64()
    return [
        pa.field("step", i),
        pa.field("proposer_id", lid), pa.field("responder_id", lid),
        pa.field("proposer_x", i), pa.field("proposer_y", i),
        pa.field("responder_x", i), pa.field("responder_y", i),
        pa.field("proposer_signal", s), pa.field("responder_signal", s),
        pa.field("demand", f), pa.field("threshold", f),
        pa.field("war", b), pa.field("winner_id", lid),
        pa.field("proposer_payoff", f), pa.field("responder_payoff", f),
        pa.field("proposer_true_capability", f),
        pa.field("responder_true_capability", f),
        pa.field("proposer_true_war_cost", f),
        pa.field("responder_true_war_cost", f),
    ]


CV_PANEL_SCHEMA = _schema(_cv_panel_fields())
CV_ARREST_SCHEMA = _schema(_cv_arrest_fields())
PD_PANEL_SCHEMA = _schema(_pd_panel_fields())
IC_DYAD_SCHEMA = _schema(_ic_dyad_fields())


def flatten_dyad_outcome(outcome: dict) -> dict:
    """Reshape a model dyad outcome into a flat IC_DYAD_SCHEMA row.

    The model carries positions as ``(x, y)`` tuples. Splitting them into
    columns keeps the table usable from any query engine, and lets rows
    extracted from historical ``records.jsonl`` share one schema with rows
    written live by a running model.
    """
    row = {k: v for k, v in outcome.items() if not k.endswith("_pos")}
    for side in ("proposer", "responder"):
        pos = outcome.get(f"{side}_pos") or (None, None)
        row[f"{side}_x"], row[f"{side}_y"] = pos[0], pos[1]
    return row


# --- Writer --------------------------------------------------------------

class AgentPanelWriter:
    """Append-only, batched writer for one panel table.

    Args:
        path: Destination file. Suffix is forced to match the backend.
        schema: pyarrow schema describing every column. Ignored when pyarrow
            is unavailable, in which case the column order comes from
            ``field_names``.
        field_names: Column order for the CSV fallback. Defaults to the
            schema's names.
        batch_size: Rows buffered before a flush.
        enabled: When False the writer accepts and discards everything.
    """

    def __init__(self, path, schema=None, field_names: Optional[list] = None,
                 batch_size: int = 20000, enabled: bool = True):
        self.enabled = enabled
        self.rows_written = 0
        self._buffer: list = []
        self._batch_size = batch_size
        self._writer = None
        self._csv_file = None
        self._csv_writer = None

        if not self.enabled:
            self.path = None
            self._names = []
            return

        self._schema = schema if _HAVE_PYARROW else None
        self._names = field_names or (
            list(schema.names) if (schema is not None and _HAVE_PYARROW) else []
        )
        if not self._names:
            raise ValueError("AgentPanelWriter needs a schema or field_names")

        path = Path(path)
        suffix = ".parquet" if self._schema is not None else ".csv"
        self.path = path.with_suffix(suffix)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        """Buffer one row; flush when the batch is full."""
        if not self.enabled:
            return
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self._schema is not None:
            columns = [
                pa.array([r.get(name) for r in self._buffer], type=field.type)
                for name, field in zip(self._schema.names, self._schema)
            ]
            batch = pa.RecordBatch.from_arrays(columns, schema=self._schema)
            if self._writer is None:
                self._writer = pq.ParquetWriter(self.path, self._schema,
                                                compression="snappy")
            self._writer.write_batch(batch)
        else:
            if self._csv_writer is None:
                self._csv_file = open(self.path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._names, extrasaction="ignore")
                self._csv_writer.writeheader()
            self._csv_writer.writerows(self._buffer)
            self._csv_file.flush()

        self.rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> dict:
        """Flush the tail, close the file, and return a small summary."""
        if not self.enabled:
            return {"enabled": False, "rows": 0, "path": None}
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        return {
            "enabled": True,
            "rows": self.rows_written,
            "path": str(self.path) if self.rows_written else None,
        }

    def __enter__(self) -> "AgentPanelWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def disabled_writer() -> AgentPanelWriter:
    """A writer that swallows every row — used when panel logging is off."""
    return AgentPanelWriter(path=None, enabled=False)


# --- Wiring ---------------------------------------------------------------
#
# Model constructor argument -> (file stem, schema), per game. The runner and
# the replay scripts both build their writers from this table, so adding a
# panel means declaring it in one place.

PANEL_TABLES: dict = {
    "pd": {"panel_writer": ("agent_panel", PD_PANEL_SCHEMA)},
    "cv": {"panel_writer": ("agent_panel", CV_PANEL_SCHEMA),
           "arrest_writer": ("arrests", CV_ARREST_SCHEMA)},
    "ic": {"dyad_writer": ("dyads", IC_DYAD_SCHEMA)},
}


def make_panel_writers(game: str, directory, prefix: str = "",
                       enabled: bool = True) -> dict:
    """Build the panel writers a game's model constructor accepts.

    Returns a mapping ready to splat into the constructor. An unknown game or
    ``enabled=False`` yields an empty mapping, so callers never need to know
    which games have panels.
    """
    if not enabled:
        return {}
    directory = Path(directory)
    return {
        arg: AgentPanelWriter(directory / f"{prefix}{stem}", schema=schema)
        for arg, (stem, schema) in PANEL_TABLES.get(game, {}).items()
    }


def close_panel_writers(writers: dict) -> dict:
    """Close every writer, returning ``{argument: summary}``."""
    return {arg: writer.close() for arg, writer in writers.items()}
