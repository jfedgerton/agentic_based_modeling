"""Tests for the streaming agent-panel writer."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.utils.agent_panel import (
    CV_PANEL_SCHEMA,
    AgentPanelWriter,
    disabled_writer,
)


SIMPLE_SCHEMA = pa.schema([
    pa.field("step", pa.int32()),
    pa.field("agent_id", pa.int64()),
    pa.field("label", pa.string()),
    pa.field("value", pa.float64()),
])


def _row(step, agent_id, label="a", value=1.0):
    return {"step": step, "agent_id": agent_id, "label": label, "value": value}


def test_writes_all_rows_across_multiple_batches(tmp_path):
    path = tmp_path / "panel.parquet"
    # batch_size below the row count forces more than one flush.
    with AgentPanelWriter(path, schema=SIMPLE_SCHEMA, batch_size=10) as w:
        for i in range(25):
            w.append(_row(step=i, agent_id=i * 2))

    table = pq.read_table(path)
    assert table.num_rows == 25
    assert table.column("step").to_pylist() == list(range(25))
    assert table.column("agent_id").to_pylist() == [i * 2 for i in range(25)]


def test_close_returns_summary_and_flushes_partial_batch(tmp_path):
    path = tmp_path / "panel.parquet"
    w = AgentPanelWriter(path, schema=SIMPLE_SCHEMA, batch_size=10)
    for i in range(13):  # 10 flushed, 3 left buffered
        w.append(_row(step=i, agent_id=i))
    summary = w.close()

    assert summary["enabled"] is True
    assert summary["rows"] == 13
    assert pq.read_table(path).num_rows == 13


def test_missing_keys_become_null_not_errors(tmp_path):
    """A row that omits a column must not abort a run mid-simulation."""
    path = tmp_path / "panel.parquet"
    with AgentPanelWriter(path, schema=SIMPLE_SCHEMA, batch_size=100) as w:
        w.append({"step": 1, "agent_id": 7})  # label and value absent

    table = pq.read_table(path)
    assert table.column("label").to_pylist() == [None]
    assert table.column("value").to_pylist() == [None]


def test_disabled_writer_is_a_noop(tmp_path):
    w = disabled_writer()
    for i in range(1000):
        w.append(_row(step=i, agent_id=i))
    summary = w.close()

    assert summary == {"enabled": False, "rows": 0, "path": None}
    assert list(tmp_path.iterdir()) == []


def test_no_file_created_when_nothing_appended(tmp_path):
    """An empty panel should not leave a stray zero-row file behind."""
    path = tmp_path / "panel.parquet"
    summary = AgentPanelWriter(path, schema=SIMPLE_SCHEMA).close()

    assert summary["rows"] == 0
    assert summary["path"] is None
    assert not path.exists()


def test_requires_schema_or_field_names(tmp_path):
    with pytest.raises(ValueError, match="schema or field_names"):
        AgentPanelWriter(tmp_path / "panel.parquet")


def test_cv_schema_records_decision_time_and_post_move_position():
    """Both positions are needed: agents move after deciding."""
    names = set(CV_PANEL_SCHEMA.names)
    assert {"x", "y", "moved_to_x", "moved_to_y"} <= names
    # activation_idx is what makes the sequential, shuffled activation order
    # reconstructable offline.
    assert "activation_idx" in names
    assert {"cops_nearby", "actives_nearby", "quiets_nearby"} <= names
