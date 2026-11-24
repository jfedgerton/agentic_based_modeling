# Analysis-ready run outputs

Per-run files for every simulation in the paper (107 run folders, 89 distinct runs; replays and excluded runs are
flagged by `analysis/R/01_run_index.R`, not removed here). Each run folder holds `config.json`, `summary.json`,
`time_series.csv`, and where the run produced them, `dyads.parquet` (crisis game) and `agent_panel.parquet`.

The folder layout mirrors the original batches (`one_simulation/`, `five_simulation/`, `second_step/`), so the R
pipeline runs against it directly. The figures built from the agents' coded reasons (Figs. 4 and 5B) also need the
coded-rationale tables in `data/derived_text/`, which were produced from the raw transcripts:

    mkdir -p analysis/derived/text && cp -r data/derived_text/* analysis/derived/text/
    ABM_ROOT=$(pwd)/data/runs Rscript analysis/R/run_all.R

Raw model transcripts (`llm_interactions.jsonl`, about 23 GB) and full decision logs (`records.jsonl`, `run.log`) are
not stored here because of their size; they will be archived in a public repository on publication.

The classic crisis-game runs with bluffing on (`ic_classic_seed42_76348bc0`, `ic_classic_seed42_851e2d79`) contain the
rerun with the corrected proposer demand rule, p_self + E[k] (commit 52033d1). The original runs, produced with
1 - p_self + E[k], are kept outside the repository.
