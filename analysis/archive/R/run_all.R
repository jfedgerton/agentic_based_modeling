# run_all.R
# Master script. Runs every stage in order, sequentially -- each stage's output
# is the next stage's checked input, and a failure stops the run right there.
#
# Usage (from inside the r_pipeline/ folder):
#   Rscript R/run_all.R
# On Jared's Mac, with the pipeline at ~/Dropbox/ABM/paper/r_pipeline, no env
# vars are needed: ABM_ROOT defaults to ../.. (the ABM/ folder).
# Elsewhere: ABM_ROOT=/path/to/ABM PIPELINE_ROOT=/path/to/r_pipeline Rscript R/run_all.R

PIPELINE_ROOT <- normalizePath(Sys.getenv("PIPELINE_ROOT", unset = "."), mustWork = TRUE)
Sys.setenv(PIPELINE_ROOT = PIPELINE_ROOT)

scripts <- c(
  "00_config.R",
  "01_build_run_index.R",        # one row per run + exclusion gate (parse failure > 5% / incomplete)
  "02_extract_timeseries.R",     # stacks every time_series.csv
  "03_convergence_metrics.R",    # LLM arm vs classic (RMSE, KS)
  "04_pd_strategies.R",          # per-agent PD strategy labels
  "05_cv_threshold_recovery.R",  # logistic recovery of CV activation threshold (classic ~0.101)
  "06_framing_divergence.R",     # role-player vs reasoner/calculator
  "07_ic_analysis.R",            # interstate conflict (out of sample)
  "08_cv_longrun.R",             # 500-step CV
  "09_provider_table.R",
  "10_figures.R",                # earlier PNAS-draft figures (kept for reference)
  "11_tables.R",                 # earlier PNAS-draft tables + SI run completeness
  "12_provider_divergence.R",    # HEADLINE: GPT vs DeepSeek vs classic, exact permutation tests
  "13_decision_rules.R",         # mechanism: PD transition rule, CV activation curve
  "14_figures_divergence.R",     # Fig 1-4 + Extended Data figures for the divergence framing
  "15_tables_divergence.R",      # Table 2 + Extended Data tables
  "19_waffle_deviations.R",      # Fig: stated reasons behind deviations (needs derived/text/ from tools/text_analysis/run_text_analysis.sh)
  "16_export_manuscript.R",      # copy figs/tables into manuscript/, zip for Overleaf
  "17_alluvial_design.R",        # alluvial plot of design -> outcomes (2-col and 2-row facets)
  "18_eda_gallery.R"             # exploratory figure gallery (figures/eda/, figures/eda_gallery.pdf)
)

start_time <- Sys.time()
for (s in scripts) {
  path <- file.path(PIPELINE_ROOT, "R", s)
  cat(sprintf("\n===== Running %s =====\n", s))
  t0 <- Sys.time()
  local(source(path, local = TRUE))
  cat(sprintf("===== %s done in %.1fs =====\n", s, as.numeric(Sys.time() - t0, units = "secs")))
}
cat(sprintf("\nPipeline complete in %.1fs. Outputs under %s/{derived,figures,tables,logs}\n",
            as.numeric(Sys.time() - start_time, units = "secs"), PIPELINE_ROOT))
