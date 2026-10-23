# 00_config.R
# Paths, constants, and shared helpers. Every R script sources this file; run_all.R runs them in order.
# Required R packages (checked by run_all.R): dplyr, tidyr, purrr, readr, tibble, jsonlite, nanoparquet,
# ggplot2, patchwork, scales, ggalluvial, RColorBrewer. Versions used for each run are written to
# logs/sessionInfo.txt.

SEED <- 123L
set.seed(SEED)

# ---- Paths -------------------------------------------------------------
# PIPELINE_ROOT = this repository (run from inside it). ABM_ROOT = the folder holding the raw simulation
# runs (one_simulation/, five_simulation/, second_step/). It defaults to two levels up, which is right when
# the repository sits at ABM/paper/llm_abm_pipeline/; otherwise set ABM_ROOT=/path/to/ABM.
PIPELINE_ROOT <- normalizePath(Sys.getenv("PIPELINE_ROOT", unset = "."), mustWork = TRUE)
ABM_ROOT <- normalizePath(Sys.getenv("ABM_ROOT", unset = file.path(PIPELINE_ROOT, "..", "..")), mustWork = TRUE)

RUN_BATCHES <- list(
  one_seed       = file.path(ABM_ROOT, "one_simulation", "outputs_agent_info"),
  five_seed      = file.path(ABM_ROOT, "five_simulation", "PD_CV_outputs"),
  cv_chinese     = file.path(ABM_ROOT, "second_step", "CV_Chinese_outputs"),
  cv_500step     = file.path(ABM_ROOT, "second_step", "CV_500steps"),
  ic_text_signal = file.path(ABM_ROOT, "second_step", "IC_text_signal_outputs")
)

OUT_DERIVED <- file.path(PIPELINE_ROOT, "derived")
OUT_FIGURES <- file.path(PIPELINE_ROOT, "figures")
OUT_TABLES  <- file.path(PIPELINE_ROOT, "tables")
OUT_LOGS    <- file.path(PIPELINE_ROOT, "logs")

dir.create(OUT_DERIVED, showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_FIGURES, showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_TABLES,  showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_LOGS,    showWarnings = FALSE, recursive = TRUE)

# ---- Data-quality gate ---------------------------------------------------
PARSE_FAILURE_EXCLUSION_THRESHOLD <- 0.05   # a run is excluded if more than 5% of its LLM calls fail to parse
STEADY_STATE_STEPS <- 51:100                 # steady state = mean over steps 51-100 of a 100-step run

# PDF device: cairo_pdf embeds fonts cleanly but needs Cairo, which some R builds on macOS lack; fall back to pdf()
PDF_DEVICE <- if (isTRUE(capabilities("cairo"))) grDevices::cairo_pdf else grDevices::pdf

# ---- Known column contracts (from ABM/one_simulation/readme_OUTPUT.md) --
EXPECTED_COLS <- list(
  time_series_pd = c("cooperation_rate", "mean_payoff", "spatial_clustering"),
  time_series_cv = c("rebellion_rate", "quiet_rate", "jailed_rate", "arrests_this_step"),
  time_series_ic = c("war_frequency", "mean_payoff", "welfare_loss", "bluff_rate", "bluff_success_rate"),
  agent_panel_cv = c("step", "activation_idx", "agent_id", "agent_type", "x", "y",
                      "state_before", "action", "moved_to_x", "moved_to_y", "hardship",
                      "risk_aversion", "legitimacy", "grievance", "cops_nearby",
                      "actives_nearby", "quiets_nearby", "arrest_prob", "jail_term",
                      "confidence", "parse_failed"),
  agent_panel_pd = c("step", "agent_id", "agent_type", "x", "y", "action_before", "action",
                      "payoff", "num_neighbors", "coop_count", "defect_count", "coop_rate",
                      "best_neighbor_action"),
  arrests_cv     = c("step", "activation_idx", "cop_id", "cop_x", "cop_y",
                      "target_id", "target_x", "target_y", "jail_term"),
  dyads_ic       = c("step", "proposer_id", "responder_id", "proposer_x", "proposer_y",
                      "responder_x", "responder_y", "proposer_signal", "responder_signal",
                      "demand", "threshold", "war", "winner_id", "proposer_payoff",
                      "responder_payoff", "proposer_true_capability", "proposer_true_war_cost",
                      "responder_true_capability", "responder_true_war_cost")
)

# ---- Helpers -------------------------------------------------------------

#' Fail loudly with a clear, specific message. Never let a script continue
#' on a violated assumption.
assert_that <- function(cond, msg) {
  if (!isTRUE(cond)) stop(msg, call. = FALSE)
}

assert_columns <- function(df, expected, label) {
  missing <- setdiff(expected, names(df))
  assert_that(
    length(missing) == 0,
    sprintf("[%s] missing expected columns: %s", label, paste(missing, collapse = ", "))
  )
}

#' Infer game from a run directory's basename ("pd_...", "cv_...", "ic_...").
#' This is the one place that relies on the run_experiment.py naming
#' convention "{game}_{arm}_seed{seed}_{hex}" -- if that convention ever
#' changes, this is the only line that needs to change.
infer_game <- function(run_dir_name) {
  prefix <- sub("_.*$", "", run_dir_name)
  assert_that(prefix %in% c("pd", "cv", "ic"),
              sprintf("Cannot infer game from run directory name: %s", run_dir_name))
  prefix
}

log_msg <- function(script, ...) {
  msg <- sprintf("[%s] %s", script, paste0(...))
  cat(msg, "\n")
  cat(msg, "\n", file = file.path(OUT_LOGS, paste0(script, ".log")), append = TRUE)
}
