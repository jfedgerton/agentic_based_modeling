# 03_convergence_metrics.R
# Fig 1 / Table 1 input: for every (game, agent_type, provider) cell, compare
# its trajectory of the game's primary outcome metric against the matching
# classic baseline. Produces trajectory RMSE always, and a seed-level KS
# distance + permutation-test p-value (seed 123) wherever both sides have
# more than one seed to draw a distribution from.
#
# Explicit, hard-coded baseline map: some batches (CV Chinese, IC text-signal)
# don't ship their own classic run, so they borrow one from elsewhere. This is
# written out rather than inferred so it's auditable at a glance.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)
library(tidyr)
library(purrr)

panel_ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)

PRIMARY_METRIC <- c(pd = "cooperation_rate", cv = "rebellion_rate", ic = "war_frequency")

# batch -> which batch's classic run to use as baseline for that batch's LLM runs
BASELINE_BATCH_MAP <- c(
  one_seed       = "one_seed",
  five_seed      = "five_seed",
  cv_chinese     = "five_seed",   # classic is language-invariant (rule-based); borrow the multi-seed English classic
  cv_500step     = "cv_500step",  # has its own classic at 500 steps
  ic_text_signal = "one_seed"     # borrow the one-seed IC classic (bluffing-enabled variant)
)

#' Two-sample permutation test on the difference in means, seed 123.
#' Returns a two-sided p-value. NA if either group has < 2 observations.
permutation_test <- function(x, y, n_perm = 5000, seed = SEED) {
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  set.seed(seed)
  observed <- mean(x) - mean(y)
  pooled <- c(x, y)
  n_x <- length(x)
  n_total <- length(pooled)
  perm_diffs <- replicate(n_perm, {
    idx <- sample(n_total, n_x)
    mean(pooled[idx]) - mean(pooled[-idx])
  })
  mean(abs(perm_diffs) >= abs(observed))
}

#' KS distance between two numeric samples. NA if either has < 2 obs.
ks_distance <- function(x, y) {
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  suppressWarnings(ks.test(x, y)$statistic[[1]])
}

compute_cell_metrics <- function(game, agent_type, provider, batch, metric_col) {
  baseline_batch <- BASELINE_BATCH_MAP[[batch]]

  llm_rows <- panel_ts %>%
    filter(game == !!game, agent_type == !!agent_type, provider == !!provider, batch == !!batch)
  classic_rows <- panel_ts %>%
    filter(game == !!game, agent_type == "classic", batch == !!baseline_batch)

  if (nrow(llm_rows) == 0 || nrow(classic_rows) == 0) return(NULL)

  # Trajectory RMSE: mean-over-seeds value at each step, compared step-for-step
  # over the overlapping step range only.
  llm_traj <- llm_rows %>% group_by(step) %>% summarise(val = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  classic_traj <- classic_rows %>% group_by(step) %>% summarise(val = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop")
  joined <- inner_join(llm_traj, classic_traj, by = "step", suffix = c("_llm", "_classic"))
  assert_that(nrow(joined) > 0,
              sprintf("No overlapping steps between %s/%s/%s and its baseline (%s)", game, agent_type, provider, baseline_batch))
  rmse <- sqrt(mean((joined$val_llm - joined$val_classic)^2))

  # Seed-level distributions: one scalar (mean over steps) per seed/run.
  llm_seed_means <- llm_rows %>% group_by(run_id) %>% summarise(m = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop") %>% pull(m)
  classic_seed_means <- classic_rows %>% group_by(run_id) %>% summarise(m = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop") %>% pull(m)

  tibble(
    game = game, agent_type = agent_type, provider = provider, batch = batch,
    baseline_batch = baseline_batch, metric = metric_col,
    n_seeds_llm = length(llm_seed_means), n_seeds_classic = length(classic_seed_means),
    rmse = rmse,
    ks_distance = ks_distance(llm_seed_means, classic_seed_means),
    perm_p_value = permutation_test(llm_seed_means, classic_seed_means)
  )
}

cells <- panel_ts %>%
  filter(agent_type != "classic") %>%
  distinct(game, agent_type, provider, batch)

log_msg("03", sprintf("Computing convergence metrics for %d cells", nrow(cells)))

results <- cells %>%
  pmap_dfr(function(game, agent_type, provider, batch) {
    metric_col <- PRIMARY_METRIC[[game]]
    compute_cell_metrics(game, agent_type, provider, batch, metric_col)
  })

assert_that(nrow(results) > 0, "No convergence metrics computed -- check panel_timeseries.csv")

write.csv(results, file.path(OUT_DERIVED, "convergence_metrics.csv"), row.names = FALSE)
log_msg("03", sprintf("Wrote convergence_metrics.csv: %d rows", nrow(results)))
log_msg("03", sprintf("  n_seeds_llm==1 rows (no permutation test possible): %d", sum(results$n_seeds_llm == 1)))
