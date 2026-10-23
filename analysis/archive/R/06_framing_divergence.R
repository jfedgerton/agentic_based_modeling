# 06_framing_divergence.R
# Fig 3: framing effects. For each (game, provider), contrast the role-player
# arm against reasoner and against calculator on the game's primary metric --
# identical payoffs/mechanics, only the prompt wrapper differs, so any
# divergence is attributable to framing alone.
#
# Same permutation-test machinery as 03_convergence_metrics.R (seed 123),
# reused here rather than re-implemented, since PD/CV five-seed data supports
# a real distributional test while IC (single seed) can only report a point
# effect with NA for the p-value.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)
library(purrr)

panel_ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)
PRIMARY_METRIC <- c(pd = "cooperation_rate", cv = "rebellion_rate", ic = "war_frequency")

permutation_test <- function(x, y, n_perm = 5000, seed = SEED) {
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  set.seed(seed)
  observed <- mean(x) - mean(y)
  pooled <- c(x, y); n_x <- length(x); n_total <- length(pooled)
  perm_diffs <- replicate(n_perm, {
    idx <- sample(n_total, n_x)
    mean(pooled[idx]) - mean(pooled[-idx])
  })
  mean(abs(perm_diffs) >= abs(observed))
}

#' Percentile bootstrap CI (95%) on the difference in means, seed 123.
#' NA/NA if either group has < 2 observations.
bootstrap_ci <- function(x, y, n_boot = 2000, seed = SEED) {
  if (length(x) < 2 || length(y) < 2) return(c(NA_real_, NA_real_))
  set.seed(seed)
  diffs <- replicate(n_boot, {
    xb <- sample(x, length(x), replace = TRUE)
    yb <- sample(y, length(y), replace = TRUE)
    mean(xb) - mean(yb)
  })
  unname(quantile(diffs, c(0.025, 0.975)))
}

seed_means <- function(rows, metric_col) {
  rows %>% group_by(run_id) %>% summarise(m = mean(.data[[metric_col]], na.rm = TRUE), .groups = "drop") %>% pull(m)
}

# Only compare within the same batch, so the contrast is never confounded by
# a different run generation/config vintage.
cells <- panel_ts %>%
  filter(agent_type == "roleplayer") %>%
  distinct(game, provider, batch)

log_msg("06", sprintf("Computing framing contrasts for %d (game, provider, batch) cells", nrow(cells)))

compute_contrast <- function(game, provider, batch, against) {
  metric_col <- PRIMARY_METRIC[[game]]
  rp_rows <- panel_ts %>% filter(game == !!game, provider == !!provider, batch == !!batch, agent_type == "roleplayer")
  vs_rows <- panel_ts %>% filter(game == !!game, provider == !!provider, batch == !!batch, agent_type == !!against)
  if (nrow(rp_rows) == 0 || nrow(vs_rows) == 0) return(NULL)

  rp_means <- seed_means(rp_rows, metric_col)
  vs_means <- seed_means(vs_rows, metric_col)
  ci <- bootstrap_ci(rp_means, vs_means)

  tibble(
    game = game, provider = provider, batch = batch, metric = metric_col,
    contrast = paste0("roleplayer_vs_", against),
    n_seeds_roleplayer = length(rp_means), n_seeds_other = length(vs_means),
    mean_roleplayer = mean(rp_means), mean_other = mean(vs_means),
    effect = mean(rp_means) - mean(vs_means),
    ci_low = ci[1], ci_high = ci[2],
    perm_p_value = permutation_test(rp_means, vs_means)
  )
}

results <- cells %>%
  pmap_dfr(function(game, provider, batch) {
    bind_rows(
      compute_contrast(game, provider, batch, "reasoner"),
      compute_contrast(game, provider, batch, "calculator")
    )
  })

assert_that(nrow(results) > 0, "Framing divergence produced no rows")

write.csv(results, file.path(OUT_DERIVED, "framing_divergence.csv"), row.names = FALSE)
log_msg("06", sprintf("Wrote framing_divergence.csv: %d rows (%d without a p-value due to n_seeds==1)",
                       nrow(results), sum(is.na(results$perm_p_value))))
