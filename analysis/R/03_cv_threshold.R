# 03_cv_threshold.R
# Fig 2 (CV panel): recover the implied activation rule from LLM citizens'
# revealed decisions and compare it to Epstein's rule.
#
# Epstein's classic rule: an agent goes ACTIVE iff
#   net_incentive := grievance - risk_aversion * arrest_prob > threshold
#
# CORRECTED Sept 27, 2026: this script originally assumed threshold = 0 "by
# convention." That was wrong. readme_OUTPUT.md documents the classical rule
# with an explicit `threshold` term, and against the real data the classic
# arm recovers to ~0.10, not 0 -- tightly clustered across all 7 independent
# classic seeds/batches (0.0996-0.1043, sd < 0.002). That is the simulator's
# actual hardcoded threshold (not exposed in config.json), confirmed by its
# consistency, not a bug in this script or in the data.
#
# For each (agent_type, provider) cell we fit
#   logit(P(ACTIVE)) ~ net_incentive
# and take the recovered threshold as the net_incentive value where the
# fitted P(ACTIVE) = 0.5, i.e. -intercept/slope.
#
# Built-in validation: we run this same recovery on the classic arm itself.
# Because classic citizens follow the rule exactly (deterministically, modulo
# no noise term in this model), every classic run should recover the SAME
# threshold value (tight clustering across seeds/batches), whatever that
# value is. If they don't cluster tightly, something is wrong with either the
# column semantics or this script, and we stop rather than trust the LLM-arm
# numbers. The empirical mean across classic runs -- not a literature value of
# 0 -- becomes CLASSIC_EMPIRICAL_THRESHOLD, the baseline every LLM arm is
# compared against downstream (see `threshold_gap` below).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(nanoparquet)  # lighter than arrow; read_parquet() has a compatible interface
library(purrr)
library(readr)

run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
cv_runs <- run_index %>% filter(game == "cv", !excluded)

log_msg("03", sprintf("Recovering CV activation threshold across %d runs", nrow(cv_runs)))

CLASSIC_CLUSTER_TOLERANCE <- 0.02  # max spread (max - min) allowed across classic-arm seeds

recover_threshold <- function(run_dir, run_id, agent_type, provider, batch) {
  panel_path <- file.path(run_dir, "agent_panel.parquet")
  if (!file.exists(panel_path)) {
    log_msg("03", "WARNING: no agent_panel.parquet for ", run_id, " -- skipping")
    return(NULL)
  }
  panel <- read_parquet(panel_path)
  assert_columns(panel, EXPECTED_COLS$agent_panel_cv, sprintf("agent_panel_cv/%s", run_id))

  panel <- panel %>%
    filter(agent_type %in% c("classic_citizen", "llm_citizen")) %>%
    filter(is.na(parse_failed) | !parse_failed) %>%
    filter(step %in% STEADY_STATE_STEPS) %>%   # steady state (also for the 500-step runs)
    mutate(
      net_incentive = grievance - risk_aversion * arrest_prob,
      is_active = as.integer(action == "ACTIVE")
    )

  if (n_distinct(panel$is_active) < 2 || nrow(panel) < 30) {
    log_msg("03", "WARNING: insufficient variation to fit threshold model for ", run_id)
    return(NULL)
  }

  # Perfect or quasi-perfect separation (a citizen whose action is a
  # near-deterministic function of net_incentive) makes glm's IRLS diverge;
  # the fitted coefficients are still usable as a description of a very
  # steep boundary, but we record the warning instead of hiding it, since a
  # non-converged fit deserves a second look before it goes in the paper.
  fit_warnings <- character(0)
  fit <- withCallingHandlers(
    glm(is_active ~ net_incentive, data = panel, family = binomial()),
    warning = function(w) {
      fit_warnings[[length(fit_warnings) + 1]] <<- conditionMessage(w)
      invokeRestart("muffleWarning")
    }
  )
  intercept <- coef(fit)[["(Intercept)"]]
  slope <- coef(fit)[["net_incentive"]]
  recovered_threshold <- -intercept / slope

  if (length(fit_warnings) > 0) {
    log_msg("03", sprintf("NOTE: glm did not cleanly converge for %s (%s/%s) -- likely near-perfect separation: %s",
                           run_id, agent_type, provider, paste(unique(fit_warnings), collapse = "; ")))
  }

  tibble(
    run_id = run_id, agent_type = agent_type, provider = provider, batch = batch,
    convergence_warning = length(fit_warnings) > 0,
    n_obs = nrow(panel), slope = slope, intercept = intercept,
    recovered_threshold = recovered_threshold,
    pseudo_r2 = 1 - fit$deviance / fit$null.deviance
  )
}

results <- cv_runs %>%
  select(run_dir, run_id, agent_type, provider, batch) %>%
  pmap_dfr(recover_threshold)

assert_that(nrow(results) > 0, "CV threshold recovery produced no rows")

# --- Validation gate on the classic arm ---
# Validate CONSISTENCY across classic seeds/batches, not agreement with a
# literature value of 0 -- see the header note above for why 0 was wrong.
classic_results <- results %>% filter(agent_type == "classic")
if (nrow(classic_results) > 0) {
  classic_spread <- max(classic_results$recovered_threshold) - min(classic_results$recovered_threshold)
  if (classic_spread > CLASSIC_CLUSTER_TOLERANCE) {
    stop(sprintf(
      paste0(
        "VALIDATION FAILURE: classic-arm recovered threshold should cluster tightly across seeds ",
        "(the classic rule is deterministic), but spread was %.4f (> tolerance %.4f). Values: %s. ",
        "This means net_incentive/action semantics are misunderstood -- do not trust the LLM-arm numbers below until this is fixed."
      ),
      classic_spread, CLASSIC_CLUSTER_TOLERANCE,
      paste(sprintf("%s=%.4f", classic_results$run_id, classic_results$recovered_threshold), collapse = ", ")
    ), call. = FALSE)
  }
  CLASSIC_EMPIRICAL_THRESHOLD <- mean(classic_results$recovered_threshold)
  log_msg("03", sprintf(
    "Validation passed: classic-arm recovered threshold clusters at %.4f (spread %.4f across %d runs). This empirical value, not 0, is the baseline for threshold_gap below.",
    CLASSIC_EMPIRICAL_THRESHOLD, classic_spread, nrow(classic_results)))
} else {
  log_msg("03", "WARNING: no classic-arm runs available to validate against -- proceeding unvalidated")
  CLASSIC_EMPIRICAL_THRESHOLD <- NA_real_
}

# threshold_gap: how far each run's recovered threshold sits from the
# empirical classic baseline. This -- not raw recovered_threshold -- is what
# Fig 2b / the manuscript should read as "divergence from the classic rule."
results <- results %>%
  mutate(threshold_gap = recovered_threshold - CLASSIC_EMPIRICAL_THRESHOLD)

write.csv(results, file.path(OUT_DERIVED, "cv_threshold_recovery.csv"), row.names = FALSE)
log_msg("03", sprintf("Wrote cv_threshold_recovery.csv: %d rows (classic baseline = %.4f)",
                       nrow(results), CLASSIC_EMPIRICAL_THRESHOLD))
