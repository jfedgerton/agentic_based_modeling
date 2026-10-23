# 07_decision_rules.R
# Mechanism for the divergence result: WHAT decision rule is each provider's
# agent actually using? Agent-level, from agent_panel.parquet, five_seed batch.
#
#   PD: empirical response function P(COOPERATE at t | neighbourhood cooperation
#       rate coop_rate, own action_before). A payoff-maximising (classic)
#       agent defects at every neighbourhood composition -> flat line at 0.
#   CV: empirical activation curve P(ACTIVE | net_incentive), net_incentive =
#       grievance - risk_aversion * arrest_prob, among citizens eligible to act
#       (not jailed entering the step). The classic rule is a step at the
#       empirical threshold ~0.101 (see 03_cv_threshold.R).
#
# Binned (not modelled) on purpose: the point of the figure is to show the raw
# shape of each rule -- a step, a smeared step, or no dependence at all -- and a
# parametric fit would impose the shape we're trying to look at. Parse-failed
# decisions are dropped: they are the classical rule by construction
# (readme_OUTPUT.md line 139), so keeping them would pull LLM arms toward classic.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)
library(purrr)
library(nanoparquet)

run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
runs <- run_index %>% filter(batch == "five_seed", !excluded, game %in% c("pd", "cv"))
assert_that(nrow(runs) > 0, "No five_seed PD/CV runs found")
if (nrow(runs) != 70) log_msg("07", sprintf("WARNING: %d five_seed PD+CV runs, paper data has 70", nrow(runs)))

PD_BINS <- seq(0, 1, by = 0.125)          # neighbourhood coop rate; 8 neighbours -> 1/8 steps
CV_BINS <- seq(-0.6, 1.0, by = 0.05)      # net incentive

# ---- PD response function ---------------------------------------------------
pd_rows <- list()
for (i in which(runs$game == "pd")) {
  r <- runs[i, ]
  panel <- read_parquet(file.path(r$run_dir, "agent_panel.parquet"))
  assert_columns(panel, EXPECTED_COLS$agent_panel_pd, sprintf("agent_panel_pd/%s", r$run_id))
  if ("parse_failed" %in% names(panel)) panel <- panel %>% filter(is.na(parse_failed) | !parse_failed)
  pd_rows[[length(pd_rows) + 1]] <- panel %>%
    filter(step > 1) %>%   # step 1 has no history to respond to
    mutate(coop_bin = cut(coop_rate, breaks = PD_BINS, include.lowest = TRUE),
           cooperated = as.integer(action == "COOPERATE"),
           prev = if_else(action_before == "COOPERATE", "was cooperating", "was defecting")) %>%
    group_by(coop_bin, prev) %>%
    summarise(n = n(), n_coop = sum(cooperated), .groups = "drop") %>%
    mutate(run_id = r$run_id, agent_type = r$agent_type, provider = r$provider)
}
pd_resp <- bind_rows(pd_rows) %>%
  group_by(agent_type, provider, prev, coop_bin) %>%
  summarise(n = sum(n), p_coop = sum(n_coop) / sum(n), .groups = "drop") %>%
  mutate(bin_mid = (as.numeric(sub("[\\(\\[]([^,]+),.*", "\\1", coop_bin)) +
                      as.numeric(sub(".*,([^]]+)\\]", "\\1", coop_bin))) / 2)
assert_that(all(pd_resp$p_coop >= 0 & pd_resp$p_coop <= 1), "PD response probabilities out of [0,1]")
write.csv(pd_resp, file.path(OUT_DERIVED, "pd_response_function.csv"), row.names = FALSE)
log_msg("07", sprintf("PD response function: %d cells", nrow(pd_resp)))

# ---- PD transition rule: P(C at t | own action t-1, best neighbour's action t-1)
# This is the cleaner summary of the PD rule (Fig. 3a). The classic spatial-PD
# update is "imitate the best-scoring neighbour" (Nowak & May); a payoff-
# maximising agent defects in every cell. What GPT does instead is visible
# directly in the off-diagonal cells (e.g. re-extending cooperation after its
# own defection even when the best neighbour defected).
pd_trans_rows <- list()
for (i in which(runs$game == "pd")) {
  r <- runs[i, ]
  panel <- read_parquet(file.path(r$run_dir, "agent_panel.parquet"))
  if ("parse_failed" %in% names(panel)) panel <- panel %>% filter(is.na(parse_failed) | !parse_failed)
  pd_trans_rows[[length(pd_trans_rows) + 1]] <- panel %>%
    filter(step %in% STEADY_STATE_STEPS) %>%   # steady state, as everywhere else in the paper
    group_by(action_before, best_neighbor_action) %>%
    summarise(n = n(), n_coop = sum(action == "COOPERATE"), .groups = "drop") %>%
    mutate(run_id = r$run_id, agent_type = r$agent_type, provider = r$provider)
}
pd_trans <- bind_rows(pd_trans_rows) %>%
  group_by(agent_type, provider, action_before, best_neighbor_action) %>%
  summarise(n = sum(n), p_coop = sum(n_coop) / sum(n), .groups = "drop")
write.csv(pd_trans, file.path(OUT_DERIVED, "pd_transition_rule.csv"), row.names = FALSE)
log_msg("07", "PD transition rule:\n", paste(capture.output(print(as.data.frame(pd_trans), digits = 3)), collapse = "\n"))

# ---- CV activation curve ----------------------------------------------------
cv_rows <- list()
for (i in which(runs$game == "cv")) {
  r <- runs[i, ]
  panel <- read_parquet(file.path(r$run_dir, "agent_panel.parquet"))
  assert_columns(panel, EXPECTED_COLS$agent_panel_cv, sprintf("agent_panel_cv/%s", r$run_id))
  cv_rows[[length(cv_rows) + 1]] <- panel %>%
    filter(agent_type %in% c("classic_citizen", "llm_citizen"),
           is.na(parse_failed) | !parse_failed,
           state_before != "JAILED",
           step %in% STEADY_STATE_STEPS) %>%
    mutate(net_incentive = grievance - risk_aversion * arrest_prob,
           ni_bin = cut(net_incentive, breaks = CV_BINS, include.lowest = TRUE),
           active = as.integer(action == "ACTIVE")) %>%
    filter(!is.na(ni_bin)) %>%
    group_by(ni_bin) %>%
    summarise(n = n(), n_active = sum(active), .groups = "drop") %>%
    mutate(run_id = r$run_id, agent_type = r$agent_type, provider = r$provider)
}
cv_curve <- bind_rows(cv_rows) %>%
  group_by(agent_type, provider, ni_bin) %>%
  summarise(n = sum(n), p_active = sum(n_active) / sum(n), .groups = "drop") %>%
  mutate(bin_mid = (as.numeric(sub("[\\(\\[]([^,]+),.*", "\\1", ni_bin)) +
                      as.numeric(sub(".*,([^]]+)\\]", "\\1", ni_bin))) / 2)
assert_that(nrow(cv_curve) > 0, "CV activation curve is empty -- check state_before / action labels")
write.csv(cv_curve, file.path(OUT_DERIVED, "cv_activation_curve.csv"), row.names = FALSE)

# Sanity check the classic arm IS a step at ~0.10: below 0.05 nobody activates,
# above 0.15 (almost) everybody eligible does. If this fails, the state/action
# semantics are wrong and the LLM curves can't be trusted.
# Threshold taken from 05's empirical recovery (0.101 on the real data; 0 on the
# synthetic test set, which encodes the literal textbook rule).
classic_thr <- read_csv(file.path(OUT_DERIVED, "cv_threshold_recovery.csv"), show_col_types = FALSE) %>%
  filter(agent_type == "classic") %>% pull(recovered_threshold) %>% mean()
classic_lo <- cv_curve %>% filter(agent_type == "classic", bin_mid < classic_thr - 0.05, n >= 50)
classic_hi <- cv_curve %>% filter(agent_type == "classic", bin_mid > classic_thr + 0.05, n >= 50)
log_msg("07", sprintf("Classic CV activation (threshold %.3f): max P(active) below thr-0.05 = %.3f; min P(active) above thr+0.05 = %.3f",
                      classic_thr, max(classic_lo$p_active), if (nrow(classic_hi)) min(classic_hi$p_active) else NA))
assert_that(max(classic_lo$p_active) < 0.05, "Classic CV agents activate below the threshold -- semantics wrong")
log_msg("07", "Done.")
