# 04_pd_strategies.R
# Fig 2 (PD panel): classify each PD agent's revealed strategy from its
# agent_panel.parquet action history, then tabulate the strategy-type
# distribution by mode x provider.
#
# Classification rules (concrete, hard-coded thresholds -- documented here,
# not buried in a config file):
#   always_cooperate    : COOPERATE in every step after burn-in (step > 1)
#   always_defect        : DEFECT in every step after burn-in
#   imitator              : action(t) == best_neighbor_action(t-1) on > 80% of steps
#                            (matches the imitation-dynamics baseline in the
#                            Methods section)
#   conditional_cooperator: P(cooperate | neighbor coop_rate >= 0.5) minus
#                            P(cooperate | neighbor coop_rate < 0.5) > 0.2
#                            i.e. a tit-for-tat-like responsiveness to the
#                            local cooperation rate
#   other                  : none of the above

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(nanoparquet)  # lighter than arrow; read_parquet() has a compatible interface
library(purrr)
library(readr)
library(tidyr)

run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
pd_runs <- run_index %>% filter(game == "pd", !excluded)

log_msg("04", sprintf("Classifying PD strategies across %d runs", nrow(pd_runs)))

BURN_IN <- 1L
IMITATION_THRESHOLD <- 0.80
CONDITIONAL_COOP_GAP <- 0.20

classify_run <- function(run_dir, run_id, agent_type, provider, batch) {
  panel_path <- file.path(run_dir, "agent_panel.parquet")
  if (!file.exists(panel_path)) {
    log_msg("04", "WARNING: no agent_panel.parquet for ", run_id, " -- skipping strategy classification")
    return(NULL)
  }
  panel <- read_parquet(panel_path)
  assert_columns(panel, EXPECTED_COLS$agent_panel_pd, sprintf("agent_panel_pd/%s", run_id))

  panel <- panel %>%
    filter(step > BURN_IN) %>%
    arrange(agent_id, step) %>%
    group_by(agent_id) %>%
    mutate(lag_best_neighbor_action = lag(best_neighbor_action)) %>%
    ungroup()

  agent_stats <- panel %>%
    group_by(agent_id) %>%
    summarise(
      n_steps = n(),
      coop_rate_own = mean(action == "COOPERATE"),
      imitation_match = mean(action == lag_best_neighbor_action, na.rm = TRUE),
      p_coop_high_nbr = mean(action[coop_rate >= 0.5] == "COOPERATE"),
      p_coop_low_nbr = mean(action[coop_rate < 0.5] == "COOPERATE"),
      .groups = "drop"
    ) %>%
    mutate(
      conditional_gap = p_coop_high_nbr - p_coop_low_nbr,
      strategy = case_when(
        coop_rate_own >= 0.99 ~ "always_cooperate",
        coop_rate_own <= 0.01 ~ "always_defect",
        imitation_match >= IMITATION_THRESHOLD ~ "imitator",
        !is.na(conditional_gap) & conditional_gap > CONDITIONAL_COOP_GAP ~ "conditional_cooperator",
        TRUE ~ "other"
      )
    )

  agent_stats$run_id <- run_id
  agent_stats$agent_type <- agent_type
  agent_stats$provider <- provider
  agent_stats$batch <- batch
  agent_stats
}

strategy_by_agent <- pd_runs %>%
  select(run_dir, run_id, agent_type, provider, batch) %>%
  pmap_dfr(classify_run)

assert_that(nrow(strategy_by_agent) > 0, "PD strategy classification produced no rows")

write.csv(strategy_by_agent, file.path(OUT_DERIVED, "pd_strategy_by_agent.csv"), row.names = FALSE)

strategy_distribution <- strategy_by_agent %>%
  count(agent_type, provider, batch, strategy) %>%
  group_by(agent_type, provider, batch) %>%
  mutate(share = n / sum(n)) %>%
  ungroup()

write.csv(strategy_distribution, file.path(OUT_DERIVED, "pd_strategy_distribution.csv"), row.names = FALSE)
log_msg("04", sprintf("Wrote pd_strategy_by_agent.csv (%d agents) and pd_strategy_distribution.csv (%d rows)",
                       nrow(strategy_by_agent), nrow(strategy_distribution)))
