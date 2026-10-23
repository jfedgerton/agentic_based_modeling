# 04_interstate_conflict.R
# Fig 4, table ED2 and the main-text crisis-game table: interstate-conflict out-of-sample test. Because ic_*_deepseek runs
# all exceed the 5% parse-failure exclusion gate (51-61% in this dataset --
# see run_index.csv), filtering on `!excluded` automatically restricts the
# main-text IC analysis to GPT, matching the outline's stated design. Nothing
# here special-cases "drop DeepSeek" -- it falls out of the same gate every
# other exclusion uses.
#
# Two outputs:
#   ic_dynamics.csv        -- classic vs calculator/reasoner/roleplayer (GPT),
#                              war_frequency / welfare_loss / bluff_rate /
#                              bluff_success_rate, from the one-seed batch.
#   ic_text_signal_ablation.csv -- the free-form-text-signal role-player
#                              ablation (second_step/IC_text_signal_outputs),
#                              reported separately since it's a signal-form
#                              variant, not part of the core 3-arm design.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)
library(tidyr)

run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
panel_ts  <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)

IC_METRICS <- c("war_frequency", "mean_payoff", "welfare_loss", "bluff_rate", "bluff_success_rate")

ic_core <- panel_ts %>%
  filter(game == "ic", batch == "one_seed") %>%
  select(run_id, agent_type, provider, step, all_of(IC_METRICS))

assert_that(nrow(ic_core) > 0, "No IC core (one_seed) data found")
llm_providers_remaining <- unique(ic_core$provider[ic_core$agent_type != "classic"])
if (!(length(llm_providers_remaining) == 1 && llm_providers_remaining == "openai")) {
  log_msg("04", sprintf(
    "NOTE: expected the parse-failure gate to leave IC main-text arms GPT-only, but providers remaining are: %s -- re-check run_index.csv exclusions before trusting this table",
    paste(llm_providers_remaining, collapse = ", ")))
}

n_classic <- n_distinct(ic_core$run_id[ic_core$agent_type == "classic"])
assert_that(n_classic == 1, sprintf("Expected one bluffing-enabled classic crisis-game run, found %d", n_classic))
assert_that(all(table(ic_core$run_id) == 100), "Every crisis-game run should have 100 steps")

# Steady state, steps 51-100, as in the canonical games. Brackets in the table are the lowest and highest
# per-step values over those steps: within-run fluctuation, not uncertainty across seeds (one run per arm).
ic_dynamics_summary <- ic_core %>%
  filter(step %in% STEADY_STATE_STEPS) %>%
  group_by(agent_type, provider) %>%
  summarise(across(all_of(IC_METRICS), list(mean = ~mean(.x, na.rm = TRUE), lo = ~min(.x, na.rm = TRUE),
                                            hi = ~max(.x, na.rm = TRUE)), .names = "{.col}_{.fn}"),
            n_steps = n(), .groups = "drop") %>%
  rename_with(~sub("_mean$", "", .x), ends_with("_mean"))
assert_that(all(ic_dynamics_summary$n_steps == 50), "Steady-state window should be 50 steps per arm")

write.csv(ic_core, file.path(OUT_DERIVED, "ic_dynamics_timeseries.csv"), row.names = FALSE)
write.csv(ic_dynamics_summary, file.path(OUT_DERIVED, "ic_dynamics_summary.csv"), row.names = FALSE)
log_msg("04", sprintf("Wrote IC core dynamics: %d timeseries rows, %d summary rows (arms: %s)",
                       nrow(ic_core), nrow(ic_dynamics_summary),
                       paste(unique(ic_core$agent_type), collapse = ", ")))

# Main-text table: mean [lowest, highest per-step value], steps 51-100
IC_ARMS <- c(classic = "Rule-based benchmark", calculator = "GPT calculator", reasoner = "GPT reasoner",
             roleplayer = "GPT role-player")
assert_that(setequal(ic_dynamics_summary$agent_type, names(IC_ARMS)), "Unexpected crisis-game arms")
cell <- function(m, lo, hi, d) sprintf(paste0("%.", d, "f [%.", d, "f, %.", d, "f]"), m, lo, hi)
tab_rows <- sapply(names(IC_ARMS), function(a) {
  r <- ic_dynamics_summary %>% filter(agent_type == a)
  sprintf("%s & %s & %s & %s \\\\", IC_ARMS[[a]], cell(r$war_frequency, r$war_frequency_lo, r$war_frequency_hi, 2),
          cell(r$mean_payoff, r$mean_payoff_lo, r$mean_payoff_hi, 2), cell(r$bluff_rate, r$bluff_rate_lo, r$bluff_rate_hi, 2))
})
writeLines(c(
  "\\begin{table}[t]", "\\centering",
  "\\caption{\\textbf{Outcomes in the crisis-bargaining game.} Means over steps 51--100 of a single run per arm; brackets give the lowest and highest per-step values over those steps, which describe within-run fluctuation rather than uncertainty across seeds. The rule-based benchmark has bluffing enabled, as do all GPT runs. DeepSeek runs are excluded (Supplementary Materials).}",
  "\\label{tab:ic}", "\\begin{tabular}{lccc}", "\\hline",
  "Arm & War frequency & Mean payoff & Bluff rate \\\\", "\\hline", tab_rows, "\\hline", "\\end{tabular}", "\\end{table}"),
  file.path(OUT_TABLES, "table_ic.tex"))
log_msg("04", "Wrote tables/table_ic.tex")

# --- Text-signal ablation ---
ic_text <- panel_ts %>%
  filter(game == "ic", batch == "ic_text_signal") %>%
  select(run_id, agent_type, provider, step, all_of(IC_METRICS))

if (nrow(ic_text) == 0) {
  log_msg("04", "WARNING: no ic_text_signal data survived exclusion -- ablation table will be empty")
} else {
  log_msg("04", sprintf("IC text-signal ablation: providers retained after exclusion gate: %s",
                         paste(unique(ic_text$provider), collapse = ", ")))
}

ic_text_summary <- ic_text %>%
  filter(step %in% STEADY_STATE_STEPS) %>%   # same steady-state window as the core arms
  group_by(agent_type, provider) %>%
  summarise(across(all_of(IC_METRICS), ~mean(.x, na.rm = TRUE)), n_steps = n(), .groups = "drop")

write.csv(ic_text_summary, file.path(OUT_DERIVED, "ic_text_signal_ablation.csv"), row.names = FALSE)
log_msg("04", sprintf("Wrote ic_text_signal_ablation.csv: %d rows", nrow(ic_text_summary)))

# Cross-reference against categorical-signal roleplayer (same provider) from
# the core batch, so the ablation table is directly comparable in the writeup.
categorical_rp <- ic_dynamics_summary %>% filter(agent_type == "roleplayer") %>% mutate(signal_form = "categorical")
text_rp <- ic_text_summary %>% mutate(signal_form = "free_form_text")
signal_comparison <- bind_rows(categorical_rp, text_rp)
write.csv(signal_comparison, file.path(OUT_DERIVED, "ic_signal_form_comparison.csv"), row.names = FALSE)
log_msg("04", sprintf("Wrote ic_signal_form_comparison.csv: %d rows", nrow(signal_comparison)))
