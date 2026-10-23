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

# ---- Bluff rate corrected for the calculator signal labels ----------------------------------------------------------
# The calculator prompt offers "S1 (low value_a), S2 (medium value_a), or S3 (high value_a)", but the simulation code at
# tag v1.0-experiments stores them as S1 -> STRONG, S2 -> MODERATE, S3 -> WEAK (_SIGNAL_SURFACE_TO_CANONICAL in
# src/agents/llm_interstate_conflict.py). Decide prompts reverse the same mapping, so calculator agents saw the labels
# they chose and war frequency and payoffs are unaffected. The logged bluff_rate and bluff_success_rate (a weak agent,
# true capability < 0.5, whose stored signal is STRONG) therefore count calculator agents who signal LOW.
# Both are recomputed here from dyads.parquet using the meaning of the broadcast signal (calculator: stored WEAK = S3 =
# high; every other arm: stored STRONG = high). Check: with the stored labels the recomputation must reproduce the
# logged values exactly, for every run and step.
ic_runs <- run_index %>% filter(game == "ic", batch == "one_seed", !excluded)
assert_that(setequal(ic_runs$run_id, unique(ic_core$run_id)), "Crisis-game runs in run_index and panel differ")
bluff_fix <- list()
for (i in seq_len(nrow(ic_runs))) {
  run <- ic_runs[i, ]
  dy <- nanoparquet::read_parquet(file.path(run$run_dir, "dyads.parquet"))
  sides <- bind_rows(
    tibble(step = dy$step, id = dy$proposer_id, sig = dy$proposer_signal, cap = dy$proposer_true_capability, war = dy$war),
    tibble(step = dy$step, id = dy$responder_id, sig = dy$responder_signal, cap = dy$responder_true_capability, war = dy$war))
  agents <- distinct(sides, step, id, sig, cap)
  assert_that(nrow(agents) == nrow(distinct(sides, step, id)), sprintf("%s: an agent has two signals in one step", run$run_id))
  high_stored <- "STRONG"
  high_meant  <- if (run$agent_type == "calculator") "WEAK" else "STRONG"
  rate <- function(x) if (length(x) == 0) 0 else mean(x)
  per_step <- agents %>% group_by(step) %>%
    summarise(n_agents = n(),
              bluff_check = rate(sig[cap < 0.5] == high_stored),
              bluff_fixed = rate(sig[cap < 0.5] == high_meant), .groups = "drop") %>%
    left_join(sides %>% group_by(step) %>%
                summarise(success_check = rate(!war[cap < 0.5 & sig == high_stored]),
                          success_fixed = rate(!war[cap < 0.5 & sig == high_meant]), .groups = "drop"), by = "step") %>%
    mutate(run_id = run$run_id)
  assert_that(n_distinct(per_step$n_agents) == 1 && nrow(per_step) == 100,
              sprintf("%s: agent count varies across steps or steps are missing in dyads.parquet", run$run_id))
  bluff_fix[[i]] <- per_step
}
bluff_fix <- bind_rows(bluff_fix)
ic_core <- ic_core %>% left_join(bluff_fix, by = c("run_id", "step"))
assert_that(!anyNA(ic_core$bluff_fixed), "Corrected bluff rate missing for some crisis-game steps")
assert_that(max(abs(ic_core$bluff_rate - ic_core$bluff_check)) < 1e-9,
            "Recomputed bluff rate with stored labels does not reproduce the logged bluff_rate")
assert_that(max(abs(ic_core$bluff_success_rate - ic_core$success_check)) < 1e-9,
            "Recomputed bluff success with stored labels does not reproduce the logged bluff_success_rate")
ic_core <- ic_core %>%
  mutate(bluff_rate_logged = bluff_rate, bluff_success_rate_logged = bluff_success_rate,
         bluff_rate = bluff_fixed, bluff_success_rate = success_fixed) %>%
  select(-n_agents, -bluff_check, -bluff_fixed, -success_check, -success_fixed)
assert_that(all(abs(ic_core$bluff_rate - ic_core$bluff_rate_logged) < 1e-9 | ic_core$agent_type == "calculator"),
            "Bluff correction changed an arm other than the calculator")
log_msg("04", sprintf("Calculator bluff rate, steps 51-100: logged %.3f, corrected %.3f",
                      mean(ic_core$bluff_rate_logged[ic_core$agent_type == "calculator" & ic_core$step %in% STEADY_STATE_STEPS]),
                      mean(ic_core$bluff_rate[ic_core$agent_type == "calculator" & ic_core$step %in% STEADY_STATE_STEPS])))

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
