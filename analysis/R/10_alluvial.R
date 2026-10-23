# 10_alluvial.R
# Alluvial plot of the experimental design and steady-state outcomes.
# Axes, left to right: Agent (classic ABM, GPT-5.4-mini, DeepSeek-V4-Flash)
#   -> Prompt (rule-based, calculator, reasoner, role-player) -> Outcome.
# Each arm carries 100 units, split by its steady-state outcome shares
# (mean over steps 51-100, then over the 5 seeds; five_seed batch).
# Flows are coloured by model (classic grey, GPT blue, DeepSeek orange) so the
# outcome column shows which model's agents end up where.
# PD outcomes: cooperate / defect. CV outcomes: quiet / active / jailed.
# Faceted by game. Two layouts: ncol = 2 (side by side), nrow = 2 (stacked).
# IC is left out: DeepSeek IC arms fail the parse gate and IC has 1 seed per arm.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(ggalluvial)

# ---- Steady-state outcome shares per arm ----------------------------------
ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE) %>%
  filter(batch == "five_seed", !excluded, step > n_steps / 2)
assert_that(nrow(ts) > 0, "No five_seed tail rows found")

run_means <- ts %>%
  group_by(game, agent_type, provider, run_id) %>%
  summarise(cooperate = mean(cooperation_rate), active = mean(rebellion_rate),
            jailed = mean(jailed_rate), quiet = mean(quiet_rate), .groups = "drop")

arm_means <- run_means %>%
  group_by(game, agent_type, provider) %>%
  summarise(n_seeds = n(), across(c(cooperate, active, jailed, quiet), mean), .groups = "drop")
N_SEEDS <- unique(arm_means$n_seeds)
assert_that(length(N_SEEDS) == 1, "Arms have different numbers of seeds")
if (N_SEEDS != 5) log_msg("10", sprintf("WARNING: %d seeds per arm, not 5 (fine for the synthetic test set, not for the paper)", N_SEEDS))
assert_that(nrow(arm_means) == 14, "Expected 14 arms (2 games x [classic + 2 providers x 3 modes])")

# CV shares must sum to 1
cv_chk <- arm_means %>% filter(game == "cv") %>% mutate(s = active + jailed + quiet)
assert_that(all(abs(cv_chk$s - 1) < 1e-6), "CV quiet + active + jailed does not sum to 1")

# Cross-check against the headline numbers from 06_provider_divergence.R
tail_ref <- read_csv(file.path(OUT_DERIVED, "run_tail_metrics.csv"), show_col_types = FALSE) %>%
  group_by(game, agent_type, provider) %>%
  summarise(ref_coop = mean(cooperation_rate), ref_mob = mean(mobilized_share), .groups = "drop")
chk <- arm_means %>% left_join(tail_ref, by = c("game", "agent_type", "provider"))
assert_that(all(abs(chk$cooperate[chk$game == "pd"] - chk$ref_coop[chk$game == "pd"]) < 1e-6),
            "PD cooperation shares disagree with run_tail_metrics.csv")
assert_that(all(abs((chk$active + chk$jailed)[chk$game == "cv"] - chk$ref_mob[chk$game == "cv"]) < 1e-6),
            "CV mobilized shares disagree with run_tail_metrics.csv")
log_msg("10", "Outcome shares match run_tail_metrics.csv for all 14 arms")

# ---- Long format: one row per arm x outcome --------------------------------
AGENT_LAB  <- c(mock = "Classic\nABM", openai = "GPT-5.4-\nmini", deepseek = "DeepSeek-\nV4-Flash")
PROMPT_LAB <- c(classic = "Rule-\nbased", calculator = "Calculator", reasoner = "Reasoner", roleplayer = "Role-\nplayer")
MODEL_LAB  <- c(mock = "Classic", openai = "GPT-5.4-mini", deepseek = "DeepSeek-V4-Flash")
# RColorBrewer Dark2: green (1), orange (2), slate blue (3)
MODEL_COL  <- c(Classic = "#1B9E77", `GPT-5.4-mini` = "#7570B3", `DeepSeek-V4-Flash` = "#D95F02")
SHORT_AGENT  <- c(mock = "Classic", openai = "GPT", deepseek = "DS")
SHORT_PROMPT <- c(classic = "rule", calculator = "calc", reasoner = "reas", roleplayer = "role")
GAME_LAB <- c(pd = "Spatial prisoner's dilemma", cv = "Civil violence (Epstein Model I)")
OUTCOME_LEVELS <- c("Cooperate", "Defect", "Quiet", "Active", "Jailed")

pd_long <- arm_means %>% filter(game == "pd") %>%
  transmute(game, agent_type, provider, Cooperate = cooperate, Defect = 1 - cooperate) %>%
  pivot_longer(c(Cooperate, Defect), names_to = "outcome", values_to = "share")
cv_long <- arm_means %>% filter(game == "cv") %>%
  transmute(game, agent_type, provider, Quiet = quiet, Active = active, Jailed = jailed) %>%
  pivot_longer(c(Quiet, Active, Jailed), names_to = "outcome", values_to = "share")

alluv <- bind_rows(pd_long, cv_long) %>%
  mutate(pct = 100 * share,
         agent = factor(AGENT_LAB[provider], levels = AGENT_LAB),
         prompt = factor(PROMPT_LAB[agent_type], levels = PROMPT_LAB),
         model = factor(MODEL_LAB[provider], levels = MODEL_LAB),
         outcome = factor(outcome, levels = OUTCOME_LEVELS),
         game = factor(GAME_LAB[game], levels = GAME_LAB),
         )

# Each arm must contribute exactly 100 units
arm_tot <- alluv %>% group_by(game, agent, prompt) %>% summarise(t = sum(pct), .groups = "drop")
assert_that(all(abs(arm_tot$t - 100) < 1e-6), "An arm does not sum to 100%")
write_csv(alluv %>% select(game, agent, prompt, outcome, pct), file.path(OUT_DERIVED, "alluvial_shares.csv"))

# ---- Plot -------------------------------------------------------------------
W <- 0.30  # stratum width
make_alluvial <- function(facet_layout) {
  ggplot(alluv, aes(axis1 = agent, axis2 = prompt, axis3 = outcome, y = pct)) +
    geom_alluvium(aes(fill = model), width = W, alpha = 0.75, colour = NA,
                  curve_type = "sigmoid", knot.pos = 0.4) +
    geom_stratum(width = W, fill = "grey96", colour = "grey70", linewidth = 0.25) +
    # stratum names: inside the box when tall enough, otherwise just right of it
    geom_text(stat = "stratum",
              aes(label = after_stat(ifelse(ymax - ymin >= 25, as.character(stratum), ""))),
              size = 2.6, lineheight = 0.85, fontface = "bold", colour = "grey15") +
    geom_text(stat = "stratum",
              aes(label = after_stat(ifelse(ymax - ymin < 25, as.character(stratum), ""))),
              size = 2.3, hjust = 0, nudge_x = W / 2 + 0.03, fontface = "bold") +
    scale_x_discrete(limits = c("Agent", "Prompt", "Outcome"), expand = expansion(add = c(0.2, 0.35))) +
    scale_fill_manual(values = MODEL_COL, name = NULL) +
    facet_layout +
    # No embedded caption: the figure caption lives in main.tex (Fig. 1, fig:design)
    labs(y = NULL, x = NULL) +
    theme_minimal(base_size = 10) +
    theme(panel.grid = element_blank(), strip.text = element_text(face = "bold", size = 11),
          legend.position = "bottom", legend.text = element_text(size = 9),
          axis.text.y = element_blank(), axis.text.x = element_text(size = 9, face = "bold"))
}

p_cols <- make_alluvial(facet_wrap(~ game, ncol = 2))
p_rows <- make_alluvial(facet_wrap(~ game, nrow = 2))
ggsave(file.path(OUT_FIGURES, "alluvial_design_2col.pdf"), p_cols, width = 12, height = 6.5, bg = "white")
ggsave(file.path(OUT_FIGURES, "alluvial_design_2col.png"), p_cols, width = 12, height = 6.5, dpi = 200, bg = "white")
ggsave(file.path(OUT_FIGURES, "alluvial_design_2row.pdf"), p_rows, width = 8, height = 11, bg = "white")
ggsave(file.path(OUT_FIGURES, "alluvial_design_2row.png"), p_rows, width = 8, height = 11, dpi = 200, bg = "white")
log_msg("10", "Wrote alluvial_design_2col and alluvial_design_2row (pdf + png)")
