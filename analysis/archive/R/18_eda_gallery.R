# 18_eda_gallery.R
# Exploratory figure gallery for Jared to look through. Descriptive only:
# no new estimators, every plot is drawn from files already in derived/.
# Output: figures/eda/eda_XX_*.png and one combined figures/eda_gallery.pdf.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

OUT_EDA <- file.path(OUT_FIGURES, "eda")
dir.create(OUT_EDA, showWarnings = FALSE, recursive = TRUE)
rd <- function(f) read_csv(file.path(OUT_DERIVED, f), show_col_types = FALSE)

# ---- Shared styling (matches 14_figures_divergence.R) ----------------------
PROV_COL <- c(Classic = "grey45", GPT = "#0072B2", DeepSeek = "#D55E00")
PROV_LAB <- c(mock = "Classic", openai = "GPT", deepseek = "DeepSeek")
MODE_LAB <- c(classic = "Rule-based", calculator = "Calculator", reasoner = "Reasoner", roleplayer = "Role-player")
GAME_LAB <- c(pd = "Prisoner's dilemma", cv = "Civil violence", ic = "Interstate conflict")
ARM_LEVELS <- c("Classic\nrule-based",
                "GPT\ncalculator", "GPT\nreasoner", "GPT\nrole-player",
                "DeepSeek\ncalculator", "DeepSeek\nreasoner", "DeepSeek\nrole-player")
label_arms <- function(df) df %>% mutate(
  who = factor(PROV_LAB[provider], levels = names(PROV_COL)),
  mode = factor(MODE_LAB[agent_type], levels = MODE_LAB),
  arm = factor(ifelse(agent_type == "classic", "Classic\nrule-based",
                      paste0(PROV_LAB[provider], "\n", tolower(MODE_LAB[agent_type]))),
               levels = ARM_LEVELS))
theme_eda <- theme_minimal(base_size = 10) +
  theme(strip.text = element_text(face = "bold"), strip.placement = "outside",
        legend.position = "bottom", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 11),
        plot.caption = element_text(size = 7, hjust = 0, colour = "grey30"))
plots <- list()
save_eda <- function(p, name, w = 9, h = 5.5) {
  ggsave(file.path(OUT_EDA, paste0(name, ".png")), p, width = w, height = h, dpi = 170)
  plots[[name]] <<- p
  log_msg("18", "Wrote ", name)
}

# ---- Data -------------------------------------------------------------------
ts_all <- rd("panel_timeseries.csv") %>% filter(!excluded) %>%
  mutate(mobilized = rebellion_rate + jailed_rate)
ts5 <- ts_all %>% filter(batch == "five_seed") %>% label_arms()
N_RUNS <- n_distinct(ts5$run_id)
assert_that(N_RUNS %% 14 == 0, "five_seed runs are not a multiple of 14 arms")
if (N_RUNS != 70) log_msg("18", sprintf("WARNING: %d five_seed runs, not 70 (fine for the synthetic test set, not for the paper)", N_RUNS))
tail5 <- rd("run_tail_metrics.csv") %>% label_arms()
assert_that(nrow(tail5) == N_RUNS, "run_tail_metrics.csv rows do not match five_seed runs")

# ---- 01 Time series, primary outcome, every seed ----------------------------
ts_prim <- ts5 %>%
  mutate(y = ifelse(game == "pd", cooperation_rate, mobilized),
         game = factor(GAME_LAB[game], levels = GAME_LAB))
classic_rep <- ts_prim %>% filter(agent_type == "classic") %>% select(-mode) %>%
  tidyr::crossing(mode = factor(MODE_LAB[-1], levels = MODE_LAB))
ts_plot <- bind_rows(ts_prim %>% filter(agent_type != "classic"), classic_rep)
ts_mean <- ts_plot %>% group_by(game, mode, who, step) %>% summarise(y = mean(y), .groups = "drop")
p <- ggplot(ts_plot, aes(step, y, colour = who)) +
  geom_line(aes(group = run_id), alpha = 0.25, linewidth = 0.3) +
  geom_line(data = ts_mean, linewidth = 0.9) +
  geom_vline(xintercept = 50.5, linetype = "dotted", colour = "grey50") +
  facet_grid(game ~ mode, scales = "free_y") +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "Primary outcome over time, every seed (thin) and the 5-seed mean (thick)",
       x = "Step", y = "PD cooperation rate / CV mobilized share",
       caption = "Classic arm repeated in every column for reference. Dotted line marks the start of the steady-state window (steps 51-100).") +
  theme_eda
save_eda(p, "eda_01_timeseries_all_seeds", 10, 6)

# ---- 02 Steady state, one dot per seed ---------------------------------------
st <- tail5 %>% mutate(y = ifelse(game == "pd", cooperation_rate, mobilized_share),
                       game = factor(GAME_LAB[game], levels = GAME_LAB))
p <- ggplot(st, aes(arm, y, colour = who)) +
  geom_point(position = position_jitter(width = 0.12, height = 0, seed = SEED), size = 1.8, alpha = 0.8) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.45, linewidth = 0.3, colour = "black") +
  facet_wrap(~ game, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "Steady-state outcome by arm, one dot per seed",
       x = NULL, y = "Mean over steps 51-100", caption = "Bar = mean of the 5 seeds.") +
  theme_eda
save_eda(p, "eda_02_steady_state_by_seed", 9, 7)

# ---- 03 CV composition: quiet / active / jailed ------------------------------
comp <- ts5 %>% filter(game == "cv", step > 50) %>%
  group_by(arm, who) %>%
  summarise(Quiet = mean(quiet_rate), Active = mean(rebellion_rate), Jailed = mean(jailed_rate), .groups = "drop") %>%
  pivot_longer(c(Quiet, Active, Jailed), names_to = "state", values_to = "share") %>%
  mutate(state = factor(state, levels = c("Jailed", "Active", "Quiet")))
p <- ggplot(comp, aes(arm, share, fill = state)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = ifelse(share >= 0.04, sprintf("%.0f%%", 100 * share), "")),
            position = position_stack(vjust = 0.5), size = 2.8) +
  scale_fill_manual(values = c(Quiet = "#56B4E9", Active = "#CC79A7", Jailed = "#F0E442"), name = NULL) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Civil violence: where citizens are in steady state",
       x = NULL, y = "Share of citizens (steps 51-100, 5 seeds)") +
  theme_eda
save_eda(p, "eda_03_cv_composition", 9, 5)

# ---- 04 CV arrests per step over time ----------------------------------------
arr <- ts5 %>% filter(game == "cv") %>% group_by(arm, who, step) %>%
  summarise(arrests = mean(arrests_this_step), .groups = "drop")
p <- ggplot(arr, aes(step, arrests, colour = who)) +
  geom_line(linewidth = 0.6) +
  facet_wrap(~ arm, nrow = 2) +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "Civil violence: arrests per step (5-seed mean)", x = "Step", y = "Arrests") +
  theme_eda
save_eda(p, "eda_04_cv_arrests", 10, 5.5)

# ---- 05 PD payoff and clustering ---------------------------------------------
pdm <- tail5 %>% filter(game == "pd") %>%
  select(arm, who, run_id, `Mean payoff` = mean_payoff, `Spatial clustering` = spatial_clustering,
         `Cooperation rate` = cooperation_rate) %>%
  pivot_longer(-c(arm, who, run_id), names_to = "metric", values_to = "value")
p <- ggplot(pdm, aes(arm, value, colour = who)) +
  geom_point(position = position_jitter(width = 0.12, height = 0, seed = SEED), size = 1.8) +
  facet_wrap(~ metric, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "Prisoner's dilemma: payoff, clustering and cooperation by seed", x = NULL, y = NULL) +
  theme_eda
save_eda(p, "eda_05_pd_payoff_clustering", 9, 8)

# ---- 06 CV activation curve --------------------------------------------------
act <- rd("cv_activation_curve.csv") %>% label_arms() %>% filter(n >= 50)
p <- ggplot(act, aes(bin_mid, p_active, colour = who)) +
  geom_vline(xintercept = 0, linetype = "dotted", colour = "grey40") +
  geom_vline(xintercept = 0.1, linetype = "dashed", colour = "grey40") +
  geom_line(linewidth = 0.6) + geom_point(aes(size = n), alpha = 0.6) +
  facet_wrap(~ arm, nrow = 2) +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  theme(panel.spacing.x = unit(1, "lines")) +
  scale_size_area(max_size = 3, guide = "none") +
  scale_x_continuous(breaks = c(-0.4, -0.2, 0, 0.2)) +
  labs(title = "Civil violence: P(active | net incentive) by arm",
       x = "Net incentive, G - R x P", y = "P(active)",
       caption = "Dashed line = Epstein threshold (0.1). Dotted line = 0, the threshold implied by the calculator prompt. Bins with fewer than 50 decisions dropped; point size = n.") +
  theme_eda
save_eda(p, "eda_06_cv_activation_curves", 10, 5.5)

# ---- 07 PD transition rule heatmap -------------------------------------------
tr <- rd("pd_transition_rule.csv") %>% label_arms() %>%
  mutate(cell = paste0("Own: ", tolower(action_before), "\nBest neighbour: ", tolower(best_neighbor_action)))
p <- ggplot(tr, aes(arm, cell, fill = p_coop)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.2f\nn=%s", p_coop, format(n, big.mark = ","))), size = 2.4) +
  scale_fill_gradient(low = "grey95", high = "#009E73", limits = c(0, 1), name = "P(cooperate)") +
  labs(title = "Prisoner's dilemma: P(cooperate next step | own and best neighbour's last action)",
       x = NULL, y = NULL, caption = "Empty cells = state never observed for that arm.") +
  theme_eda + theme(panel.grid = element_blank())
save_eda(p, "eda_07_pd_transition_heatmap", 10, 5)

# ---- 08 CV recovered threshold per run --------------------------------------
thr <- rd("cv_threshold_recovery.csv") %>% filter(batch == "five_seed") %>% label_arms()
p <- ggplot(thr, aes(arm, recovered_threshold, colour = who, shape = convergence_warning)) +
  geom_hline(yintercept = 0, linetype = "dotted", colour = "grey40") +
  geom_hline(yintercept = 0.1, linetype = "dashed", colour = "grey40") +
  geom_point(position = position_jitter(width = 0.1, height = 0, seed = SEED), size = 2.2) +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 4), name = "Logit separation warning") +
  labs(title = "Civil violence: threshold recovered from each run's decisions",
       x = NULL, y = "Recovered threshold",
       caption = "Arms with too little variation to fit are missing. Dashed = 0.1 (Epstein); dotted = 0.") +
  theme_eda
save_eda(p, "eda_08_cv_threshold_by_run", 9, 5)

# ---- 09 Parse-failure rate, every run ----------------------------------------
ri <- rd("run_index.csv") %>% filter(provider != "mock") %>% label_arms() %>%
  mutate(game = factor(GAME_LAB[game], levels = GAME_LAB))
p <- ggplot(ri, aes(arm, 100 * parse_fail_rate + 0.001, colour = who, shape = batch)) +
  geom_hline(yintercept = 5, linetype = "dashed", colour = "firebrick") +
  geom_point(position = position_jitter(width = 0.12, height = 0, seed = SEED), size = 2) +
  scale_y_log10(labels = function(x) paste0(signif(x, 2), "%")) +
  facet_wrap(~ game, ncol = 1) +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "Parse-failure rate by run (log scale)", x = NULL, y = "Parse failures",
       caption = "Dashed line = 5% exclusion gate. 0.001 added so runs with zero failures can be shown on the log scale.") +
  theme_eda
save_eda(p, "eda_09_parse_failures", 9, 8)

# ---- 10 IC time series and steady-state metrics (GPT and classic only) -------
ic_ts <- rd("ic_dynamics_timeseries.csv") %>% label_arms()
p <- ggplot(ic_ts, aes(step, war_frequency, colour = arm, group = run_id)) +
  geom_line(linewidth = 0.6) +
  scale_colour_manual(values = c("Classic\nrule-based" = "grey45", "GPT\ncalculator" = "#9ecae1",
                                 "GPT\nreasoner" = "#3182bd", "GPT\nrole-player" = "#08306b"), name = NULL) +
  labs(title = "Interstate conflict: war frequency over time (one seed per arm, GPT only)",
       x = "Step", y = "War frequency",
       caption = "DeepSeek IC arms fail the 5% parse gate and are excluded.") +
  theme_eda
save_eda(p, "eda_10_ic_war_timeseries", 9, 5)

ic_sum <- rd("ic_dynamics_summary.csv") %>% label_arms() %>%
  select(arm, `War frequency` = war_frequency, `Bluff rate` = bluff_rate,
         `Bluff success` = bluff_success_rate, `Welfare loss` = welfare_loss) %>%
  pivot_longer(-arm, names_to = "metric", values_to = "value")
p <- ggplot(ic_sum, aes(arm, value)) +
  geom_col(fill = "#0072B2", width = 0.6) +
  geom_text(aes(label = signif(value, 3)), vjust = -0.3, size = 2.8) +
  facet_wrap(~ metric, scales = "free_y", nrow = 2) +
  labs(title = "Interstate conflict: summary metrics by arm", x = NULL, y = NULL) +
  theme_eda
save_eda(p, "eda_11_ic_metrics", 9, 6)

# ---- 12 CV 500-step runs -----------------------------------------------------
long <- rd("cv_500step_timeseries.csv") %>% label_arms() %>% mutate(mobilized = rebellion_rate + jailed_rate)
p <- ggplot(long, aes(step, mobilized, colour = who)) +
  geom_line(linewidth = 0.5) +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "Civil violence over 500 steps (calculator arms, one seed)", x = "Step", y = "Mobilized share") +
  theme_eda
save_eda(p, "eda_12_cv_500_steps", 9, 4.5)

# ---- 13 CV English vs Chinese prompts (seed-matched where available) --------
lang <- ts_all %>% filter(game == "cv", batch %in% c("cv_chinese", "five_seed"), step > n_steps / 2,
                          agent_type != "classic") %>%
  group_by(batch, language, run_id, agent_type, provider) %>%
  summarise(mob = mean(mobilized), .groups = "drop") %>% label_arms() %>%
  mutate(language = factor(ifelse(language == "zh", "Chinese (1 seed)", "English (5 seeds)"),
                           levels = c("English (5 seeds)", "Chinese (1 seed)")))
p <- ggplot(lang, aes(arm, mob, colour = who, shape = language)) +
  geom_point(position = position_dodge(width = 0.5), size = 2.2, alpha = 0.8) +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  scale_shape_manual(values = c(16, 17), name = NULL) +
  labs(title = "Civil violence: English vs Chinese prompts, steady-state mobilized share",
       x = NULL, y = "Mobilized share (second half of run)") +
  theme_eda
save_eda(p, "eda_13_cv_language", 9, 5)

# ---- 14 LLM calls per run (cost proxy) ---------------------------------------
calls <- rd("run_index.csv") %>% filter(provider != "mock", batch == "five_seed") %>% label_arms() %>%
  mutate(game = factor(GAME_LAB[game], levels = GAME_LAB))
p <- ggplot(calls, aes(arm, total_llm_interactions / 1000, colour = who)) +
  geom_point(position = position_jitter(width = 0.1, height = 0, seed = SEED), size = 2) +
  facet_wrap(~ game, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = PROV_COL, name = NULL) +
  labs(title = "LLM calls per run, five-seed batch", x = NULL, y = "Thousands of calls",
       caption = "Includes cache hits; temperature 0 with caching means repeated states may not hit the API.") +
  theme_eda
save_eda(p, "eda_14_llm_calls", 9, 6)

# ---- Combined PDF -------------------------------------------------------------
pdf(file.path(OUT_FIGURES, "eda_gallery.pdf"), width = 10, height = 7)
for (nm in names(plots)) print(plots[[nm]])
dev.off()
log_msg("18", "Wrote eda_gallery.pdf with ", length(plots), " pages")
