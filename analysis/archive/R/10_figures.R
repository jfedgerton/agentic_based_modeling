# 10_figures.R
# Fig 1-5 as publication PDFs. Style rules (Jared's stated preference):
#   - never angle x-axis text; wrap long categorical labels onto two lines
#     and left-align them instead
#   - facet_wrap strip text may be angled, but never moved off the top
# wrap_labels() below is the one place that enforces the label-wrapping rule;
# every figure that needs categorical x-axis labels calls it.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)
library(ggplot2)
library(stringr)

theme_set(theme_minimal(base_size = 11))

wrap_labels <- function(labels, width = 12) str_wrap(labels, width = width)

save_fig <- function(plot, name, width = 7, height = 4.5) {
  path <- file.path(OUT_FIGURES, paste0(name, ".pdf"))
  ggsave(path, plot, width = width, height = height, device = cairo_pdf)
  log_msg("10", "Wrote ", path)
}

# ---- Fig 1: convergence trajectories (PD + CV, classic vs calculator) ----
panel_ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)

fig1_data <- panel_ts %>%
  filter(batch == "five_seed", agent_type %in% c("classic", "calculator")) %>%
  mutate(
    metric_val = if_else(game == "pd", cooperation_rate, rebellion_rate),
    metric_name = if_else(game == "pd", "Cooperation rate", "Rebellion rate"),
    arm_label = if_else(agent_type == "classic", "Classic", paste0("Calculator (", provider, ")"))
  ) %>%
  group_by(game, metric_name, arm_label, step) %>%
  summarise(mean_val = mean(metric_val, na.rm = TRUE),
            lo = mean_val - sd(metric_val, na.rm = TRUE),
            hi = mean_val + sd(metric_val, na.rm = TRUE),
            .groups = "drop")

fig1 <- ggplot(fig1_data, aes(step, mean_val, color = arm_label, fill = arm_label)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.7) +
  facet_wrap(~metric_name, scales = "free_y") +
  labs(x = "Step", y = NULL, color = "Arm", fill = "Arm",
       title = "Fig 1. Convergence under formal payoffs (5-seed bands)") +
  theme(legend.position = "bottom")
save_fig(fig1, "fig1_convergence")

# ---- Fig 2a: PD strategy-type distribution ----
pd_strat <- read_csv(file.path(OUT_DERIVED, "pd_strategy_distribution.csv"), show_col_types = FALSE)
pd_strat <- pd_strat %>%
  filter(agent_type != "classic") %>%
  mutate(arm_provider = wrap_labels(paste0(agent_type, "\n", provider), width = 20))

fig2a <- ggplot(pd_strat, aes(x = arm_provider, y = share, fill = strategy)) +
  geom_col(position = "stack") +
  labs(x = NULL, y = "Share of agents", fill = "Strategy",
       title = "Fig 2a. PD strategy-type distribution by mode x provider") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
save_fig(fig2a, "fig2a_pd_strategies")

# ---- Fig 2b: CV recovered activation threshold vs the classic baseline ----
# The dashed line is the EMPIRICAL classic-arm threshold (~0.10, recovered
# from data -- see 05_cv_threshold_recovery.R header), not the literature
# value of 0. That assumption was wrong and was corrected Sept 27, 2026.
cv_thresh <- read_csv(file.path(OUT_DERIVED, "cv_threshold_recovery.csv"), show_col_types = FALSE)
classic_baseline <- cv_thresh %>% filter(agent_type == "classic") %>% pull(recovered_threshold) %>% mean()
cv_thresh_plot <- cv_thresh %>%
  filter(agent_type != "classic") %>%
  mutate(arm_provider = wrap_labels(paste0(agent_type, "\n", provider), width = 20))

fig2b <- ggplot(cv_thresh_plot, aes(x = arm_provider, y = recovered_threshold)) +
  geom_hline(yintercept = classic_baseline, linetype = "dashed", color = "grey40") +
  geom_point(size = 2.5) +
  labs(x = NULL, y = "Recovered activation threshold",
       title = sprintf("Fig 2b. CV recovered threshold vs. classic baseline (dashed = %.3f, empirical)", classic_baseline)) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
save_fig(fig2b, "fig2b_cv_threshold")

# ---- Fig 3: framing effect plot ----
framing <- read_csv(file.path(OUT_DERIVED, "framing_divergence.csv"), show_col_types = FALSE)
framing_plot <- framing %>%
  mutate(label = wrap_labels(paste0(game, " / ", provider, "\n", contrast), width = 18))

fig3 <- ggplot(framing_plot, aes(x = label, y = effect, color = game)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_pointrange(aes(ymin = ci_low, ymax = ci_high), na.rm = TRUE) +
  geom_point(data = filter(framing_plot, is.na(ci_low)), size = 2.5, shape = 1) +
  labs(x = NULL, y = "Effect (role-player - contrast arm)", color = "Game",
       title = "Fig 3. Framing effects: role-player vs. reasoner/calculator") +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
save_fig(fig3, "fig3_framing_effects", width = 8, height = 5)

# ---- Fig 4: IC dynamics, classic vs GPT arms ----
ic_ts <- read_csv(file.path(OUT_DERIVED, "ic_dynamics_timeseries.csv"), show_col_types = FALSE)
ic_long <- ic_ts %>%
  tidyr::pivot_longer(cols = c(war_frequency, welfare_loss, bluff_rate, bluff_success_rate),
                       names_to = "metric", values_to = "value") %>%
  mutate(metric = str_replace_all(metric, "_", " "))

fig4 <- ggplot(ic_long, aes(step, value, color = agent_type)) +
  geom_line(linewidth = 0.6, alpha = 0.85) +
  facet_wrap(~metric, scales = "free_y") +
  labs(x = "Step", y = NULL, color = "Arm",
       title = "Fig 4. IC out-of-sample dynamics (GPT only; DeepSeek excluded, >50% parse failure)") +
  theme(legend.position = "bottom")
save_fig(fig4, "fig4_ic_dynamics", width = 8, height = 6)

# ---- Fig 5: CV long-run stability (500 steps) ----
cv500 <- read_csv(file.path(OUT_DERIVED, "cv_500step_timeseries.csv"), show_col_types = FALSE)
cv500 <- cv500 %>% mutate(arm_label = if_else(agent_type == "classic", "Classic", paste0("Calculator (", provider, ")")))

fig5 <- ggplot(cv500, aes(step, rebellion_rate, color = arm_label)) +
  geom_line(linewidth = 0.6, alpha = 0.85) +
  geom_vline(xintercept = 100, linetype = "dotted", color = "grey50") +
  labs(x = "Step", y = "Rebellion rate", color = "Arm",
       title = "Fig 5. CV long-run stability to 500 steps (1 seed/arm; dotted = 100-step horizon used elsewhere)") +
  theme(legend.position = "bottom")
save_fig(fig5, "fig5_cv_longrun")

log_msg("10", "All figures written to ", OUT_FIGURES)
