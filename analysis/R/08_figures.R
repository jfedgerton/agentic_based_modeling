# 08_figures.R
# Main-text and Extended Data figures for the provider-divergence framing.
#
# Style rules (Jared): x-axis text never angled -- long labels are wrapped onto
# two lines; facet strips stay on top. Colour: GPT = #0072B2, DeepSeek =
# #D55E00 (validated colour-blind safe, dE 21.9 protan), classic = grey/black
# dashed. Shape is a redundant cue (GPT circle, DeepSeek triangle) so identity
# never rides on colour alone.
#
# Sizes: 7.2 in wide = Nature/Science double column (183 mm).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(patchwork)
library(scales)

COL <- c("GPT (OpenAI)" = "#0072B2", "DeepSeek" = "#D55E00", "Classic (rule-based)" = "grey25")
SHP <- c("GPT (OpenAI)" = 16, "DeepSeek" = 17, "Classic (rule-based)" = 15)
MODE_LAB <- c(calculator = "Calculator\n(formal payoffs)",
              reasoner   = "Reasoner\n(descriptive rules)",
              roleplayer = "Role-player\n(narrative scenario)")
SQRT_BREAKS <- c(0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)
prov_lab <- function(p) recode(p, openai = "GPT (OpenAI)", deepseek = "DeepSeek", mock = "Classic (rule-based)")

theme_set(theme_minimal(base_size = 9) +
            theme(panel.grid.minor = element_blank(),
                  legend.position = "bottom",
                  strip.text = element_text(face = "bold", size = 9),
                  plot.title = element_text(face = "bold", size = 10),
                  axis.text.x = element_text(angle = 0, hjust = 0.5)))

save_fig <- function(plot, name, width = 7.2, height = 4.2) {
  path <- file.path(OUT_FIGURES, paste0(name, ".pdf"))
  ggsave(path, plot, width = width, height = height, device = PDF_DEVICE)
  log_msg("08", "Wrote ", path)
}

run_tail <- read_csv(file.path(OUT_DERIVED, "run_tail_metrics.csv"), show_col_types = FALSE)
pairwise <- read_csv(file.path(OUT_DERIVED, "provider_divergence_pairwise.csv"), show_col_types = FALSE)

# =============================================================================
# FIG 1 -- the headline: per-seed tail outcome, provider x mode, vs classic
# =============================================================================
llm_tail <- run_tail %>% filter(agent_type != "classic") %>%
  mutate(Agent = prov_lab(provider), mode = factor(MODE_LAB[agent_type], levels = MODE_LAB))
classic_tail <- run_tail %>% filter(agent_type == "classic") %>%
  group_by(game) %>%
  summarise(cmean = mean(if_else(game == "pd", cooperation_rate, mobilized_share)),
            cmin = min(if_else(game == "pd", cooperation_rate, mobilized_share)),
            cmax = max(if_else(game == "pd", cooperation_rate, mobilized_share)), .groups = "drop")

fig1_panel <- function(g, ylab, title, ytrans = "identity") {
  d <- llm_tail %>% filter(game == g) %>%
    mutate(y = if (g == "pd") cooperation_rate else mobilized_share)
  cl <- classic_tail %>% filter(game == g)
  dm <- d %>% group_by(mode, Agent) %>% summarise(y = mean(y), .groups = "drop")
  ggplot(d, aes(x = mode, y = y, colour = Agent, shape = Agent)) +
    geom_hline(yintercept = cl$cmean, linetype = "dashed", colour = COL[["Classic (rule-based)"]]) +
    geom_point(position = position_jitterdodge(jitter.width = 0.12, dodge.width = 0.55, seed = SEED),
               size = 1.3, alpha = 0.55) +
    geom_point(data = dm, position = position_dodge(width = 0.55), size = 3) +
    scale_colour_manual(values = COL, drop = FALSE) +
    scale_shape_manual(values = SHP, drop = FALSE) +
    scale_y_continuous(trans = ytrans, breaks = if (ytrans == "sqrt") SQRT_BREAKS else waiver()) +
    labs(x = NULL, y = ylab, title = title, colour = NULL, shape = NULL)
}
# One row, two panels (Jared, Oct 22): capital panel letters; civil violence on a
# linear axis with evenly spaced breaks (no square-root compression below 0.10).
fig1a <- fig1_panel("pd", "Cooperation rate (steps 51-100)",
                    "A  Spatial prisoner's dilemma") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2))
fig1b <- fig1_panel("cv", "Mobilized share, active + jailed\n(steps 51-100)",
                    "B  Civil violence") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2))
fig1 <- (fig1a | fig1b) + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
fig1 <- fig1 + plot_annotation(
  caption = "Small points: individual seeds (5 per arm). Large points: arm means. Dashed line: classic rule-based agents (5 seeds).")
save_fig(fig1, "fig1_provider_divergence", height = 3.6)

# =============================================================================
# FIG 2 -- trajectories, provider x mode, seed ranges, classic dashed
# =============================================================================
panel_ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)
traj <- panel_ts %>%
  filter(batch == "five_seed", !excluded, game %in% c("pd", "cv")) %>%
  mutate(y = if_else(game == "pd", cooperation_rate, 1 - quiet_rate)) %>%
  group_by(game, agent_type, provider, step) %>%
  summarise(ym = mean(y), ylo = min(y), yhi = max(y), .groups = "drop") %>%
  mutate(Agent = prov_lab(provider))
classic_traj <- traj %>% filter(agent_type == "classic") %>% select(-agent_type)
llm_traj <- traj %>% filter(agent_type != "classic") %>%
  mutate(mode = factor(MODE_LAB[agent_type], levels = MODE_LAB))
classic_rep <- tidyr::crossing(classic_traj, mode = factor(MODE_LAB, levels = MODE_LAB))

fig2_row <- function(g, ylab, title, ytrans = "identity") {
  ggplot(filter(llm_traj, game == g), aes(step, ym, colour = Agent, fill = Agent)) +
    geom_ribbon(aes(ymin = ylo, ymax = yhi), alpha = 0.18, colour = NA) +
    geom_line(linewidth = 0.6) +
    geom_line(data = filter(classic_rep, game == g), linetype = "dashed", linewidth = 0.5) +
    facet_wrap(~mode, nrow = 1) +
    scale_colour_manual(values = COL) + scale_fill_manual(values = COL) +
    scale_y_continuous(trans = ytrans, breaks = if (ytrans == "sqrt") SQRT_BREAKS else waiver()) +
    labs(x = "Step", y = ylab, title = title, colour = NULL, fill = NULL)
}
fig2 <- (fig2_row("pd", "Cooperation rate", "A  Prisoner's dilemma") /
           fig2_row("cv", "Mobilized share\n(sqrt scale)", "B  Civil violence", "sqrt")) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
fig2 <- fig2 + plot_annotation(caption = "Lines: mean over 5 seeds; ribbons: min-max across seeds; dashed: classic rule-based agents.")
save_fig(fig2, "fig2_trajectories", height = 5.4)

# =============================================================================
# FIG 3 -- mechanism: the decision rule each provider actually uses
# =============================================================================
pd_trans <- read_csv(file.path(OUT_DERIVED, "pd_transition_rule.csv"), show_col_types = FALSE) %>%
  filter(n >= 20) %>%
  mutate(Agent = prov_lab(provider),
         # Faceted by model (strips on the right), so rows only need the short framing name (Jared, Oct 23)
         model_lab = factor(recode(provider, openai = "GPT", deepseek = "DeepSeek", mock = "ABM"),  # ABM = classic rule-based agent
                            levels = c("GPT", "DeepSeek", "ABM")),
         arm_lab = factor(if_else(agent_type == "classic", "Classic rule", sub("\n.*", "", MODE_LAB[agent_type])),
                          levels = rev(c("Calculator", "Reasoner", "Role-player", "Classic rule"))),
         cell = paste0("Own: ", tolower(substr(action_before, 1, 1)),
                       " / best nbr: ", tolower(substr(best_neighbor_action, 1, 1))))
# Heat map: reversed magma (cream = 0, near-black = 1), perceptually uniform, colour-blind safe and ordered in greyscale
# (Jared, Oct 23). Cell labels switch to white on dark cells by luminance.
MAGMA <- viridisLite::magma(101, direction = -1, begin = 0.05, end = 0.97)
lum <- function(h) { m <- grDevices::col2rgb(h) / 255; drop(c(0.2126, 0.7152, 0.0722) %*% m) }
pd_trans <- pd_trans %>% mutate(txt = if_else(lum(MAGMA[round(p_coop * 100) + 1]) < 0.45, "white", "grey10"))
assert_that(!anyNA(pd_trans$model_lab) && !anyNA(pd_trans$arm_lab), "Fig 3A: unknown provider or framing label")
fig3a <- ggplot(pd_trans, aes(x = cell, y = arm_lab, fill = p_coop)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  facet_grid(model_lab ~ ., scales = "free_y", space = "free_y") +
  geom_text(aes(label = sprintf("%.2f", p_coop), colour = txt), size = 3.4, fontface = "bold") +
  scale_colour_identity() +
  scale_fill_gradientn(colours = MAGMA, limits = c(0, 1), name = "P(cooperate)") +
  scale_x_discrete(labels = function(x) gsub(" / ", "\n", x)) +
  labs(x = "State at t-1 (c = cooperate, d = defect)", y = NULL,
       title = "A  PD: probability of cooperating given own and best neighbour's last action") +
  theme(panel.grid = element_blank(), axis.text = element_text(size = 9, colour = "black"),
        strip.text.y = element_text(size = 9, face = "bold", angle = -90), panel.spacing.y = unit(4, "pt"),
        axis.title = element_text(size = 9.5), legend.text = element_text(size = 8.5),
        legend.title = element_text(size = 9), legend.key.width = unit(28, "pt"))

cv_curve <- read_csv(file.path(OUT_DERIVED, "cv_activation_curve.csv"), show_col_types = FALSE) %>%
  filter(n >= 50, bin_mid >= -0.3, bin_mid <= 0.5) %>%
  mutate(Agent = prov_lab(provider))
classic_curve <- cv_curve %>% filter(agent_type == "classic") %>% select(-agent_type)
llm_curve <- cv_curve %>% filter(agent_type != "classic") %>%
  mutate(mode = factor(MODE_LAB[agent_type], levels = MODE_LAB))
classic_curve_rep <- tidyr::crossing(classic_curve, mode = factor(MODE_LAB, levels = MODE_LAB))
thr <- read_csv(file.path(OUT_DERIVED, "cv_threshold_recovery.csv"), show_col_types = FALSE) %>%
  filter(agent_type == "classic") %>% pull(recovered_threshold) %>% mean()
# Black-and-white safe: each agent has its own colour (Okabe-Ito), line type and marker; the classic rule is a dashed
# step with open squares, GPT a solid line with circles, DeepSeek a dot-dash line with triangles.
LTY3 <- c("Classic (rule-based)" = "dashed", "DeepSeek" = "dotdash", "GPT (OpenAI)" = "solid")
SHP3 <- c("Classic (rule-based)" = 0, "DeepSeek" = 17, "GPT (OpenAI)" = 16)
fig3b <- ggplot(llm_curve, aes(bin_mid, p_active, colour = Agent, shape = Agent, linetype = Agent)) +
  geom_vline(xintercept = thr, linetype = "dotted", colour = "grey40") +
  geom_step(data = classic_curve_rep, direction = "mid", linewidth = 0.5) +
  geom_point(data = classic_curve_rep, size = 1.4) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.4) +
  facet_wrap(~mode, nrow = 1) +
  scale_colour_manual(values = COL) + scale_shape_manual(values = SHP3) + scale_linetype_manual(values = LTY3) +
  theme(legend.key.width = unit(24, "pt"), axis.text = element_text(size = 9, colour = "black"),
        axis.title = element_text(size = 9.5), legend.text = element_text(size = 9)) +
  labs(x = "Net incentive to rebel (grievance - risk aversion x arrest probability)",
       y = "P(active)", colour = NULL, shape = NULL, linetype = NULL,
       title = "B  CV: probability of activating by net incentive")  # line meanings are in the legend and caption
fig3 <- (fig3a / fig3b) + plot_layout(heights = c(1.2, 1)) & theme(legend.position = "bottom")
# Panels A and B share left edges, so B's axis column is as wide as A's row labels and B's y title would sit far from
# its axis. Move B's y title next to B's own axis numbers (Jared, Oct 23).
g3 <- patchwork::patchworkGrob(fig3)
ylab_rows <- which(grepl("^ylab-l", g3$layout$name))
ylab_B <- ylab_rows[which.max(g3$layout$t[ylab_rows])]
axis_B <- which(grepl("^axis-l", g3$layout$name) & g3$layout$t == g3$layout$t[ylab_B])
assert_that(length(axis_B) == 1, "Fig 3: could not find panel B's y axis in the patchwork layout")
gB <- ggplotGrob(fig3b)
wB <- gB$widths[gB$layout$l[gB$layout$name == "axis-l-1-1"]]
assert_that(length(wB) == 1, "Fig 3: could not measure panel B's axis-label width")
g3$grobs[[ylab_B]] <- grid::nullGrob()
g3 <- gtable::gtable_add_grob(g3, grid::textGrob("P(active)", x = grid::unit(1, "npc") - wB - grid::unit(4, "pt"),
                                                  rot = 90, vjust = 0, gp = grid::gpar(fontsize = 9.5)),
                              t = g3$layout$t[ylab_B], b = g3$layout$b[ylab_B], l = g3$layout$l[axis_B],
                              name = "ylab-B-moved", clip = "off")
# Both titles start where the axes start: put each title in the first panel column of the shared layout
title_rows <- which(grepl("^title-", g3$layout$name))
panel_cols <- g3$layout$l[grepl("^panel", g3$layout$name) & g3$layout$name != "panel-area"]
assert_that(length(title_rows) == 2 && length(panel_cols) > 0, "Fig 3: unexpected title or panel layout")
g3$layout$l[title_rows] <- min(panel_cols)
save_fig(g3, "fig3_decision_rules", height = 6.6)

# =============================================================================
# FIG S1 -- parse failures by game and model
# =============================================================================
run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
pf <- run_index %>%
  filter(provider != "mock", batch %in% c("one_seed", "five_seed"), !excluded_duplicate) %>%  # one copy of each replayed run
  mutate(Agent = prov_lab(provider),
         game_lab = recode(game, pd = "Prisoner's\ndilemma", cv = "Civil\nviolence", ic = "Interstate conflict\n(novel, out of sample)"),
         game_lab = factor(game_lab, levels = c("Prisoner's\ndilemma", "Civil\nviolence", "Interstate conflict\n(novel, out of sample)")),
         pfr = pmax(parse_fail_rate, 1e-5))
fig4a <- ggplot(pf, aes(game_lab, pfr, colour = Agent, shape = Agent)) +
  geom_hline(yintercept = PARSE_FAILURE_EXCLUSION_THRESHOLD, linetype = "dashed", colour = "grey40") +
  geom_point(position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.5, seed = SEED), size = 1.8, alpha = 0.8) +
  annotate("text", x = 0.55, y = PARSE_FAILURE_EXCLUSION_THRESHOLD * 1.6, label = "5% exclusion gate",
           hjust = 0, size = 2.6, colour = "grey30") +
  scale_y_log10(labels = label_percent(accuracy = 0.001), breaks = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1)) +
  scale_colour_manual(values = COL) + scale_shape_manual(values = SHP) +
  labs(x = NULL, y = "Share of decisions that failed to parse\n(log scale; floor 0.001%)", colour = NULL, shape = NULL,
       title = NULL)
# Supplement: parse failures by game and model (moved out of the main crisis-game figure)
save_fig(fig4a + theme(legend.position = "bottom"), "figS_parse_failures", height = 3.4)

# Fig. 5 (crisis-game trajectories and reasons) is drawn by R/12_fig5_crisis.R

# =============================================================================
# EXTENDED DATA
# =============================================================================
# ED1 -- multivariate divergence triangle
mv <- read_csv(file.path(OUT_DERIVED, "provider_divergence_multivariate.csv"), show_col_types = FALSE) %>%
  mutate(game_lab = factor(recode(game, pd = "Prisoner's dilemma", cv = "Civil violence"),
                           levels = c("Prisoner's dilemma", "Civil violence")),
         mode_lab = factor(MODE_LAB[mode], levels = rev(MODE_LAB)),
         side = factor(side, levels = c("d(GPT, classic)", "d(DeepSeek, classic)", "d(GPT, DeepSeek)")),
         lab = sprintf("%.2f\np=%.3f", distance, exact_p))
ed1 <- ggplot(mv, aes(side, mode_lab, fill = distance)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = lab), size = 2.5, colour = if_else(mv$distance > 2.5, "white", "grey15")) +
  facet_wrap(~game_lab) +
  scale_fill_gradient(low = "#EAF2F8", high = "#0B3C5D", name = "Standardized\ncentroid distance") +
  scale_x_discrete(labels = function(x) gsub(", ", ",\n", x)) +
  labs(x = NULL, y = NULL, title = "Divergence triangle: distance between arm centroids (z-scored tail metrics)") +
  theme(panel.grid = element_blank())
save_fig(ed1, "ed1_divergence_triangle", height = 3.4)

# ED2 -- CV 500-step long-run check
cv500 <- read_csv(file.path(OUT_DERIVED, "cv_500step_timeseries.csv"), show_col_types = FALSE) %>%
  mutate(Agent = prov_lab(provider), mobilized = 1 - quiet_rate)
ed2 <- ggplot(cv500, aes(step, mobilized, colour = Agent)) +
  geom_line(linewidth = 0.5) +
  geom_vline(xintercept = 100, linetype = "dotted", colour = "grey50") +
  scale_colour_manual(values = COL) +
  scale_y_continuous(trans = "sqrt", breaks = SQRT_BREAKS) +
  labs(x = "Step", y = "Mobilized share (sqrt scale)", colour = NULL,
       title = "CV calculator arms to 500 steps (1 seed each); dotted = 100-step horizon used in the main analysis")
save_fig(ed2, "ed2_cv_longrun", height = 3.2)

# ED3 -- language robustness check: Chinese-language CV prompts (1 seed) vs English (5 seeds).
# One panel per model, so the reader compares the framing pattern within a model across
# languages rather than reading the EN-CN gap as a finding (Maël, Oct 16 comment).
zh <- panel_ts %>% filter(batch == "cv_chinese", !excluded, step >= 51) %>%
  group_by(agent_type, provider) %>% summarise(mobilized = mean(1 - quiet_rate), .groups = "drop") %>%
  mutate(language = "Chinese prompt (1 seed)")
en_seed <- run_tail %>% filter(game == "cv", agent_type != "classic") %>%
  transmute(agent_type, provider, mobilized = mobilized_share, language = "English prompt (5 seeds)")
en <- en_seed %>% group_by(agent_type, provider, language) %>% summarise(mobilized = mean(mobilized), .groups = "drop")
assert_that(nrow(zh) == 6 && nrow(en) == 6, "ED3 expects 6 Chinese and 6 English arms")
lab_fix <- function(d) d %>% mutate(Agent = prov_lab(provider), mode = factor(MODE_LAB[agent_type], levels = MODE_LAB))
zh <- lab_fix(zh); en <- lab_fix(en); en_seed <- lab_fix(en_seed)
seg <- inner_join(en %>% select(Agent, mode, en = mobilized), zh %>% select(Agent, mode, zh = mobilized), by = c("Agent", "mode"))
ed3 <- ggplot() +
  geom_segment(data = seg, aes(x = mode, xend = mode, y = en, yend = zh), colour = "grey60", linewidth = 0.4) +
  geom_point(data = en_seed, aes(mode, mobilized, colour = Agent), size = 1.2, alpha = 0.35,
             position = position_jitter(width = 0.08, height = 0, seed = SEED)) +
  geom_point(data = en, aes(mode, mobilized, colour = Agent, shape = language), size = 2.6) +
  geom_point(data = zh, aes(mode, mobilized, colour = Agent, shape = language), size = 2.6, stroke = 0.9) +
  facet_wrap(~ Agent) +
  scale_colour_manual(values = COL, guide = "none") + scale_shape_manual(values = c(1, 16)) +
  scale_y_continuous(trans = "sqrt", breaks = SQRT_BREAKS) +
  labs(x = NULL, y = "Mobilized share, steps 51-100 (sqrt scale)", shape = NULL,
       title = "CV robustness check: Chinese-language prompts (one seed) vs English prompts (five seeds)",
       caption = "Faint dots = individual English seeds. Chinese-language arms have one seed each and are descriptive only.") +
  theme(legend.position = "bottom", plot.caption = element_text(size = 7, hjust = 0))
save_fig(ed3, "ed3_language_vs_provider", height = 3.8)

log_msg("08", "All divergence figures written.")
