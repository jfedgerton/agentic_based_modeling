# 12_fig5_crisis.R
# Fig. 5, the crisis-bargaining game (GPT only; DeepSeek runs fail the parse gate, fig. S1).
#   A  War frequency and bluff rate per step, GPT framings vs the classic rule-based agent (one run per arm).
#      Bluff rate is the corrected one from R/04 (calculator signal labels).
#   B  One waffle per framing: the reasons GPT agents give when they signal more strength than their true
#      capability (codebook v1, T4; python/05_ic_signal_reasons.py). Each square is 1% of those signals.
# Output: figures/fig4_out_of_sample.pdf (file name kept so main.tex does not change).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(patchwork)

FONT <- "Helvetica"
PT <- 1 / .pt
FIG_W <- 5.6; FIG_H <- 5.2  # B's waffles fill the width (aligned with the B label); each waffle slightly taller than wide

# ---- The text analysis must cover exactly the GPT crisis-game runs kept by R/01 --------------------------------------
run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
kept_gpt <- run_index %>% filter(game == "ic", batch == "one_seed", !excluded, provider == "openai") %>% pull(run_id)
TEXT_RUNS <- c("ic_calculator_seed32_f954c956", "ic_reasoner_seed52_562ca987", "ic_roleplayer_seed62_ad69da08")
assert_that(setequal(kept_gpt, TEXT_RUNS),
            "python/05_ic_signal_reasons.py RUNS no longer match the GPT crisis-game runs kept in run_index.csv")

# ---- Panel A: trajectories ----------------------------------------------------------------------------------------
ic_ts <- read_csv(file.path(OUT_DERIVED, "ic_dynamics_timeseries.csv"), show_col_types = FALSE)
assert_that("bluff_rate_logged" %in% names(ic_ts), "ic_dynamics_timeseries.csv lacks the bluff correction from R/04")
ic_ts <- ic_ts %>%
  group_by(agent_type, provider, step) %>%
  summarise(across(c(war_frequency, bluff_rate), mean), .groups = "drop") %>%
  pivot_longer(c(war_frequency, bluff_rate), names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(recode(metric, war_frequency = "War frequency", bluff_rate = "Bluff rate"),
                         levels = c("War frequency", "Bluff rate")),
         arm = if_else(agent_type == "classic", "Classic (rule-based)",
                       paste0("GPT ", recode(agent_type, roleplayer = "role-player"))))
# Colour-blind safe (Dark2 hues, checked with colorspacious CVD simulation) and readable in black and white:
# each arm also has its own line type and point shape (points every 10 steps).
IC_COL <- c("Classic (rule-based)" = "black", "GPT calculator" = "#D95F02", "GPT reasoner" = "#7570B3",
            "GPT role-player" = "#1B9E77")
IC_LTY <- c("Classic (rule-based)" = "dashed", "GPT calculator" = "solid", "GPT reasoner" = "dotted",
            "GPT role-player" = "dotdash")
IC_SHP <- c("Classic (rule-based)" = 0, "GPT calculator" = 4, "GPT reasoner" = 2, "GPT role-player" = 16)
assert_that(setequal(unique(ic_ts$arm), names(IC_COL)), "Unexpected crisis-game arms")
# Steps 1-50 are burn-in (light grey band behind the lines); every reported crisis-game number averages steps 51-100
# (STEADY_STATE_STEPS). Band labels sit in the empty top strip of the war-frequency panel, above every line (Jared, Oct 27).
assert_that(max(ic_ts$value[ic_ts$metric == "War frequency"]) < 0.9, "War-frequency lines reach the label strip at y = 0.95")
BURN_END <- min(STEADY_STATE_STEPS) - 0.5
band_lab <- tibble(metric = factor("War frequency", levels = levels(ic_ts$metric)),
                   x = c(BURN_END / 2, (BURN_END + 100) / 2), y = 0.95,
                   label = c("Burn-in", "Steady state (averaged)"))
pA <- ggplot(ic_ts, aes(step, value, colour = arm, linetype = arm, shape = arm)) +
  annotate("rect", xmin = -Inf, xmax = BURN_END, ymin = -Inf, ymax = Inf, fill = "grey50", alpha = 0.1) +  # translucent, so gridlines show through
  geom_text(data = band_lab, aes(x, y, label = label), inherit.aes = FALSE, size = 7 * PT, family = FONT,
            colour = "black") +
  geom_line(linewidth = 0.45) +
  geom_point(data = filter(ic_ts, step %% 10 == 5), size = 1.4, stroke = 0.6) +
  facet_wrap(~metric, nrow = 1) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  scale_colour_manual(values = IC_COL, name = NULL) +
  scale_linetype_manual(values = IC_LTY, name = NULL) +  # one legend: colour, line type and shape share the name
  scale_shape_manual(values = IC_SHP, name = NULL) +
  labs(x = "Step", y = NULL, title = "A  Outcomes") +
  theme_minimal(base_size = 8, base_family = FONT) +
  theme(panel.grid.minor = element_blank(), strip.text = element_text(size = 8, face = "bold"),
        axis.text = element_text(size = 8, colour = "black"), axis.title = element_text(size = 8),
        plot.title = element_text(face = "bold", size = 10), plot.title.position = "panel",
        legend.position = "bottom", legend.text = element_text(size = 7.5), legend.key.width = unit(22, "pt"),
        legend.margin = margin(0, 0, 0, 0), plot.margin = margin(2, 4, 6, 2))

# ---- Panel B: reasons for overstating strength --------------------------------------------------------------------
cells_path <- file.path(OUT_DERIVED, "text", "ic_signal_cells.csv")
assert_that(file.exists(cells_path), "Missing derived/text/ic_signal_cells.csv: run python/05_ic_signal_reasons.py first")
dev <- read_csv(cells_path, show_col_types = FALSE)
REASONS <- c("Misreads own strength", "Misreads the rules", "Bluffs openly", "Deterrence or bargaining leverage",
             "Keeps past signal or payoff", "Other or unclassified")
assert_that(all(dev$reason %in% REASONS), "Unknown reason in ic_signal_cells.csv")
chk <- dev %>% group_by(mode) %>% summarise(s = sum(share), .groups = "drop")
assert_that(nrow(chk) == 3 && all(abs(chk$s - 1) < 1e-6), "Reason shares must sum to 1 in each of the three framings")
PROMPT_LAB <- c(calculator = "Calculator", reasoner = "Reasoner", roleplayer = "Role-player")
cells <- dev %>% distinct(mode, n_overstating, n_valid_signals) %>%
  mutate(prompt = factor(PROMPT_LAB[mode], levels = PROMPT_LAB),
         note = sprintf("Overstates its strength\n(n = %s, %.0f%% of signals)",
                        format(n_overstating, big.mark = ",", trim = TRUE), 100 * n_overstating / n_valid_signals))
# Largest-remainder rounding to 100 squares; squares fill from the largest reason down, Other last (as in Fig. 4)
waffle <- dev %>% mutate(reason = factor(reason, levels = REASONS)) %>%
  arrange(mode, reason == "Other or unclassified", desc(share), reason) %>%
  group_by(mode) %>% mutate(raw = 100 * share, n = floor(raw)) %>%
  mutate(n = n + (rank(-(raw - n), ties.method = "first") <= (100 - sum(n)))) %>%
  uncount(n) %>% mutate(i = row_number() - 1, x = i %% 10, y = 9 - i %/% 10) %>% ungroup() %>%
  mutate(prompt = factor(PROMPT_LAB[mode], levels = PROMPT_LAB))
assert_that(all(table(waffle$mode) == 100), "A framing does not have exactly 100 squares")
# Dark2 throughout (teal darkened so it separates from pink under deuteranopia). colorspacious CVD simulation,
# severity 100: smallest pairwise difference 7.5 (tritanopia, orange vs pink), 9.7 (protanopia, purple vs pink).
PAL <- c("Misreads own strength" = "#D95F02", "Misreads the rules" = "#7570B3", "Bluffs openly" = "#E7298A",
         "Deterrence or bargaining leverage" = "#0E6E52", "Keeps past signal or payoff" = "#E6AB02",
         "Other or unclassified" = "#BDBDBD")
LEG_LAB <- c("Misreads own strength" = "Misreads own\nstrength", "Misreads the rules" = "Misreads the\nrules",
             "Bluffs openly" = "Bluffs openly", "Deterrence or bargaining leverage" = "Deterrence or\nbargaining leverage",
             "Keeps past signal or payoff" = "Keeps past signal\nor payoff", "Other or unclassified" = "Other or\nunclassified")
pB <- ggplot(waffle, aes(x, y)) +
  geom_tile(aes(fill = reason), colour = "white", linewidth = 0.3) +
  geom_text(data = cells, aes(x = 4.5, y = -0.55, label = note), vjust = 1, size = 6.5 * PT, lineheight = 0.95,
            family = FONT, colour = "black") +
  coord_cartesian(xlim = c(-0.5, 9.5), ylim = c(-2.1, 9.5), expand = FALSE) +  # fills the width; FIG_W/FIG_H keep tiles square
  facet_wrap(~prompt, nrow = 1) +
  scale_fill_manual(values = PAL, labels = LEG_LAB, name = NULL, drop = FALSE) +
  guides(fill = guide_legend(nrow = 1)) +
  labs(x = NULL, y = NULL, title = "B  Reasons GPT agents give for overstating their strength") +
  theme_minimal(base_size = 7, base_family = FONT) +
  theme(axis.text = element_blank(), panel.grid = element_blank(), panel.spacing.x = unit(10, "pt"),
        strip.text = element_text(size = 8, face = "bold"), plot.title = element_text(face = "bold", size = 10),
        plot.title.position = "panel", legend.position = "bottom", legend.justification = "center", legend.text = element_text(size = 6.5, lineheight = 0.9), legend.spacing.x = unit(2, "pt"),
        legend.key.size = unit(8, "pt"), legend.margin = margin(0, 0, 0, 0), plot.margin = margin(2, 2, 2, 2))

p <- pA / pB + plot_layout(heights = c(1, 1.25))
ggsave(file.path(OUT_FIGURES, "fig4_out_of_sample.pdf"), p, width = FIG_W, height = FIG_H, device = PDF_DEVICE)
ggsave(file.path(OUT_FIGURES, "fig4_out_of_sample.png"), p, width = FIG_W, height = FIG_H, dpi = 600, bg = "white")
log_msg("12", sprintf("Wrote fig4_out_of_sample (Fig. 5, crisis game), %.1f x %.1f in", FIG_W, FIG_H))
