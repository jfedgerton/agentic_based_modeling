# 11_waffle.R
# One waffle per deviation cell: every arm where LLM agents depart from the
# rule-based benchmark, broken down by what the agents said in those decisions.
# 100 squares = 100% of the deviating decisions in that cell (largest-remainder
# rounding, so categories under 0.5% can show no square).
# Input: derived/text/deviation_cells.csv (tools/text_analysis/02_summarise.py).
# Colours: RColorBrewer Dark2, one colour per stated reason, fixed across cells.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(RColorBrewer)

dev_path <- file.path(OUT_DERIVED, "text", "deviation_cells.csv")
assert_that(file.exists(dev_path), "Missing derived/text/deviation_cells.csv: run tools/text_analysis/run_text_analysis.sh first")
dev <- read_csv(dev_path, show_col_types = FALSE)
assert_that(nrow(dev) > 0, "deviation_cells.csv is empty")
chk <- dev %>% group_by(cell) %>% summarise(s = sum(share), .groups = "drop")
assert_that(all(abs(chk$s - 1) < 1e-3), "Reason shares do not sum to 1 within a cell")
assert_that(n_distinct(dev$cell) == 8, "Expected 8 deviation cells")
# Reasons come from codebook v1 (tools/text_analysis/CODEBOOK.md), applied by 03_label_deviations.py

# Order sets the fill order within each waffle and the legend order (Jared, Oct 23: coordination before
# misreads and reciprocity in the prisoner's dilemma calculator cell)
REASONS <- c("Says the other option is better", "Coordination", "Misreads or miscalculates payoffs", "Reciprocity",
             "Follows its own calculation", "Arrest risk, no fellow protesters",
             "Arrest risk, other reasons", "No fellow protesters, no arrest risk", "Other or unclassified")
assert_that(all(dev$reason %in% REASONS), "Unknown reason label in deviation_cells.csv")
DARK2 <- brewer.pal(8, "Dark2")

# Layout (Jared, Oct 22): facet_grid(model ~ prompt) within each game; panel A prisoner's dilemma,
# panel B civil violence. Only cells with a deviation are drawn.
library(patchwork)
PROMPT_LAB <- c(calculator = "Calculator", reasoner = "Reasoner", roleplayer = "Role-player")
MODELS <- c("GPT", "DeepSeek")
# One benchmark for every cell: the classic rule-based agent's action (Jared, Oct 23)
DEV_LAB <- c("cooperates where the classic rule defects" = "Cooperates, classic rule defects",
             "active where the classic rule is quiet" = "Active, classic rule quiet",
             "quiet where the classic rule is active" = "Quiet, classic rule active")
cells <- dev %>% distinct(cell, game, mode, model, deviation, n_decisions) %>%
  mutate(prompt = factor(PROMPT_LAB[mode], levels = PROMPT_LAB), model = factor(model, levels = MODELS),
         note = sprintf("%s\n(n = %s)", DEV_LAB[deviation], format(n_decisions, big.mark = ",", trim = TRUE)))
assert_that(nrow(cells) == 8, "Expected 8 deviation cells")
assert_that(!any(is.na(cells$prompt)) && !any(is.na(DEV_LAB[cells$deviation])), "Unknown prompt or deviation label")
# Panel A shows only the prisoner's dilemma cells with a deviation (GPT calculator and role-player);
# "no deviation" cells are stated in the text (Jared, Oct 22). Each panel has its own legend and palette.

# Largest-remainder rounding to exactly 100 squares per cell
waffle <- dev %>% mutate(reason = factor(reason, levels = REASONS)) %>% arrange(cell, reason) %>%
  group_by(cell) %>% mutate(raw = 100 * share, n = floor(raw)) %>%
  mutate(n = n + (rank(-(raw - n), ties.method = "first") <= (100 - sum(n)))) %>%
  uncount(n) %>% mutate(i = row_number() - 1, x = i %% 10, y = 9 - i %/% 10) %>% ungroup() %>%
  mutate(prompt = factor(PROMPT_LAB[mode], levels = PROMPT_LAB), model = factor(model, levels = MODELS))
assert_that(all(table(waffle$cell) == 100), "A cell does not have exactly 100 squares")
assert_that(all(waffle$model[waffle$game == "pd"] == "GPT"), "Prisoner's dilemma deviations expected for GPT only")

# Separate legends per panel, Dark2 throughout. Reasons shared by both panels keep one colour; reasons
# unique to a panel get Dark2 colours not used by the other panel, so no colour means two different things.
PAL <- c("Says the other option is better" = DARK2[1], "Misreads or miscalculates payoffs" = DARK2[2],
         "Reciprocity" = DARK2[3], "Coordination" = DARK2[4],                       # prisoner's dilemma only
         "Follows its own calculation" = DARK2[5], "Arrest risk, no fellow protesters" = DARK2[6],
         "Arrest risk, other reasons" = "#386CB0",  # ColorBrewer Accent blue; Dark2 brown was too close to Dark2 yellow  # civil violence only
         "No fellow protesters, no arrest risk" = DARK2[7],                         # civil violence only (codebook v1)
         "Other or unclassified" = DARK2[8])
assert_that(setequal(names(PAL), REASONS) && !any(duplicated(PAL)), "Palette must cover every reason once")
panel_cols <- function(g) PAL[REASONS[REASONS %in% waffle$reason[waffle$game == g]]]

# Sized for Science's full page width (3 columns, about 18.3 cm = 7.2 in) at final print size, so every
# font size below is the printed size. Helvetica-equivalent sans (TeX Gyre Heros on Linux, Helvetica on macOS).
FIG_W <- 7.2; FIG_H <- 4.1; FONT <- "Helvetica"
PT <- 1 / .pt  # converts points to ggplot's mm text units
# Legend labels, broken onto two lines where long
LEG_LAB <- c("Says the other option is better" = "Says the other\noption is better",
             "Misreads or miscalculates payoffs" = "Misreads or\nmiscalculates payoffs",
             "Reciprocity" = "Reciprocity", "Coordination" = "Coordination",
             "Follows its own calculation" = "Follows its own\ncalculation",
             "Arrest risk, no fellow protesters" = "Arrest risk, no\nfellow protesters",
             "Arrest risk, other reasons" = "Arrest risk,\nother reasons",
             "No fellow protesters, no arrest risk" = "No fellow protesters,\nno arrest risk",
             "Other or unclassified" = "Other or\nunclassified")
assert_that(setequal(names(LEG_LAB), REASONS), "Every reason needs a legend label")
# Both panels use coord_cartesian, so each waffle fills its column and lines up with the panel title.
# Panel A's lower limit puts the bottom of its waffles level with the bottom of panel B's lower row.
game_panel <- function(g, title, ylim_low, legend_rows = 2) {
  d <- filter(waffle, game == g) %>% droplevels()
  ggplot(d, aes(x, y)) +
    geom_tile(aes(fill = reason), colour = "white", linewidth = 0.3) +
    geom_text(data = filter(cells, game == g), aes(x = 4.5, y = -0.55, label = note), vjust = 1,
              size = 6.5 * PT, lineheight = 0.95, family = FONT, colour = "black") +
    coord_cartesian(xlim = c(-0.5, 9.5), ylim = c(ylim_low, 9.5), expand = FALSE) +
    facet_grid(model ~ prompt, drop = TRUE) +
    scale_fill_manual(values = panel_cols(g), labels = LEG_LAB, name = NULL) +
    guides(fill = guide_legend(nrow = legend_rows, byrow = TRUE)) +
    labs(x = NULL, y = NULL, title = title) +
    theme_minimal(base_size = 7, base_family = FONT) +
    theme(axis.text = element_blank(), panel.grid = element_blank(),
          panel.spacing.x = unit(4, "pt"), panel.spacing.y = unit(1.5, "pt"),
          strip.text = element_text(size = 8, face = "bold"),
          plot.title = element_text(face = "bold", size = 10), plot.title.position = "panel",
          legend.position = "bottom", legend.justification = "left",
          legend.text = element_text(size = 7, lineheight = 0.9), legend.key.size = unit(8, "pt"),
          legend.spacing.x = unit(3, "pt"),
          legend.margin = margin(0, 0, 0, 0), plot.margin = margin(2, 2, 2, 2))
}
p <- (game_panel("pd", "A  Prisoner's dilemma", ylim_low = -1.24, legend_rows = 2) |
      game_panel("cv", "B  Civil violence", ylim_low = -2.1, legend_rows = 2)) +
  plot_layout(widths = c(2.2, 2.8), guides = "keep")
ggsave(file.path(OUT_FIGURES, "fig_waffle_deviations.pdf"), p, width = FIG_W, height = FIG_H, device = PDF_DEVICE)
ggsave(file.path(OUT_FIGURES, "fig_waffle_deviations.png"), p, width = FIG_W, height = FIG_H, dpi = 600, bg = "white")
ggsave(file.path(OUT_FIGURES, "fig_waffle_deviations.tiff"), p, width = FIG_W, height = FIG_H, dpi = 600, bg = "white",
       compression = "lzw")
log_msg("11", sprintf("Wrote fig_waffle_deviations (pdf + 600 dpi png + tiff), %.2f x %.2f in", FIG_W, FIG_H))
