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
assert_that(n_distinct(dev$cell) == 9, "Expected 9 deviation cells (codebook v2)")
# Not drawn (Jared, Oct 26): the GPT calculator's prisoner's dilemma defections where the classic rule cooperates. Every
# one gives the same reason (follows its own calculation), so the waffle would be a single colour; the text and the
# caption report it instead. The check below fails if that stops being true.
DROP_CELL <- "pd_calculator_gpt_defects"
drop <- filter(dev, cell == DROP_CELL)
assert_that(nrow(drop) == 1 && drop$reason == "Follows its own calculation" && abs(drop$share - 1) < 1e-9,
            "The dropped calculator defection cell no longer has a single reason: draw it again")
log_msg("11", sprintf("Not drawn: %s (n = %d, 100%% follows its own calculation; reported in the text)", DROP_CELL, drop$n_decisions))
dev <- filter(dev, cell != DROP_CELL)
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
DEV_LAB <- c("cooperates where the classic rule defects" = "Cooperates,\nclassic rule defects",
             "defects where the classic rule cooperates" = "Defects,\nclassic rule cooperates",
             "active where the classic rule is quiet" = "Active,\nclassic rule quiet",
             "quiet where the classic rule is active" = "Quiet,\nclassic rule active")
# Facet rows: model in civil violence; deviation direction in the prisoner's dilemma (all PD deviations are GPT, and the
# GPT calculator arm deviates in both directions under codebook v2)
# Panel A (all GPT) is one row of two tall waffles beside panel B, with the shared legend below both (Jared, Oct 26);
# its columns are the two drawn prisoner's dilemma cells, and the notes give the agent's action
ROW_LEV <- MODELS
PD_CELLS <- c(pd_calculator_gpt_cooperates = "Calculator", pd_roleplayer_gpt_cooperates = "Role-player")
COL_LEV <- c(names(PD_CELLS), PROMPT_LAB)
COL_STRIP <- c(PD_CELLS, setNames(PROMPT_LAB, PROMPT_LAB))
cells <- dev %>% distinct(cell, game, mode, model, deviation, n_decisions) %>%
  mutate(prompt = factor(PROMPT_LAB[mode], levels = PROMPT_LAB), model = factor(model, levels = MODELS),
         row = factor(as.character(model), levels = ROW_LEV),
         col = factor(if_else(game == "pd", cell, as.character(prompt)), levels = COL_LEV),
         # Notes give only the agent's action and n; the classic rule's action is explained in the text (Jared, Oct 26)
         note = sprintf("%s\n(n = %s)", sub(",\n.*", "", DEV_LAB[deviation]), format(n_decisions, big.mark = ",", trim = TRUE)),
         # the narrow prisoner's dilemma rectangles need one more line break
         note = note)
assert_that(nrow(cells) == 8, "Expected 8 drawn deviation cells (9 minus the dropped calculator defections)")
assert_that(setequal(cells$cell[cells$game == "pd"], names(PD_CELLS)), "Unexpected prisoner's dilemma cells")
assert_that(!any(is.na(cells$prompt)) && !any(is.na(DEV_LAB[cells$deviation])), "Unknown prompt or deviation label")
# Panel A shows only the prisoner's dilemma cells with a deviation (GPT calculator and role-player);
# "no deviation" cells are stated in the text (Jared, Oct 22). Each panel has its own legend and palette.

# Largest-remainder rounding to exactly 100 squares per cell
# Within each waffle, squares fill from the largest reason to the smallest (Jared, Oct 23); 'Other or unclassified'
# always comes last. The legend keeps the fixed REASONS order.
waffle <- dev %>% mutate(reason = factor(reason, levels = REASONS)) %>%
  arrange(cell, reason == "Other or unclassified", desc(share), reason) %>%
  group_by(cell) %>% mutate(raw = 100 * share, n = floor(raw)) %>%
  mutate(n = n + (rank(-(raw - n), ties.method = "first") <= (100 - sum(n)))) %>%
  uncount(n) %>% mutate(i = row_number() - 1,
                        # civil violence: 10 x 10; prisoner's dilemma: tall 5 x 20 rectangles (Jared, Oct 26)
                        x = if_else(game == "pd", i %% 5, i %% 10),
                        y = if_else(game == "pd", 19 - i %/% 5, 9 - i %/% 10)) %>% ungroup() %>%
  mutate(prompt = factor(PROMPT_LAB[mode], levels = PROMPT_LAB), model = factor(model, levels = MODELS),
         row = factor(as.character(model), levels = ROW_LEV),
         col = factor(if_else(game == "pd", cell, as.character(prompt)), levels = COL_LEV))
assert_that(all(table(waffle$cell) == 100), "A cell does not have exactly 100 squares")
assert_that(all(waffle$model[waffle$game == "pd"] == "GPT"), "Prisoner's dilemma deviations expected for GPT only")

# Separate legends per panel, Dark2 throughout. Reasons shared by both panels keep one colour; reasons
# unique to a panel get Dark2 colours not used by the other panel, so no colour means two different things.
PAL <- c("Says the other option is better" = DARK2[1], "Misreads or miscalculates payoffs" = DARK2[2],
         "Reciprocity" = DARK2[3], "Coordination" = DARK2[4],                       # prisoner's dilemma only
         "Follows its own calculation" = "#4D004B", "Arrest risk, no fellow protesters" = DARK2[6],  # plum: strongest colour-blind contrast
         "Arrest risk, other reasons" = "#386CB0",  # ColorBrewer Accent blue; Dark2 brown was too close to Dark2 yellow  # civil violence only
         "No fellow protesters, no arrest risk" = DARK2[7],                         # civil violence only (codebook v1)
         "Other or unclassified" = "#BDBDBD")                                     # light grey, distinct from the plum
assert_that(setequal(names(PAL), REASONS) && !any(duplicated(PAL)), "Palette must cover every reason once")
USED <- REASONS[REASONS %in% waffle$reason]

# Sized for Science's full page width (3 columns, about 18.3 cm = 7.2 in) at final print size, so every
# font size below is the printed size. Helvetica-equivalent sans (TeX Gyre Heros on Linux, Helvetica on macOS).
FIG_W <- 7.0; FIG_H <- 4.0; A_WIDTH <- 1.3; A_YLIM <- -2.4; B_YLIM <- -2.6; FONT <- "Helvetica"  # measured, see layout note below
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
    geom_text(data = filter(cells, game == g), aes(x = if (g == "pd") 2 else 4.5, y = -0.55, label = note), vjust = 1,
              size = 6.5 * PT, lineheight = 0.95, family = FONT, colour = "black") +
    coord_cartesian(xlim = if (g == "pd") c(-0.5, 4.5) else c(-0.5, 9.5),
                    ylim = c(ylim_low, if (g == "pd") 19.5 else 9.5), expand = FALSE) +
    facet_grid(row ~ col, drop = TRUE, labeller = labeller(col = COL_STRIP)) +
    # One shared legend for both panels (same colour = same reason everywhere), collected below the figure
    scale_fill_manual(values = PAL[USED], limits = USED, labels = LEG_LAB, name = NULL, drop = FALSE) +
    guides(fill = guide_legend(nrow = legend_rows, byrow = TRUE)) +
    labs(x = NULL, y = NULL, title = title) +
    theme_minimal(base_size = 7, base_family = FONT) +
    theme(axis.text = element_blank(), panel.grid = element_blank(),
          panel.spacing.x = unit(4, "pt"), panel.spacing.y = unit(1.5, "pt"),
          strip.text = element_text(size = 8, face = "bold"), strip.clip = "off",  # A's narrow strips must not cut "Role-player"
          plot.title = element_text(face = "bold", size = 10), plot.title.position = "panel",
          legend.position = "bottom", legend.justification = "left",
          legend.text = element_text(size = 6.2, lineheight = 0.9), legend.key.size = unit(7, "pt"),
          legend.spacing.x = unit(3, "pt"),
          legend.margin = margin(0, 0, 0, 0), plot.margin = margin(2, 2, 2, 2))
}
# Layout: A is two tall 5 x 20 rectangles beside B's 3 x 2 grid of 10 x 10 waffles, one shared legend below.
# A_YLIM puts the bottoms of A's rectangles level with the bottoms of B's lower waffles; A_WIDTH and FIG_W give every
# B's tiles near square (width/height 0.96), A's rectangles wide (1.74), B's rows close together; 7.0 x 4.0 in, within Science's 7.2 in full width (Jared, Oct 26; measured from the PNG).
p <- (game_panel("pd", "A  Prisoner's dilemma", ylim_low = A_YLIM, legend_rows = 1) |
      game_panel("cv", "B  Civil violence", ylim_low = B_YLIM, legend_rows = 1)) +
  plot_layout(widths = c(A_WIDTH, 2), guides = "collect") & theme(legend.position = "bottom", legend.justification = "center")
ggsave(file.path(OUT_FIGURES, "fig_waffle_deviations.pdf"), p, width = FIG_W, height = FIG_H, device = PDF_DEVICE)
ggsave(file.path(OUT_FIGURES, "fig_waffle_deviations.png"), p, width = FIG_W, height = FIG_H, dpi = 600, bg = "white")
ggsave(file.path(OUT_FIGURES, "fig_waffle_deviations.tiff"), p, width = FIG_W, height = FIG_H, dpi = 600, bg = "white",
       compression = "lzw")
log_msg("11", sprintf("Wrote fig_waffle_deviations (pdf + 600 dpi png + tiff), %.2f x %.2f in", FIG_W, FIG_H))
