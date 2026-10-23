# 11_tables.R
# LaTeX (booktabs) tables for the manuscript, written to paste directly into
# main.tex (\usepackage{booktabs} is already in the template).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)
library(xtable)

write_booktabs <- function(df, path, caption, label, digits = NULL) {
  xt <- xtable(df, caption = caption, label = label, digits = digits)
  print(xt, file = path, include.rownames = FALSE, booktabs = TRUE,
        caption.placement = "top", table.placement = "h")
  log_msg("11", "Wrote ", path)
}

# ---- Table 1a: convergence vs. classic ----
t1a <- read_csv(file.path(OUT_DERIVED, "table1_vs_classic.csv"), show_col_types = FALSE)
write_booktabs(
  t1a, file.path(OUT_TABLES, "table1a_vs_classic.tex"),
  caption = "Convergence metrics vs. classic baseline, by game, mode, and provider.",
  label = "tab:convergence-vs-classic"
)

# ---- Table 1b: framing contrasts ----
t1b <- read_csv(file.path(OUT_DERIVED, "table1_framing.csv"), show_col_types = FALSE)
write_booktabs(
  t1b, file.path(OUT_TABLES, "table1b_framing.tex"),
  caption = "Framing effects: role-player vs. reasoner/calculator, by game and provider.",
  label = "tab:framing-contrasts"
)

# ---- SI Appendix B: run completeness / parse-failure diagnostics ----
run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
appendix_b <- run_index %>%
  group_by(batch, game, agent_type, provider) %>%
  summarise(
    n_runs = n(),
    n_excluded = sum(excluded),
    mean_parse_fail_rate = round(mean(parse_fail_rate, na.rm = TRUE), 4),
    .groups = "drop"
  ) %>%
  arrange(batch, game, agent_type, provider)

write_booktabs(
  appendix_b, file.path(OUT_TABLES, "appendixB_run_completeness.tex"),
  caption = "Run completeness and parse-failure rates by batch, game, mode, and provider. Rows with mean parse-failure rate above 5\\% are excluded from main-text analyses (see run\\_index.csv).",
  label = "tab:appendix-b-completeness"
)

# ---- SI: PD strategy distribution ----
pd_strat <- read_csv(file.path(OUT_DERIVED, "pd_strategy_distribution.csv"), show_col_types = FALSE)
pd_strat_wide <- pd_strat %>%
  mutate(share = round(share, 3)) %>%
  tidyr::pivot_wider(id_cols = c(agent_type, provider, batch), names_from = strategy, values_from = share, values_fill = 0)

write_booktabs(
  pd_strat_wide, file.path(OUT_TABLES, "si_pd_strategy_distribution.tex"),
  caption = "PD agent strategy-type shares by mode and provider.",
  label = "tab:si-pd-strategies"
)

log_msg("11", "All tables written to ", OUT_TABLES)
