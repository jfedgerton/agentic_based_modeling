# 09_provider_table.R
# Table 1: convergence metrics by game x mode x provider. Combines
# convergence_metrics.csv (vs. classic) with framing_divergence.csv
# (roleplayer vs. reasoner/calculator) into the single publication table
# outline.md specifies for section 2.5.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)

convergence <- read_csv(file.path(OUT_DERIVED, "convergence_metrics.csv"), show_col_types = FALSE)
framing <- read_csv(file.path(OUT_DERIVED, "framing_divergence.csv"), show_col_types = FALSE)

# Panel A: mode vs. classic (validation)
table1_vs_classic <- convergence %>%
  transmute(
    game, agent_type, provider, batch,
    comparison = "vs_classic",
    n_seeds = n_seeds_llm,
    rmse = round(rmse, 4),
    ks_distance = round(ks_distance, 4),
    p_value = round(perm_p_value, 4)
  )

# Panel B: roleplayer vs. reasoner/calculator (framing effect)
table1_framing <- framing %>%
  transmute(
    game, agent_type = contrast, provider, batch,
    comparison = "framing_contrast",
    n_seeds = n_seeds_roleplayer,
    effect = round(effect, 4),
    ci_low = round(ci_low, 4),
    ci_high = round(ci_high, 4),
    p_value = round(perm_p_value, 4)
  )

write.csv(table1_vs_classic, file.path(OUT_DERIVED, "table1_vs_classic.csv"), row.names = FALSE)
write.csv(table1_framing, file.path(OUT_DERIVED, "table1_framing.csv"), row.names = FALSE)
log_msg("09", sprintf("Wrote table1_vs_classic.csv (%d rows) and table1_framing.csv (%d rows)",
                       nrow(table1_vs_classic), nrow(table1_framing)))
