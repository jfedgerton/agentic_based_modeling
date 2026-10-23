# 05_cv_longrun.R
# New main-text content (per Jared's direction to move this out of SI):
# CV long-run stability at 500 steps, classic vs calculator (both providers).
#
# IMPORTANT LIMITATION, stated plainly rather than buried: this batch has
# exactly 1 seed per arm (classic, calculator-openai, calculator-deepseek) --
# there is no seed-level replication at 500 steps, so this script reports
# descriptive trajectory statistics only. No permutation test, no CI. If a
# reviewer will expect uncertainty bands here, that requires new runs, not a
# smarter script.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(readr)

panel_ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)

longrun <- panel_ts %>% filter(game == "cv", batch == "cv_500step")
assert_that(nrow(longrun) > 0, "No CV 500-step data found in panel_timeseries.csv")

n_seeds_per_arm <- longrun %>% distinct(agent_type, provider, run_id) %>% count(agent_type, provider)
log_msg("05", "CV 500-step seed count per arm (expect 1 everywhere):")
for (i in seq_len(nrow(n_seeds_per_arm))) {
  log_msg("05", sprintf("  %s/%s: n=%d", n_seeds_per_arm$agent_type[i], n_seeds_per_arm$provider[i], n_seeds_per_arm$n[i]))
}

# Early window (steps 1-100, matching the other batches' run length) vs late
# window (steps 401-500): compare mean and within-window SD of rebellion_rate
# to ask whether the process is still evolving or has settled into a regime.
EARLY_WINDOW <- c(1, 100)
LATE_WINDOW <- c(401, 500)

stability <- longrun %>%
  mutate(window = case_when(
    step >= EARLY_WINDOW[1] & step <= EARLY_WINDOW[2] ~ "early_1_100",
    step >= LATE_WINDOW[1] & step <= LATE_WINDOW[2] ~ "late_401_500",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(window)) %>%
  group_by(agent_type, provider, window) %>%
  summarise(
    mean_rebellion_rate = mean(rebellion_rate, na.rm = TRUE),
    sd_rebellion_rate = sd(rebellion_rate, na.rm = TRUE),
    mean_jailed_rate = mean(jailed_rate, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(longrun, file.path(OUT_DERIVED, "cv_500step_timeseries.csv"), row.names = FALSE)
write.csv(stability, file.path(OUT_DERIVED, "cv_500step_stability.csv"), row.names = FALSE)
log_msg("05", sprintf("Wrote cv_500step_timeseries.csv (%d rows) and cv_500step_stability.csv (%d rows)",
                       nrow(longrun), nrow(stability)))
