# 02_timeseries.R
# Read every run's time_series.csv, tag it with run-index metadata, and stack
# into one long panel. This is the input to convergence metrics (03),
# framing/divergence tests (06), IC analysis (07), and the 500-step check (08).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(purrr)
library(readr)

run_index <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
log_msg("02", sprintf("Loaded run index: %d rows (%d excluded)", nrow(run_index), sum(run_index$excluded)))

read_one_timeseries <- function(run_dir, run_id, game, batch) {
  path <- file.path(run_dir, "time_series.csv")
  if (!file.exists(path)) {
    log_msg("02", "WARNING: no time_series.csv for ", run_id, " -- skipping")
    return(NULL)
  }
  df <- read_csv(path, show_col_types = FALSE)

  expected <- switch(game,
    pd = EXPECTED_COLS$time_series_pd,
    cv = EXPECTED_COLS$time_series_cv,
    ic = EXPECTED_COLS$time_series_ic
  )
  assert_columns(df, expected, sprintf("time_series/%s/%s", game, run_id))

  # time_series.csv is written 0-indexed; normalize to a 1-based `step` to
  # match agent_panel.parquet / dyads.parquet / arrests.parquet (see
  # readme_OUTPUT.md trap #2). If a `step` column already exists, trust it;
  # otherwise the row index (0-based) + 1 is the step.
  if (!"step" %in% names(df)) {
    df$step <- seq_len(nrow(df))
  } else {
    df$step <- df$step + 1L
  }

  df$run_id <- run_id
  df$batch <- batch
  df
}

# Join key is (run_id, batch), not run_id alone: run_id is a hash-suffixed
# directory name that is unique within a batch by construction, but nothing
# guarantees global uniqueness across batches. Joining on run_id alone risks
# a silent many-to-many join if two batches ever produce the same run_id.
panel_ts <- run_index %>%
  filter(!excluded) %>%
  select(run_id, run_dir, game, batch) %>%
  pmap_dfr(function(run_id, run_dir, game, batch) read_one_timeseries(run_dir, run_id, game, batch))

assert_that(nrow(panel_ts) > 0, "Extracted time-series panel is empty")

dup_check <- run_index %>% count(run_id, batch) %>% filter(n > 1)
assert_that(nrow(dup_check) == 0,
            sprintf("Duplicate (run_id, batch) in run_index -- cannot join safely: %s",
                    paste(dup_check$run_id, collapse = ", ")))

panel_ts <- panel_ts %>%
  left_join(run_index, by = c("run_id", "batch"), suffix = c("", ".idx"))

write.csv(panel_ts, file.path(OUT_DERIVED, "panel_timeseries.csv"), row.names = FALSE)
log_msg("02", sprintf("Wrote panel_timeseries.csv: %d rows across %d runs",
                       nrow(panel_ts), n_distinct(panel_ts$run_id)))

# Sanity check per game: row count should be n_steps (allow off-by-one from
# 0/1 indexing quirks, but never wildly off).
check <- panel_ts %>%
  group_by(run_id, game, n_steps) %>%
  summarise(n_rows = n(), .groups = "drop") %>%
  mutate(diff = abs(n_rows - n_steps))
bad <- check %>% filter(diff > 1)
if (nrow(bad) > 0) {
  log_msg("02", "WARNING: runs whose time_series row count does not match configured n_steps:")
  for (i in seq_len(nrow(bad))) {
    log_msg("02", sprintf("  %s (%s): n_rows=%d expected=%d", bad$run_id[i], bad$game[i], bad$n_rows[i], bad$n_steps[i]))
  }
}
