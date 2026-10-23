# 01_run_index.R
# Walk every run directory across all five batches, parse config.json +
# summary.json, and build one manifest row per run with data-quality flags.
# This is the single source of truth every downstream script joins against --
# nothing downstream re-derives exclusion logic.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(jsonlite)
library(dplyr)
library(purrr)
library(tibble)

log_msg("01", "Building run index across batches: ", paste(names(RUN_BATCHES), collapse = ", "))

read_run_row <- function(run_dir, batch_name) {
  run_id <- basename(run_dir)
  config_path  <- file.path(run_dir, "config.json")
  summary_path <- file.path(run_dir, "summary.json")

  assert_that(file.exists(config_path),
              sprintf("Run %s has no config.json -- every staged run must have one", run_id))
  cfg <- fromJSON(config_path)

  complete <- file.exists(summary_path)
  smry <- if (complete) fromJSON(summary_path) else list(
    total_llm_interactions = NA_integer_,
    total_parse_failures = NA_integer_,
    runtime_seconds = NA_real_
  )

  game     <- infer_game(run_id)
  agent    <- cfg$agent$type
  provider <- cfg$llm$provider
  model    <- cfg$llm$model
  # The folder name carries the command-line seed; config.json's random_seed is a stale default (always 42)
  seed     <- as.integer(sub(".*_seed([0-9]+)_.*", "\\1", run_id))
  seed_config <- cfg$experiment$random_seed
  n_steps  <- cfg$experiment$num_steps
  language <- if (!is.null(cfg$civil_violence$language)) cfg$civil_violence$language else NA_character_
  signal_form <- if (!is.null(cfg$interstate_conflict$signal_form)) cfg$interstate_conflict$signal_form else NA_character_
  bluffing <- if (!is.null(cfg$interstate_conflict$bluffing_enabled)) cfg$interstate_conflict$bluffing_enabled else NA

  total_llm   <- if (is.null(smry$total_llm_interactions)) NA_integer_ else smry$total_llm_interactions
  total_fail  <- if (is.null(smry$total_parse_failures)) NA_integer_ else smry$total_parse_failures
  parse_fail_rate <- if (!is.na(total_llm) && total_llm > 0) total_fail / total_llm else 0

  tibble(
    batch = batch_name,
    run_id = run_id,
    run_dir = run_dir,
    game = game,
    agent_type = agent,
    provider = provider,
    model = model,
    seed = seed,
    seed_config = seed_config,
    n_steps = n_steps,
    language = language,
    ic_signal_form = signal_form,
    ic_bluffing_enabled = bluffing,
    total_llm_interactions = total_llm,
    total_parse_failures = total_fail,
    parse_fail_rate = parse_fail_rate,
    is_complete = complete,
    excluded_incomplete = !complete,
    excluded_parse_failure = !is.na(parse_fail_rate) & parse_fail_rate > PARSE_FAILURE_EXCLUSION_THRESHOLD,
    # Benchmark must use the same game settings as the LLM runs it is compared with. Every LLM crisis-game run
    # has bluffing enabled, so the one classic crisis-game run with bluffing disabled is not a valid benchmark.
    excluded_config_mismatch = game == "ic" & agent == "classic" & isFALSE(bluffing),
    excluded = !complete | (!is.na(parse_fail_rate) & parse_fail_rate > PARSE_FAILURE_EXCLUSION_THRESHOLD) |
      (game == "ic" & agent == "classic" & isFALSE(bluffing))
  )
}

all_rows <- imap(RUN_BATCHES, function(batch_dir, batch_name) {
  if (!dir.exists(batch_dir)) {
    log_msg("01", "WARNING: batch dir missing, skipping: ", batch_dir)
    return(NULL)
  }
  run_dirs <- list.dirs(batch_dir, recursive = FALSE)
  log_msg("01", sprintf("%s: %d run directories found", batch_name, length(run_dirs)))
  map_dfr(run_dirs, read_run_row, batch_name = batch_name)
}) %>% bind_rows()

assert_that(nrow(all_rows) > 0, "Run index is empty -- check ABM_ROOT and RUN_BATCHES paths")
assert_that(!any(is.na(all_rows$seed)), "Could not parse the seed from a run folder name")

# The one_seed folder also holds byte-identical replays of five_seed seed-42 runs (same time_series.csv),
# some of them twice. Keep one copy: within each game x agent x provider, a run whose time_series.csv
# matches an earlier one (five_seed first) is excluded as a duplicate. Duplicates are only searched
# within the same arm, because different arms can legitimately produce identical series (e.g. the
# all-defect prisoner's dilemma equilibrium of classic and DeepSeek agents).
all_rows <- all_rows %>%
  mutate(ts_md5 = unname(tools::md5sum(file.path(run_dir, "time_series.csv"))),
         batch_order = if_else(batch == "five_seed", 1L, 2L)) %>%
  arrange(game, agent_type, provider, batch_order, batch, run_id) %>%
  group_by(game, agent_type, provider) %>%
  mutate(excluded_duplicate = !is.na(ts_md5) & duplicated(ts_md5)) %>%
  ungroup() %>%
  mutate(excluded = excluded | excluded_duplicate) %>%
  select(-batch_order)
assert_that(!any(all_rows$excluded_duplicate & all_rows$batch == "five_seed"),
            "Two five_seed runs of the same arm have identical time series")
log_msg("01", sprintf("Duplicate replays excluded: %d (%s)", sum(all_rows$excluded_duplicate),
                       paste(sort(unique(all_rows$batch[all_rows$excluded_duplicate])), collapse = ", ")))
# Within each batch and game, every run kept for analysis must share the same crisis-game settings
cfg_check <- all_rows %>% filter(!excluded, game == "ic") %>% group_by(batch) %>%
  summarise(n_bluff = n_distinct(ic_bluffing_enabled), n_signal = n_distinct(ic_signal_form), .groups = "drop")
assert_that(all(cfg_check$n_bluff == 1 & cfg_check$n_signal == 1),
            "Crisis-game runs kept for analysis differ in bluffing or signal form within a batch")

# Sanity checks that should never fail on this dataset; if they do, something
# about the data changed and every downstream script's assumptions are stale.
assert_that(all(all_rows$game %in% c("pd", "cv", "ic")), "Unexpected game value in run index")
assert_that(all(all_rows$provider %in% c("openai", "deepseek", "mock") | is.na(all_rows$provider)),
            "Unexpected provider value in run index")

n_excluded <- sum(all_rows$excluded)
log_msg("01", sprintf("Total runs indexed: %d | excluded: %d (incomplete: %d, parse-failure>%.0f%%: %d, benchmark config mismatch: %d)",
                       nrow(all_rows), n_excluded,
                       sum(all_rows$excluded_incomplete),
                       PARSE_FAILURE_EXCLUSION_THRESHOLD * 100,
                       sum(all_rows$excluded_parse_failure & !all_rows$excluded_incomplete),
                       sum(all_rows$excluded_config_mismatch)))

if (n_excluded > 0) {
  log_msg("01", "Excluded runs:")
  excl <- all_rows %>% filter(excluded) %>% select(batch, run_id, game, agent_type, provider, is_complete, parse_fail_rate)
  for (i in seq_len(nrow(excl))) {
    log_msg("01", sprintf("  %s/%s [%s/%s/%s] complete=%s parse_fail_rate=%.2f",
                           excl$batch[i], excl$run_id[i], excl$game[i], excl$agent_type[i],
                           excl$provider[i], excl$is_complete[i], excl$parse_fail_rate[i]))
  }
}

write.csv(all_rows, file.path(OUT_DERIVED, "run_index.csv"), row.names = FALSE)
log_msg("01", sprintf("Wrote %s (%d rows)", file.path(OUT_DERIVED, "run_index.csv"), nrow(all_rows)))
