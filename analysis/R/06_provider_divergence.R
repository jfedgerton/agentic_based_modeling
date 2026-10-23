# 06_provider_divergence.R
# Headline analysis for the reframed paper: do OpenAI (GPT) and DeepSeek agents
# diverge from EACH OTHER and from the classic rule-based baseline, game by
# game and prompt mode by prompt mode?
#
# Unit of analysis = one run (one seed). Only the five_seed batch is used here:
# it is the only batch with the full 2 providers x 3 modes x 5 seeds + classic
# design for PD and CV. IC has no DeepSeek arm that survives the parse-failure
# gate, so the out-of-sample divergence is reported separately (parse-failure
# rates, 04_interstate_conflict.R) rather than forced into this frame.
#
# Per run we take TAIL means (steps 51-100, i.e. after the burn-in) of each
# game's behavioural metrics:
#   PD: cooperation_rate, spatial_clustering, mean_payoff
#   CV: rebellion_rate, jailed_rate, mobilized_share (= 1 - quiet_rate = active + jailed)
# Primary metric (the one used in Fig. 1 and in-text numbers):
#   PD = cooperation_rate
#   CV = mobilized_share. Rebellion_rate alone is tiny and noisy because rebels
#        are jailed within a step or two; the share of citizens who are active
#        OR jailed is the stable footprint of rebellion. Confirmed by Jared
#        Sept 27, 2026. (Rebellion rate alone gives the same direction.)
#
# Three analyses, all seed 123:
#   (A) Pairwise contrasts on the primary metric for every (game, mode):
#       GPT - classic, DeepSeek - classic, GPT - DeepSeek. Exact permutation
#       test (all choose(10,5) = 252 relabelings of 5 vs 5 seeds), two-sided.
#       With 5 seeds per arm the SMALLEST attainable p is 2/252 = 0.0079 --
#       stated in every table note so nobody reads p = 0.008 as "barely".
#   (B) Multivariate divergence: z-score the three metrics within game (across
#       all five_seed runs of that game), take the Euclidean distance between
#       arm centroids. Gives the "divergence triangle" d(GPT,classic),
#       d(DeepSeek,classic), d(GPT,DeepSeek) per (game, mode), with an exact
#       permutation p for each side.
#   (C) Framing sensitivity by provider: how far each provider's primary metric
#       moves across modes, and the provider x framing interaction
#       [(roleplayer - calculator)_GPT - (roleplayer - calculator)_DeepSeek]
#       (and the same for reasoner), with a seed-level bootstrap 95% CI
#       (2,000 draws, resampling seeds within each of the four arms).

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)
library(purrr)

TAIL_START <- 51
PRIMARY_METRIC <- c(pd = "cooperation_rate", cv = "mobilized_share")
METRICS <- list(
  pd = c("cooperation_rate", "spatial_clustering", "mean_payoff"),
  cv = c("rebellion_rate", "jailed_rate", "mobilized_share")
)
MODES <- c("calculator", "reasoner", "roleplayer")
N_BOOT <- 2000

panel_ts <- read_csv(file.path(OUT_DERIVED, "panel_timeseries.csv"), show_col_types = FALSE)

# ---- Per-run tail means ------------------------------------------------------
five <- panel_ts %>%
  filter(batch == "five_seed", !excluded, game %in% c("pd", "cv")) %>%
  mutate(mobilized_share = if_else(game == "cv", 1 - quiet_rate, NA_real_))

# run_id encodes the replication seed ("..._seed43_..."); config.json's
# random_seed is the BASE seed (42) for every replication, so do not group on it.
five <- five %>% mutate(rep_seed = as.integer(sub(".*_seed([0-9]+)_.*", "\\1", run_id)))
assert_that(!any(is.na(five$rep_seed)), "Could not parse replication seed from a five_seed run_id")

# Tail = second half of each run (steps 51-100 on the real 100-step data;
# TAIL_START is recorded for the paper, the rule itself is "second half" so the
# synthetic test set with shorter runs exercises the same code path).
five <- five %>% group_by(run_id) %>% mutate(run_len = max(step)) %>% ungroup()
assert_that(all(five$run_len[five$batch == "five_seed"] == max(five$run_len)), "five_seed runs differ in length")
run_tail <- five %>%
  filter(step > run_len / 2) %>%
  group_by(game, agent_type, provider, run_id, rep_seed) %>%
  summarise(across(c(cooperation_rate, spatial_clustering, mean_payoff,
                     rebellion_rate, jailed_rate, mobilized_share), ~ mean(.x, na.rm = TRUE)),
            n_tail_steps = n(), .groups = "drop") %>%
  mutate(arm = if_else(agent_type == "classic", "classic", paste(provider, agent_type, sep = "_")))

# Design check: every (game, LLM arm) and classic should have exactly 5 runs
design <- run_tail %>% count(game, agent_type, provider)
log_msg("06", "Five-seed design cells:\n", paste(capture.output(print(design, n = 50)), collapse = "\n"))
N_SEEDS <- unique(design$n)
assert_that(length(N_SEEDS) == 1 && N_SEEDS >= 2,
            "Every five_seed cell must have the same number of runs (>= 2) -- check exclusions / staging")
if (N_SEEDS != 5) log_msg("06", sprintf("WARNING: %d seeds per cell, not 5 (fine for the synthetic test set, not for the paper)", N_SEEDS))
MIN_P <- 2 / choose(2 * N_SEEDS, N_SEEDS)
assert_that(length(unique(run_tail$n_tail_steps)) == 1, "Runs have different numbers of tail steps")
if (unique(run_tail$n_tail_steps) != 50) log_msg("06", "WARNING: tail is not steps 51-100 (fine for synthetic data only)")

write.csv(run_tail, file.path(OUT_DERIVED, "run_tail_metrics.csv"), row.names = FALSE)

# ---- Helpers -----------------------------------------------------------------
# Exact two-sided permutation p for a difference in means (all relabelings).
exact_perm_p <- function(x, y) {
  pooled <- c(x, y); n_x <- length(x)
  combos <- combn(length(pooled), n_x)
  obs <- abs(mean(x) - mean(y))
  perm <- apply(combos, 2, function(idx) abs(mean(pooled[idx]) - mean(pooled[-idx])))
  # tolerance guards against floating-point ties being counted as "less extreme"
  mean(perm >= obs - 1e-12)
}

# Exact permutation p for a statistic computed on two groups of rows of a matrix.
exact_perm_p_matrix <- function(mx, my, stat_fn) {
  pooled <- rbind(mx, my); n_x <- nrow(mx)
  combos <- combn(nrow(pooled), n_x)
  obs <- stat_fn(mx, my)
  perm <- apply(combos, 2, function(idx) stat_fn(pooled[idx, , drop = FALSE], pooled[-idx, , drop = FALSE]))
  mean(perm >= obs - 1e-12)
}

centroid_dist <- function(mx, my) sqrt(sum((colMeans(mx) - colMeans(my))^2))

get_vals <- function(g, arm_name, metric) {
  run_tail %>% filter(game == g, arm == arm_name) %>% arrange(rep_seed) %>% pull(!!metric)
}

# ---- (A) Pairwise contrasts on the primary metric ----------------------------
pairwise <- list()
for (g in c("pd", "cv")) {
  m <- PRIMARY_METRIC[[g]]
  classic_vals <- get_vals(g, "classic", m)
  for (mode in MODES) {
    gpt <- get_vals(g, paste0("openai_", mode), m)
    ds  <- get_vals(g, paste0("deepseek_", mode), m)
    contrasts <- list(
      c("GPT - classic", "gpt", "classic"),
      c("DeepSeek - classic", "ds", "classic"),
      c("GPT - DeepSeek", "gpt", "ds")
    )
    vals <- list(gpt = gpt, ds = ds, classic = classic_vals)
    for (ct in contrasts) {
      x <- vals[[ct[2]]]; y <- vals[[ct[3]]]
      pairwise[[length(pairwise) + 1]] <- tibble(
        game = g, mode = mode, metric = m, contrast = ct[1],
        mean_x = mean(x), mean_y = mean(y), diff = mean(x) - mean(y),
        sd_x = sd(x), sd_y = sd(y),
        exact_p = exact_perm_p(x, y), min_attainable_p = MIN_P
      )
    }
  }
}
pairwise <- bind_rows(pairwise)
write.csv(pairwise, file.path(OUT_DERIVED, "provider_divergence_pairwise.csv"), row.names = FALSE)
log_msg("06", "Pairwise contrasts:\n", paste(capture.output(print(pairwise %>% select(game, mode, contrast, mean_x, mean_y, diff, exact_p), n = 50)), collapse = "\n"))

# ---- (B) Multivariate divergence triangle ------------------------------------
multivar <- list()
for (g in c("pd", "cv")) {
  mets <- METRICS[[g]]
  zdat <- run_tail %>% filter(game == g) %>%
    mutate(across(all_of(mets), ~ {
      s <- sd(.x); if (s == 0) 0 else (.x - mean(.x)) / s
    }))
  get_mat <- function(arm_name) as.matrix(zdat %>% filter(arm == arm_name) %>% select(all_of(mets)))
  classic_m <- get_mat("classic")
  for (mode in MODES) {
    gpt_m <- get_mat(paste0("openai_", mode)); ds_m <- get_mat(paste0("deepseek_", mode))
    sides <- list(
      list("d(GPT, classic)", gpt_m, classic_m),
      list("d(DeepSeek, classic)", ds_m, classic_m),
      list("d(GPT, DeepSeek)", gpt_m, ds_m)
    )
    for (sd_ in sides) {
      multivar[[length(multivar) + 1]] <- tibble(
        game = g, mode = mode, side = sd_[[1]],
        distance = centroid_dist(sd_[[2]], sd_[[3]]),
        exact_p = exact_perm_p_matrix(sd_[[2]], sd_[[3]], centroid_dist),
        metrics = paste(mets, collapse = "+")
      )
    }
  }
}
multivar <- bind_rows(multivar)
write.csv(multivar, file.path(OUT_DERIVED, "provider_divergence_multivariate.csv"), row.names = FALSE)
log_msg("06", "Divergence triangle:\n", paste(capture.output(print(multivar, n = 50)), collapse = "\n"))

# ---- (C) Framing sensitivity and provider x framing interaction ------------
set.seed(SEED)
boot_did <- function(a1, a0, b1, b0) {
  # (a1 - a0) - (b1 - b0), resampling seeds within each arm
  replicate(N_BOOT, {
    (mean(sample(a1, replace = TRUE)) - mean(sample(a0, replace = TRUE))) -
      (mean(sample(b1, replace = TRUE)) - mean(sample(b0, replace = TRUE)))
  })
}

sensitivity <- list()
interaction <- list()
for (g in c("pd", "cv")) {
  m <- PRIMARY_METRIC[[g]]
  for (prov in c("openai", "deepseek")) {
    means <- sapply(MODES, function(md) mean(get_vals(g, paste0(prov, "_", md), m)))
    sensitivity[[length(sensitivity) + 1]] <- tibble(
      game = g, provider = prov, metric = m,
      calculator = means[["calculator"]], reasoner = means[["reasoner"]], roleplayer = means[["roleplayer"]],
      range_across_modes = max(means) - min(means)
    )
  }
  for (contrast_mode in c("reasoner", "roleplayer")) {
    a1 <- get_vals(g, paste0("openai_", contrast_mode), m);   a0 <- get_vals(g, "openai_calculator", m)
    b1 <- get_vals(g, paste0("deepseek_", contrast_mode), m); b0 <- get_vals(g, "deepseek_calculator", m)
    obs <- (mean(a1) - mean(a0)) - (mean(b1) - mean(b0))
    draws <- boot_did(a1, a0, b1, b0)
    interaction[[length(interaction) + 1]] <- tibble(
      game = g, metric = m, framing_contrast = paste0(contrast_mode, " - calculator"),
      gpt_effect = mean(a1) - mean(a0), deepseek_effect = mean(b1) - mean(b0),
      interaction = obs,
      ci_low = unname(quantile(draws, 0.025)), ci_high = unname(quantile(draws, 0.975))
    )
  }
}
sensitivity <- bind_rows(sensitivity)
interaction <- bind_rows(interaction)
write.csv(sensitivity, file.path(OUT_DERIVED, "framing_sensitivity_by_provider.csv"), row.names = FALSE)
write.csv(interaction, file.path(OUT_DERIVED, "provider_by_framing_interaction.csv"), row.names = FALSE)
log_msg("06", "Framing sensitivity:\n", paste(capture.output(print(sensitivity)), collapse = "\n"))
log_msg("06", "Provider x framing interaction:\n", paste(capture.output(print(interaction)), collapse = "\n"))

# ---- Sanity checks that fail loudly -----------------------------------------
assert_that(nrow(pairwise) == 2 * 3 * 3, "Expected 18 pairwise contrasts (2 games x 3 modes x 3)")
assert_that(all(pairwise$exact_p >= MIN_P - 1e-9 & pairwise$exact_p <= 1),
            "Exact permutation p outside [min attainable, 1] -- permutation code is wrong")
assert_that(nrow(multivar) == 18, "Expected 18 divergence-triangle sides")
log_msg("06", "Done.")
