# 00_pd_classic_rule.R
# The classic prisoner's dilemma rule applied to every agent-step of every prisoner's dilemma run (steps 51-100):
# copy the action of whichever agent scored highest last round among the agent itself and its 8 Moore neighbors
# (ClassicPDAgent.decide in src/agents/classic_pd.py at v1.0-experiments; 20 x 20 torus). This is the benchmark the
# text analysis (python/03_label_deviations.py) compares each LLM decision with, so a prisoner's dilemma deviation is
# "the agent did something other than what the classic rule would do in the same state", as in civil violence.
# When the top scorers tie but chose different actions, the classic agent's choice depends on Mesa's neighbor order;
# those agent-steps are marked AMBIGUOUS and left out of the deviation counts (the count is logged).
# Check: in the classic runs, every non-ambiguous agent-step must reproduce the classic agent's logged action.
# Runs before the Python text analysis (run_all.sh). Output: derived/text/pd_classic_rule.csv.gz
# Usage: Rscript R/00_pd_classic_rule.R   (reads ABM_ROOT and PIPELINE_ROOT like the other stages)

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(nanoparquet)

pd_dirs <- sort(Sys.glob(file.path(ABM_ROOT, "five_simulation", "PD_CV_outputs", "pd_*")))
assert_that(length(pd_dirs) == 35, sprintf("Expected 35 prisoner's dilemma runs (7 arms x 5 seeds), found %d", length(pd_dirs)))
W <- 20; H <- 20
shift <- function(m, dx, dy) {  # value at (x + dx, y + dy) on the torus, for every cell (x, y)
  m[((seq_len(W) - 1 + dx) %% W) + 1, ((seq_len(H) - 1 + dy) %% H) + 1]
}
out <- list(); checks <- list()
for (d in pd_dirs) {
  run <- basename(d)
  p <- read_parquet(file.path(d, "agent_panel.parquet"))
  p <- p[p$step %in% STEADY_STATE_STEPS, ]
  assert_that(nrow(p) == 400 * length(STEADY_STATE_STEPS), sprintf("%s: expected 400 agents x 50 steps", run))
  assert_that(all(p$x %in% 0:19) && all(p$y %in% 0:19), sprintf("%s: grid is not 20 x 20", run))
  res <- vector("list", length(STEADY_STATE_STEPS))
  for (k in seq_along(STEADY_STATE_STEPS)) {
    s <- p[p$step == STEADY_STATE_STEPS[k], ]
    pay <- matrix(NA_real_, W, H); coop <- matrix(NA, W, H)
    pay[cbind(s$x + 1, s$y + 1)] <- s$payoff
    coop[cbind(s$x + 1, s$y + 1)] <- s$action_before == "COOPERATE"
    assert_that(!anyNA(pay) && !anyNA(coop), sprintf("%s step %d: grid not fully occupied", run, STEADY_STATE_STEPS[k]))
    offs <- expand.grid(dx = -1:1, dy = -1:1)   # includes (0, 0): the agent itself
    best <- matrix(-Inf, W, H)
    for (i in seq_len(nrow(offs))) best <- pmax(best, shift(pay, offs$dx[i], offs$dy[i]))
    any_c <- matrix(FALSE, W, H); any_d <- matrix(FALSE, W, H)
    for (i in seq_len(nrow(offs))) {
      top <- shift(pay, offs$dx[i], offs$dy[i]) == best
      a <- shift(coop, offs$dx[i], offs$dy[i])
      any_c <- any_c | (top & a); any_d <- any_d | (top & !a)
    }
    rule <- ifelse(any_c & any_d, "AMBIGUOUS", ifelse(any_c, "COOPERATE", "DEFECT"))
    res[[k]] <- data.frame(run = run, step = s$step, agent_id = s$agent_id, action = s$action,
                           rule_action = rule[cbind(s$x + 1, s$y + 1)], stringsAsFactors = FALSE)
  }
  r <- bind_rows(res)
  if (grepl("^pd_classic_", run)) {   # the classic agents must reproduce their own rule
    ok <- r$rule_action == "AMBIGUOUS" | r$rule_action == r$action
    assert_that(all(ok), sprintf("%s: classic agents disagree with the recomputed rule in %d agent-steps", run, sum(!ok)))
  }
  checks[[run]] <- data.frame(run = run, n = nrow(r), ambiguous = sum(r$rule_action == "AMBIGUOUS"),
                              deviates = sum(r$rule_action != "AMBIGUOUS" & r$rule_action != r$action))
  out[[run]] <- r
}
rule_tab <- bind_rows(out)
dir.create(file.path(OUT_DERIVED, "text"), showWarnings = FALSE, recursive = TRUE)
con <- gzfile(file.path(OUT_DERIVED, "text", "pd_classic_rule.csv.gz"), "w")
write.csv(rule_tab[, c("run", "step", "agent_id", "rule_action")], con, row.names = FALSE)
close(con)
chk <- bind_rows(checks)
print(chk)
log_msg("00", sprintf("PD classic rule: %d agent-steps, %d ambiguous (%.2f%%); wrote derived/text/pd_classic_rule.csv.gz",
                      nrow(rule_tab), sum(chk$ambiguous), 100 * sum(chk$ambiguous) / nrow(rule_tab)))
