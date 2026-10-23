# 09_tables.R
# LaTeX (booktabs) tables for the provider-divergence framing. Written by hand
# with sprintf rather than xtable so the layout (multi-column headers, p-values
# in parentheses, table notes) is exactly what goes in the paper.

source(file.path(Sys.getenv("PIPELINE_ROOT", "."), "R", "00_config.R"))
library(dplyr)
library(tidyr)
library(readr)

fmt_num <- function(x, d = 3) ifelse(abs(x) < 10^(-d) & x != 0, formatC(x, format = "e", digits = 1), formatC(x, format = "f", digits = d))
fmt_p <- function(p) ifelse(p < 0.01, sprintf("%.3f", p), sprintf("%.2f", p))
write_tex <- function(lines, name) {
  path <- file.path(OUT_TABLES, name)
  writeLines(lines, path)
  log_msg("09", "Wrote ", path)
}
MODE_TEX <- c(calculator = "Calculator", reasoner = "Reasoner", roleplayer = "Role-player")

# ---- Table 2: provider divergence by game and mode ---------------------------
pw <- read_csv(file.path(OUT_DERIVED, "provider_divergence_pairwise.csv"), show_col_types = FALSE)
t2 <- pw %>%
  mutate(cell = paste0(fmt_num(diff), " (", fmt_p(exact_p), ")")) %>%
  select(game, mode, contrast, cell) %>%
  pivot_wider(names_from = contrast, values_from = cell)
means <- pw %>%
  filter(contrast %in% c("GPT - classic", "DeepSeek - classic")) %>%
  mutate(who = if_else(contrast == "GPT - classic", "gpt", "ds")) %>%
  select(game, mode, who, mean_x, classic = mean_y) %>%
  pivot_wider(names_from = who, values_from = mean_x)
t2 <- left_join(means, t2, by = c("game", "mode")) %>%
  mutate(game = factor(game, levels = c("pd", "cv")), mode = factor(mode, levels = names(MODE_TEX))) %>%
  arrange(game, mode)
assert_that(nrow(t2) == 6, "Table 2 should have 6 rows (2 games x 3 modes)")

rows <- c()
for (g in c("pd", "cv")) {
  hdr <- if (g == "pd") "\\multicolumn{7}{l}{\\textit{Prisoner's dilemma: cooperation rate}} \\\\"
         else "\\midrule\n\\multicolumn{7}{l}{\\textit{Civil violence: mobilized share (active + jailed)}} \\\\"
  rows <- c(rows, hdr)
  for (i in which(t2$game == g)) {
    r <- t2[i, ]
    rows <- c(rows, sprintf("\\quad %s & %s & %s & %s & %s & %s & %s \\\\",
                            MODE_TEX[[as.character(r$mode)]], fmt_num(r$classic), fmt_num(r$gpt), fmt_num(r$ds),
                            r$`GPT - classic`, r$`DeepSeek - classic`, r$`GPT - DeepSeek`))
  }
}
write_tex(c(
  "\\begin{table}[t]", "\\centering", "\\small",
  "\\caption{Provider divergence by game and prompt mode. Arm means of the primary outcome over steps 51--100; differences with exact two-sided permutation $p$-values in parentheses (5 vs.\\ 5 seeds, all 252 relabelings).}",
  "\\label{tab:divergence}",
  "\\begin{tabular}{lcccccc}", "\\toprule",
  " & \\multicolumn{3}{c}{Arm mean} & \\multicolumn{3}{c}{Difference ($p$)} \\\\",
  "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
  "Mode & Classic & GPT & DeepSeek & GPT $-$ classic & DeepSeek $-$ classic & GPT $-$ DeepSeek \\\\",
  "\\midrule", rows, "\\bottomrule", "\\end{tabular}",
  "\\par\\smallskip\\footnotesize\\raggedright Note: with 5 seeds per arm the smallest attainable exact $p$ is $2/252 = 0.008$ (both tails). Seeds vary the environment only (temperature 0). Classic arm is identical across modes (it receives no prompt).",
  "\\end{table}"), "table2_provider_divergence.tex")

# ---- ED Table 1: provider x framing interaction ------------------------------
inter <- read_csv(file.path(OUT_DERIVED, "provider_by_framing_interaction.csv"), show_col_types = FALSE)
sens <- read_csv(file.path(OUT_DERIVED, "framing_sensitivity_by_provider.csv"), show_col_types = FALSE)
rows <- c()
for (i in seq_len(nrow(inter))) {
  r <- inter[i, ]
  rows <- c(rows, sprintf("%s & %s & %s & %s & %s [%s, %s] \\\\",
                          if_else(r$game == "pd", "PD", "CV"), gsub(" - ", " $-$ ", r$framing_contrast),
                          fmt_num(r$gpt_effect), fmt_num(r$deepseek_effect),
                          fmt_num(r$interaction), fmt_num(r$ci_low), fmt_num(r$ci_high)))
}
rows2 <- c()
for (i in seq_len(nrow(sens))) {
  r <- sens[i, ]
  rows2 <- c(rows2, sprintf("%s & %s & %s & %s & %s & %s \\\\",
                            if_else(r$game == "pd", "PD", "CV"), if_else(r$provider == "openai", "GPT", "DeepSeek"),
                            fmt_num(r$calculator), fmt_num(r$reasoner), fmt_num(r$roleplayer), fmt_num(r$range_across_modes)))
}
write_tex(c(
  "\\begin{table}[t]", "\\centering", "\\small",
  "\\caption{Framing sensitivity by provider. Top: primary-outcome mean by mode and the range across modes. Bottom: provider $\\times$ framing interaction, $(\\text{GPT effect}) - (\\text{DeepSeek effect})$, with 95\\% seed-bootstrap CIs (2{,}000 draws, seed 123).}",
  "\\label{tab:ed-framing}",
  "\\begin{tabular}{llcccc}", "\\toprule",
  "Game & Provider & Calculator & Reasoner & Role-player & Range \\\\", "\\midrule", rows2,
  "\\midrule",
  "Game & Framing contrast & GPT effect & DeepSeek effect & \\multicolumn{2}{c}{Interaction [95\\% CI]} \\\\", "\\midrule",
  sub("\\\\\\\\$", "", rows) %>% paste0(" \\\\") %>% sub("& ([^&]*) \\\\\\\\$", "& \\\\multicolumn{2}{c}{\\1} \\\\\\\\", .),
  "\\bottomrule", "\\end{tabular}", "\\end{table}"), "ed_table1_framing_interaction.tex")

# ---- ED Table 2: interstate conflict (out of sample) -------------------------
ic <- read_csv(file.path(OUT_DERIVED, "ic_dynamics_summary.csv"), show_col_types = FALSE)
ri <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE)
ic_pf <- ri %>% filter(game == "ic", batch == "one_seed") %>%
  group_by(agent_type, provider) %>% summarise(pfr = mean(parse_fail_rate), excluded = any(excluded), .groups = "drop")
rows <- c()
for (arm in c("classic", "calculator", "reasoner", "roleplayer")) {
  r <- ic %>% filter(agent_type == arm)
  pf_gpt <- ic_pf %>% filter(agent_type == arm, provider %in% c("openai", "mock")) %>% pull(pfr)
  pf_ds  <- ic_pf %>% filter(agent_type == arm, provider == "deepseek") %>% pull(pfr)
  rows <- c(rows, sprintf("%s & %s & %s & %s & %s & %s & %s \\\\",
                          if (arm == "classic") "Classic" else paste0("GPT ", tolower(MODE_TEX[[arm]])),
                          fmt_num(r$war_frequency), fmt_num(r$welfare_loss, 1), fmt_num(r$bluff_rate),
                          fmt_num(r$bluff_success_rate), sprintf("%.1f\\%%", 100 * pf_gpt[1]),
                          if (length(pf_ds)) sprintf("%.1f\\%% (excl.)", 100 * pf_ds) else "---"))
}
write_tex(c(
  "\\begin{table}[t]", "\\centering", "\\small",
  "\\caption{Out-of-sample game (interstate crisis bargaining), one seed per arm. Behavioural outcomes are GPT only: every DeepSeek arm exceeds the 5\\% parse-failure gate, and failed decisions fall back to the classical rule, so its outcomes are not LLM behaviour.}",
  "\\label{tab:ed-ic}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
  "\\begin{tabular}{lcccccc}", "\\toprule",
  " & War & Welfare & Bluff & Bluff & \\multicolumn{2}{c}{Parse failure} \\\\",
  "\\cmidrule(lr){6-7}",
  "Arm & frequency & loss & rate & success & GPT & DeepSeek \\\\",
  "\\midrule", rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"), "ed_table2_interstate_conflict.tex")

# ---- Table S3: run completeness and parse failures (moved here from the retired 11_tables.R) -----------
# One row per batch x game x arm. Duplicate replays (see 01_run_index.R) are not counted as runs.
ri_all <- read_csv(file.path(OUT_DERIVED, "run_index.csv"), show_col_types = FALSE) %>% filter(!excluded_duplicate)
s3 <- ri_all %>%
  group_by(batch, game, agent_type, provider) %>%
  summarise(n_runs = n(), n_excluded = sum(excluded), pf = mean(parse_fail_rate, na.rm = TRUE), .groups = "drop") %>%
  arrange(batch, game, agent_type, provider)
s3_rows <- sprintf("%s & %s & %s & %s & %d & %d & %.4f \\\\", gsub("_", "\\\\_", s3$batch), s3$game,
                   s3$agent_type, s3$provider, s3$n_runs, s3$n_excluded, s3$pf)
write_tex(c(
  "\\begin{table}[t]", "\\centering", "\\footnotesize",
  sprintf("\\caption{Run completeness and parse-failure rates by batch, game, mode and provider (%d distinct runs). A run is excluded if more than 5\\%% of its LLM calls fail to parse, or if it is a rule-based benchmark whose game settings differ from the LLM runs it is compared with.}", nrow(ri_all)),
  "\\label{tab:appendix-b-completeness}", "\\begin{tabular}{lllrrrr}", "\\toprule",
  "Batch & Game & Mode & Provider & Runs & Excluded & Mean parse failure \\\\", "\\midrule", s3_rows,
  "\\bottomrule", "\\end{tabular}", "\\end{table}"), "appendixB_run_completeness.tex")

log_msg("09", "Done.")
