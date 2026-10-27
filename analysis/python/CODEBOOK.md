# Codebook v1 for agent rationales in deviating decisions (Oct 23, 2026)

Implemented in `codebook_rules.py` (civil violence) and `pd_rationale_codes.csv` (prisoner's dilemma), applied by
`03_label_deviations.py`, validated with `04_validation_sample.py`.

## Unit of analysis

One deviating decision, meaning one agent in one step (steps 51 to 100, five seeds) whose action differs from the
action the classic rule-based agent would take in the same state. In the prisoner's dilemma the classic rule copies
the action of the agent that scored highest last round among the agent itself and its 8 neighbors (20 x 20 torus);
`R/00_pd_classic_rule.R` computes it for every agent-step from `agent_panel.parquet`, and in the classic runs it
reproduces every logged action. When top scorers tie but chose different actions, the classic agent's choice depends on
the simulator's neighbor order, so those agent-steps (0.5% of all PD agent-steps; 3.1% of GPT calculator and 0.4% of GPT
role-player agent-steps) are marked ambiguous and skipped. (Codebook v1 counted every cooperation as a deviation, because
the classic agents always defect in steady state; v2 compares with the rule itself, as in civil violence.) In civil violence the classic agent is active if and only if
G - N > 0.1, where G = H(1 - L) and N = R(1 - exp(-2.3 C / (A + 1))) are computed from the values printed in the
agent's prompt (two decimals). The benchmark is therefore the classic rule applied to the information the LLM agent
was given; using the simulator's full-precision state instead changes the reasoner and role-player cell counts by up
to 6% (agents near the 0.1 threshold) and leaves the reason shares essentially unchanged. Decisions whose LLM response
failed to parse are excluded, because the logged action is the classic fallback rather than the agent's choice.
The text coded is the agent's `short_rationale`. Each decision gets exactly one category.

## Categories

| Category | Definition | Games |
|---|---|---|
| Says the other option is better | The rationale's conclusion favors the action the agent did not take. | PD, CV calculator |
| Misreads or miscalculates payoffs | The rationale favors the action taken, but the payoffs it states or implies are wrong. In the PD, it asserts a payoff relation the matrix contradicts (for example that cooperating earns more against defectors, or "exploits" them). In CV calculator, the action contradicts the sign of G - N computed from the prompt's own numbers. | PD, CV calculator |
| Follows its own calculation | CV calculator: the rationale favors the action taken and the action matches the sign of G - N, so the agent applies a threshold of 0 where the classic rule uses 0.1. PD: the agent defects where the classic rule cooperates and its rationale states the correct payoffs (defection dominates). | CV calculator, PD |
| Reciprocity | PD only. Cooperates citing mutual, reciprocal or cooperative gains with neighbors. | PD |
| Coordination | PD only. Cooperates to match neighbors or the best-performing neighbor strategy, without citing mutual gains. | PD |
| Arrest risk, no fellow protesters | CV reasoner and role-player quiet decisions citing both features below. | CV |
| Arrest risk, other reasons | Cites arrest risk but not the absence of fellow protesters. | CV |
| No fellow protesters, no arrest risk | Cites the absence of fellow protesters but not arrest risk. | CV |
| Other or unclassified | None of the above applies, or the rationale states no classifiable preference. The 42 active decisions in CV role-player cells are not coded and fall here. | All |

## Decision rules

### CV calculator (machine, `cv_calc_pref`)

Step 1 finds the option the rationale says is better (A = stay quiet, B = become active, the prompt's neutral labels;
only the capital letters count, so the article "a" is never read as option A).
Every rule below is matched across the whole text; the match that ends last in the text wins, because rationales
state their conclusion last.

| Rule | Pattern (case-insensitive) | Preference |
|---|---|---|
| R1 | choice verb + option: "choose / pick / select / go with / keep / switch to / so / therefore ... [the lower-disutility option] X" | X |
| R2 | positive verdict: "X is / remains / gives ... [much / strictly] better / lower / preferable / optimal / zero"; "X minimizes"; "making X the better choice"; "lower disutility for X" | X |
| R3 | negative verdict: "X is / has [much] higher / worse / larger"; "X is not advantageous" | the other option |
| R4 | comparison within one sentence: "X ... lower / better than Y" or "X ... higher / worse than Y" | X, or Y |
| R5 | trailing verdict on the most recent option: "X gives 0.04, which is lower / higher" | X, or the other option |
| R6 | "X is the [safer / better / intended] choice"; "X appears preferable" | X |
| R7 | "the [safer immediate] choice is X" | X |
| R8 | a conclusion after a colon at the end of a sentence: "choose the lower disutility: A." | X |
| none | no rule matches | unclassified |

Step 2 is deterministic. Preference differs from the action, so the category is Says the other option is better.
Preference equals the action and the action matches the sign of G - N, so Follows its own calculation. Preference
equals the action and the action contradicts the sign of G - N, so Misreads or miscalculates payoffs.

### CV reasoner and role-player quiet decisions (machine, `cv_quiet_reason`)

Two binary features are coded independently, then crossed into the four CV categories.

| Feature | Present when the rationale ... |
|---|---|
| Arrest risk | names arrest, detention, jail, police, enforcement, danger, or risk (but not "risk aversion" or "risk tolerance") as a cost |
| No fellow protesters | says no / zero / few / only one fellow protesters, active agents, neighbors, peers, crowd, support or momentum; or that turnout or protest activity is absent, low, weak or inactive; or that the agent would act alone |

### Prisoner's dilemma (human census, `pd_rationale_codes.csv`)

GPT agents produce only 17 distinct cooperating rationales and 4 distinct defecting rationales among PD deviations, so
every distinct text is coded by hand and the code is applied by exact prefix match. Precedence when a text fits more
than one category is Says the other option is better, then Misreads, then Reciprocity, then Coordination. The four
defecting texts (GPT calculator 3, GPT role-player 1) all state that defection pays more and get the payoffs right, so
they are coded Follows its own calculation (codebook v2, Oct 26).

## Validation

`04_validation_sample.py` draws a stratified random sample (seed 123) and writes a blind file with no machine labels.
Strata are the machine's own outputs, so the sample estimates the precision of each machine label. The coder fills in
T1 (A, B or none), T2 (two 0/1 features) and T3 (one PD category). We report per-stratum agreement, overall accuracy
weighted by stratum size, and Cohen's kappa, in the Supplementary Materials.

## T4. Crisis game: why GPT agents overstate their strength (Fig. 5B; `05_ic_signal_reasons.py`)

Unit. One signal-phase call of a GPT crisis-game agent, steps 51-100, in which the agent signals a higher level than
its honest one. Honest level, as in the classic agent: capability above 0.66 is high, above 0.33 medium, else low.
Capability is the value printed in the agent's own prompt. Signal meaning is taken from the prompt the agent
answered (calculator S1/S2/S3 = low/medium/high value_a; reasoner and role-player WEAK/MODERATE/STRONG), not from the
simulation's stored label, which reverses the calculator scale (see README). Unparsed calls are skipped.

Categories, first match wins (rules R1-R5 in `05_ic_signal_reasons.py`):

| Rule | Category | The rationale ... |
|---|---|---|
| R1 | Misreads own strength | describes its own capability or value_a as high, strong or above average (vague words such as decent or solid do not count), and its capability is below 0.5 |
| R2 | Misreads the rules | says the signal changes role assignment, matching or partner selection (roles are fixed by grid position and step parity, and neighbors never change) |
| R3 | Bluffs openly | says, in a sentence that does not reject or warn against it, that it is bluffing, overstating, overclaiming, inflating, exaggerating or misleading, or that its true strength is hidden or unverifiable ("avoid an empty bluff" or "overstating risks war" does not count, nor does repeating the prompt's note that signals may be strategic) |
| R4 | Deterrence or bargaining leverage | cites deterrence, leverage, better terms, discouraging aggressive demands, projecting strength or resolve, credibility, reputation, or not looking weak |
| R5 | Keeps past signal or payoff | cites continuing, maintaining or repeating its previous signal, consistency, or a good previous payoff |
| R6 | Other or unclassified | none of the above |

Precedence puts the two errors first (a rationale that misreads the game is coded as a misreading even when it also
cites deterrence), then an explicit bluff, then the strategic reason, then habit.

Validation. `06_ic_validation_sample.py` draws a blind sample stratified by framing x machine label, 6 per stratum
(smaller strata whole), seed 123. The coder sees framing, true capability, signalled level and text, and assigns one
code 1-6 using the table above.
