# Codebook v1 for agent rationales in deviating decisions (Oct 23, 2026)

Implemented in `codebook_rules.py` (civil violence) and `pd_rationale_codes.csv` (prisoner's dilemma), applied by
`03_label_deviations.py`, validated with `04_validation_sample.py`.

## Unit of analysis

One deviating decision, meaning one agent in one step (steps 51 to 100, five seeds) whose action differs from the
action the classic rule-based agent would take in the same state. In the prisoner's dilemma the classic agent always
defects, so every cooperation is a deviation. In civil violence the classic agent is active if and only if
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
| Follows its own calculation | CV calculator only. The rationale favors the action taken and the action matches the sign of G - N, so the agent applies a threshold of 0 where the classic rule uses 0.1. | CV calculator |
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

GPT agents produce only 26 distinct cooperating rationales across all PD decisions, so every distinct text is coded
by hand and the code is applied by exact prefix match. Precedence when a text fits more than one category
is Says the other option is better, then Misreads, then Reciprocity, then Coordination.

## Validation

`04_validation_sample.py` draws a stratified random sample (seed 123) and writes a blind file with no machine labels.
Strata are the machine's own outputs, so the sample estimates the precision of each machine label. The coder fills in
T1 (A, B or none), T2 (two 0/1 features) and T3 (one PD category). We report per-stratum agreement, overall accuracy
weighted by stratum size, and Cohen's kappa, in the Supplementary Materials.
