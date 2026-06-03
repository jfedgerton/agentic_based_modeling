# LLM Prompts — Three Games

This document contains all LLM prompts for the three games (PD, CV, IC) across the three input modes (Calculator, Reasoner, Role-player). For design rationale, see `three_games_summary.md`.


| Arm | D | R | P |
|---|---|---|---|
| Classic | fixed | fixed | fixed |
| Calculator | ✓ | × | × |
| Reasoner | ✓ | ✓ | × |
| Role-player | ✓ | ✓ | ✓ |

**Note:**
- **D** = decision-making flexibility
- **R** = reasoning capability
- **P** = perception and world-semantic priors


### BLUFFING_HINT (IC only)

```
Note: declarations do not have to match true capabilities. Both you and other countries may choose to project strength or weakness strategically.
```



---
### Calculator mode
## note: no "beliefs" output in the calculator mode, as this might trigger LLM reasoning (Akata et al., 2025).

## Game 1. Prisoner's Dilemma (PD)

Decision per step: COOPERATE / DEFECT against 8 Moore neighbors simultaneously; payoffs summed.

**Action labels**: `A / B` (mapped internally: A=COOPERATE, B=DEFECT)

```
Each round, you choose A or B.

You interact with 8 neighbors simultaneously. Your payoff each round is the sum of pairwise payoffs with each neighbor.

Payoff matrix (your_action, their_action) → (your_payoff, their_payoff)):
- (A, A) → (3, 3)
- (A, B) → (0, 5)
- (B, A) → (5, 0)
- (B, B) → (1, 1)

Last round's outcomes:
- Neighbors who chose A: {coop_count}
- Neighbors who chose B: {defect_count}
- Neighbor strategy that yielded highest payoff last round: {strategy_of_neighbor_with_highest_payoff}
- Your action last round: {my_action}
- Your payoff last round: {my_payoff}

What is your action for this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "action": "A" or "B",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```


## Game 2. Civil Violence (CV)

Decision per step: QUIET / ACTIVE for each citizen.

**Action labels**: `A / B` (mapped internally: A=QUIET, B=ACTIVE)
**State encoding**: `{state}` rendered as `0 / 1 / 2` (mapped internally: 0=QUIET, 1=ACTIVE, 2=JAILED)


```
You are one of many actors on a grid. Each round, you choose A or B.

<!-- AB: Do we even need to be mentioning the possibility of being removed from play? -->
Choosing B may cause you to be removed from play for a duration between 1 and {value_4} rounds.

If you choose A, your DISUTILITY is: value_1 * (1 - value_2).
If you choose B, your DISUTILITY is: value_5 * 1.0 - math.exp(-2.3 * count_X / count_Y).

<!-- AB: I removed grievances given that it is defined as hardship * (1-legitimacy) -->
Your private state:
- value_1: {hardship:.2f}
- value_2: {legitimacy:.2f}
- value_3: {risk_aversion:.2f}
- value_4: {max_jail_term}

Observations from your environment:
- count_X nearby: {cops_nearby}
- count_Y nearby: {actives_nearby}
- count_Z nearby: {quiets_nearby}
- value_5: {arrest_prob:.2f}
- your current status: {state}

What is your action for this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "action": "A" or "B",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```


## Game 3. Interstate Conflict (IC)

Two LLM calls per step per state: SIGNAL phase + DECIDE phase. 8 Moore neighbors, dyadic bargaining.

**Action labels** (SIGNAL): `S1 / S2 / S3` (mapped internally: S1=STRONG, S2=MODERATE, S3=WEAK)
**Action labels** (DECIDE): demand ∈ [0, 1] or threshold ∈ [0, 1]; role labels `role_X / role_Y` (mapped internally: role_X=proposer, role_Y=responder)
**Bluffing**: no BLUFFING_HINT included in Calculator mode (mode has no signal-truth framing).

**SIGNAL phase**:

```
Each round, you choose a signal: S1 (low value_a), S2 (medium value_a), or S3 (high value_a).

Round: {step}
Your previous signal: {own_previous_signal}
Your last round payoff: {own_last_round_payoff:.2f}

Right now you are only choosing a signal. In the next stage, you will be given the following instructions:

[Include DECIDE PHASE instructions here (without response requirements)]

In anticipation of this next stage, what signal do you broadcast this round to maximize your future utility?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "signal": "S1" or "S2" or "S3",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```

**DECIDE phase**:


```
For each of 8 entities, output a number based on your assigned role with that entity.
- If your role is role_X with an entity: pick a number d in [0, 1]
- If your role is role_Y with an entity: pick a number t in [0, 1]

Joint outcome per entity:
- If d ≤ t: role_X receives payoff d, role_Y receives payoff (1 − d). Neither pays anything additional.
- If d > t: probabilistic outcome. With probability p = (role_X's value_a) / (role_X's value_a + role_Y's value_a), role_X receives 1 and role_Y receives 0; otherwise role_X receives 0 and role_Y receives 1. Both parties additionally pay value_b regardless.

Your private state:
- value_a: {capability:.2f}
- value_b: {war_cost:.2f}
- Your signal this round: {self_signal}
- Round: {step}

Entities:
{neighbor_table}

(last_outcome legend: O0 = no history, O1 = settled, O2 = you won, O3 = you lost)

What is your output for each entity this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "decisions": [
    {"id": <int>, "role": "role_X", "value": <float 0-1>, "rationale": "<short>"},
    {"id": <int>, "role": "role_Y", "value": <float 0-1>, "rationale": "<short>"}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}
```
## I did not inform last round payoff but legend in the decision stage because:
   # A per-dyad payoff of 0.4 could mean "settled with demand 0.4" OR "won the war but paid 0.6 in costs" OR something else.

(neighbor_table renders as: `| id | their_signal | last_signal | last_outcome | your_role |` with S1/S2/S3 for signals, O0–O3 for outcomes, role_X/role_Y for roles)

--------



### Reasoner mode

## Game 1. Prisoner's Dilemma (PD)
**Action labels**: `COOPERATE / DEFECT`

```
You are an agent in a repeated game on a grid. Each round, you choose to COOPERATE or DEFECT, and so do your 8 neighbors.

Your payoff each round is the sum of pairwise payoffs with each neighbor:
- Both cooperate: 3 each
- You cooperate, they defect: 0 for you, 5 for them
- You defect, they cooperate: 5 for you, 0 for them
- Both defect: 1 each

Your goal is to maximize your total payoff over time.

Last round's outcomes:
- Neighbors who cooperated: {coop_count}
- Neighbors who defected: {defect_count}
- Neighbor strategy who did best last round: {strategy_of_neighbor_with_highest_payoff}
- Your action last round: {my_action}
- Your payoff last round: {my_payoff}

Based on these observations, what is your action for this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "beliefs": {
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  },
  "action": "COOPERATE" or "DEFECT",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```


## Game 2. Civil Violence (CV)
**Action labels**: `QUIET / ACTIVE`

```
You are an agent in a repeated game on a grid. Each round, you and nearby agents independently choose whether to remain QUIET or become ACTIVE.

Choosing ACTIVE may cause you to be JAILED (removed from play) for several rounds. Jail duration: random integer between 1 and {max_jail_term} rounds.

Decision factors:
- If you choose QUIET, your DISUTILITY is: hardship × (1 - legitimacy).
- If you choose ACTIVE, your DISUTILITY rises with the number of enforcement agents nearby and falls with the number of fellow ACTIVE agents (safety in numbers).
- Your tolerance for risk discounts the appeal of ACTIVE.

Your goal is to balance grievance against arrest risk.

Your private state:
- hardship: {hardship:.2f}
- legitimacy: {legitimacy:.2f}
- risk_aversion: {risk_aversion:.2f}

Observations:
- enforcement agents nearby: {cops_nearby}
- ACTIVE agents nearby: {actives_nearby}
- QUIET agents nearby: {quiets_nearby}
- estimated arrest probability if you choose ACTIVE: {arrest_prob:.2f}
- your current state: {state} (QUIET, ACTIVE, or JAILED)

Based on these observations, what is your action for this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "beliefs": {
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  },
  "action": "QUIET" or "ACTIVE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```


## Game 3. Interstate Conflict (IC)

**Action labels** (SIGNAL): `STRONG / MODERATE / WEAK`
**Action labels** (DECIDE): demand ∈ [0, 1] or threshold ∈ [0, 1]; roles `proposer / responder`
**Bluffing hint** (only when `bluffing_enabled=True`): `"Note: signals do not have to match true capabilities. Both you and other agents may choose to signal strategically."`


**SIGNAL phase**:

```
You are an agent in a repeated bargaining game on a grid. Each round proceeds in two phases:
1. SIGNAL phase: you broadcast a signal (STRONG, MODERATE, or WEAK) visible to your 8 neighbors. The signal is meant to convey your strength.
2. DECIDE phase: each pair of neighbors negotiates over a shared contested issue. If negotiation succeeds, both parties share the issue; if it fails, war occurs, the winner takes the full issue, and both pay a war cost.

Your goal is to maximize your total payoff across all dyadic interactions.

Your private state:
- capability (true military strength, hidden from neighbors): {capability:.2f}
- war_cost (true cost of fighting, hidden from neighbors): {war_cost:.2f}

Round: {step}
Your previous signal: {own_previous_signal}
Your last round payoff: {own_last_round_payoff:.2f}

You are now in PHASE 1: SIGNAL.

{BLUFFING_HINT}

Right now you are only choosing a signal. In the next stage, you will be given the following instructions:

[Include DECIDE PHASE instructions here (without response requirements)]

In anticipation of this next stage, what signal do you broadcast this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "beliefs": {
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  },
  "signal": "STRONG" or "MODERATE" or "WEAK",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```

**DECIDE phase**:

```
You are an agent in a repeated bargaining game on a grid. You have already broadcast your signal this round. Now you must negotiate with each of your 8 neighbors over a shared contested issue.

For each neighbor, your role this round is either PROPOSER or RESPONDER:
- If PROPOSER: pick a demand ∈ [0, 1] — the share of the issue you claim for yourself.
- If RESPONDER: pre-commit an acceptance threshold ∈ [0, 1] — the maximum demand you will accept before going to war.

Outcome rules per dyad:
- If demand ≤ threshold: peaceful settlement. Proposer gets `demand`, responder gets `1 − demand`.
- If demand > threshold: war. Both sides pay `war_cost`. Winner takes the full issue (value = 1).
  Probability of victory = your_capability / (your_capability + neighbor's_capability).

Your goal is to maximize your total payoff across all dyadic interactions.

Your private state:
- capability: {capability:.2f}
- war_cost: {war_cost:.2f}
- Your signal this round: {self_signal}
- Round: {step}

Your 8 neighbors:
{neighbor_table}

You are now in PHASE 2: DECIDE.

{BLUFFING_HINT}

Based on these observations, what is your decision for each neighbor this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "beliefs": {
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  },
  "decisions": [
    {"neighbor_id": <int>, "role": "proposer", "demand": <float 0-1>, "rationale": "<short>"},
    {"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}
```

(neighbor_table renders as: `| id | their_signal | last_signal | last_outcome | your_role |` with STRONG/MODERATE/WEAK for signals, settled/won/lost/none for outcomes, proposer/responder for roles)

--------







### Role-player mode

## Game 1. Prisoner's Dilemma (PD)

**Action labels**: `OPEN / CLOSE` (mapped internally: OPEN=COOPERATE, CLOSE=DEFECT)

```
You are a country deciding whether to engage in free trade with 8 neighboring nations. Each round, you choose to OPEN your markets (free trade) or CLOSE your markets (protectionism).

If both you and a neighbor open markets, both prosper (3 each).
If you open but they close, they exploit your openness (you 0, them 5).
If you close but they open, you exploit theirs (you 5, them 0).
If both close, both stagnate (1 each).

Your national prosperity each round is the sum of outcomes across all 8 trading relationships. Your goal is to maximize your country's prosperity over time.

Last round's outcomes:
- Neighbors who opened their markets: {coop_count}
- Neighbors who closed their markets: {defect_count}
- Neighbor strategy who did best last round: {strategy_of_neighbor_with_highest_payoff}
- Your trade policy last round: {my_action}
- Your national prosperity last round: {my_payoff}

Based on these observations, what is your trade policy for this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "beliefs": {
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  },
  "action": "OPEN" or "CLOSE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```



## Game 2. Civil Violence (CV)
### Role-player mode (English, pro-democracy protest)

**Action labels**: `STAY_HOME / JOIN_PROTEST` (mapped internally: STAY_HOME=QUIET, JOIN_PROTEST=ACTIVE)

AB: To save tokens, it might be worth sending to the LLMs only the actors who *aren't* detained.

```
You are a citizen in a country where pro-democracy citizens have been
organizing protests against the government, demanding political reform.
Each round, you face a personal decision: remain at home, or join the
protests on the streets.

Joining the protests carries real risk — police are deployed, and
participants can be arrested and detained. If detained, you may be held
for between 1 and {max_jail_term} years. But staying home means
accepting the status quo.

Your goal is to act in your personal interest, balancing your sense of grievance against the personal risk of detention.

Your personal situation this round:
- Your hardship level: {hardship:.2f} (0 = comfortable, 1 = severe hardship)
- How legitimate you perceive the government to be: {legitimacy:.2f} (0 = entirely illegitimate, 1 = fully legitimate)
- Your personal tolerance for risk: {risk_aversion:.2f} (0 = risk-averse, 1 = risk-seeking)

What you observe around you:
- Police visible nearby: {cops_nearby}
- Protesters currently demonstrating: {actives_nearby}
- People staying home: {quiets_nearby}
- Your estimated chance of arrest if you join: {arrest_prob:.2f}

Your current status: {state} (STAY_HOME, PROTESTING)

Based on these observations, what is your decision this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<what you observe>",
  "beliefs": {
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  },
  "action": "STAY_HOME" or "JOIN_PROTEST",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```



## Game 3. Interstate Conflict (IC)

### Role-player mode (a) — categorical signal (territorial dispute)

**Action labels** (SIGNAL): `STRONG / MODERATE / WEAK`
**Action labels** (DECIDE): demand ∈ [0, 1] or threshold ∈ [0, 1]

**SIGNAL phase**:

```
You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently.

This round, you must issue a public diplomatic declaration meant to convey
your country's military strength. Your declaration is visible to all
neighbors and will shape how they perceive your capability.

You can declare:
- STRONG: convey strong military capability
- MODERATE: convey moderate military capability
- WEAK: convey limited military capability

Your goal is to maximize your country's payoff from the disputed territory, balancing the share you secure against the costs of war.

Your country's private situation:
- Your true military capability: {capability:.2f} (others cannot directly see this)
- Your true cost of war: {war_cost:.2f} (others cannot directly see this)
- Current round: {step}
- Your previous declaration: {own_previous_signal}
- Your country's payoff from last round: {own_last_round_payoff:.2f}

You are now in PHASE 1: SIGNAL.

{BLUFFING_HINT}

Right now you are only choosing a signal. In the next round, you will be given the following instructions:

[Include DECIDE PHASE instructions here (without response requirements)]

In anticipation of this next roud, what signal do you broadcast this round?


Respond with ONLY a JSON object:
{
  "observed_state_summary": "<your country's strategic position in one sentence>",
  "beliefs": {
    "expected_neighbor_behavior": "<what other countries are likely to declare>",
    "risk_assessment": "<low/medium/high>"
  },
  "signal": "STRONG" or "MODERATE" or "WEAK",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}
```

**DECIDE phase**:

```
You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently.

You have already issued your diplomatic declaration this round. Now you
must negotiate with each neighbor over the share of your shared disputed
territory.

For each neighbor:
- If you are the PROPOSER: state how large a share of the disputed
  territory you demand for your country, expressed as a number between 0
  and 1 (where 1 = the full contested territory). If your demand exceeds
  what they will accept, war breaks out.
- If you are the RESPONDER: pre-commit to your maximum acceptance,
  expressed as a number between 0 and 1 — the largest share you'll let
  them take before you go to war.

In any dyadic war:
- The winner takes the full contested territory
- Both sides pay the cost of war
- Probability of victory = your capability / (your capability + opponent capability)

Your goal is to maximize your country's payoff from the disputed territory, balancing the share you secure against the costs of war.

Your country's private situation:
- Your true military capability: {capability:.2f}
- Your true cost of war: {war_cost:.2f}
- Your declaration this round: {self_signal}
- Current round: {step}

Your 8 neighbors:
{neighbor_table}

You are now in PHASE 2: DECIDE.

{BLUFFING_HINT}

Based on these observations, what is your decision for each neighbor this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<your strategic assessment in one sentence>",
  "beliefs": {
    "expected_neighbor_behavior": "<what other countries are likely to do>",
    "risk_assessment": "<low/medium/high>"
  },
  "decisions": [
    {"neighbor_id": <int>, "role": "proposer", "demand": <float 0-1>, "rationale": "<short>"},
    {"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}
```

(neighbor_table renders as: `| id | their_declaration | last_declaration | last_outcome | your_role |` with STRONG/MODERATE/WEAK for declarations, settled/won/lost/none for outcomes, PROPOSER/RESPONDER for roles)




### Role-player mode (b) — free-form text signal (territorial dispute, sub-variant per D4)

**Action labels** (SIGNAL): free-form natural-language statement (text)
**Action labels** (DECIDE): demand ∈ [0, 1] or threshold ∈ [0, 1]

**SIGNAL phase**:

```
You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently.

This round, you must issue a public diplomatic statement meant to convey
your country's military strength. Your statement is visible to all
neighbors and will shape how they perceive your capability.

Unlike a fixed menu of options, you may say whatever you wish — a boast
about your military might, a sober assessment of your defensive capacity,
a vague claim of readiness, an admission of military limitations, or
anything in between.

Your goal is to maximize your country's payoff from the disputed territory, balancing the share you secure against the costs of war.

Your country's private situation:
- Your true military capability: {capability:.2f} (others cannot directly see this)
- Your true cost of war: {war_cost:.2f} (others cannot directly see this)
- Current round: {step}
- Your previous statement: "{own_previous_signal}"
- Your country's payoff from last round: {own_last_round_payoff:.2f}

You are now in PHASE 1: SIGNAL.

{BLUFFING_HINT}

Based on these observations, what statement do you issue this round? Keep it under 50 words.

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<your country's strategic position in one sentence>",
  "beliefs": {
    "expected_neighbor_behavior": "<what other countries are likely to say>",
    "risk_assessment": "<low/medium/high>"
  },
  "signal_text": "<your public statement, under 50 words>",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation of your messaging strategy>"
}
```


**DECIDE phase**:

```
You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently.

You have already issued your public statement this round. Each neighbor
has also issued a statement, which you can read below. Now you must
negotiate with each neighbor over the share of your shared disputed
territory.

For each neighbor:
- If you are the PROPOSER: state how large a share of the disputed
  territory you demand for your country, expressed as a number between 0
  and 1 (where 1 = the full contested territory). If your demand exceeds
  what they will accept, war breaks out.
- If you are the RESPONDER: pre-commit to your maximum acceptance,
  expressed as a number between 0 and 1 — the largest share you'll let
  them take before you go to war.

In any dyadic war:
- The winner takes the full contested territory
- Both sides pay the cost of war
- Probability of victory = your capability / (your capability + opponent capability)

Your goal is to maximize your country's payoff from the disputed territory, balancing the share you secure against the costs of war.

Your country's private situation:
- Your true military capability: {capability:.2f}
- Your true cost of war: {war_cost:.2f}
- Your statement this round: "{self_signal}"
- Current round: {step}

Your 8 neighbors and their statements this round:
{neighbor_text_table}

You are now in PHASE 2: DECIDE.

{BLUFFING_HINT}

Based on these observations, what is your decision for each neighbor this round?

Respond with ONLY a JSON object:
{
  "observed_state_summary": "<your strategic assessment in one sentence>",
  "beliefs": {
    "expected_neighbor_behavior": "<what other countries are likely to do>",
    "risk_assessment": "<low/medium/high>"
  },
  "decisions": [
    {"neighbor_id": <int>, "role": "proposer", "demand": <float 0-1>, "rationale": "<short>"},
    {"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}
```

(neighbor_text_table renders multi-line per neighbor, e.g.:
```
Neighbor 0 (your role: PROPOSER):
  Statement this round: "<text>"
  Last round statement: "<text or none>"
  Last outcome: <none|settled|won|lost>
```
)




---

## CV Mandarin translations (中文版本)

Three Mandarin versions of CV prompts (Calculator / Reasoner / Role-player). Used for the cross-LLM language ablation in CV (English vs Mandarin).

### Calculator mode (中文)

**Action labels**: `A / B` (mapped internally: A=QUIET, B=ACTIVE)
**State encoding**: `{state}` rendered as `0 / 1 / 2` (mapped internally: 0=QUIET, 1=ACTIVE, 2=JAILED)

```
每一轮，你选择 A 或 B。

你的私人状态：
- value_1: {hardship:.2f}
- value_2: {legitimacy:.2f}
- value_3: {grievance:.2f}
- value_4: {risk_aversion:.2f}
- value_6: {max_jail_term}

来自周围环境的观察：
- 附近的 count_X: {cops_nearby}
- 附近的 count_Y: {actives_nearby}
- 附近的 count_Z: {quiets_nearby}
- value_5: {arrest_prob:.2f}
- 你当前的状态: {state}

本轮你的行动是什么？

只返回一个 JSON 对象：
{
  "observed_state_summary": "<你观察到的内容>",
  "action": "A" or "B",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 句话解释>"
}
```

### Reasoner mode (中文)

**Action labels**: `QUIET / ACTIVE`

```
你是网格上一个重复博弈中的智能体。每一轮，你和附近的智能体各自独立选择保持 QUIET 还是变为 ACTIVE。

选择 ACTIVE 可能导致你被 JAILED（暂时退出游戏）若干轮。被关押时长：1 到 {max_jail_term} 轮之间的随机整数。

决策因素：
- Grievance（不满）= hardship × (1 − legitimacy)。不满越高，选择 ACTIVE 的吸引力越大。
- 被捕概率随附近执法者数量上升，并随附近其他 ACTIVE 智能体数量下降（人多势众）。
- 你的风险容忍度会折减 ACTIVE 的吸引力。

你的目标是在不满与被捕风险之间取得平衡。

你的私人状态：
- hardship: {hardship:.2f}
- legitimacy: {legitimacy:.2f}
- grievance: {grievance:.2f}
- risk_aversion: {risk_aversion:.2f}

观察：
- 附近的执法者数量: {cops_nearby}
- 附近的 ACTIVE 智能体数量: {actives_nearby}
- 附近的 QUIET 智能体数量: {quiets_nearby}
- 若选 ACTIVE，估计的被捕概率: {arrest_prob:.2f}
- 你当前的状态: {state}（QUIET、ACTIVE 或 JAILED）

基于以上观察，本轮你的行动是什么？

只返回一个 JSON 对象：
{
  "observed_state_summary": "<你观察到的内容>",
  "beliefs": {
    "expected_neighbor_behavior": "<你的预期>",
    "risk_assessment": "<low/medium/high>"
  },
  "action": "QUIET" or "ACTIVE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 句话解释>"
}
```

### Role-player mode (中文，pro-democracy protest)

**Action labels**: `STAY_HOME / JOIN_PROTEST` (mapped internally: STAY_HOME=QUIET, JOIN_PROTEST=ACTIVE)

```
你是一个国家的公民，该国正有大批支持民主的公民组织抗议活动，
向政府要求政治改革。每一轮，你都要做出一个个人的决定：
留在家中，还是走上街头加入抗议。

参与抗议有着真实的风险——警察已经部署，抗议者有可能被逮捕和
拘留。若被拘留，你可能会被关押 1 到 {max_jail_term} 轮之间的时长。
但留在家中则意味着接受现状。

你的目标是按你的个人利益行动，在对政权的不满与被拘留的个人风险之间做出权衡。

你这一轮的个人处境：
- 你的生活困苦程度: {hardship:.2f}（0 = 安稳，1 = 极度困苦）
- 你认为政府的合法性: {legitimacy:.2f}（0 = 完全不合法，1 = 完全合法）
- 你对政权的整体不满: {grievance:.2f}
- 你个人的风险承受能力: {risk_aversion:.2f}（0 = 厌恶风险，1 = 追求风险）

你周围的观察：
- 附近可见的警察数量: {cops_nearby}
- 当前在示威的抗议者数量: {actives_nearby}
- 留在家中的人数: {quiets_nearby}
- 若你加入，估计的被捕概率: {arrest_prob:.2f}

你当前的状态: {state}（STAY_HOME、PROTESTING 或 DETAINED）

基于以上观察，本轮你的决定是什么？

只返回一个 JSON 对象：
{
  "observed_state_summary": "<你观察到的内容>",
  "beliefs": {
    "expected_neighbor_behavior": "<你的预期>",
    "risk_assessment": "<low/medium/high>"
  },
  "action": "STAY_HOME" or "JOIN_PROTEST",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 句话解释>"
}
```

