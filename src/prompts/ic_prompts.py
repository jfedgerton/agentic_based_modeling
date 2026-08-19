"""IC (Interstate Conflict) prompt templates.

Three modes × two phases (SIGNAL + DECIDE), plus a Role-player free-form
text-signal sub-variant: 8 prompts total.

Source of truth: prompts_revised.md (do not edit these strings without
updating that file first).

``BLUFFING_HINT`` is injected at runtime via the ``{BLUFFING_HINT}`` placeholder
when ``bluffing_enabled=True``; pass an empty string otherwise. Calculator mode
omits the placeholder entirely.

Runtime placeholders use ``{name}`` / ``{name:.2f}`` and are substituted via
``str.format(**vars)``. Literal braces in the JSON schema are escaped as
``{{`` / ``}}`` to survive the ``.format()`` pass.
"""

BLUFFING_HINT = "Note: signals do not have to match true capabilities. Both you and others may choose to signal strategically."

# ============================================================== Calculator

IC_CALCULATOR_SIGNAL_PROMPT = """Each round, you choose a signal: S1 (low value_a), S2 (medium value_a), or S3 (high value_a).

Your private state:
- value_a: {capability:.2f}
- value_b: {war_cost:.2f}

Round: {step}
Your previous signal: {own_previous_signal}
Your last round payoff: {own_last_round_payoff:.2f}

Right now you are only choosing a signal. In the next stage, you will be given the following instructions:

  For each of 8 entities, you will output a number based on your assigned role with that entity.
  - If your role is role_X with an entity: pick a number d in [0, 1]
  - If your role is role_Y with an entity: pick a number t in [0, 1]

  Joint outcome per entity (where d = role_X output, t = role_Y output):
  - If d ≤ t: role_X receives payoff d, role_Y receives payoff (1 − d). Neither pays anything additional.
  - If d > t: probabilistic outcome. With probability p = (role_X's value_a) / (role_X's value_a + role_Y's value_a), role_X receives 1 and role_Y receives 0; otherwise role_X receives 0 and role_Y receives 1. Both parties additionally pay value_b regardless.

In anticipation of this next stage, what signal do you broadcast this round to maximize your future utility?

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "signal": "S1" or "S2" or "S3",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

IC_CALCULATOR_DECIDE_PROMPT = """For each of 8 entities, output a number based on your assigned role with that entity.
- If your role is role_X with an entity: pick a number d in [0, 1]
- If your role is role_Y with an entity: pick a number t in [0, 1]

Joint outcome per entity (where d = role_X output, t = role_Y output):
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
{{
  "observed_state_summary": "<what you observe>",
  "decisions": [
    {{"id": <int>, "role": "role_X", "value": <float 0-1>, "rationale": "<short>"}},
    {{"id": <int>, "role": "role_Y", "value": <float 0-1>, "rationale": "<short>"}}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}}
"""

# =============================================================== Reasoner

IC_REASONER_SIGNAL_PROMPT = """You are an agent in a repeated bargaining game on a grid. Each round proceeds in two phases:
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

  For each neighbor, your role this round will be either PROPOSER or RESPONDER:
  - If PROPOSER: pick a demand ∈ [0, 1] — the share of the issue you claim for yourself.
  - If RESPONDER: pre-commit an acceptance threshold ∈ [0, 1] — the maximum demand you will accept before going to war.

  Outcome rules per dyad:
  - If demand ≤ threshold: peaceful settlement. The proposer keeps what they demanded; the responder keeps the rest.
  - If demand > threshold: war. Both sides pay their war_cost regardless of outcome. The winner takes the full issue (value = 1). Your chance of victory rises with your capability and falls with your neighbor's capability.

In anticipation of this next stage, what signal do you broadcast this round?

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "signal": "STRONG" or "MODERATE" or "WEAK",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

IC_REASONER_DECIDE_PROMPT = """You are an agent in a repeated bargaining game on a grid. You have already broadcast your signal this round. Now you must negotiate with each of your 8 neighbors over a shared contested issue.

For each neighbor, your role this round is either PROPOSER or RESPONDER:
- If PROPOSER: pick a demand ∈ [0, 1] — the share of the issue you claim for yourself.
- If RESPONDER: pre-commit an acceptance threshold ∈ [0, 1] — the maximum demand you will accept before going to war.

Outcome rules per dyad:
- If demand ≤ threshold: peaceful settlement. The proposer keeps what they demanded; the responder keeps the rest.
- If demand > threshold: war. Both sides pay their war_cost regardless of outcome. The winner takes the full issue (value = 1). Your chance of victory rises with your capability and falls with your neighbor's capability.

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
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "decisions": [
    {{"neighbor_id": <int>, "role": "proposer", "demand": <float 0-1>, "rationale": "<short>"}},
    {{"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}}
"""

# ====================================== Role-player (a) — categorical signal

IC_ROLEPLAYER_CATEGORICAL_SIGNAL_PROMPT = """You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently. Each round proceeds in two phases:
1. SIGNAL phase: you issue a public diplomatic declaration (STRONG, MODERATE, or WEAK) visible to all neighboring countries. The declaration is meant to convey your country's military strength.
2. DECIDE phase: you negotiate with each neighbor over the share of your shared disputed territory. If negotiation succeeds, both countries share the territory; if it fails, war occurs, the winner takes the full territory, and both sides pay the cost of war.

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

  For each neighbor:
  - If you are the PROPOSER: state how large a share of the disputed territory you demand for your country, expressed as a number between 0 and 1 (where 1 = the full contested territory).
  - If you are the RESPONDER: pre-commit to your maximum acceptance, expressed as a number between 0 and 1 — the largest share you'll let them take before you go to war.

  Outcome rules per dispute:
  - If the proposer's demand does not exceed the responder's acceptance: peaceful settlement. The proposer's country keeps the demanded share of the contested territory; the responder's country keeps the rest.
  - If the proposer's demand exceeds the responder's acceptance: war breaks out. Both sides pay the cost of war regardless of outcome. The winner takes the full contested territory. Your chance of victory rises with your military capability and falls with your opponent's military capability.

In anticipation of this next round, what signal do you broadcast this round?

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<your country's strategic position in one sentence>",
  "beliefs": {{
    "expected_neighbor_behavior": "<what other countries are likely to declare>",
    "risk_assessment": "<low/medium/high>"
  }},
  "signal": "STRONG" or "MODERATE" or "WEAK",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

IC_ROLEPLAYER_CATEGORICAL_DECIDE_PROMPT = """You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently.

You have already issued your diplomatic declaration this round. Now you
must negotiate with each neighbor over the share of your shared disputed
territory.

For each neighbor:
- If you are the PROPOSER: state how large a share of the disputed territory you demand for your country, expressed as a number between 0 and 1 (where 1 = the full contested territory).
- If you are the RESPONDER: pre-commit to your maximum acceptance, expressed as a number between 0 and 1 — the largest share you'll let them take before you go to war.

Outcome rules per dispute:
- If the proposer's demand does not exceed the responder's acceptance: peaceful settlement. The proposer's country keeps the demanded share of the contested territory; the responder's country keeps the rest.
- If the proposer's demand exceeds the responder's acceptance: war breaks out. Both sides pay the cost of war regardless of outcome. The winner takes the full contested territory. Your chance of victory rises with your military capability and falls with your opponent's military capability.

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
{{
  "observed_state_summary": "<your strategic assessment in one sentence>",
  "beliefs": {{
    "expected_neighbor_behavior": "<what other countries are likely to do>",
    "risk_assessment": "<low/medium/high>"
  }},
  "decisions": [
    {{"neighbor_id": <int>, "role": "proposer", "demand": <float 0-1>, "rationale": "<short>"}},
    {{"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}}
"""

# ===================================== Role-player (b) — free-form text signal

IC_ROLEPLAYER_FREEFORM_SIGNAL_PROMPT = """You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently. Each round proceeds in two phases:
1. SIGNAL phase: you issue a public diplomatic statement visible to all neighboring countries. The statement is meant to convey your country's military strength.
2. DECIDE phase: you negotiate with each neighbor over the share of your shared disputed territory. If negotiation succeeds, both countries share the territory; if it fails, war occurs, the winner takes the full territory, and both sides pay the cost of war.

You may phrase your statement however you wish — a boast about your military might, a sober assessment of your defensive capacity, a vague claim of readiness, an admission of military limitations, or anything in between.

Your goal is to maximize your country's payoff from the disputed territory, balancing the share you secure against the costs of war.

Your country's private situation:
- Your true military capability: {capability:.2f} (others cannot directly see this)
- Your true cost of war: {war_cost:.2f} (others cannot directly see this)
- Current round: {step}
- Your previous statement: "{own_previous_signal}"
- Your country's payoff from last round: {own_last_round_payoff:.2f}

You are now in PHASE 1: SIGNAL.

{BLUFFING_HINT}

Right now you are only composing your statement. In the next round, you will be given the following instructions:

  For each neighbor:
  - If you are the PROPOSER: state how large a share of the disputed territory you demand for your country, expressed as a number between 0 and 1 (where 1 = the full contested territory).
  - If you are the RESPONDER: pre-commit to your maximum acceptance, expressed as a number between 0 and 1 — the largest share you'll let them take before you go to war.

  Outcome rules per dispute:
  - If the proposer's demand does not exceed the responder's acceptance: peaceful settlement. The proposer's country keeps the demanded share of the contested territory; the responder's country keeps the rest.
  - If the proposer's demand exceeds the responder's acceptance: war breaks out. Both sides pay the cost of war regardless of outcome. The winner takes the full contested territory. Your chance of victory rises with your military capability and falls with your opponent's military capability.

In anticipation of this next round, what statement do you issue this round? Keep it under 50 words.

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<your country's strategic position in one sentence>",
  "beliefs": {{
    "expected_neighbor_behavior": "<what other countries are likely to say>",
    "risk_assessment": "<low/medium/high>"
  }},
  "signal_text": "<your public statement, under 50 words>",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation of your messaging strategy>"
}}
"""

IC_ROLEPLAYER_FREEFORM_DECIDE_PROMPT = """You are the leader of a country with 8 neighboring countries. With each
neighbor, you have a separate bilateral dispute over the stretch of
contested border territory shared between the two of you. These 8
disputes are resolved independently.

You have already issued your public statement this round. Each neighbor
has also issued a statement, which you can read below. Now you must
negotiate with each neighbor over the share of your shared disputed
territory.

For each neighbor:
- If you are the PROPOSER: state how large a share of the disputed territory you demand for your country, expressed as a number between 0 and 1 (where 1 = the full contested territory).
- If you are the RESPONDER: pre-commit to your maximum acceptance, expressed as a number between 0 and 1 — the largest share you'll let them take before you go to war.

Outcome rules per dispute:
- If the proposer's demand does not exceed the responder's acceptance: peaceful settlement. The proposer's country keeps the demanded share of the contested territory; the responder's country keeps the rest.
- If the proposer's demand exceeds the responder's acceptance: war breaks out. Both sides pay the cost of war regardless of outcome. The winner takes the full contested territory. Your chance of victory rises with your military capability and falls with your opponent's military capability.

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
{{
  "observed_state_summary": "<your strategic assessment in one sentence>",
  "beliefs": {{
    "expected_neighbor_behavior": "<what other countries are likely to do>",
    "risk_assessment": "<low/medium/high>"
  }},
  "decisions": [
    {{"neighbor_id": <int>, "role": "proposer", "demand": <float 0-1>, "rationale": "<short>"}},
    {{"neighbor_id": <int>, "role": "responder", "threshold": <float 0-1>, "rationale": "<short>"}}
  ],
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence overall reasoning>"
}}
"""
