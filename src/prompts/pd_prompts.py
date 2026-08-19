"""PD prompt templates for Calculator / Reasoner / Role-player modes.

Source of truth: prompts_revised.md (do not edit these strings without
updating that file first).

Runtime placeholders use ``{name}`` (single braces) and are substituted via
``str.format(**vars)``. Literal braces in the JSON schema are escaped as
``{{`` / ``}}`` to survive the ``.format()`` pass.
"""

PD_CALCULATOR_PROMPT = """Each round, you choose A or B.

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
{{
  "observed_state_summary": "<what you observe>",
  "action": "A" or "B",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

PD_REASONER_PROMPT = """You are an agent in a repeated game on a grid. Each round, you choose to COOPERATE or DEFECT, and so do your 8 neighbors.

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
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "COOPERATE" or "DEFECT",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

PD_ROLEPLAYER_PROMPT = """You are a country deciding whether to engage in free trade with 8 neighboring nations. Each round, you choose to OPEN your markets or CLOSE your markets.

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
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "OPEN" or "CLOSE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""
