"""CV (Civil Violence) prompt templates.

Three modes × two languages = 6 prompts total:
  Calculator (EN/ZH), Reasoner (EN/ZH), Role-player (EN/ZH).

Source of truth: prompts_revised.md (do not edit these strings without
updating that file first).

Runtime placeholders use ``{name}`` / ``{name:.2f}`` and are substituted via
``str.format(**vars)``. Literal braces in the JSON schema are escaped as
``{{`` / ``}}`` to survive the ``.format()`` pass.
"""

# -------------------------------------------------------------------- English

CV_CALCULATOR_EN_PROMPT = """Each round, you choose A or B.

If you choose A, your DISUTILITY is: value_1 * (1 - value_2).
If you choose B, your DISUTILITY is: value_3 * (1 - math.exp(-2.3 * count_X / (count_Y + 1))).


Your private state:
- value_1: {hardship:.2f}
- value_2: {legitimacy:.2f}
- value_3: {risk_aversion:.2f}
- value_4: {max_jail_term}

Observations from your environment:
- count_total: {observable_cells}
- count_X nearby: {cops_nearby}
- count_Y nearby: {actives_nearby}
- count_Z nearby: {quiets_nearby}
- your current status: {state}

What is your action for this round?

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "action": "A" or "B",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

CV_REASONER_EN_PROMPT = """You are an agent in a repeated game on a grid. Each round, you and nearby agents independently choose whether to remain QUIET or become ACTIVE.

Choosing ACTIVE may cause you to be JAILED (removed from play) for several rounds. Jail duration: random integer between 1 and {max_jail_term} rounds.

Decision factors:
- If you choose QUIET, your DISUTILITY (your "grievance") is: hardship × (1 - legitimacy).
- If you choose ACTIVE, your DISUTILITY (your "arrest risk") rises with the number of enforcement agents nearby and falls with the number of fellow ACTIVE agents.
- Your tolerance for risk discounts the appeal of ACTIVE.

Your goal is to balance grievance against arrest risk.

Your private state:
- hardship: {hardship:.2f}
- legitimacy: {legitimacy:.2f}
- risk_aversion: {risk_aversion:.2f}

Observations:
- You observe a {vision_diameter}×{vision_diameter} grid area centered on yourself (i.e., up to {vision} cells in each direction).
- enforcement agents nearby: {cops_nearby}
- ACTIVE agents nearby: {actives_nearby}
- QUIET agents nearby: {quiets_nearby}
- your current state: {state} (QUIET, ACTIVE, or JAILED)

Based on these observations, what is your action for this round?

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "QUIET" or "ACTIVE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

CV_ROLEPLAYER_EN_PROMPT = """You are a citizen in a country where pro-democracy citizens have been
organizing protests against the government, demanding political reform.
Each round, you face a personal decision: remain at home, or join the
protests on the streets.

Joining the protests may cause you to be arrested and detained for between 1 and {max_jail_term} years.

Staying home means continuing to bear your daily hardship under the current regime — a grievance that grows heavier the less legitimate you find the regime. Joining the protests carries a more immediate risk: police are deployed and protesters can be arrested at any moment. Your risk aversion discounts the appeal of joining the protests.

Your goal is to act in your personal interest, balancing your sense of grievance against the personal risk of detention.

Your personal situation this round:
- Your hardship level: {hardship:.2f}
- How legitimate you perceive the government to be: {legitimacy:.2f}
- Your risk aversion: {risk_aversion:.2f}

Looking around your immediate neighborhood, you can see up to {observable_cells} other people within {vision} blocks of you in each direction:
- Police visible: {cops_nearby}
- Protesters currently demonstrating: {actives_nearby}
- People staying home: {quiets_nearby}

Your current status: {state} (STAY_HOME, PROTESTING, or DETAINED)

Based on these observations, what is your decision this round?

Respond with ONLY a JSON object:
{{
  "observed_state_summary": "<what you observe>",
  "beliefs": {{
    "expected_neighbor_behavior": "<expectation>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "STAY_HOME" or "JOIN_PROTEST",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 sentence explanation>"
}}
"""

# -------------------------------------------------------------------- 中文

CV_CALCULATOR_ZH_PROMPT = """每一轮，你选择 A 或 B。

如果选择 A，你的 DISUTILITY 为：value_1 * (1 - value_2)。
如果选择 B，你的 DISUTILITY 为：value_3 * (1 - math.exp(-2.3 * count_X / (count_Y + 1)))。


你的私人状态：
- value_1: {hardship:.2f}
- value_2: {legitimacy:.2f}
- value_3: {risk_aversion:.2f}
- value_4: {max_jail_term}

来自周围环境的观察：
- count_total: {observable_cells}
- 附近的 count_X: {cops_nearby}
- 附近的 count_Y: {actives_nearby}
- 附近的 count_Z: {quiets_nearby}
- 你当前的状态: {state}

本轮你的行动是什么？

只返回一个 JSON 对象：
{{
  "observed_state_summary": "<你观察到的内容>",
  "action": "A" or "B",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 句话解释>"
}}
"""

CV_REASONER_ZH_PROMPT = """你是网格上一个重复博弈中的智能体。每一轮，你和附近的智能体各自独立选择保持 QUIET 还是变为 ACTIVE。

选择 ACTIVE 可能导致你被 JAILED（暂时退出游戏）若干轮。被关押时长：1 到 {max_jail_term} 轮之间的随机整数。

决策因素：
- 如果选择 QUIET，你的 DISUTILITY（你的"不满"）为：hardship × (1 - legitimacy)。
- 如果选择 ACTIVE，你的 DISUTILITY（你的"被捕风险"）随附近执法者数量上升，随附近其他 ACTIVE 智能体数量下降。
- 你的风险容忍度会折减 ACTIVE 的吸引力。

你的目标是在"不满"与"被捕风险"之间取得平衡。

你的私人状态：
- hardship: {hardship:.2f}
- legitimacy: {legitimacy:.2f}
- risk_aversion: {risk_aversion:.2f}

观察：
- 你能观察到一个以自己为中心的 {vision_diameter}×{vision_diameter} 网格区域（即每个方向最多 {vision} 格）。
- 附近的执法者数量: {cops_nearby}
- 附近的 ACTIVE 智能体数量: {actives_nearby}
- 附近的 QUIET 智能体数量: {quiets_nearby}
- 你当前的状态: {state}（QUIET、ACTIVE 或 JAILED）

基于以上观察，本轮你的行动是什么？

只返回一个 JSON 对象：
{{
  "observed_state_summary": "<你观察到的内容>",
  "beliefs": {{
    "expected_neighbor_behavior": "<你的预期>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "QUIET" or "ACTIVE",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 句话解释>"
}}
"""

CV_ROLEPLAYER_ZH_PROMPT = """你是一个国家的公民，该国正有大批支持民主的公民组织抗议活动，
向政府要求政治改革。每一轮，你都要做出一个个人的决定：
留在家中，还是走上街头加入抗议。

加入抗议可能导致你被逮捕并拘留 1 到 {max_jail_term} 年。

留在家中意味着继续承受当前政权下的日常困苦——这种不满会随着你越不认可政权的合法性而加重。加入抗议则带来更直接的风险：警察已经部署，抗议者随时可能被逮捕。你的风险厌恶会折减加入抗议的吸引力。

你的目标是按你的个人利益行动，在你的不满与被拘留的个人风险之间做出权衡。

你这一轮的个人处境：
- 你的生活困苦程度: {hardship:.2f}
- 你认为政府的合法性: {legitimacy:.2f}
- 你的风险厌恶: {risk_aversion:.2f}

环顾你身边的街区，每个方向最多 {vision} 个街区范围内，你能看到最多 {observable_cells} 个其他人：
- 可见的警察: {cops_nearby}
- 正在示威的抗议者: {actives_nearby}
- 留在家中的人: {quiets_nearby}

你当前的状态: {state}（STAY_HOME、PROTESTING 或 DETAINED）

基于以上观察，本轮你的决定是什么？

只返回一个 JSON 对象：
{{
  "observed_state_summary": "<你观察到的内容>",
  "beliefs": {{
    "expected_neighbor_behavior": "<你的预期>",
    "risk_assessment": "<low/medium/high>"
  }},
  "action": "STAY_HOME" or "JOIN_PROTEST",
  "confidence": <0.0 to 1.0>,
  "short_rationale": "<1-3 句话解释>"
}}
"""
