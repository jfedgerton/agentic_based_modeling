# Three Games Design Summary

---

## Context

This project benchmarks classic rule-based ABMs against LLM agents across three classical social simulations: **Prisoner's Dilemma**, **Civil Violence**, and **Interstate Conflict**. For each game, LLM agents are run under three input modes — **Calculator** (numeric payoffs only), **Reasoner** (formal game mechanics, without naming the game), and **Role-player** (discursive scenario framing) — to test how LLM-empowered agents diverge from classic rule-based agents.


**Mechanism contrast**:
- *Classic*: follows a fixed, game-specific rule.
- *LLM (Calculator)*: receives only numeric payoffs and aggregated counts, with abstract action labels and no game framing.
- *LLM (Reasoner)*: receives a formal description of the game mechanics with semantic action labels, without naming the game.
- *LLM (Role-player)*: receives a discursive scenario framing tailored to each game.


**Where LLM might matter**:
The literature identifies three potential sources of divergence between classic rule-based ABMs and LLM agents:
- how LLMs perceive the world,
- how LLMs reason, and
- how LLMs make decisions.

Our four-arm design (Classic + three LLM modes) isolates each source through pairwise comparisons:

- **Classic vs. Calculator** isolates LLM **decision-making**: both receive the same numeric inputs, but the LLM is free to choose any action while Classic follows a fixed rule.
- **Calculator vs. Reasoner** isolates LLM **reasoning**: Reasoner adds formal game mechanics and semantic action labels (e.g., "COOPERATE / DEFECT"), enabling the LLM to reason game-theoretically about the structure of the interaction.
- **Reasoner vs. Role-player** isolates LLM **perception**: Role-player adds a discursive scenario, allowing the LLM to apply domain-specific priors and world knowledge.

Stacking these modes incrementally allows us to attribute any observed Classic-vs-LLM divergence to specific layers — decision-making, reasoning, or perception — rather than treating the LLM as a single black box.


---

## Game 1. Prisoner's Dilemma Grid (PD)

**Setup**: 20×20 torus; Binary action (COOPERATE / DEFECT) played against all 8 Moore neighbors each step; payoffs summed.

**What it tests**: emergence of cooperation through spatial clustering.

**What varies**:

*Core IV (across all modes)*:
- LLM input mode (Calculator / Reasoner / Role-player)

*Additional IVs (within Reasoner / Role-player modes only)*:
- Payoff matrix (default: CC=3, CD=0, DC=5, DD=1)
- Initial cooperation probability

**Key metrics**: cooperation_rate, spatial_clustering

---

## Game 2. Epstein Civil Violence (CV)

**Setup**: 20×20 sparse grid with movement. Citizens decide QUIET / ACTIVE based on grievance vs. arrest risk; cops arrest active citizens.

**What it tests**: punctuated equilibrium in collective rebellion under heterogeneous grievance + repressive risk.


**What varies**:

*Core IV (across all modes)*:
- LLM input mode (Calculator / Reasoner / Role-player)

*Additional IVs (within Reasoner / Role-player modes only)*:
- Prompt language (English vs. Mandarin)
- Regime legitimacy (default 0.82)
- Citizen and cop densities, vision radius (measured in grid cells, within which an agent observes nearby agents)
- Max jail term

**Cross-LLM angle**: rebellion is a politically charged topic, and different LLMs may have systematically different rebellion propensities depending on their training and alignment context (e.g., US-trained models vs. models trained in authoritarian information environments). We vary the prompt language (English vs. Mandarin) to probe cross-LLM differences and within-LLM differences across prompt languages.

**Key metrics**: rebellion_rate, arrest_count, jailed_rate

---

## Game 3. Interstate Conflict (Fearon 1995)

**Setup**: 20×20 torus; every cell is a state with **private**  information about military capability and war cost. Each step:
1. *Signal phase*: every state broadcasts a global signal (categorical / continuous).
2. *Decide phase*: every Moore-neighbor dyad runs a take-it-or-leave-it bargain.
3. *Resolve phase*: peaceful settlement, or war (winner takes the issue, both pay war_cost).

**What it tests**: war as bargaining failure under private information and strategic signaling.

**What varies**:
*Core IV (across all modes)*:
- LLM input mode (Calculator / Reasoner / Role-player)

*Additional IVs (within Reasoner / Role-player modes only)*:
- `bluffing_enabled` (global switch; off = forced honest baseline)
- `bluffing_rate` (Classic only; measured as output for LLM)
- War-cost distribution
- The form of the signal (numeric or generative text)

**Key metrics**: war_frequency, mean_payoff, welfare_loss, bluff_rate, bluff_success_rate, signal_informativeness (mutual information between signal and true capability)

---


## Central framings of the research question and main contribution 

**RQ**: How does LLM-empowered agency diverge from classic rule-based agency across canonical social simulations, and which dimensions of LLM capability — perception, reasoning, or decision-making — drive the divergence?

**Contribution** (I'll frame this more critically):
- *Methodological*: A four-arm benchmark design (Classic + three LLM modes — Calculator, Reasoner, Role-player) that decomposes the LLM-vs-Classic gap into three identifiable layers — perception, reasoning, and decision-making — through controlled prompt ablation.
- *Empirical*: Systematic evidence across three canonical social simulations (PD, CV, Interstate Conflict) on the magnitude and source of LLM divergence from rule-based agents, including a cross-LLM angle testing whether politically aligned training shifts emergent behavior in politically charged scenarios (Civil Violence).

---

## Implementation status

I will update all three games to fit the new structure if the current design looks good. Open items:

- **Prompt standardization across games.** This will be relatively straightforward for the Calculator and Reasoner modes but harder for the Role-player mode, where each game requires a distinct narrative. We will also need a careful interpretation strategy for cross-game LLM differences: are they driven by the specific context we constructed (a prompt-design artifact), or by systematic variation in how the LLM perceives different domains (a meaningful empirical finding)?
   A related question is how strongly we can justify the boundaries between “perception,” “reasoning,” and “decision-making” across the three LLM modes. We may want to revisit these distinctions once the prompts are developed.

- **One-shot vs. repeated games.** All three games are currently implemented as one-shot interactions. They can be extended to a repeated-game version, though this would substantially increase our costs.

- **Text generation in LLM modes.** We all know LLMs advance in their text-generation capabilities. The current setup allows LLM-generated text only in Interstate Conflict (in the form of strategic signaling). I wonder whether we should expand this aspect further across games. I’ll review the relevant literature before refining this part of the design. Please feel free to share your thoughts on this. 


---

## Appendix — Example LLM prompts for the three input modes

Illustrated with PD; 

### Calculator mode

```
Choose between two actions: A or B. You interact with 8 neighbors simultaneously.

Payoff matrix (your_action, their_action → your_payoff):
- (A, A) → 3
- (A, B) → 0
- (B, A) → 5
- (B, B) → 1

This round:
- Neighbors choosing A: 5
- Neighbors choosing B: 3
- Your previous payoff: 24

Output: A or B
```

### Reasoner mode 

```
You are an agent in a repeated game on a grid. Each round, you choose to
COOPERATE or DEFECT, and so do your 8 neighbors.

Your payoff each round = sum of pairwise payoffs with each neighbor.
- Both cooperate: 3 each
- You cooperate, they defect: 0 for you, 5 for them
- You defect, they cooperate: 5 for you, 0 for them
- Both defect: 1 each

This round:
- Neighbors who cooperated: 5
- Neighbors who defected: 3
- Your previous payoff: 24

Output: COOPERATE or DEFECT
```

### Role-player mode

```
You are a country deciding whether to engage in free trade with 8
neighboring nations. Each round, you choose to:
- OPEN your markets (free trade), or
- CLOSE your markets (protectionism)

If both you and a neighbor open markets, both prosper (3 each).
If you open but they close, they exploit your openness (you 0, them 5).
If you close but they open, you exploit theirs (you 5, them 0).
If both close, both stagnate (1 each).

This round:
- 5 neighbors opened markets last round
- 3 closed markets
- Your last round's prosperity: 24

Output: OPEN or CLOSE
```
