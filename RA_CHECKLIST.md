# Graduate Research Assistant Checklist

## Phase 1: Environment Setup and Validation

### Setup
- [ ] Clone repository and create virtual environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set API keys in environment: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- [ ] Verify installation: `pytest tests/ -v`

### Classic Model Validation
- [ ] Run classic PD Grid: `python run_experiment.py --model pd_grid --agent classic --steps 200 --seeds 42 43 44 45 46`
- [ ] Confirm cooperation dynamics match Nowak & May (1992) patterns
- [ ] Run classic Civil Violence: `python run_experiment.py --model civil_violence --agent classic --steps 200 --seeds 42 43 44 45 46`
- [ ] Confirm punctuated equilibrium pattern in rebellion dynamics (Epstein 2002)
- [ ] Document any deviations from expected canonical behavior

### Mock LLM Validation
- [ ] Run mock LLM PD: `python run_experiment.py --model pd_grid --agent llm --provider mock --steps 50`
- [ ] Verify LLM outputs parse correctly (check `logs/` for parse failures)
- [ ] Run mock hybrid PD: `python run_experiment.py --model pd_grid --agent hybrid --provider mock --steps 50`
- [ ] Confirm hybrid override logic works (check decision_source in logs)

## Phase 2: Small-Scale LLM Simulations

### Initial LLM Runs
- [ ] Run small PD Grid with real LLM: `python run_experiment.py --model pd_grid --agent llm --provider openai --llm-model gpt-4o-mini --steps 20 --seeds 42`
- [ ] Inspect `logs/` directory: verify prompts, responses, and parsed actions
- [ ] Check malformed response rate (should be < 5%)
- [ ] Estimate per-run API cost from token counts in `summary.json`

### Prompt Caching Verification
- [ ] Run same configuration twice, confirm cache hits on second run
- [ ] Compare runtime with and without caching (`--no-cache` flag)

### Temperature Sensitivity (Small Scale)
- [ ] Run with temperature=0.0, 0.3, 0.7, 1.0 (3 seeds each, 20 steps)
- [ ] Record cooperation rate variance at each temperature
- [ ] Document if temperature significantly affects emergent behavior

## Phase 3: Full Benchmark Runs

### PD Grid Benchmark
- [ ] Classic: 10 seeds × 200 steps
- [ ] LLM (gpt-4o-mini): 10 seeds × 200 steps
- [ ] Hybrid: 10 seeds × 200 steps
- [ ] Record runtime, API cost, and cooperation rate for each run
- [ ] Generate cooperation rate time series plots per architecture

### Civil Violence Benchmark
- [ ] Classic: 10 seeds × 200 steps
- [ ] LLM: 10 seeds × 200 steps
- [ ] Hybrid: 10 seeds × 200 steps
- [ ] Record rebellion rate, arrest rate, and temporal dynamics
- [ ] Generate rebellion rate time series plots per architecture

### Cross-Provider Comparison (Optional)
- [ ] Run PD Grid LLM with Anthropic (claude-sonnet): 5 seeds × 100 steps
- [ ] Compare cooperation dynamics between OpenAI and Anthropic providers

## Phase 4: Analysis and Figures

### Compute Metrics
- [ ] Cooperation rates by architecture (mean, std across seeds)
- [ ] Spatial clustering coefficients
- [ ] Rebellion rates by architecture
- [ ] Runtime comparison table
- [ ] Seed sensitivity (coefficient of variation across seeds)
- [ ] Temperature sensitivity curves
- [ ] Malformed response rates

### Generate Figures
- [ ] Figure 1: Cooperation rate time series (classic vs LLM vs hybrid)
- [ ] Figure 2: Spatial clustering evolution
- [ ] Figure 3: Rebellion rate dynamics comparison
- [ ] Figure 4: Runtime and cost comparison bar chart
- [ ] Figure 5: Temperature sensitivity plot
- [ ] Figure 6: Rationale theme frequency distribution

### Rationale Analysis
- [ ] Extract rationale themes from LLM interaction logs
- [ ] Compute theme frequency by action type
- [ ] Identify whether rationale themes predict actions

## Phase 5: Robustness Checks

- [ ] Seed sensitivity: verify low coefficient of variation for classic agents
- [ ] Temperature sweep: 0.0, 0.1, 0.3, 0.5, 0.7, 1.0 (5 seeds each)
- [ ] Prompt variation: test 2-3 prompt formulation variants
- [ ] Grid size sensitivity: test 10×10, 20×20, 40×40
- [ ] Check that LLM agents never receive richer information than classic agents

## Phase 6: Documentation and Replication Package

- [ ] Update README with final instructions
- [ ] Document all experiment configurations used
- [ ] Create `configs/` directory with named configuration files for each experiment
- [ ] Archive all raw logs and results
- [ ] Write replication instructions (step-by-step commands to reproduce all figures)
- [ ] Verify a clean clone + run reproduces baseline results
