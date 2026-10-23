"""01_rationales_by_run.py -- one pass over each LLM run's interaction log.

Python 3.10 standard library only. Read-only on the run folders.
For every five-seed PD and CV run (steps 51-100, parse failures skipped), writes derived/text/per_run/<run>.json with
  (a) CV only: the agent's own incentive G - N from the prompt values, and action by incentive; for calculator
      decisions that contradict the sign of G - N, what the rationale prefers (codebook rules, codebook_rules.py),
  (b) word counts over distinct rationales (for 02's log-odds comparison),
  (c) PD only: every distinct rationale with its count and action (for exact hand coding).
Usage: python3 01_rationales_by_run.py <run_dir> <out_dir>
"""
import json, re, math, sys, os, collections, hashlib

run_dir, out_dir = sys.argv[1], sys.argv[2]
run = os.path.basename(run_dir.rstrip('/'))
game, mode = run.split('_')[0], run.split('_')[1]
assert game in ('pd', 'cv') and mode in ('calculator', 'reasoner', 'roleplayer'), run
seed = int(re.search(r'seed(\d+)', run).group(1))  # the folder seed is the command-line seed; config.json logs a stale default
provider = json.load(open(os.path.join(run_dir, 'config.json')))['llm']['provider']
assert provider in ('openai', 'deepseek'), provider

# Stated preference (CV calculator only) comes from the codebook rules, so every output uses one classifier.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codebook_rules import cv_calc_pref

def is_parse_failure(parsed):
    """A failed LLM response is logged with the classic fallback action, so the action alone cannot flag it.
    The simulator marks it in the rationale ("LLM parse failure: ...") and the state summary."""
    parsed = parsed or {}
    return (str(parsed.get('short_rationale') or '').startswith('LLM parse failure')
            or str(parsed.get('observed_state_summary') or '').startswith('Parse failure'))

# ---- CV incentive: G = H(1 - L), N = R(1 - exp(-2.3 C / (A + 1))), same formula in every framing
CVKEYS = {'calculator': ('value_1', 'value_2', 'value_3', 'count_X nearby', 'count_Y nearby'),
          'reasoner': ('- hardship', '- legitimacy', '- risk_aversion', 'enforcement agents nearby', 'ACTIVE agents nearby'),
          'roleplayer': ('Your hardship level', 'How legitimate you perceive the government to be', 'Your risk aversion',
                         'Police visible', 'Protesters currently demonstrating')}

def num(prompt, key):
    m = re.search(re.escape(key) + r'[^:\n]*:\s*([-+]?[0-9]*\.?[0-9]+)', prompt)
    return float(m.group(1)) if m else None

STOP = set('''the a an and or of to in on for with is are was were be been being it its this that these those as at by from
than then so if but not no yes i my me we our you your they their them which who what when where why how would could should
will can may might must do does did have has had also only more most less very much many some any each all both other such
same own just there here into over under again once while because about against between through during before after above
below up down out off since until per round rounds'''.split())
LABELS = set('a b quiet active stay staying home remain remaining join joining protest protests protesting value count status '
             'cooperate cooperating cooperation defect defecting defection open opening close closing markets'.split())
TOK = re.compile(r'[a-z]+(?:-[a-z]+)?')

res = dict(run=run, game=game, mode=mode, provider=provider, seed=seed, n=0, n_parsed=0, act_B=0, n_unique=0)
cv = dict(n=0, neg=0, B_neg=0, band=0, B_band=0, above=0, B_above=0, wrong=0, wrong_agrees=0, wrong_disagrees=0, wrong_uncl=0)
tokens, seen, inventory = collections.Counter(), set(), collections.Counter()
inv_action = {}
with open(os.path.join(run_dir, 'llm_interactions.jsonl')) as fh:
    for line in fh:
        r = json.loads(line)
        if r.get('step', 0) <= 50:
            continue
        res['n'] += 1
        action = (r.get('parsed') or {}).get('action')
        if action not in ('COOPERATE', 'DEFECT', 'QUIET', 'ACTIVE') or is_parse_failure(r.get('parsed')):
            continue  # parse failure: the logged action is the classic fallback, not the LLM's choice
        res['n_parsed'] += 1
        act = 'A' if action in ('COOPERATE', 'QUIET') else 'B'
        res['act_B'] += act == 'B'
        rat = str((r.get('parsed') or {}).get('short_rationale') or '').strip()
        if game == 'cv':
            vals = [num(r.get('prompt', ''), k) for k in CVKEYS[mode]]
            assert None not in vals, (run, r['step'], r['agent_id'])
            H, L, R, C, A_ = vals
            m = H * (1 - L) - R * (1 - math.exp(-2.3 * C / (A_ + 1)))
            cv['n'] += 1
            key = 'neg' if m < 0 else ('band' if m <= 0.1 else 'above')
            cv[key] += 1; cv['B_' + key] += act == 'B'
            if mode == 'calculator':  # the calculator prompt implies threshold 0: B is optimal iff m > 0
                if act != ('B' if m > 0 else 'A'):
                    pref = cv_calc_pref(rat)[0]
                    cv['wrong'] += 1
                    cv['wrong_uncl' if pref is None else ('wrong_agrees' if pref == act else 'wrong_disagrees')] += 1
        low = rat.lower()
        h = hashlib.md5(low.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            tokens.update(t for t in TOK.findall(low) if len(t) >= 3 and t not in STOP and t not in LABELS)
        if game == 'pd':
            inventory[rat] += 1; inv_action[rat] = action

assert res['n_parsed'] > 0.95 * res['n'], f'{run}: parse rate below 95%'
res['n_unique'] = len(seen)
res['cv'] = cv if game == 'cv' else None
res['tokens'] = dict(tokens)
res['pd_inventory'] = [dict(text=t, count=c, action=inv_action[t]) for t, c in inventory.most_common()] if game == 'pd' else None
os.makedirs(os.path.join(out_dir, 'per_run'), exist_ok=True)
json.dump(res, open(os.path.join(out_dir, 'per_run', run + '.json'), 'w'))
print(run, provider, 'parsed', res['n_parsed'], 'unique', res['n_unique'])
