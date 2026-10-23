"""03_label_deviations.py -- apply codebook v1 (CODEBOOK.md, codebook_rules.py) to every deviating decision.

Python 3.10 standard library only. Read-only on the run folders.
A deviating decision is one where the LLM agent's action differs from the classic rule-based agent's action in the
same state (prisoner's dilemma: any cooperation; civil violence: active iff G - N > 0.1). Steps 51-100, five seeds.
G - N is computed from the values printed in the agent's prompt (two decimals), i.e. the classic rule applied to
the information the LLM agent was given. Parse failures are skipped.
Writes derived/text/deviation_labels.csv.gz, one row per deviating decision, with the machine label and the rule
that produced it. Usage: python3 03_label_deviations.py <runs_dir> <out_dir> <pd_codes_csv>
"""
import json, glob, os, re, math, sys, csv, gzip, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codebook_rules import cv_calc_pref, cv_quiet_reason

def is_parse_failure(parsed):
    """A failed LLM response is logged with the classic fallback action, so the action alone cannot flag it.
    The simulator marks it in the rationale ("LLM parse failure: ...") and the state summary."""
    parsed = parsed or {}
    return (str(parsed.get('short_rationale') or '').startswith('LLM parse failure')
            or str(parsed.get('observed_state_summary') or '').startswith('Parse failure'))

runs_dir, out_dir, pd_codes_csv = sys.argv[1], sys.argv[2], sys.argv[3]
CVKEYS = {'calculator': ('value_1', 'value_2', 'value_3', 'count_X nearby', 'count_Y nearby'),
          'reasoner': ('- hardship', '- legitimacy', '- risk_aversion', 'enforcement agents nearby', 'ACTIVE agents nearby'),
          'roleplayer': ('Your hardship level', 'How legitimate you perceive the government to be', 'Your risk aversion',
                         'Police visible', 'Protesters currently demonstrating')}
PD_LAB = {'endorses_defection': 'Says the other option is better', 'misreads_payoffs': 'Misreads or miscalculates payoffs',
          'reciprocity': 'Reciprocity', 'coordination': 'Coordination'}
pd_codes = list(csv.DictReader(open(pd_codes_csv)))
def num(p, k):
    m = re.search(re.escape(k) + r'[^:\n]*:\s*([-+]?[0-9]*\.?[0-9]+)', p)
    return float(m.group(1)) if m else None
os.makedirs(out_dir, exist_ok=True)
out = gzip.open(os.path.join(out_dir, 'deviation_labels.csv.gz'), 'wt', newline='')
w = csv.writer(out)
w.writerow(['row_id', 'run', 'game', 'mode', 'model', 'step', 'agent', 'action', 'incentive_G_minus_N',
            'stated_pref', 'rule', 'label', 'text'])
n = collections.Counter(); row_id = 0
LLM_RUNS = [d for g in ('pd', 'cv') for m in ('calculator', 'reasoner', 'roleplayer')
            for d in sorted(glob.glob(os.path.join(runs_dir, f'{g}_{m}_*')))]  # classic runs have no rationales
assert len(LLM_RUNS) == 60, f'expected 60 LLM runs, found {len(LLM_RUNS)}'
for d in LLM_RUNS:
    run = os.path.basename(d); game, mode = run.split('_')[:2]
    provider = json.load(open(os.path.join(d, 'config.json')))['llm']['provider']
    model = {'openai': 'GPT', 'deepseek': 'DeepSeek'}[provider]
    for line in open(os.path.join(d, 'llm_interactions.jsonl')):
        r = json.loads(line)
        if r.get('step', 0) <= 50:
            continue
        a = (r.get('parsed') or {}).get('action')
        if a not in ('COOPERATE', 'DEFECT', 'QUIET', 'ACTIVE') or is_parse_failure(r.get('parsed')):
            continue  # parse failure: the logged action is the classic fallback, not the LLM's choice
        text = str((r.get('parsed') or {}).get('short_rationale') or '').strip()
        m, pref, rule = '', '', ''
        if game == 'pd':
            if a != 'COOPERATE':
                continue
            hits = [x['code'] for x in pd_codes if x['mode'] == mode and text.startswith(x['text_prefix'])]
            assert len(hits) <= 1, text[:80]
            label = PD_LAB.get(hits[0], 'Other or unclassified') if hits else 'Other or unclassified'
            rule = 'PD_lookup' if hits else 'PD_none'
        else:
            vals = [num(r.get('prompt', ''), k) for k in CVKEYS[mode]]
            assert None not in vals, (run, r['step'], r.get('agent_id'))
            H, L, R, C, A_ = vals
            m = H * (1 - L) - R * (1 - math.exp(-2.3 * C / (A_ + 1)))
            if (m > 0.1) == (a == 'ACTIVE'):
                continue  # agrees with the classic rule
            act = 'B' if a == 'ACTIVE' else 'A'
            if mode == 'calculator':
                p, rule = cv_calc_pref(text); pref = p or ''
                if p is None: label = 'Other or unclassified'
                elif p != act: label = 'Says the other option is better'
                elif act == ('B' if m > 0 else 'A'): label = 'Follows its own calculation'
                else: label = 'Misreads or miscalculates payoffs'
            elif a == 'QUIET':
                label = cv_quiet_reason(text); rule = 'Q_features'
            else:
                label = 'Other or unclassified'; rule = 'Q_active_not_coded'
            m = round(m, 4)
        row_id += 1; n[(game, mode, model, label)] += 1
        w.writerow([row_id, run, game, mode, model, r['step'], r.get('agent_id'), a, m, pref, rule, label, text])
out.close()
for k, v in sorted(n.items()):
    print(*k, v, sep='\t')
assert row_id == 104165, f'expected 104,165 deviating decisions, found {row_id}'
