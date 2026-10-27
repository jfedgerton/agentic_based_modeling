"""02_summarise.py -- tables from 01's per-run files. Python 3.10 standard library only.

Writes to derived/text/:
  cv_incentive_by_arm.csv      CV action by the agent's own incentive G - N; calculator errors vs. rationale (codebook rules)
  pd_rationale_inventory.csv   every distinct PD rationale with its share of decisions, action, and hand code
  pd_cooperation_sources.csv   GPT PD cooperation broken down by hand code
  logodds_gpt_vs_deepseek.csv  distinctive words, GPT vs DeepSeek within each CV framing (Monroe et al. 2008)
Usage: python3 02_summarise.py <out_dir> <codes_csv>
"""
import json, glob, math, collections, csv, os, sys
out_dir, codes_csv = sys.argv[1], sys.argv[2]
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, 'per_run', '*.json')))]
assert len(runs) == 60, f'expected 60 LLM runs (2 games x 3 framings x 2 models x 5 seeds), found {len(runs)}'
arms = collections.defaultdict(list)
for r in runs:
    arms[(r['game'], r['mode'], r['provider'])].append(r)
assert len(arms) == 12 and all(len(v) == 5 for v in arms.values()), 'every arm needs 5 seeds'
W = lambda name: csv.writer(open(os.path.join(out_dir, name), 'w', newline=''))

# 2. CV action by own incentive
w = W('cv_incentive_by_arm.csv')
w.writerow(['mode', 'provider', 'decisions', 'pct_active_incentive_below_0', 'pct_active_0_to_0.1', 'pct_active_above_0.1',
            'calc_wrong', 'calc_wrong_rationale_disagrees', 'calc_wrong_rationale_agrees', 'calc_wrong_unclassified'])
for k in sorted(arms):
    if k[0] != 'cv':
        continue
    s = collections.Counter()
    for r in arms[k]:
        s.update(r['cv'])
    pct = lambda a, b: round(100 * s[a] / s[b], 2) if s[b] else ''
    w.writerow([k[1], k[2], s['n'], pct('B_neg', 'neg'), pct('B_band', 'band'), pct('B_above', 'above'),
                s['wrong'], s['wrong_disagrees'], s['wrong_agrees'], s['wrong_uncl']])
    print(f"CV {k[1]:10s} {k[2]:8s} active|<0 {pct('B_neg','neg')}%  active|0-0.1 {pct('B_band','band')}%  active|>0.1 {pct('B_above','above')}%"
          + (f"  wrong {s['wrong']}, rationale disagrees {s['wrong_disagrees']} ({100*s['wrong_disagrees']/max(s['wrong'],1):.1f}%)" if k[1] == 'calculator' else ''))

# 3. PD: every distinct rationale, joined to the hand codes
codes = list(csv.DictReader(open(codes_csv)))
w = W('pd_rationale_inventory.csv')
w.writerow(['mode', 'provider', 'share_of_decisions', 'action', 'code', 'text'])
w2 = W('pd_cooperation_sources.csv')
w2.writerow(['mode', 'code', 'share_of_decisions', 'share_of_cooperation'])
for mode in ('calculator', 'reasoner', 'roleplayer'):
    for prov in ('openai', 'deepseek'):
        inv = collections.Counter(); act = {}
        for r in arms[('pd', mode, prov)]:
            for e in r['pd_inventory']:
                inv[e['text']] += e['count']; act[e['text']] = e['action']
        tot = sum(inv.values()); coop = sum(c for t, c in inv.items() if act[t] == 'COOPERATE')
        by_code = collections.Counter()
        for t, c in inv.most_common():
            hits = [x['code'] for x in codes if x['mode'] == mode and t.startswith(x['text_prefix'])]
            assert len(hits) <= 1, t[:60]
            code = hits[0] if hits else ''
            if prov == 'openai' and act[t] == 'COOPERATE' and c / tot >= 0.005:
                assert code, f'uncoded GPT cooperation rationale with share >= 0.5%: {t[:80]}'
            if act[t] == 'COOPERATE':
                by_code[code or 'uncoded (<0.5% each)'] += c
            w.writerow([mode, prov, round(c / tot, 5), act[t], code, t])
        if prov == 'openai' and coop:
            for code, c in by_code.most_common():
                w2.writerow([mode, code, round(c / tot, 4), round(c / coop, 4)])
                print(f'PD {mode:10s} GPT cooperation from {code:22s} {100*c/tot:5.1f}% of decisions, {100*c/coop:5.1f}% of cooperation')
        print(f'PD {mode:10s} {prov:8s} distinct rationales {len(inv)} (pooled over 5 seeds)')

# 4. Log-odds with an informative Dirichlet prior (pooled corpus, a0 = 1000), CV only (PD texts are too few)
def logodds(y1, y2, a0=1000.0):
    prior = y1 + y2; P = sum(prior.values()); n1, n2 = sum(y1.values()), sum(y2.values()); out = []
    for wd, cnt in prior.items():
        a = a0 * cnt / P; f1, f2 = y1.get(wd, 0), y2.get(wd, 0)
        d = math.log((f1 + a) / (n1 + a0 - f1 - a)) - math.log((f2 + a) / (n2 + a0 - f2 - a))
        out.append((d / math.sqrt(1 / (f1 + a) + 1 / (f2 + a)), wd, f1, f2))
    return sorted(out)
w = W('logodds_gpt_vs_deepseek.csv')
w.writerow(['mode', 'word', 'z_gpt_minus_deepseek', 'count_gpt', 'count_deepseek'])
for mode in ('calculator', 'reasoner', 'roleplayer'):
    y = {}
    for prov in ('openai', 'deepseek'):
        y[prov] = collections.Counter()
        for r in arms[('cv', mode, prov)]:
            y[prov].update(r['tokens'])
    res = logodds(y['openai'], y['deepseek'])
    for z, wd, a, b in res:
        w.writerow([mode, wd, round(z, 2), a, b])
    print(f'CV {mode:10s} GPT words:', ', '.join(x[1] for x in res[::-1][:10]))
    print(f'CV {mode:10s} DeepSeek words:', ', '.join(x[1] for x in res[:10]))

# 5. Deviation cells: what agents said in the decisions where they depart from the rule-based benchmark.
#    Labels come from codebook v1 (CODEBOOK.md), applied to every deviating decision by 03_label_deviations.py.
#    One row per cell x reason; shares sum to 1 within a cell. Feeds R/19_waffle_deviations.R.
import gzip
lab_path = os.path.join(out_dir, 'deviation_labels.csv.gz')
assert os.path.exists(lab_path), 'run 03_label_deviations.py before 02_summarise.py'
# A direction is shown as its own waffle cell if it holds at least 5% of the arm's deviating decisions, so each label
# (e.g. "Quiet, classic rule active") and n describe exactly the decisions shown. This keeps both directions for the
# GPT calculator prisoner's dilemma arm (cooperates where the classic rule defects, and defects where it cooperates)
# and drops the few opposite-direction decisions elsewhere (civil violence: GPT calculator 15 quiet, GPT role-player
# 37 active, DeepSeek role-player 5 active; prisoner's dilemma: GPT role-player 10 defections), counted in the log.
lab_rows = list(csv.DictReader(gzip.open(lab_path, 'rt')))
assert len(lab_rows) == 98657, 'expected 98,657 deviating decisions (codebook v2)'
def direction(r):
    return {'COOPERATE': 'cooperates', 'DEFECT': 'defects', 'ACTIVE': 'dev_active', 'QUIET': 'dev_quiet'}[r['action']]
dirs = collections.defaultdict(collections.Counter)
for r in lab_rows:
    dirs[(r['game'], r['mode'], r['model'])][direction(r)] += 1
shown = {(k, d) for k, v in dirs.items() for d, c in v.items() if c >= 0.05 * sum(v.values())}
cells = collections.defaultdict(collections.Counter)
for r in lab_rows:
    k = (r['game'], r['mode'], r['model'])
    if (k, direction(r)) in shown:
        cells[k + (direction(r),)][r['label']] += 1
dropped = sum(c for k, v in dirs.items() for d, c in v.items() if (k, d) not in shown)
assert dropped == 67, f'expected 67 decisions in minority directions (57 civil violence, 10 prisoner\'s dilemma), found {dropped}'
assert len(cells) == 9, f'expected 9 waffle cells, found {len(cells)}'
DEV_DIR = {'cooperates': 'cooperates where the classic rule defects', 'defects': 'defects where the classic rule cooperates',
           'dev_active': 'active where the classic rule is quiet', 'dev_quiet': 'quiet where the classic rule is active'}
rows = []
for (game, mode, model, d), c in sorted(cells.items()):
    n = sum(c.values())
    for reason, x in c.most_common():
        rows.append([f'{game}_{mode}_{model.lower()}_{d}', game, mode, model, DEV_DIR[d], n, reason, round(x / n, 5)])
for k, v in sorted(dirs.items()):
    print(f'{k[0]} {k[1]:10s} {k[2]:8s} deviations by direction {dict(v)}; shown: {sorted(d for (kk, d) in shown if kk == k)}')
w = W('deviation_cells.csv')
w.writerow(['cell', 'game', 'mode', 'model', 'deviation', 'n_decisions', 'reason', 'share'])
w.writerows(rows)
for cell in dict.fromkeys(r[0] for r in rows):
    print(f'{cell:24s}', '; '.join(f'{r[6]} {100*r[7]:.0f}%' for r in rows if r[0] == cell), f'(n={[r[5] for r in rows if r[0]==cell][0]})')
