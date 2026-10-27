"""04_validation_sample.py -- stratified, blind validation sample for codebook v1 (seed 123).

Three coding tasks, each validating the one judgment the machine makes (everything else is deterministic):
  T1  CV calculator: which option does the rationale say is better? (A, B, or neither)  strata = model x machine pref
  T2  CV reasoner/role-player quiet decisions: does the rationale cite arrest risk? the absence of fellow protesters?
      strata = machine feature pattern (both / risk only / no-protesters only); the large 'both' stratum is split
      evenly over the four model x framing cells
  T3  PD: census of every distinct deviating rationale (cooperations where the classic rule defects and defections
      where it cooperates; no sampling), coded into the PD categories
Strata smaller than n are taken whole. Writes validation/validation_blind.csv (for the human coder, no machine
labels) and validation/validation_key.csv (machine labels, joined later by row_id).
Usage: python3 04_validation_sample.py <deviation_labels.csv.gz> <out_dir> [n_per_stratum=50] [--dry-run]
       python3 04_validation_sample.py <deviation_labels.csv.gz> <out_dir> --rekey
--rekey keeps the drawn items (and any hand codes in validation_blind.csv) and only refreshes the machine labels in
validation_key.csv after the codebook rules change. Row ids are stable because the set of deviating decisions is.
"""
import csv, gzip, random, sys, os, collections
src, out_dir = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 50
DRY = '--dry-run' in sys.argv
rng = random.Random(123)
rows = list(csv.DictReader(gzip.open(src, 'rt')))
assert len(rows) == 98657, len(rows)  # codebook v2
if '--rekey' in sys.argv:
    by_id = {r['row_id']: r for r in rows}
    key_path = os.path.join(out_dir, 'validation_key.csv')
    old = list(csv.DictReader(open(key_path)))
    changed = 0
    for k in old:
        if k['task'] in ('T1', 'T2'):
            r = by_id[k['row_id']]
            assert r['model'] == k['model'] and r['mode'] == k['mode'] and r['action'] == k['action'], k['item']
            changed += (r['stated_pref'], r['label']) != (k['stated_pref'], k['label'])
            k.update(stated_pref=r['stated_pref'], rule=r['rule'], label=r['label'])
    w = csv.DictWriter(open(key_path, 'w', newline=''), fieldnames=list(old[0].keys()))
    w.writeheader(); w.writerows(old)
    print(f'Rekeyed {len(old)} items; machine label changed for {changed}')
    sys.exit(0)
strata = collections.defaultdict(list)
for r in rows:
    if r['game'] == 'cv' and r['mode'] == 'calculator':
        strata[('T1', r['model'], 'pref_' + (r['stated_pref'] or 'none'))].append(r)
    elif r['game'] == 'cv' and r['action'] == 'QUIET':
        pat = {'Arrest risk, no fellow protesters': 'risk+alone', 'Arrest risk, other reasons': 'risk_only',
               'No fellow protesters, no arrest risk': 'alone_only', 'Other or unclassified': 'neither'}[r['label']]
        key = ('T2', pat, r['model'] + '_' + r['mode']) if pat == 'risk+alone' else ('T2', pat, 'all')
        strata[key].append(r)
sample = []
for k in sorted(strata):
    v = strata[k]
    n = N // 4 if k[1] == 'risk+alone' else N  # the 'both' stratum is split over four cells
    take = v if len(v) <= n else rng.sample(v, n)
    print(f'{k[0]}  {k[1]:<10} {k[2]:<20} population {len(v):>6}  sampled {len(take):>3}')
    sample += [(k, r) for r in take]
# T3 census: one row per distinct PD cooperating rationale, weighted by how often it occurs
pd = collections.Counter(); first = {}
for r in rows:
    if r['game'] == 'pd':
        pd[(r['mode'], r['text'])] += 1; first[(r['mode'], r['text'])] = r
print(f'T3  census     distinct PD rationales {len(pd):>3}  (covering {sum(pd.values())} decisions)')
print(f'TOTAL items to code: {len(sample) + len(pd)}')
if DRY:
    sys.exit(0)
os.makedirs(out_dir, exist_ok=True)
rng.shuffle(sample)  # strata interleaved so the coder cannot infer the machine label from position
items = [(k[0], r) for k, r in sample] + [('T3', dict(first[k], decisions=c)) for k, c in sorted(pd.items())]
with open(os.path.join(out_dir, 'validation_blind.csv'), 'w', newline='') as fb, \
     open(os.path.join(out_dir, 'validation_key.csv'), 'w', newline='') as fk:
    b = csv.writer(fb); kk = csv.writer(fk)
    b.writerow(['item', 'task', 'game_framing', 'text', 'T1_pref_A_B_none', 'T2_cites_arrest_risk_0_1',
                'T2_cites_no_fellow_protesters_0_1', 'T3_code', 'coder_note'])
    kk.writerow(['item', 'task', 'row_id', 'model', 'mode', 'action', 'stated_pref', 'rule', 'label', 'decisions'])
    for i, (task, r) in enumerate(items, 1):
        b.writerow([i, task, f"{r['game']} {r['mode']}", r['text'], '', '', '', '', ''])
        kk.writerow([i, task, r['row_id'], r['model'], r['mode'], r['action'], r['stated_pref'], r['rule'], r['label'],
                     r.get('decisions', 1)])
print('Wrote', os.path.join(out_dir, 'validation_blind.csv'))
