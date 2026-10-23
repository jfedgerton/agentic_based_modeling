"""06_ic_validation_sample.py -- blind validation sample for the crisis-game signal reasons (codebook v1, T4; seed 123).

Strata are framing x machine label (3 x 6). Each stratum gives n items (default 6; smaller strata are taken whole),
so every reason is checked in every framing, about 100 items in all. The coder sees the framing, the agent's true
capability (as printed in its prompt), the level it signalled and the rationale, never the machine label.
Writes validation/ic_validation_blind.csv (for the coder) and validation/ic_validation_key.csv (machine labels).
Usage: python3 06_ic_validation_sample.py <ic_signal_labels.csv.gz> <out_dir> [n_per_stratum=6]
       python3 06_ic_validation_sample.py <ic_signal_labels.csv.gz> <out_dir> --rekey
--rekey keeps the drawn items (and any hand codes) and only refreshes the machine labels after a rule change.
"""
import csv, gzip, os, random, sys, collections

src, out_dir = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 6
rows = list(csv.DictReader(gzip.open(src, "rt")))
assert rows, "no overstating signals found"
for r in rows:
    r["row_id"] = f"{r['mode']}|{r['step']}|{r['agent_id']}"
assert len({r["row_id"] for r in rows}) == len(rows), "row ids are not unique"
key_path = os.path.join(out_dir, "ic_validation_key.csv")

if "--rekey" in sys.argv:
    by_id = {r["row_id"]: r for r in rows}
    old = list(csv.DictReader(open(key_path)))
    changed = 0
    for k in old:
        r = by_id[k["row_id"]]
        changed += r["label"] != k["label"]
        k.update(rule=r["rule"], label=r["label"])
    w = csv.DictWriter(open(key_path, "w", newline=""), fieldnames=list(old[0].keys()))
    w.writeheader(); w.writerows(old)
    print(f"Rekeyed {len(old)} crisis-game items; machine label changed for {changed}")
    sys.exit(0)

rng = random.Random(123)
strata = collections.defaultdict(list)
for r in rows:
    strata[(r["mode"], r["label"])].append(r)
sample = []
for k in sorted(strata):
    v = strata[k]
    take = v if len(v) <= N else rng.sample(v, N)
    print(f"T4  {k[0]:<11} {k[1]:<36} population {len(v):>6}  sampled {len(take):>2}")
    sample += take
rng.shuffle(sample)  # strata interleaved so the coder cannot infer the machine label from position
print(f"TOTAL crisis-game items to code: {len(sample)}")
os.makedirs(out_dir, exist_ok=True)
LEVEL = {"0": "low", "1": "medium", "2": "high"}
FRAMING = {"calculator": "calculator", "reasoner": "reasoner", "roleplayer": "role-player"}
with open(os.path.join(out_dir, "ic_validation_blind.csv"), "w", newline="") as fb, open(key_path, "w", newline="") as fk:
    b = csv.writer(fb); kk = csv.writer(fk)
    b.writerow(["item", "task", "framing", "true_capability", "signalled_level", "text", "T4_code_1_to_6", "coder_note"])
    kk.writerow(["item", "task", "row_id", "mode", "capability", "rule", "label"])
    for i, r in enumerate(sample, 1):
        b.writerow([i, "T4", FRAMING[r["mode"]], r["capability"], LEVEL[r["signal_level"]], r["text"], "", ""])
        kk.writerow([i, "T4", r["row_id"], r["mode"], r["capability"], r["rule"], r["label"]])
print("Wrote", os.path.join(out_dir, "ic_validation_blind.csv"))
