"""05_ic_signal_reasons.py -- crisis game (interstate conflict): why GPT agents overstate their strength (Fig. 5B).

Reads the signal-phase calls of the three GPT crisis-game runs (one per framing), steps 51-100, and keeps every
signal that claims more strength than the agent's true capability (the rule-based benchmark's honest signal).
Each kept rationale gets one reason from codebook v1, section T4 (CODEBOOK.md); first matching rule wins.

Honest signal level, as in the classic agent (src/agents/classic_interstate_conflict.py at v1.0-experiments):
capability > 0.66 high, > 0.33 medium, else low. Capability is read from the agent's own prompt (2 decimals),
the value the agent saw. Signal meaning comes from the prompt the agent answered: calculator S1/S2/S3 = low/medium/
high value_a; reasoner and role-player WEAK/MODERATE/STRONG. (The simulation stores calculator S1 as STRONG and S3 as
WEAK; this script never uses the stored labels.) Calls whose JSON did not parse are skipped, because the logged
signal is then the classic fallback, not the model's choice.

Usage: python3 05_ic_signal_reasons.py <ABM_ROOT> <out_dir>
Writes <out_dir>/ic_signal_labels.csv.gz (one row per overstating signal) and <out_dir>/ic_signal_cells.csv.
"""
import csv, gzip, json, os, re, sys, collections

abm_root, out_dir = sys.argv[1], sys.argv[2]
BASE = os.path.join(abm_root, "one_simulation", "outputs_agent_info")
# GPT crisis-game runs kept by R/01_run_index.R (the DeepSeek runs fail the 5% parse gate); R/12 checks this list
RUNS = {"calculator": "ic_calculator_seed32_f954c956",
        "reasoner": "ic_reasoner_seed52_562ca987",
        "roleplayer": "ic_roleplayer_seed62_ad69da08"}
LEVEL = {"calculator": {"S1": 0, "S2": 1, "S3": 2},
         "reasoner": {"WEAK": 0, "MODERATE": 1, "STRONG": 2},
         "roleplayer": {"WEAK": 0, "MODERATE": 1, "STRONG": 2}}
CAP_RE = re.compile(r"-\s*(?:value_a|capability \(true military strength[^:]*|Your true military capability)\s*:\s*([0-9.]+)")
STEPS = range(51, 101)

# ---- Codebook v1, T4: reasons for overstating strength (first match wins) ------------------------------------------
STRENGTH_WORD = r"(?:high|strong|above[- ]average)"  # vague words (decent, solid, good) are not coded as misreadings
OWN = r"(?:value_a|capability|military capability|strength|military strength|position)"
MOD = r"(?:relatively |fairly |quite |reasonably |still |moderately )?"
R1_OWN_HIGH = re.compile(  # applies only when capability < 0.5; must describe the agent's OWN capability
    r"(?:\b(?:my|our|your)|\bwith(?: a| an)?|\bgiven(?: a| an| my| our| your)?) (?:true )?" + MOD + STRENGTH_WORD + r" (?:true )?" + OWN
    + r"|\b(?:my|our|your) (?:true )?" + OWN + r"(?: of [0-9.]+)? (?:is|remains) " + MOD + r"(?:" + STRENGTH_WORD + r"|(?:slightly |well )?above (?:average|the midpoint|0\.5))"
    + r"|\b(?:with|given) (?:a |an )?" + OWN + r"(?: of [0-9.]+)? (?:slightly |well )?above (?:average|the midpoint|0\.5)"
    # bare forms: "value_a is relatively high", "value_a relatively high", "a relatively strong value_a" (not "a high value_a type")
    + r"|\bvalue_a (?:is |being )?" + MOD + STRENGTH_WORD
    + r"|\ban? " + MOD + STRENGTH_WORD + r" value_a(?! type| posture| stance| signal| agent| image)", re.I)
R2_RULES = re.compile(
    r"role assignments?|(?:favorable|better|advantageous|future|preferred) (?:role )?assignments?|"
    r"assigned (?:the |a |to )?(?:more |higher|better|favorable|stronger|preferred|advantageous)|(?:favorable|better|safer|advantageous) "
    r"(?:role|matching|match|partner)|matching|partner selection|be assigned|role_X partner|choose (?:you|me) as", re.I)
R3_BLUFF = re.compile(
    r"bluff|overstat|overclaim|exaggerat|inflat|misleading|(?:true|actual|real) (?:strength|capability|capabilities|value_a) (?:is|are) "
    r"(?:not visible|hidden|private|unobserv|not observ)|(?:not|cannot be) (?:directly )?(?:visible|observ|seen|verified)|"
    r"(?:cannot|can't|do not|don't) (?:directly )?(?:see|observe|verify)|hidden from|"
    r"more than (?:my|our) (?:true|actual)|than (?:i|we) (?:really|actually) (?:am|are)", re.I)
# A bluff word in a sentence that rejects or warns against bluffing ("avoid an empty bluff", "overstating risks war")
# does not count; R3 needs a sentence that uses it without these cues.
R3_REJECT = re.compile(r"\b(?:avoid\w*|without|rather than|instead of|could|would|risk\w*|unnecessar\w*|empty|"
                       r"implausible|too|enough to|provok\w*|invit\w*)\b", re.I)
R4_DETER = re.compile(
    r"deter|leverage|bargaining (?:position|power|terms|outcomes|posture|stance)|better (?:terms|settlements?|shares?|deals?|outcomes)|"
    r"favorable (?:terms|settlements?|shares?|outcomes|treatment)|discourag|aggressive demands|(?:tougher|higher|excessive) demands|"
    r"invite (?:exploitation|tougher|higher|unfavorable|aggress)|project(?:ing|s)? (?:strength|resolve|firmness|power|confidence)|"
    r"resolve|credib|(?:look|appear|seem)(?:ing|s)? (?:weak|vulnerable|hesitant)|perceived as (?:a |the )?(?:strong|high|tough)|"
    r"exploit|push(?:ed)? around|concede|concessions|reputation|stronger (?:type|standing|posture|image)|"
    r"(?:position|present)(?:s|ing)? (?:you|me|us|myself|yourself|ourselves) as (?:the |a )?(?:strong|high|stronger|tough)", re.I)
R5_KEEP = re.compile(
    r"continu|maintain|keep(?:ing)? (?:the |my |our |a |this )?(?:same |strong |high |previous )?(?:signal|declaration|posture|stance)|"
    r"consisten|already (?:been )?(?:signal|declar|established|using)|previous(?:ly)? (?:signal|declar|strong|S3)|prior (?:signal|declaration)|"
    r"last (?:round|signal|declaration|choice)|repeat|stay(?:ing)? with|stick(?:ing)? with|payoff (?:was|has been|improved|is)|"
    r"(?:good|strong|solid|positive|favorable|decent) (?:last[- ]round )?payoff", re.I)
LABELS = ["Misreads own strength", "Misreads the rules", "Bluffs openly", "Deterrence or bargaining leverage",
          "Keeps past signal or payoff", "Other or unclassified"]


def label(text, cap):
    if cap < 0.5 and R1_OWN_HIGH.search(text):
        return LABELS[0], "R1"
    if R2_RULES.search(text):
        return LABELS[1], "R2"
    if any(R3_BLUFF.search(sent) and not R3_REJECT.search(sent) for sent in re.split(r"(?<=[.!?;])\s+", text)):
        return LABELS[2], "R3"
    if R4_DETER.search(text):
        return LABELS[3], "R4"
    if R5_KEEP.search(text):
        return LABELS[4], "R5"
    return LABELS[5], "R6"


os.makedirs(out_dir, exist_ok=True)
rows, totals = [], {}
for mode, run in RUNS.items():
    n = collections.Counter()
    with open(os.path.join(BASE, run, "llm_interactions.jsonl")) as fh:
        for line in fh:
            if '"phase": "signal"' not in line:
                continue
            r = json.loads(line)
            p = r.get("parsed") or {}
            if p.get("phase") != "signal" or r["step"] not in STEPS:
                continue
            n["signal_calls"] += 1
            rp = p.get("raw_parsed") or {}
            s = rp.get("signal")
            if s not in LEVEL[mode]:
                n["parse_failure"] += 1
                continue
            m = CAP_RE.search(r["prompt"])
            assert m, f"{run} step {r['step']} agent {r['agent_id']}: capability not found in prompt"
            cap = float(m.group(1))
            honest = 2 if cap > 0.66 else 1 if cap > 0.33 else 0
            lv = LEVEL[mode][s]
            if lv > honest:
                n["overstates"] += 1
                text = (rp.get("short_rationale") or "").strip()
                lab, rule = label(text, cap)
                rows.append(dict(mode=mode, step=r["step"], agent_id=r["agent_id"], capability=cap, honest_level=honest,
                                 signal_level=lv, rule=rule, label=lab, text=text))
            elif lv < honest:
                n["understates"] += 1
            else:
                n["honest"] += 1
    assert n["signal_calls"] == 50 * 400, f"{run}: expected 20,000 signal calls in steps 51-100, found {n['signal_calls']}"
    assert n["parse_failure"] < 0.05 * n["signal_calls"], f"{run}: signal parse failures above 5%"
    totals[mode] = n
    print(f"{mode:<11} signals {n['signal_calls']}  parse failures {n['parse_failure']}  honest {n['honest']}  "
          f"overstates {n['overstates']}  understates {n['understates']}")

with gzip.open(os.path.join(out_dir, "ic_signal_labels.csv.gz"), "wt", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(os.path.join(out_dir, "ic_signal_cells.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mode", "reason", "n", "share", "n_overstating", "n_valid_signals"])
    for mode in RUNS:
        c = collections.Counter(r["label"] for r in rows if r["mode"] == mode)
        tot = sum(c.values())
        assert tot == totals[mode]["overstates"]
        valid = totals[mode]["signal_calls"] - totals[mode]["parse_failure"]
        for lab in LABELS:
            w.writerow([mode, lab, c[lab], c[lab] / tot, tot, valid])
        print(f"{mode:<11} " + "  ".join(f"{lab} {100 * c[lab] / tot:.1f}%" for lab in LABELS))
print("Wrote", os.path.join(out_dir, "ic_signal_labels.csv.gz"), "and ic_signal_cells.csv")
