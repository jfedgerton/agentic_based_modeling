"""Codebook rules (v1). Pure functions, standard library only. See CODEBOOK.md for definitions."""
import re

# ---------- Civil violence calculator: which option does the rationale say is better? ----------
# Options are the prompt's neutral labels: A = stay quiet, B = become active.
OPT = r'(?:option |action |choice )?\(?(?P<o>(?-i:[AB]))\)?'  # option letters are capitals only (not the article "a")
ADV = r'(?:(?:clearly|much|far|substantially|significantly|slightly|marginally|strictly|still|thus|therefore|so|now|actually|also|the|a|more|even|considerably|notably|noticeably|somewhat|overall|immediate|current)\s+)*'
MOD = r'(?:(?:the|a|lower|lower-disutility|less|least|costly|cheaper|safer|better|best|optimal|preferred|immediate|current|disutility-minimizing|minimum)[- ]?){0,4}\s*'
GOOD = (r'(?:better|preferable|preferred|optimal|the best|best|the better (?:choice|option)|the optimal (?:choice|option)|'
        r'lower|smaller|less costly|less harmful|cheaper|zero|near[- ]zero|negligible|essentially zero|extremely low|very low|'
        r'(?:zero|near[- ]zero|negligible|essentially zero|extremely low|very low) disutility|the lower(?:-disutility)? (?:option|choice)|dominant|superior|the clear choice)')
BAD = r'(?:worse|higher|larger|greater|more costly|costlier|riskier|inferior|suboptimal|the worse (?:choice|option))'
SAME_SENT = r'(?:(?!\.\s|\.$|;)[^\n])'          # any character that does not end the sentence
NO_OPT = r'(?:(?!\b(?-i:[AB])\b)(?!\.\s|\.$|;)[^\n])'  # ... and is not itself an option label
CHOICE_NOUN = r'the\s+(?:(?!worse|riskier|costlier|inferior|suboptimal)[\w-]+\s+){0,3}(?:choice|option|action|pick)'
RULES = [
    # R1 explicit choice: "choose A", "I choose the lower-disutility option A", "go with B", "so B"
    ('R1_choice', re.compile(r"(?:choose|choosing|chose|pick|picking|select|selecting|go with|going with|opt for|opting for|"
                             r"prefer|i will|i'll|i would|i'd|decide to|deciding on|keep|stick with|stay with|switch to|"
                             r"\bso\b|hence|therefore|thus)\s+(?:to\s+)?" + MOD + OPT + r'\b', re.I), 'same'),
    # R2 positive verdict on an option: "B is strictly better", "A is lower by a clear margin", "B minimizes disutility"
    ('R2_good', re.compile(r'\b' + OPT + r"(?:'s disutility| disutility)?\s+(?:is|remains|would be|looks|seems|appears|gives a|yields a|has a|has|gives|yields|produces|offers)\s+" + ADV + GOOD + r'\b', re.I), 'same'),
    ('R2_min', re.compile(r'\b' + OPT + r'\s+(?:minimizes|minimises|reduces|lowers)\b', re.I), 'same'),
    ('R2_making', re.compile(r'(?:making|makes|leaving|leaves)\s+' + OPT + r'\s+(?:the\s+)?' + ADV + GOOD, re.I), 'same'),
    # R3 negative verdict: "B has much higher disutility", "B is worse" -> the other option
    ('R3_bad', re.compile(r'\b' + OPT + r"(?:'s disutility| disutility)?\s+(?:is|remains|would be|looks|seems|appears|has|gives|yields|produces|carries|would give|would yield)\s+" + ADV + r'(?:a\s+)?' + ADV + BAD + r'\b', re.I), 'other'),
    # R4 comparison against the other option: "lower than A's" / "higher than B's" with the subject option earlier in the clause
    ('R4_than', re.compile(r'\b' + OPT + r'\b' + SAME_SENT + r'{0,160}?\b(?P<cmp>lower|smaller|less|better|cheaper|higher|larger|greater|more|worse|costlier)\s+than\s+(?:for\s+|that of\s+|option\s+|action\s+)?(?P<o2>(?-i:[AB]))\b', re.I), 'cmp'),
    # R5 trailing verdict on the most recent option: "B gives 0.04, which is lower" / "B ... which is slightly higher"
    ('R5_which_good', re.compile(r'\b' + OPT + r'\b' + NO_OPT + r'{0,160}?,?\s+which\s+is\s+' + ADV + r'(?:lower|smaller|less)\b(?!\s+than)', re.I), 'same'),
    ('R5_which_bad', re.compile(r'\b' + OPT + r'\b' + NO_OPT + r'{0,160}?,?\s+which\s+is\s+' + ADV + r'(?:higher|larger|greater|worse)\b(?!\s+than)', re.I), 'other'),
    # R6 option named as the choice: "B is the intended choice", "B is the safer near-tie choice"
    ('R6_choice_noun', re.compile(r'\b' + OPT + r'\s+(?:is|remains|would be|seems|appears)\s+(?:(?:still|likely|probably|actually|clearly|effectively|therefore|thus|also)\s+)*' + CHOICE_NOUN + r'\b', re.I), 'same'),
    # R7 reversed order: "the safer immediate choice is A", "the better action is A"
    ('R7_noun_is', re.compile(r'\bthe\s+(?:(?!worse|riskier|costlier|inferior|suboptimal)[\w-]+\s+){0,3}(?:choice|option|action)\s+(?:is|remains|would be)\s+(?:still\s+|clearly\s+)?' + OPT + r'\b', re.I), 'same'),
    ('R6_preferable', re.compile(r'\b' + OPT + r'\b' + NO_OPT + r'{0,80}?\b(?:appears|seems|is|looks)\s+(?:more\s+|the\s+more\s+|clearly\s+)?(?:preferable|preferred|advantageous|favorable)\b', re.I), 'same'),
    ('R2_lower_of', re.compile(r'\b(?:lower|smaller|less|minimal|minimum)\s+disutility\s+(?:from|for|of|with|under)\s+' + OPT + r'\b', re.I), 'same'),
    # R8 conclusion after a colon at the end of a sentence: "choose the lower disutility: A."
    ('R8_colon', re.compile(r':\s*' + OPT + r'\s*(?:\.|$)', re.I), 'same'),
    ('R3_not_adv', re.compile(r'\b' + OPT + r'\s+is\s+not\s+(?:advantageous|better|preferable|optimal|favorable)\b', re.I), 'other'),
]

def cv_calc_pref(text):
    """Return (pref in {'A','B',None}, rule_id). Among all rule matches, the last one in the text wins."""
    best = None
    for rid, pat, kind in RULES:
        for m in pat.finditer(text):
            o = m.group('o').upper()
            if kind == 'same':
                p = o
            elif kind == 'other':
                p = 'B' if o == 'A' else 'A'
            else:  # comparison
                o2 = m.group('o2').upper()
                if o2 == o:
                    continue
                low = m.group('cmp').lower() in ('lower', 'smaller', 'less', 'better', 'cheaper')
                p = o if low else o2
            pos = m.end()
            if best is None or pos > best[0]:
                best = (pos, p, rid)
    if best:
        return best[1], best[2]
    return None, 'none'

# ---------- Civil violence reasoner / role-player: why stay quiet? Two independent features ----------
# RISK: the rationale cites arrest, detention, jail, police or enforcement as a cost.
RISK = re.compile(r'\b(?:arrest\w*|detain\w*|detention|jail\w*|imprison\w*|police\w*|policed|policing|enforcement|enforcers?|cops?|crackdown|repression|danger\w*|riskier|risky|risks?(?!\s*-?\s*(?:aversion|averse|tolerance|tolerant|appetite)))\b', re.I)
# ALONE: the rationale cites the absence or weakness of fellow protesters / active agents.
_Q = r'(?:fellow |nearby |visible |visibly |other |current |ongoing |active |local |existing |concurrent |apparent |observed |clear |an |a )*'
_W = (r'(?:activists?|protesters?|protests?|protesting|demonstrators?|demonstrations?|active agents?|agents? (?:are |is )?active|'
      r'crowds?|allies|support|supporters|momentum|protest (?:activity|wave|movement|presence|support|momentum|turnout)|collective action|'
      r'participants?|co-participants?|co-actors?|peers?|neighbou?rs|turnout|mobili[sz]ation|cover|active|protest)')
ALONE = re.compile(
    r'\bno ' + _Q + _W + r'|\bnone of (?:my |the )?neighbou?rs\b|\bzero ' + _Q + _W +
    r'|\b0 ' + _Q + _W + r'|\bno one (?:nearby |around |else )?(?:is |currently |already )*(?:active|protesting|demonstrating|joining)\b'
    r'|\bno one (?:around |nearby )?is (?:visibly |currently )*(?:protesting|active)\b|\bonly (?:one|1|two|2|a few) (?:protesters?|active)\b'
    r'|\b(?:participation|turnout) (?:nearby |locally )?is (?:zero|low|minimal)\b|\bweakly supported\b'
    r'|\bnobody (?:nearby |else )?(?:is )?(?:active|protesting|joining)\b|\bwithout (?:a |any )?' + _Q + _W +
    r'|\b(?:absence|lack) of (?:any |a |visible |nearby |active )*' + _Q + _W + r'|\balone\b|\bisolated\b'
    r'|\b' + _W + r'[^.]{0,40}\b(?:absent|weak|nonexistent|missing|not visible|lacking|limited|sparse)\b'
    r'|\bmost (?:neighbou?rs|people|others) (?:are )?(?:staying home|quiet)\b|\bmany neighbou?rs staying home\b'
    r'|\bfew (?:if any )?(?:protesters|active)\b'
    r'|\b(?:low|zero|very low|little|limited|weak|minimal) (?:current |local |visible |immediate )*(?:protest )?(?:turnout|presence|momentum|activity|mobili[sz]ation|participation|collective \w+)\b'
    r'|\bturnout (?:is |remains )?(?:currently )?(?:zero|low|minimal)\b|\b(?:inactive|not (?:visibly )?growing|no (?:protest|demonstration) (?:is )?underway)\b', re.I)

def cv_quiet_reason(text):
    """Return one of the four quiet-reason categories from the two features."""
    r, a = bool(RISK.search(text)), bool(ALONE.search(text))
    if r and a:
        return 'Arrest risk, no fellow protesters'
    if r:
        return 'Arrest risk, other reasons'
    if a:
        return 'No fellow protesters, no arrest risk'
    return 'Other or unclassified'
