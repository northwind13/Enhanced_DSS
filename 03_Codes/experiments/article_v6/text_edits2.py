"""Cross-run tracked edits: (a) phrase replacements that may span runs, (b) parenthetical figure/table references -> prose."""
import re, sys, copy, datetime
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "{http://www.w3.org/XML/1998/namespace}"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id = [12000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag)); [e.set(w(k), v) for k, v in (attrib or {}).items()]; [e.append(c) for c in kids]; return e
def run(text, rpr=None):
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); t = E("t"); t.text = text; t.set(XML + "space", "preserve"); r.append(t); return r
def ins(*k): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *k)
def dele_run(r):
    d = E("del", {"id": nid(), "author": AUTHOR, "date": DATE}); r2 = copy.deepcopy(r)
    for t in r2.iter(w("t")): t.tag = w("delText")
    d.append(r2); return d
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot()

def live_runs(p):
    """text runs of p in order: top-level and inside w:ins (not deleted), that carry a w:t"""
    out = []
    for c in p:
        if c.tag == w("r") and c.find(w("t")) is not None: out.append(c)
        elif c.tag == w("ins"):
            out.extend(r for r in c if r.tag == w("r") and r.find(w("t")) is not None)
    return out

def split_run(r, a, b):
    """split run r so that [a:b) of its text becomes its own run; returns that middle run"""
    t = r.find(w("t")); txt = t.text; rpr = r.find(w("rPr")); parent = r.getparent(); idx = parent.index(r)
    pieces = [txt[:a], txt[a:b], txt[b:]]; made = []
    parent.remove(r)
    for k, s in enumerate(pieces):
        if s == "" and k != 1: continue
        nr = run(s, rpr); parent.insert(idx, nr); idx += 1; made.append((k, nr))
    return dict(made)[1]

def replace_span(p, old, new):
    runs = live_runs(p); txt = ""; pos = []
    for r in runs:
        s = r.find(w("t")).text or ""; pos.append((len(txt), len(txt) + len(s), r)); txt += s
    a = txt.find(old)
    if a < 0: return False
    b = a + len(old); hit = [(s, e, r) for s, e, r in pos if e > a and s < b]
    rpr = hit[0][2].find(w("rPr")); first = None
    for s, e, r in hit:
        mid = split_run(r, max(a, s) - s, min(b, e) - s)
        d = dele_run(mid); mid.getparent().replace(mid, d)
        if first is None: first = d
    first.addprevious(ins(run(new, rpr))) if first.getparent().tag != w("ins") else first.addprevious(run(new, rpr))
    return True

# (a) phrase edits (cross-run); each applied once, skipped when the new wording is already present
import importlib.util
EDITS = [
 ("Decision support systems exist for parts of this task", "Decision support systems (DSSs) exist for parts of this task"),
 ("DSS\u2019s exist for parts of this task", "Decision support systems (DSSs) exist for parts of this task"),
 ("Decision support systems (DSS) are systems of systems (SoS).", "Decision support systems (DSSs) are systems of systems (SoS)."),
 ("These works motivate a decision support system that generates", "These works motivate a DSS that generates"),
 ("a pretrained language model proposes a new rule", "a pretrained large language model (LLM) proposes a new rule"),
 ("let language models weight a fixed hierarchy", "let LLMs weight a fixed hierarchy"),
 ("multi-agent language models produce free-text response plans", "multi-agent LLMs produce free-text response plans"),
 ("a language-model agent that automates", "an LLM agent that automates"),
 ("The proposer is a pretrained language model used in context only", "The proposer is a pretrained LLM used in context only"),
 ("the language-model proposer behind the full gate chain", "the LLM proposer behind the full gate chain"),
 ("The proposals of a language model are not bitwise reproducible", "The proposals of an LLM are not bitwise reproducible"),
 ("lets a pretrained language model extend that doctrine", "lets a pretrained LLM extend that doctrine"),
 ("Evolving Takagi-Sugeno models create and tune rules online", "Evolving Takagi-Sugeno (TS) models create and tune rules online"),
 ("zero-order Takagi-Sugeno inference turns rule activations", "zero-order TS inference turns rule activations"),
 ("Wildfire response is carried out by a system of systems", "Wildfire response is carried out by a system of systems (SoS)"),
 ("presents DisasterAware as a system of systems with five kinds of participant", "presents DisasterAware as an SoS with five kinds of participant"),
 ("DisasterAware as a system of systems. The simulation core owns", "DisasterAware as an SoS. The simulation core owns"),
 ("Predefined alternatives in centralized architectures: the system ranks", "Predefined alternatives in centralized architectures. The system ranks"),
]
def para_text(p): return "".join((r.find(w("t")).text or "") for r in live_runs(p))
for old, new in EDITS:
    pars = list(root.iter(w("p")))
    if any(new in para_text(p) for p in pars): print("phrase already applied:", old[:40]); continue
    ok = any(replace_span(p, old, new) for p in pars)
    print("phrase", old[:40], ok)

# (b) parenthetical cross references
PHRASE = {"Fig. 3": ", as depicted in ", "Fig": ", as shown in ", "Table": ", listed in "}
count = 0
for p in list(root.iter(w("p"))):
    changed = True
    while changed:
        changed = False
        runs = live_runs(p)
        for i, r in enumerate(runs):
            t = r.find(w("t")); s = t.text or ""
            m = re.search(r" \((Fig\. |Table )$", s)
            if not m: continue
            kind = m.group(1).strip().rstrip(".")
            # which number follows? (cached text of the field after this run)
            nxt_txt = "".join((x.find(w("t")).text or "") for x in runs[i + 1:i + 3])
            key = "Fig. 3" if kind == "Fig" and nxt_txt.startswith("3") else kind
            phrase = PHRASE[key]
            mid = split_run(r, m.start(), m.start() + 2)              # the " (" piece
            d = dele_run(mid); mid.getparent().replace(mid, d)
            (d.addprevious(ins(run(phrase, r.find(w("rPr"))))) if d.getparent().tag != w("ins") else d.addprevious(run(phrase)))
            # closing parenthesis: first later run whose text starts with ")"
            for r2 in live_runs(p):
                s2 = r2.find(w("t")).text or ""
                if r2 is not r and s2.startswith(")") and live_runs(p).index(r2) > live_runs(p).index(d.getparent() if d.getparent().tag == w("p") else d) if False else True:
                    pass
            # simpler: scan runs after the deleted piece in document order
            after = False
            for r2 in live_runs(p):
                s2 = r2.find(w("t")).text or ""
                if not after:
                    if r2.getparent() is d.getparent() and d.getparent().index(r2) > d.getparent().index(d): after = True
                    elif r2.getparent().tag == w("ins") and d.getparent().index(r2.getparent()) > d.getparent().index(d): after = True
                    else: continue
                if s2.startswith(")"):
                    c = split_run(r2, 0, 1); dc = dele_run(c); c.getparent().replace(c, dc); break
            count += 1; changed = True; break
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("parenthetical refs rewritten:", count)
