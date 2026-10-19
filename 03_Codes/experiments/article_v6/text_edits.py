"""Tracked in-run text replacements (abbreviations, colon fixes). Usage: python text_edits.py <unpacked dir>"""
import re, sys, copy, datetime
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "{http://www.w3.org/XML/1998/namespace}"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id = [11000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag)); [e.set(w(k), v) for k, v in (attrib or {}).items()]; [e.append(c) for c in kids]; return e
def run(text, rpr=None):
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); t = E("t"); t.text = text; t.set(XML + "space", "preserve"); r.append(t); return r
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot()
EDITS = [  # (old, new, max occurrences in order of appearance; None = all)
 ("Decision support systems exist for parts of this task", "Decision support systems (DSSs) exist for parts of this task", 1),
 ("These works motivate a decision support system that generates", "These works motivate a DSS that generates", 1),
 ("a pretrained language model proposes a new rule", "a pretrained large language model (LLM) proposes a new rule", 1),
 ("let language models weight a fixed hierarchy", "let LLMs weight a fixed hierarchy", 1),
 ("multi-agent language models produce free-text response plans", "multi-agent LLMs produce free-text response plans", 1),
 ("a language-model agent that automates", "an LLM agent that automates", 1),
 ("The proposer is a pretrained language model used in context only", "The proposer is a pretrained LLM used in context only", 1),
 ("the language-model proposer behind the full gate chain", "the LLM proposer behind the full gate chain", 1),
 ("The proposals of a language model are not bitwise reproducible", "The proposals of an LLM are not bitwise reproducible", 1),
 ("lets a pretrained language model extend that doctrine", "lets a pretrained LLM extend that doctrine", 1),
 ("Evolving Takagi-Sugeno models create and tune rules online", "Evolving Takagi-Sugeno (TS) models create and tune rules online", 1),
 ("zero-order Takagi-Sugeno inference turns rule activations", "zero-order TS inference turns rule activations", 1),
 ("Wildfire response is carried out by a system of systems", "Wildfire response is carried out by a system of systems (SoS)", 1),
 ("presents DisasterAware as a system of systems with five kinds of participant", "presents DisasterAware as an SoS with five kinds of participant", 1),
 ("DisasterAware as a system of systems. The simulation core owns", "DisasterAware as an SoS. The simulation core owns", 1),
 ("Predefined alternatives in centralized architectures: the system ranks", "Predefined alternatives in centralized architectures. The system ranks", 1),
]
done = []
for old, new, maxn in EDITS:
    n = 0
    for p in list(root.iter(w("p"))):
        for r in list(p.iter(w("r"))):
            if r.getparent().tag == w("del"): continue
            t = r.find(w("t"))
            if t is None or not t.text or old not in t.text: continue
            txt = t.text; a = txt.index(old); b = a + len(old); rpr = r.find(w("rPr"))
            # common prefix/suffix so only the changed words are marked
            i = 0
            while i < len(old) and i < len(new) and old[i] == new[i]: i += 1
            j = 0
            while j < len(old) - i and j < len(new) - i and old[-1 - j] == new[-1 - j]: j += 1
            o_mid, n_mid = old[i:len(old) - j], new[i:len(new) - j]
            parent = r.getparent(); idx = parent.index(r); parent.remove(r)
            seq = [run(txt[:a] + old[:i], rpr)]
            if o_mid: d = E("del", {"id": nid(), "author": AUTHOR, "date": DATE}); dr = run(o_mid, rpr); dr.find(w("t")).tag = w("delText"); d.append(dr); seq.append(d)
            if n_mid: seq.append(E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, run(n_mid, rpr)))
            tail = old[len(old) - j:] + txt[b:]
            if tail: seq.append(run(tail, rpr))
            for k, e in enumerate(seq): parent.insert(idx + k, e)
            n += 1
            if maxn and n >= maxn: break
        if maxn and n >= maxn: break
    done.append((old[:40], n))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
for d in done: print(d)
