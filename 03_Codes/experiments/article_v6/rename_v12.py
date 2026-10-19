"""v12_v1 -> v12_v2: DisasterAware -> VODA (defined once in the abstract and once in the introduction), gates renumbered
G1..G8 (G2c->G3, G2d->G4, G3->G5, G4->G6, G5->G7, G5b->G8), figures regenerated. Text edits tracked; math edited in place."""
import re, sys, copy, datetime, shutil
from lxml import etree
from PIL import Image
D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; M = "http://schemas.openxmlformats.org/officeDocument/2006/math"; XML = "http://www.w3.org/XML/1998/namespace"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"; R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"; WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id = [30000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def run(text, rpr):
    r = E("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
def drun(text, rpr):
    r = run(text, rpr); r.find(w("t")).tag = w("delText"); return r
GATES = {"G2c": "G3", "G2d": "G4", "G3": "G5", "G4": "G6", "G5": "G7", "G5b": "G8"}
PAT = re.compile(r"DA is DisasterAware|DisasterAware|\bG(?:2c|2d|3|4|5b|5)\b")
def sub(m):
    s = m.group(0)
    if s == "DA is DisasterAware": return "VODA is the full architecture"
    if s == "DisasterAware": return "VODA"
    return GATES[s]
doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))
# 1. definitions: abstract and introduction (plain in-place edits of the defining sentence, tracked)
DEFS = [("This article presents an architecture with an open decision space.", "This article presents the Verified Open Decision Architecture (VODA), an architecture with an open decision space."),
        ("DisasterAware answers these limits with an architecture whose central contribution is an open decision space.", "The Verified Open Decision Architecture (VODA) answers these limits. Its central contribution is an open decision space.")]
n_def = 0
for old, new in DEFS:
    for t in body.iter(w("t")):
        if t.text and old in t.text:
            r = t.getparent()
            if r.getparent().tag == w("del"): continue
            rpr = r.find(w("rPr")); a, b = t.text.split(old, 1); par = r.getparent(); pos = par.index(r)
            if par.tag == w("ins"): t.text = a + new + b; n_def += 1; break
            par.remove(r); new_els = []
            if a: new_els.append(run(a, rpr))
            new_els.append(E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, drun(old, rpr)))
            new_els.append(E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, run(new, rpr)))
            if b: new_els.append(run(b, rpr))
            for k, el in enumerate(new_els): par.insert(pos + k, el)
            n_def += 1; break
assert n_def == 2, n_def
# 2. tracked token replacement in text runs
n_txt = 0
for t in list(body.iter(w("t"))):
    if not t.text or not PAT.search(t.text): continue
    r = t.getparent(); par = r.getparent()
    if par.tag == w("del"): continue
    if par.tag == w("ins"): t.text = PAT.sub(sub, t.text); n_txt += 1; continue
    rpr = r.find(w("rPr")); pos = par.index(r); par.remove(r); els = []; last = 0
    for m in PAT.finditer(t.text):
        if m.start() > last: els.append(run(t.text[last:m.start()], rpr))
        els.append(E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, drun(m.group(0), rpr)))
        els.append(E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, run(sub(m), rpr)))
        last = m.end(); n_txt += 1
    if last < len(t.text): els.append(run(t.text[last:], rpr))
    for k, el in enumerate(els): par.insert(pos + k, el)
# 3. math text (cannot be tracked)
n_math = 0
GP = re.compile(r"\bG(?:2c|2d|3|4|5b|5)\b")
for t in body.iter("{%s}t" % M):
    if t.text and GP.search(t.text): t.text = GP.sub(lambda m: GATES[m.group(0)], t.text); n_math += 1
# 4. figures
IMAGES = {2: "fig2_pipeline", 3: "fig4_grammar", 4: "fig3_gates", 6: "r1b_outcomes", 7: "r2b_paired", 8: "r3_costs", 9: "r4_products", 12: "r7_attribution"}
rels = etree.parse(f"{D}/word/_rels/document.xml.rels"); rmap = {e.get("Id"): e.get("Target") for e in rels.getroot()}
for k, dr in enumerate(body.iter(w("drawing")), 1):
    if k not in IMAGES: continue
    blip = next(dr.iter("{%s}blip" % A)); target = rmap[blip.get("{%s}embed" % R)]
    src = f"/home/claude/v6/fig/{IMAGES[k]}.png"; shutil.copy(src, f"{D}/word/{target}"); iw, ih = Image.open(src).size
    for ext in list(dr.iter("{%s}extent" % WP)) + list(dr.iter("{%s}ext" % A)):
        cx = int(ext.get("cx")); ext.set("cy", str(int(cx * ih / iw)))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("definitions:", n_def, "| text replacements:", n_txt, "| math runs:", n_math)
