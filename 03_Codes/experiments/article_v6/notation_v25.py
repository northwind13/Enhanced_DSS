"""v25 -> v25_v1
1. Notation: the cell argument (x,y) is dropped from every field in Sections III and IV (display and inline math), except
   in the state definition (3), the neighborhood sum (8), and the one inline mention that introduces the convention.
   A sentence stating the convention is inserted (tracked) after the explanation of (1).
2. An empty line follows every equation block (table) in Sections III and IV.
3. Figures 1 and 2 pictures are replaced by the relabeled versions; the Fig. 1 paragraph loses its Heading1 style and
   the 'A. Two Cores and the Interface' line gets Heading2 back.
usage: python3 notation_v25.py <unzipped dir>"""
import sys, re, copy, datetime, shutil
from lxml import etree
from PIL import Image
D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"; R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"; WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
def w(t): return "{%s}%s" % (W, t)
def m(t): return "{%s}%s" % (M, t)
AUTHOR = "Claude"; DATE = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [41000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def mark(tag): return E(tag, {"id": nid(), "author": AUTHOR, "date": DATE})
def run(text):
    r = E("r"); t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))
def mtext(el): return "".join(t.text or "" for t in el.iter(m("t")))
def pstyle(p):
    ppr = p.find(w("pPr"))
    if ppr is None: return ""
    ps = ppr.find(w("pStyle")); return ps.get(w("val")) if ps is not None else ""

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
kids = list(body)
def h1(prefix): return next(i for i, k in enumerate(kids) if k.tag == w("p") and pstyle(k) == "Heading1" and ptext(k).startswith(prefix))
i3, i5 = h1("III."), h1("V.")
region = kids[i3:i5]

# ---------------------------------------------------------------- 0. orphan inline math in the Section V heading (left behind when
# tracked paragraph deletions were accepted; OMML cannot be part of a tracked deletion)
h5 = kids[i5]; orphans = [om for om in h5 if om.tag == m("oMath")]
for om in orphans: h5.remove(om)
print("orphan math objects removed from the V heading:", len(orphans))
# ---------------------------------------------------------------- 1. notation
KEEP_ROWS = {"_RefEqN_state", "_RefEqN_psi"}
removed = 0; kept_intro = False
def strip_xy(scope, protect_first_inline=False):
    global removed, kept_intro
    for d_ in list(scope.iter(m("d"))):
        if mtext(d_).replace(" ", "") != "x,y": continue
        om = next(a for a in d_.iterancestors() if a.tag == m("oMath"))
        if protect_first_inline and not kept_intro and mtext(om).replace(" ", "").strip("()") == "x,y":
            kept_intro = True; continue          # "$(x,y)$ a cell of the grid": the convention statement keeps it
        d_.getparent().remove(d_); removed += 1
for el in region:
    if el.tag == w("tbl"):
        for tr in el.findall(w("tr")):
            bm = tr.find(".//" + w("bookmarkStart"))
            if bm is not None and bm.get(w("name")) in KEEP_ROWS: continue
            strip_xy(tr)
    elif el.tag == w("p"):
        strip_xy(el, protect_first_inline=True)
# equation (2): the user's copy lost the '=' between U_DSS,k and pi_DSS; restore it
for tr in (t for el in region if el.tag == w("tbl") for t in el.findall(w("tr"))):
    bm = tr.find(".//" + w("bookmarkStart"))
    if bm is None or bm.get(w("name")) != "_RefEqN_loop": continue
    for ss in list(tr.iter(m("sSub"))):
        if mtext(ss).startswith("π"):
            prev = ss.getprevious()
            if prev is not None and prev.tag == m("r") and mtext(prev).strip() == "": prev = prev.getprevious()
            if prev is not None and prev.tag == m("sSub") and mtext(prev).startswith("U"):
                sp = ss.getprevious()
                if sp.tag == m("r") and mtext(sp).strip() == "": sp.getparent().remove(sp)
                eq = copy.deepcopy(next(ss.iter(m("r")))); eq.find(m("t")).text = "="
                for c in list(eq):
                    if c.tag == m("rPr"): eq.remove(c)
                ss.addprevious(eq); print("'=' restored in (2)")
# the sentence that states the convention, after the paragraph explaining (1)
p1 = next(p for p in region if p.tag == w("p") and ptext(p).startswith("In (1)"))
ins_ = mark("ins"); ins_.append(run(" Every field is defined per cell of the grid, so the cell argument is omitted from here on; a subscript k marks a field that changes in time, and a field without it is static."))
p1.append(ins_)
# "the cell argument (x,y)" wording in that paragraph stays as it is; nothing else to edit

# ---------------------------------------------------------------- 2. empty line after each equation block
added = 0
for el in region:
    if el.tag == w("tbl") and el.find(".//" + m("oMath")) is not None:
        nxt = el.getnext()
        if nxt is not None and nxt.tag == w("p") and not ptext(nxt).strip() and nxt.find(".//" + w("drawing")) is None: continue
        sp = E("p", None, E("pPr", None, E("pStyle", {"val": "Text"}), E("spacing", {"before": "0", "after": "0"}), E("rPr", None, mark("ins"), E("sz", {"val": "12"}), E("szCs", {"val": "12"}))))
        el.addnext(sp); added += 1

# ---------------------------------------------------------------- 3. figures and the two styles
rels = etree.parse(f"{D}/word/_rels/document.xml.rels"); rmap = {e.get("Id"): e.get("Target") for e in rels.getroot()}
def swap(p, src):
    dr = p.find(".//" + w("drawing")); blip = next(dr.iter("{%s}blip" % A)); target = rmap[blip.get("{%s}embed" % R)]
    shutil.copy(src, f"{D}/word/{target}"); iw, ih = Image.open(src).size
    for ext in list(dr.iter("{%s}extent" % WP)) + list(dr.iter("{%s}ext" % A)):
        if ext.get("cx") is None: continue
        cx = int(ext.get("cx")); ext.set("cy", str(int(cx * ih / iw)))
    return target
fig1p = next(p for p in kids[:i3] if p.tag == w("p") and p.find(".//" + w("drawing")) is not None and "VODA system architecture" in ptext(p.getnext() if p.getnext() is not None else E("p")))
fig2p = next(p for p in region if p.tag == w("p") and p.find(".//" + w("drawing")) is not None)
t1 = swap(fig1p, "/home/claude/v23/fig/fig1_simcore.png"); t2 = swap(fig2p, "/home/claude/v23/fig/fig2_decisioncore.png")
ppr = fig1p.find(w("pPr")); ps = ppr.find(w("pStyle")) if ppr is not None else None
if ps is not None: ps.set(w("val"), "Text")
pa = next(p for p in region if ptext(p).strip() == "A. Two Cores and the Interface")
ppr = pa.find(w("pPr"))
if ppr is None: ppr = E("pPr"); pa.insert(0, ppr)
ps = ppr.find(w("pStyle"))
if ps is None: ps = E("pStyle"); ppr.insert(0, ps)
ps.set(w("val"), "Heading2")

doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("cell arguments removed:", removed, "| empty lines after equation blocks:", added, "| images:", t1, t2)
