"""v24 repairs before the Section III/IV rewrite:
- the Section III heading paragraph carries the floating Fig. 1 group (picture + caption text box): the group moves to its
  own paragraph after the heading, its picture is replaced by the new Fig. 1 and its geometry follows the new aspect ratio;
- the duplicate standalone 'Fig. 1.' caption paragraph is deleted (tracked);
- the Section V heading lost its first characters to a stray paste ('the simulation core ... VODA isV. Results'): repaired (tracked)."""
import sys, copy, re, datetime, shutil
from lxml import etree
from PIL import Image
D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"; MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [29000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def mark(tag): return E(tag, {"id": nid(), "author": AUTHOR, "date": DATE})
def run(text):
    r = E("r"); t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
def del_el(el):
    e2 = copy.deepcopy(el)
    for t in e2.iter(w("t")): t.tag = w("delText")
    return e2
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
# --- 1. heading III with the floating group
h3 = next(p for p in body.iter(w("p")) if "III. System Architecture" in ptext(p) and p.find(".//{%s}AlternateContent" % MC) is not None)
grp_run = next(r for r in h3.findall(w("r")) if r.find("{%s}AlternateContent" % MC) is not None)
h3.remove(grp_run)
for r in list(h3.findall(w("r"))):
    if ptext(r) != "III. System Architecture": h3.remove(r)
figp = E("p", None, E("pPr", None, E("pStyle", {"val": "Text"}), E("ind", {"firstLine": "0"}))); figp.append(grp_run); h3.addnext(figp)
# picture and geometry
src = "/home/claude/v23/fig/fig1_simcore.png"; iw, ih = Image.open(src).size
cx = 3017520; cy_pic = int(cx * ih / iw)
for xf in grp_run.iter("{%s}xfrm" % A):
    ext = xf.find("{%s}ext" % A); off = xf.find("{%s}off" % A); chext = xf.find("{%s}chExt" % A)
    if chext is not None:                          # group
        cy_grp = cy_pic - 57150 + 193040; ext.set("cy", str(cy_grp)); chext.set("cy", str(cy_grp))
    elif ext.get("cy") == "3354070":               # picture
        ext.set("cy", str(cy_pic))
    elif ext.get("cy") == "193040":                # caption text box
        off.set("y", str(cy_pic - 57150))
for e in grp_run.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"): e.set("cy", str(cy_pic - 57150 + 193040))
rels = etree.parse(f"{D}/word/_rels/document.xml.rels"); rmap = {e.get("Id"): e.get("Target") for e in rels.getroot()}
blip = next(grp_run.iter("{%s}blip" % A)); target = rmap[blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")]
shutil.copy(src, f"{D}/word/{target}")
# --- 3. heading V
h5 = next(p for p in body.iter(w("p")) if ptext(p).endswith("V. Results") and "VODA" in ptext(p))
runs = h5.findall(w("r"));
for pe in h5.findall(w("proofErr")): h5.remove(pe)
for r in runs: h5.remove(r)
d = mark("del")
for r in runs: d.append(del_el(r))
h5.append(d); i = mark("ins"); i.append(run("V. Results")); h5.append(i)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("prefix done: group moved, picture", target, "cy", cy_pic, "| heading V repaired")
