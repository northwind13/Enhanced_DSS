"""v7_v3 (user's accepted copy) -> v7_v5: dash citation groups -> comma lists (tracked), figure images refreshed (Word palette), Track Changes on."""
import re, sys, copy, datetime, shutil
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "{http://www.w3.org/XML/1998/namespace}"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"; A = "http://schemas.openxmlformats.org/drawingml/2006/main"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id = [13000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag)); [e.set(w(k), v) for k, v in (attrib or {}).items()]; [e.append(c) for c in kids]; return e
def run(text, rpr=None):
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); t = E("t"); t.text = text; t.set(XML + "space", "preserve"); r.append(t); return r
def ins(*k): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *k)
def fld(instr, cached, rpr=None):
    out = []
    for kind in ("begin", None, "separate", "cached", "end"):
        r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None)
        if kind == "cached":
            rp = r.find(w("rPr")) or E("rPr"); (r.append(rp) if rp.getparent() is None else None); rp.append(E("noProof")); t = E("t"); t.text = cached; r.append(t)
        elif kind is None: it = E("instrText"); it.text = f" {instr} "; it.set(XML + "space", "preserve"); r.append(it)
        else: r.append(E("fldChar", {"fldCharType": kind}))
        out.append(r)
    return out
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot()
# 1. "[a–b]" groups -> "[a, a+1, ..., b]"
n = 0
for p in root.iter(w("p")):
    runs = [r for r in p if r.tag == w("r")]
    for i, r in enumerate(runs):
        t = r.find(w("t"))
        if t is None or t.text != "–": continue
        # number before: last cached field result before i ; number after: first cached after i
        prev = next((x.find(w("t")).text for x in reversed(runs[:i]) if x.find(w("t")) is not None and x.find(w("t")).text and x.find(w("t")).text.strip().isdigit()), None)
        nxt = next((x.find(w("t")).text for x in runs[i + 1:] if x.find(w("t")) is not None and x.find(w("t")).text and x.find(w("t")).text.strip().isdigit()), None)
        if prev is None or nxt is None: continue
        a, b = int(prev), int(nxt)
        if b - a < 2: continue
        rpr = r.find(w("rPr")); d = E("del", {"id": nid(), "author": AUTHOR, "date": DATE}); dr = copy.deepcopy(r); dr.find(w("t")).tag = w("delText"); d.append(dr)
        new = ins(run(", ", rpr)); 
        for k in range(a + 1, b): [new.append(x) for x in fld(f"REF _RefB{k} \\r \\t \\h", str(k), rpr)]; new.append(run(", ", rpr))
        r.addprevious(d); r.addprevious(new); p.remove(r); n += 1
print("dash groups expanded:", n)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
# 2. figure images: drawings in document order -> new PNGs for figures 1-4 and 6-12
rels = etree.parse(f"{D}/word/_rels/document.xml.rels"); rmap = {e.get("Id"): e.get("Target") for e in rels.getroot()}
d = etree.parse(f"{D}/word/document.xml").getroot()
blips = [b.get("{%s}embed" % R) for b in d.iter("{%s}blip" % A)]
NEW = {1: "fig1_architecture", 2: "fig2_pipeline", 3: "fig4_grammar", 4: "fig3_gates", 6: "r1b_outcomes", 7: "r2b_paired", 8: "r3_costs", 9: "r4_products", 10: "r5_vocabulary", 11: "r6_actuators", 12: "r7_attribution"}
print("drawings:", len(blips))
for k, rid in enumerate(blips, 1):
    if k in NEW:
        target = rmap[rid]; assert target.lower().endswith(".png"), (k, target)
        shutil.copy(f"/home/claude/v6/fig/{NEW[k]}.png", f"{D}/word/{target}"); print(" replaced", k, target, "<-", NEW[k])
# 3. settings
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
