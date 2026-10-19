"""Renumber the locked CITATION controls by first appearance (live text only), rewrite the cached bibliography entries
in the new order and the RefOrder values in the sources part. usage: python3 renumber_v17.py <dir>"""
import re, sys, copy
from lxml import etree
import sources_v9, bibfmt
D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"; B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"
NS = {"w": W, "b": B}
def w(t): return "{%s}%s" % (W, t)
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def run(text, rpr=None, italic=False):
    r = E("r"); rp = copy.deepcopy(rpr) if rpr is not None else E("rPr")
    if italic: rp.insert(0, E("i"))
    if len(rp): r.append(rp)
    t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
RECS = sources_v9.load(); TAGS = bibfmt.make_tags(RECS); N_OF_TAG = {v: k for k, v in TAGS.items()}
doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
order = []; fields = []
for it in body.iter(w("instrText")):               # delInstrText (deleted text) is skipped automatically
    if "CITATION" not in (it.text or ""): continue
    tags = [a or b for a, b in re.findall(r"CITATION (\S+)|\\m (\S+)", it.text)]
    for t in tags:
        if t not in order: order.append(t)
    fields.append((it, tags))
assert set(order) == set(TAGS.values()), set(TAGS.values()) ^ set(order)
num = {t: i + 1 for i, t in enumerate(order)}
changed = 0
for it, tags in fields:
    tags = sorted(tags, key=lambda t: num[t]); new_instr = f" CITATION {tags[0]} \\l 1033" + "".join(f" \\m {t}" for t in tags[1:]) + " "
    it.text = new_instr
    for sib in it.getparent().itersiblings():
        t = sib.find(w("t"))
        if t is not None:
            new = "[" + ", ".join(str(num[x]) for x in tags) + "]"
            if t.text != new: changed += 1
            t.text = new; break
# bibliography paragraphs: keep the field runs, rewrite the entry text
bibps = [p for p in body.iter(w("p")) if p.find("w:pPr/w:pStyle", NS) is not None and p.find("w:pPr/w:pStyle", NS).get(w("val")) == "Bibliography"]
assert len(bibps) == 52, len(bibps)
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
def is_field_run(r):
    return r.find(w("fldChar")) is not None or r.find(w("instrText")) is not None
for i, p in enumerate(bibps):
    runs = [c for c in p if c.tag == w("r")]
    head = [r for r in runs if is_field_run(r) and (r.find(w("fldChar")) is None or r.find(w("fldChar")).get(w("fldCharType")) != "end")]
    tail = [r for r in runs if r.find(w("fldChar")) is not None and r.find(w("fldChar")).get(w("fldCharType")) == "end"]
    for r in runs: p.remove(r)
    for r in head: p.append(r)
    p.append(run(f"[{i + 1}]\t", rpr16))
    for r in bibfmt.entry_runs(RECS, N_OF_TAG[order[i]], rpr16, run): p.append(r)
    for r in tail: p.append(r)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
src = etree.parse(f"{D}/customXml/item2.xml")
for s in src.getroot().findall("b:Source", NS):
    s.find("b:RefOrder", NS).text = str(num[s.find("b:Tag", NS).text])
src.write(f"{D}/customXml/item2.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
moved = [(N_OF_TAG[t], num[t]) for t in order if N_OF_TAG[t] != num[t]]
print("citation controls:", len(fields), "| cached values changed:", changed, "| references renumbered (old->new):", moved)
