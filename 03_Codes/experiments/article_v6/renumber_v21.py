"""Renumber the live CITATION controls by first appearance, drop the sources that are no longer cited from the
bibliography (field result) and from the sources part. usage: python3 renumber_v21.py <dir>"""
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
dropped = sorted(N_OF_TAG[t] for t in set(TAGS.values()) - set(order))
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
bibps = [p for p in body.iter(w("p")) if p.find("w:pPr/w:pStyle", NS) is not None and p.find("w:pPr/w:pStyle", NS).get(w("val")) == "Bibliography"]
assert len(bibps) == 52, len(bibps)
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
def is_field_run(r):
    return r.find(w("fldChar")) is not None or r.find(w("instrText")) is not None
n = len(order)
# the field 'end' run sits in the last paragraph; move it to paragraph n, then drop paragraphs n+1..52 (field result, untracked)
last_runs = [c for c in bibps[-1] if c.tag == w("r")]
tail = [r for r in last_runs if r.find(w("fldChar")) is not None and r.find(w("fldChar")).get(w("fldCharType")) == "end"]
for i, p in enumerate(bibps[:n]):
    runs = [c for c in p if c.tag == w("r")]
    head = [r for r in runs if is_field_run(r) and (r.find(w("fldChar")) is None or r.find(w("fldChar")).get(w("fldCharType")) != "end")]
    for r in runs: p.remove(r)
    for r in head: p.append(r)
    p.append(run(f"[{i + 1}]\t", rpr16))
    for r in bibfmt.entry_runs(RECS, N_OF_TAG[order[i]], rpr16, run): p.append(r)
    if i == n - 1:
        for r in tail: p.append(r)
for p in bibps[n:]: p.getparent().remove(p)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
src = etree.parse(f"{D}/customXml/item1.xml"); root = src.getroot()
for s in root.findall("b:Source", NS):
    tag = s.find("b:Tag", NS).text
    if tag in num: s.find("b:RefOrder", NS).text = str(num[tag])
    else: root.remove(s)
src.write(f"{D}/customXml/item1.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
moved = [(N_OF_TAG[t], num[t]) for t in order if N_OF_TAG[t] != num[t]]
print("citation controls:", len(fields), "| kept:", n, "| dropped (old numbers):", dropped, "| cached values changed:", changed)
print("old->new:", moved)
