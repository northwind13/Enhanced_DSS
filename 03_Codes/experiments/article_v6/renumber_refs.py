"""Renumber references in order of first citation (IEEE). Works on an unpacked docx whose reference paragraphs carry _RefB<n> bookmarks."""
import re, sys
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def w(t): return "{%s}%s" % (W, t)
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot(); body = root.find(w("body"))
# reference paragraphs (bookmarked) -> exclude their own content from the citation scan
refp = {}
for bs in root.iter(w("bookmarkStart")):
    m = re.match(r"_RefB(\d+)$", bs.get(w("name")))
    if m:
        p = bs.getparent()
        while p.tag != w("p"): p = p.getparent()
        refp[int(m.group(1))] = p
order = []
for it in root.iter(w("instrText")):
    m = re.search(r"REF _RefB(\d+)\b", it.text or "")
    if m:
        n = int(m.group(1))
        if n not in order: order.append(n)
missing = [n for n in refp if n not in order]
order += sorted(missing)
new = {old: i + 1 for i, old in enumerate(order)}
print("first-citation order:", order[:12], "... uncited:", missing)
# rewrite field codes and cached values
for p in root.iter(w("p")):
    runs = list(p.iter(w("r")))
    for k, r in enumerate(runs):
        it = r.find(w("instrText"))
        if it is not None and it.text and "REF _RefB" in it.text:
            old = int(re.search(r"_RefB(\d+)", it.text).group(1)); it.text = it.text.replace(f"_RefB{old} ", f"_RefB{new[old]}X ")
            for r2 in runs[k + 1:k + 4]:
                t = r2.find(w("t"))
                if t is not None and t.text and t.text.startswith("["): t.text = f"[{new[old]}]"; break
for it in root.iter(w("instrText")):
    if it.text and "_RefB" in it.text: it.text = it.text.replace("X ", " ")
# rename bookmarks and reorder paragraphs
for bs in root.iter(w("bookmarkStart")):
    m = re.match(r"_RefB(\d+)$", bs.get(w("name")))
    if m: bs.set(w("name"), f"_RefB{new[int(m.group(1))]}")
first = refp[order[0]] if order[0] in refp else None
parent = refp[1].getparent(); anchor = refp[1].getprevious()
for p in refp.values(): parent.remove(p)
prev = anchor
for old in order:
    p = refp[old]; prev.addnext(p); prev = p
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("renumbered", len(new))
