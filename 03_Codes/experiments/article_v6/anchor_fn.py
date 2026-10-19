"""Anchor footnote 1 the way the IEEE template does: an empty 'Text' paragraph right AFTER the continuous section break
(inside the two-column section), holding the footnote reference with a custom blank mark. Anchoring it before the
section break makes Word turn the continuous break into a page break (blank first page)."""
import sys, datetime, copy
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "{http://www.w3.org/XML/1998/namespace}"
def w(t): return "{%s}%s" % (W, t)
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag)); [e.set(w(k), v) for k, v in (attrib or {}).items()]; [e.append(c) for c in kids]; return e
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot(); body = root.find(w("body"))
if root.find(".//" + w("footnoteReference")) is not None: print("anchor already present"); sys.exit()
secp = next(p for p in body.findall(w("p")) if p.find("w:pPr/w:sectPr", {"w": W}) is not None)   # the continuous break paragraph
DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
rpr = E("rPr", None, E("rFonts", {"ascii": "Times", "hAnsi": "Times"}), E("sz", {"val": "18"}), E("szCs", {"val": "18"}))
p = E("p", None, E("pPr", None, E("pStyle", {"val": "Text"}), E("ind", {"firstLine": "0"}), E("rPr", None, E("ins", {"id": "14000", "author": "Claude", "date": DATE}), *[copy.deepcopy(c) for c in rpr])))
r = E("r", None, copy.deepcopy(rpr), E("footnoteReference", {"customMarkFollows": "1", "id": "1"}), E("sym", {"font": "Symbol", "char": "F020"}))
p.append(E("ins", {"id": "14001", "author": "Claude", "date": DATE}, r))
secp.addnext(p)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True); print("footnote anchored after the section break (template position)")
