"""v8 -> v8_v1: caption fields compatible with Word's Insert Caption (labels 'Fig.' and 'TABLE'), tables and figures kept unsplit, Track Changes on."""
import re, sys
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def w(t): return "{%s}%s" % (W, t)
def E(tag, attrib=None):
    e = etree.Element(w(tag)); [e.set(w(k), v) for k, v in (attrib or {}).items()]; return e
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot(); body = root.find(w("body"))
# 1. SEQ identifiers = caption labels
n = 0
for it in root.iter(w("instrText")):
    if it.text and " SEQ Figure " in it.text: it.text = it.text.replace(" SEQ Figure ", " SEQ Fig. "); n += 1
    elif it.text and " SEQ Table " in it.text: it.text = it.text.replace(" SEQ Table ", " SEQ TABLE "); n += 1
print("SEQ fields relabelled:", n)
# 2. keep figures with their captions, keep tables unsplit
def ppr(p):
    x = p.find(w("pPr"))
    if x is None: x = E("pPr"); p.insert(0, x)
    return x
def set_keepnext(p):
    x = ppr(p)
    if x.find(w("keepNext")) is None:
        # keepNext goes right after pStyle (schema order)
        st = x.find(w("pStyle")); (st.addnext(E("keepNext")) if st is not None else x.insert(0, E("keepNext")))
kn = 0
KEEP = "--keep" in sys.argv
kids = list(body) if KEEP else []
for i, k in enumerate(kids):
    if k.tag == w("p") and k.find(".//" + w("drawing")) is not None:   # figure image paragraph -> stays with its caption
        set_keepnext(k); kn += 1
    if k.tag == w("p"):
        st = k.find("w:pPr/w:pStyle", {"w": W})
        if st is not None and st.get(w("val")) == "TableTitle": set_keepnext(k); kn += 1
    if k.tag == w("tbl"):
        rows = k.findall(w("tr"))
        for j, tr in enumerate(rows):
            trpr = tr.find(w("trPr"))
            if trpr is None: trpr = E("trPr"); tr.insert(tr.index(tr.find(w("tblPrEx"))) + 1 if tr.find(w("tblPrEx")) is not None else 0, trpr)
            if trpr.find(w("cantSplit")) is None: trpr.insert(0, E("cantSplit"))
            if j < len(rows) - 1:
                for p in tr.iter(w("p")): set_keepnext(p); kn += 1
print("keepNext set on", kn, "paragraphs")
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
# 3. settings: custom caption labels + track changes
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
s = re.sub(r"<w:captions>.*?</w:captions>", "", s, flags=re.S)
cap = ('<w:captions><w:caption w:name="Fig." w:pos="below" w:numFmt="decimal"/>'
       '<w:caption w:name="TABLE" w:pos="above" w:numFmt="upperRoman"/>'
       '<w:caption w:name="(" w:pos="below" w:numFmt="decimal"/></w:captions>')
for tag in ("<w:shapeDefaults", "<w:decimalSymbol", "<w:listSeparator", "</w:settings>"):
    if tag in s: s = s.replace(tag, cap + tag, 1); break
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("settings ok")
