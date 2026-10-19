"""Restore footnote 1 (first-page affiliation note) into footnotes.xml from an earlier version; runs marked as tracked insertions."""
import sys, re, zipfile, datetime
from lxml import etree
D, SRC = sys.argv[1], sys.argv[2]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def w(t): return "{%s}%s" % (W, t)
src = etree.fromstring(zipfile.ZipFile(SRC).read("word/footnotes.xml"))
f1 = next(f for f in src.findall(w("footnote")) if f.get(w("id")) == "1")
fn = etree.parse(f"{D}/word/footnotes.xml"); root = fn.getroot()
if any(f.get(w("id")) == "1" for f in root.findall(w("footnote"))): print("footnote 1 already present"); sys.exit()
DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); k = 14100
for p in f1.findall(w("p")):
    for r in list(p.findall(w("r"))):
        ins = etree.Element(w("ins")); ins.set(w("id"), str(k)); ins.set(w("author"), "Claude"); ins.set(w("date"), DATE); k += 1
        r.addprevious(ins); ins.append(r)
root.append(f1)
fn.write(f"{D}/word/footnotes.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("footnote 1 restored:", re.sub(r"\s+", " ", "".join(t.text or "" for t in f1.iter(w("t"))))[:90])
