"""v22_v2 -> v22_v3: remove the six continuous section breaks around the full-width figures (Fig. 1, 10, 12) and put each
figure with its caption into a text frame that spans both columns (the IEEE template mechanism)."""
import sys, re
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
def w(t): return "{%s}%s" % (W, t)
doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
secps = [p for p in body.iter(w("p")) if p.find(w("pPr")) is not None and p.find(w("pPr")).find(w("sectPr")) is not None]
assert len(secps) == 7, len(secps)
kids = list(body)
removed = 0; framed = []
for sp in secps[1:]:                    # keep the title-block break (first)
    i = kids.index(sp)
    # a one-column section ends with: [drawing paragraph][caption][this sectPr paragraph]
    prev2, prev1 = kids[i - 2], kids[i - 1]
    cap = prev1.find(w("pPr")) is not None and prev1.find(w("pPr")).find(w("pStyle")) is not None and prev1.find(w("pPr")).find(w("pStyle")).get(w("val")) == "FigureCaption"
    if cap and prev2.find(".//" + w("drawing")) is not None:
        for p in (prev2, prev1):
            ppr = p.find(w("pPr"))
            fr = etree.Element(w("framePr"))
            for k, v in (("w", "10368"), ("hSpace", "0"), ("vSpace", "180"), ("wrap", "notBeside"), ("vAnchor", "margin"), ("hAnchor", "margin"), ("xAlign", "center"), ("yAlign", "top")):
                fr.set(w(k), v)
            if "Fig. 12" in "".join(t.text or "" for t in prev1.iter(w("t"))): fr.set(w("yAlign"), "bottom")
            ins_at = 1 if ppr.find(w("pStyle")) is not None else 0
            ppr.insert(ins_at, fr)
        framed.append(''.join(t.text or '' for t in prev1.iter(w("t")))[:12])
    body.remove(sp); removed += 1
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("section breaks removed:", removed, "| framed figures:", framed, "| remaining sectPr:", len(list(body.iter(w("sectPr")))))
