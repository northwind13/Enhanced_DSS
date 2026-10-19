"""v22 -> v22_v1: shorten the reference list. Journal and publisher names are abbreviated in the sources part
(customXml/item2.xml, what Word renders), authors beyond six collapse to 'et al.', country names and page ranges of
conference papers are dropped. The rendered bibliography table (field result) is rewritten to match.
usage: python3 shorten_refs_v22.py <unzipped dir>"""
import sys, copy, re
from lxml import etree
D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"; NS = {"w": W, "b": B}
def w(t): return "{%s}%s" % (W, t)
def b(t): return "{%s}%s" % (B, t)

# tag -> field updates (None deletes the field). Journal abbreviations follow the IEEE/CASSI list.
EDITS = {
 "Turoff04":   {"JournalName": "J. Inf. Technol. Theory Appl."},
 "Bowman20":   {"JournalName": "Nat. Rev. Earth Environ."},
 "Thompson11": {"JournalName": "J. Environ. Manage."},
 "Rothermel72": {"City": "Res. Pap. INT-115, USDA Forest Service, Ogden, UT"},
 "Finney06":   {"ConferenceName": "Fuels Management: How to Measure Success (RMRS-P-41)", "City": "Portland, OR", "Pages": None},
 "NoonanWright11": {"CORP": "E. K. Noonan-Wright et al.", "JournalName": "J. Combust."},
 "Rapp20":     {"JournalName": "Int. J. Wildland Fire"},
 "Rapp26":     {"JournalName": "Fire Ecol."},
 "Arnott05":   {"JournalName": "J. Inf. Technol."},
 "Arbel82":    {"JournalName": "J. Oper. Res. Soc."},
 "Hofstede95": {"JournalName": "J. Decis. Syst."},
 "Colorni20":  {"JournalName": "J. Multi-Criteria Decis. Anal."},
 "Brill90":    {"JournalName": "IEEE Trans. Syst., Man, Cybern."},
 "Klein17":    {"City": "Cambridge, MA"},
 "Morgan90":   {"City": "Cambridge, U.K.", "Publisher": "Cambridge Univ. Press"},
 "Shafer76":   {"City": "Princeton, NJ", "Publisher": "Princeton Univ. Press"},
 "Ribeiro16":  {"ConferenceName": "Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining", "City": "San Francisco, CA", "Pages": None},
 "Sullivan09": {"JournalName": "Int. J. Wildland Fire"},
 "Pais212":    {"JournalName": "Front. For. Glob. Change"},
 "Zadeh65":    {"JournalName": "Inf. Control"},
 "Takagi85":   {"JournalName": "IEEE Trans. Syst., Man, Cybern."},
 "Angelov04":  {"JournalName": "IEEE Trans. Syst., Man, Cybern. B"},
 "French19":   {"City": "Cambridge, U.K.", "Publisher": "Cambridge Univ. Press"},
 "Missier15": {"JournalName": "Front. Psychol."},
 "Struss19":   {"ConferenceName": "Proc. Int. Workshop Qualitative Reasoning (QR)"},
 "Ding23":     {"CORP": "Y. Ding et al.", "JournalName": "Auton. Robots"},
 "Gizzi21":    {"ConferenceName": "Proc. IEEE Int. Conf. Develop. Learn. (ICDL)", "Pages": None},
 "Sutton18":   {"City": "Cambridge, MA"},
 "Xie24":      {"SourceType": "JournalArticle", "JournalName": "arXiv:2402.07877"},
 "Simon56":    {"JournalName": "Psychol. Rev."},
 "Chandra25":  {"JournalName": "Cluster Comput."},
 "Cheng26":    {"CORP": "Y. Cheng et al."},
 "Gerolimos25": {"CORP": "N. Gerolimos et al.", "JournalName": "Hum.-Centric Intell. Syst."},
 "Cui26":      {"JournalName": "Reliab. Eng. Syst. Saf."},
}
ORDER = ["Tag", "SourceType", "Guid", "Author", "Title", "JournalName", "ConferenceName", "BookTitle", "Year", "Volume", "Issue", "Pages",
         "Publisher", "City", "Institution", "ThesisType", "RefOrder"]

src = etree.parse(f"{D}/customXml/item2.xml"); root = src.getroot()
tags_seen = set()
for s in root.findall("b:Source", NS):
    tag = s.find("b:Tag", NS).text; tags_seen.add(tag)
    ed = EDITS.get(tag, {})
    for k, v in ed.items():
        if k == "CORP":
            au = s.find("b:Author/b:Author", NS)
            for c in list(au): au.remove(c)
            corp = etree.SubElement(au, b("Corporate")); corp.text = v
            continue
        el = s.find(f"b:{k}", NS)
        if v is None:
            if el is not None: s.remove(el)
            continue
        if el is None:
            el = etree.Element(b(k))
            # insert in schema order
            idx = ORDER.index(k); placed = False
            for c in s:
                nm = etree.QName(c).localname
                if nm in ORDER and ORDER.index(nm) > idx: c.addprevious(el); placed = True; break
            if not placed: s.append(el)
        el.text = v
missing = set(EDITS) - tags_seen; assert not missing, missing
src.write(f"{D}/customXml/item2.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# render like Word's IEEE 2006 style
def authors(s):
    au = s.find("b:Author/b:Author", NS)
    corp = au.find("b:Corporate", NS)
    if corp is not None: return corp.text
    names = []
    for p in au.iter(b("Person")):
        g = lambda k: (p.find(f"b:{k}", NS).text if p.find(f"b:{k}", NS) is not None else "")
        names.append(" ".join(x for x in (g("First"), g("Middle"), g("Last")) if x))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
def f(s, k): e = s.find(f"b:{k}", NS); return (e.text or "") if e is not None else ""
def render(s):
    """list of (text, italic)"""
    a = authors(s); t = f(s, "Title"); y = f(s, "Year"); typ = f(s, "SourceType"); out = []
    if typ == "JournalArticle":
        out.append((f'{a}, "{t}," ', False))
        if f(s, "JournalName"): out.append((f(s, "JournalName") + ", ", True))
        tail = []
        if f(s, "Volume"): tail.append("vol. " + f(s, "Volume"))
        if f(s, "Issue"): tail.append("no. " + f(s, "Issue"))
        if f(s, "Pages"): tail.append(("pp. " if "-" in f(s, "Pages") else "p. ") + f(s, "Pages"))
        tail.append(y); out.append((", ".join(tail) + ". ", False))
    elif typ == "Book":
        out.append((f"{a}, ", False)); out.append((t + ", ", True)); out.append((f'{f(s, "City")}: {f(s, "Publisher")}, {y}. ', False))
    elif typ == "ConferenceProceedings":
        out.append((f'{a}, "{t}," in ', False)); out.append((f(s, "ConferenceName") + ", ", True))
        out.append(((f(s, "City") + ", " if f(s, "City") else "") + y + ". ", False))
    elif typ == "Report":
        out.append((f'{a}, "{t}," {f(s, "City")}, {y}. ', False))
    else:
        out.append((f"{a}, ", False)); out.append((t + ", ", True)); out.append((y + ". ", False))
    return out
by_order = {int(f(s, "RefOrder")): s for s in root.findall("b:Source", NS)}

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
tbl = None
for t in body.iter(w("tbl")):
    if "Bibliography" in etree.tostring(t).decode(): tbl = t
rows = tbl.findall("w:tr", NS); assert len(rows) == len(by_order), (len(rows), len(by_order))
def run(text, italic):
    r = etree.Element(w("r")); rp = etree.SubElement(r, w("rPr"))
    if italic: etree.SubElement(rp, w("i")); etree.SubElement(rp, w("iCs"))
    etree.SubElement(rp, w("noProof")); t = etree.SubElement(r, w("t")); t.text = text; t.set(f"{{{XML}}}space", "preserve"); return r
total_before = total_after = 0
for i, tr in enumerate(rows, 1):
    cells = tr.findall("w:tc", NS); p = cells[1].find("w:p", NS)
    old = "".join(x.text or "" for x in p.iter(w("t"))); total_before += len(old)
    for r in p.findall("w:r", NS): p.remove(r)
    parts = render(by_order[i]); total_after += sum(len(x) for x, _ in parts)
    for text, it in parts: p.append(run(text, it))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print(f"entries: {len(rows)} | characters {total_before} -> {total_after} ({100 * (1 - total_after / total_before):.0f}% shorter)")
