"""v9 -> v9_v1: move the references to Word's own bibliography engine (IEEE style), all edits tracked.
  * customXml/item2.xml  : b:Sources with 52 structured sources (tags Turoff04 ...), SelectedStyle IEEE
  * every REF _RefBn citation (and each [a, b, c] group) -> one CITATION field inside a w:sdt (w:citation)
  * the manual reference list -> one BIBLIOGRAPHY field (cached text rendered here, Word refreshes it on update)
  * the G1 sentence cites [10], [15], [35]
usage: python3 bib_v9.py <unzipped docx dir>"""
import re, sys, copy, datetime, uuid, unicodedata
from lxml import etree
import sources_v9

D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [9000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def ins(*kids): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def dele(*kids): return E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def run(text, rpr=None, italic=False):
    r = E("r"); rp = copy.deepcopy(rpr) if rpr is not None else E("rPr")
    if italic: rp.insert(0, E("i"))
    if len(rp): r.append(rp)
    t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
def del_run(r):
    r2 = copy.deepcopy(r)
    for t in r2.iter(w("t")): t.tag = w("delText")
    for t in r2.iter(w("instrText")): t.tag = w("delInstrText")
    return r2
def fld_runs(instr, cached_runs, rpr=None):
    out = []
    for kind in ("begin", "instr", "separate"):
        r = E("r")
        if rpr is not None: r.append(copy.deepcopy(rpr))
        if kind == "instr":
            it = E("instrText"); it.text = f" {instr} "; it.set("{%s}space" % XML, "preserve"); r.append(it)
        else: r.append(E("fldChar", {"fldCharType": kind}))
        out.append(r)
    out += cached_runs
    r = E("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    r.append(E("fldChar", {"fldCharType": "end"})); out.append(r)
    return out
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))

# ------------------------------------------------------------------ sources
RECS = sources_v9.load()
def ascii_(s): return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())
def person(name):
    toks = name.split()
    i = 0
    while i < len(toks) - 1 and re.fullmatch(r"([A-Z]\.-?)+", toks[i]): i += 1
    inits = toks[:i]; last = " ".join(toks[i:])
    return last, (inits[0] if inits else ""), " ".join(inits[1:])
TAGS = {}
used = set()
for n, r in RECS.items():
    last = person(r["authors"][0])[0]; base = ascii_(last.split()[-1]) + r["year"][2:]
    tag = base; k = 2
    while tag in used: tag = f"{base}{k}"; k += 1
    used.add(tag); TAGS[n] = tag

def b(tag, text=None, *kids):
    e = etree.Element("{%s}%s" % (B, tag))
    if text is not None: e.text = text
    for c in kids: e.append(c)
    return e
def namelist(names):
    nl = b("NameList")
    for nm in names:
        last, first, mid = person(nm)
        p = b("Person", None, b("Last", last))
        if first: p.append(b("First", first))
        if mid: p.append(b("Middle", mid))
        nl.append(p)
    return nl
def source_xml(n, r):
    s = b("Source", None, b("Tag", TAGS[n]), b("SourceType", r["type"]), b("Guid", "{%s}" % str(uuid.uuid4()).upper()))
    au = b("Author", None, b("Author", None, namelist(r["authors"])))
    if r.get("editors"): au.append(b("Editor", None, namelist(r["editors"])))
    s.append(au); s.append(b("Title", r["title"]))
    if r.get("journal"): s.append(b("JournalName", r["journal"]))
    if r.get("book"): s.append(b("BookTitle", r["book"]))
    if r.get("conference"): s.append(b("ConferenceName", r["conference"]))
    if r.get("pubtitle"): s.append(b("PublicationTitle", r["pubtitle"]))
    s.append(b("Year", r["year"]))
    for key, tag in (("volume", "Volume"), ("issue", "Issue"), ("pages", "Pages"), ("publisher", "Publisher"), ("city", "City"),
                     ("institution", "Institution"), ("reporttype", "ThesisType")):
        if r.get(key): s.append(b(tag, r[key]))
    s.append(b("RefOrder", str(n)))
    return s
root = etree.Element("{%s}Sources" % B, nsmap={"b": B, None: B})
root.set("SelectedStyle", "\\IEEE2006OfficeOnline.xsl"); root.set("StyleName", "IEEE"); root.set("Version", "2006")
for n in sorted(RECS): root.append(source_xml(n, RECS[n]))
open(f"{D}/customXml/item2.xml", "wb").write(etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True))

# ------------------------------------------------------------------ cached IEEE text for the bibliography field
def fmt_authors(names):
    if len(names) > 6: return names[0] + " et al."
    if len(names) == 1: return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
def entry_runs(n, rpr):
    r = RECS[n]; a = fmt_authors(r["authors"]); t = r["title"]; y = r["year"]; out = []
    def plain(s): out.append(run(s, rpr))
    def ital(s): out.append(run(s, rpr, italic=True))
    if r["type"] == "JournalArticle":
        plain(f'{a}, "{t}," '); ital(r["journal"] or "(journal)")
        tail = ""
        if r.get("volume"): tail += f", vol. {r['volume']}"
        if r.get("issue"): tail += f", no. {r['issue']}"
        if r.get("pages"): tail += (", p. " if "-" not in r["pages"] else ", pp. ") + r["pages"]
        plain(f"{tail}, {y}.")
    elif r["type"] == "Book":
        plain(f"{a}, "); ital(t); plain(f". {r.get('city','')}: {r.get('publisher','')}, {y}.")
    elif r["type"] == "BookSection":
        plain(f'{a}, "{t}," in '); ital(r["book"])
        eds = fmt_authors(r.get("editors", [])); plain(f", {eds}, Eds. {r.get('city','')}: {r.get('publisher','')}, {y}, pp. {r['pages']}.")
    elif r["type"] == "ConferenceProceedings":
        plain(f'{a}, "{t}," in '); ital(r["conference"])
        tail = f", {r['city']}" if r.get("city") else ""
        tail += f", {y}"
        if r.get("pages"): tail += f", pp. {r['pages']}"
        plain(tail + ".")
    elif r["type"] == "Report":
        plain(f'{a}, "{t}," {r.get("reporttype","")}, {r["institution"]}, {r["city"]}, {y}.'.replace(" ,", ","))
    else:
        plain(f'{a}, "{t}," {r.get("pubtitle","")}, {y}.')
    return out

# ------------------------------------------------------------------ document
doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
rpr_noproof = E("rPr", None, E("noProof"))
def cite_sdt(nums, rpr):
    instr = f"CITATION {TAGS[nums[0]]} \\l 1033" + "".join(f" \\m {TAGS[k]}" for k in nums[1:])
    cached = ", ".join(f"[{k}]" for k in nums)
    rp = copy.deepcopy(rpr) if rpr is not None else E("rPr"); rp.append(E("noProof"))
    runs = fld_runs(instr, [run(cached, rp)], rpr)
    sdt = E("sdt", None, E("sdtPr", None, E("id", {"val": str(-(2000000000 - int(nid())))}), E("citation")), E("sdtContent", None, *runs))
    return sdt

def field_items(runs):
    """[(i_begin, i_end, n)] for every REF _RefBn field in the run list"""
    items = []; i = 0
    while i < len(runs):
        fc = runs[i].find(w("fldChar"))
        if fc is not None and fc.get(w("fldCharType")) == "begin":
            j = i + 1; n = None
            while j < len(runs):
                it = runs[j].find(w("instrText"))
                if it is not None and it.text and "REF _RefB" in it.text: n = int(re.search(r"_RefB(\d+)", it.text).group(1))
                fc2 = runs[j].find(w("fldChar"))
                if fc2 is not None and fc2.get(w("fldCharType")) == "end": break
                j += 1
            if n is not None: items.append((i, j, n))
            i = j + 1
        else: i += 1
    return items

def rtext(r):
    t = r.find(w("t")); return t.text if t is not None else None

converted = 0; groups = 0
for p in list(body.iter(w("p"))):
    runs = [r for r in p.iter(w("r"))]
    items = field_items(runs)
    if not items: continue
    # group consecutive fields separated only by ", " runs
    grp = []; cur = [items[0]]
    for a, bb in zip(items, items[1:]):
        between = runs[a[1] + 1:bb[0]]
        if all(rtext(x) is not None and rtext(x).strip() in (",", "") for x in between): cur.append(bb)
        else: grp.append(cur); cur = [bb]
    grp.append(cur)
    special = "cannot propose what the incident needs" in ptext(p)
    for g in reversed(grp):          # right to left so indices stay valid
        i0, i1 = g[0][0], g[-1][1]; nums = sorted(k[2] for k in g)
        if special and nums == [15]: nums = [10, 15, 35]
        first, last = runs[i0], runs[i1]; parent = first.getparent()
        assert parent is last.getparent() and parent.tag == w("p"), ptext(p)[:60]
        rpr = first.find(w("rPr"))
        # strip the literal brackets around a group
        before = runs[i0 - 1] if i0 > 0 else None; after = runs[i1 + 1] if i1 + 1 < len(runs) else None
        if len(g) > 1 or (before is not None and (rtext(before) or "").endswith("[")):
            if before is not None and (rtext(before) or "").endswith("["):
                before.find(w("t")).text = rtext(before)[:-1]; before.find(w("t")).set("{%s}space" % XML, "preserve")
                before.addnext(dele(del_run(run("[", before.find(w("rPr"))))))
            if after is not None and (rtext(after) or "").startswith("]"):
                after.find(w("t")).text = rtext(after)[1:]; after.find(w("t")).set("{%s}space" % XML, "preserve")
                after.addprevious(dele(del_run(run("]", after.find(w("rPr"))))))
        d = dele(); parent.insert(parent.index(first), d)
        for r in runs[i0:i1 + 1]: parent.remove(r); d.append(del_run(r))
        d.addnext(ins(cite_sdt(nums, rpr)))
        converted += len(g); groups += 1

# ------------------------------------------------------------------ reference list -> BIBLIOGRAPHY field
refps = []
for p in body.iter(w("p")):
    bs = p.find(w("bookmarkStart"))
    if bs is not None and bs.get(w("name"), "").startswith("_RefB"): refps.append(p)
assert len(refps) == 52, len(refps)
for p in refps:                                   # the leftover empty bibliography placeholder control
    for s in p.findall(w("sdt")): p.remove(s)
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
def bib_ppr(inserted=True, old=None):
    ppr = E("pPr", None, E("pStyle", {"val": "Bibliography"}))
    rp = E("rPr")
    if inserted: rp.append(E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}))
    ppr.append(rp)
    if old is not None:
        ch = E("pPrChange", {"id": nid(), "author": AUTHOR, "date": DATE}); o = copy.deepcopy(old)
        for x in o.findall(w("rPr")): o.remove(x)
        ch.append(o); ppr.append(ch)
    return ppr
def mark_deleted(p):
    for r in list(p.iter(w("r"))):
        par = r.getparent(); d = dele(); par.replace(r, d); d.append(del_run(r))
    ppr = p.find(w("pPr")); rp = ppr.find(w("rPr"))
    if rp is None: rp = E("rPr"); ppr.append(rp)
    rp.insert(0, E("del", {"id": nid(), "author": AUTHOR, "date": DATE}))
for p in refps[:-1]: mark_deleted(p)
last = refps[-1]; old_ppr = copy.deepcopy(last.find(w("pPr")))
for r in list(last.iter(w("r"))):
    par = r.getparent(); d = dele(); par.replace(r, d); d.append(del_run(r))
last.replace(last.find(w("pPr")), bib_ppr(inserted=False, old=old_ppr))
# new paragraphs for entries 1..51 before the last one; entry 52 + field end in the last paragraph
for n in range(1, 52):
    p = E("p", None, bib_ppr())
    body_runs = [run(f"[{n}]\t", rpr16)] + entry_runs(n, rpr16)
    if n == 1: body_runs = fld_runs("BIBLIOGRAPHY", [], rpr16)[:3] + body_runs
    p.append(ins(*body_runs)); last.addprevious(p)
endr = E("r", None, copy.deepcopy(rpr16), E("fldChar", {"fldCharType": "end"}))
last.append(ins(run("[52]\t", rpr16), *entry_runs(52, rpr16), endr))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ------------------------------------------------------------------ styles: Bibliography like the template reference text
s = open(f"{D}/word/styles.xml", encoding="utf-8").read()
new_bib = ('<w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>'
           '<w:uiPriority w:val="37"/><w:unhideWhenUsed/><w:pPr><w:tabs><w:tab w:val="left" w:pos="270"/></w:tabs><w:spacing w:after="20"/>'
           '<w:ind w:left="270" w:hanging="270"/><w:jc w:val="both"/></w:pPr><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:style>')
s2, k = re.subn(r'<w:style w:type="paragraph" w:styleId="Bibliography">.*?</w:style>', new_bib, s, flags=re.S)
assert k == 1
open(f"{D}/word/styles.xml", "w", encoding="utf-8").write(s2)

# ------------------------------------------------------------------ settings
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("sources:", len(RECS), "| citation fields converted:", converted, "in", groups, "CITATION fields | tags:", ", ".join(TAGS[k] for k in sorted(TAGS))[:200])
