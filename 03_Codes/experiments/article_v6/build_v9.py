"""v9 -> v9_v3 in one pass.
 1. Word bibliography engine: 52 structured sources (IEEE style) in customXml, every REF _RefBn citation group becomes one
    CITATION field (locked, cached as [7, 8] in one bracket) so the sources appear under Manage Sources / Insert Citation.
    The field swap is not tracked because the visible text does not change; everything else is tracked.
 2. Introduction rewritten with a word level diff (unchanged words stay, only the changed words are marked),
    two new paragraphs, the scope paragraph merged into the contribution paragraph, Table I folded into prose.
 3. Citations renumbered by first appearance; table captions renumbered in Roman numerals.
 4. The manual reference list becomes a BIBLIOGRAPHY field (tracked), Bibliography style matched to the template.
usage: python3 build_v9.py <unzipped v9 dir>"""
import re, sys, copy, datetime, uuid, difflib
from lxml import etree
import sources_v9, bibfmt

D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"; NS = {"w": W, "b": B}
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
def del_el(el):
    """copy of a run / sdt with its text and field codes marked deleted"""
    e2 = copy.deepcopy(el)
    for t in e2.iter(w("t")): t.tag = w("delText")
    for t in e2.iter(w("instrText")): t.tag = w("delInstrText")
    return e2
def fld_runs(instr, cached_runs, rpr=None, lock=False):
    out = []
    for kind in ("begin", "instr", "separate"):
        r = E("r")
        if rpr is not None: r.append(copy.deepcopy(rpr))
        if kind == "instr":
            it = E("instrText"); it.text = f" {instr} "; it.set("{%s}space" % XML, "preserve"); r.append(it)
        else:
            fc = E("fldChar", {"fldCharType": kind})
            if kind == "begin" and lock: fc.set(w("fldLock"), "1")
            r.append(fc)
        out.append(r)
    out += cached_runs
    r = E("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    r.append(E("fldChar", {"fldCharType": "end"})); out.append(r)
    return out
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))
def rtext(r):
    t = r.find(w("t")); return t.text if t is not None else None

RECS = sources_v9.load(); TAGS = bibfmt.make_tags(RECS); N_OF_TAG = {v: k for k, v in TAGS.items()}
def cite_sdt(tags, rpr):
    instr = f"CITATION {tags[0]} \\l 1033" + "".join(f" \\m {t}" for t in tags[1:])
    rp = copy.deepcopy(rpr) if rpr is not None else E("rPr"); rp.append(E("noProof"))
    runs = fld_runs(instr, [run("[" + ", ".join(str(N_OF_TAG[t]) for t in tags) + "]", rp)], rpr, lock=True)
    return E("sdt", None, E("sdtPr", None, E("id", {"val": str(-(2000000000 - int(nid())))}), E("citation")), E("sdtContent", None, *runs))
def sdt_tags(sdt):
    it = next(x for x in sdt.iter(w("instrText")) if "CITATION" in (x.text or ""))
    return [a or b for a, b in re.findall(r"CITATION (\S+)|\\m (\S+)", it.text)]

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))

# ============================================================ 1. REF _RefBn groups -> CITATION controls (untracked)
def field_items(runs):
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
converted = 0
for p in list(body.iter(w("p"))):
    runs = list(p.iter(w("r"))); items = field_items(runs)
    if not items: continue
    grp = []; cur = [items[0]]
    for a, bb in zip(items, items[1:]):
        between = runs[a[1] + 1:bb[0]]
        if all(rtext(x) is not None and rtext(x).strip() in (",", "") for x in between): cur.append(bb)
        else: grp.append(cur); cur = [bb]
    grp.append(cur)
    for g in reversed(grp):
        i0, i1 = g[0][0], g[-1][1]; nums = sorted(k[2] for k in g)
        first, last = runs[i0], runs[i1]; parent = first.getparent(); rpr = first.find(w("rPr"))
        before = runs[i0 - 1] if i0 > 0 else None; after = runs[i1 + 1] if i1 + 1 < len(runs) else None
        if before is not None and (rtext(before) or "").endswith("["):
            before.find(w("t")).text = rtext(before)[:-1]; before.find(w("t")).set("{%s}space" % XML, "preserve")
        if after is not None and (rtext(after) or "").startswith("]"):
            after.find(w("t")).text = rtext(after)[1:]; after.find(w("t")).set("{%s}space" % XML, "preserve")
        pos = parent.index(first)
        for r in runs[i0:i1 + 1]: parent.remove(r)
        parent.insert(pos, cite_sdt([TAGS[n] for n in nums], rpr)); converted += 1

# ============================================================ 2. Introduction, word level tracked diff
NEW = {
"P1": """ILDFIRE response is carried out by a system of systems {Turoff04}. Satellites and weather services deliver observations, a hazard model turns them into an expected fire, regional teams decide where to commit crews, aircraft, and evacuation orders, and an operator supervises the whole under a clock set by the fire {Bowman20;Thompson11}. Fire seasons are becoming longer and more severe {Bowman20}, and the decisions that matter most are taken in the first hours, when observations are sparse, the forecast is uncertain, and a wrong commitment is most costly {Thompson11;Calkin21}. A recommendation that arrives after the fire has moved is of no use, so the support has to keep pace with the incident.""",
"P2": """Decision support systems (DSSs) exist for parts of this task {Power02;Vasquez21}. Spread simulators forecast fire behaviour from fuel, terrain, and weather {Rothermel72;Finney06}. Allocation tools assign crews and aircraft against a forecast {Pais21;Orphanoudakis25}. Integrated platforms such as the Wildland Fire Decision Support System (WFDSS) bring these parts together for use in the field {NoonanWright11;Fillmore23;Calkin21}. Field studies of these tools report their adoption more often than a measured improvement of the decisions taken with them {Rapp20;Rapp26}, and a critical review of DSS research finds the same distance between what is built and what is shown to help {Arnott05}.""",
"P3": """Three structural limits repeat across these systems. First, the alternatives are predefined. A rule base, an optimisation model, or a menu of tactics fixes the set of orders before the incident, so the system can rank the alternatives it is given but cannot propose what the incident needs {NoonanWright11;Malczewski99;Levin14}. Decision analysis has long identified the generation of alternatives as the weakest step of the process {Arbel82;Colorni18;Hofstede95}, and the methods proposed for it either enumerate variations of a fixed model {Brill90;Fazlollahi01} or depend on the recall of the operator under time pressure {Klein17}. Second, uncertainty is representable only through prior distributions or belief intervals that real-time sensing cannot supply {Morgan90;Shafer76}, and the reasoning behind a recommendation is exposed, when it is exposed at all, through post-hoc attribution {Ribeiro16}. Third, integrated platforms embed the forecast and the decision logic in one tool {NoonanWright11;Calkin21}, so neither is validated on its own and the benefit of a decision mechanism cannot be separated from the quality of the forecast behind it. Taken together, these limits mean that the decision space of current systems is closed. It is fixed before the incident and cannot be verified or extended during it.""",
"P4": """DisasterAware answers these limits with an architecture whose central contribution is an open decision space. The set of orders the system may issue is not fixed in advance but grows during the incident, under verification by the hazard model. The architecture separates a simulation core, which owns the hazard state and is validated against satellite records, from a decision core that reads the state through confidence-rated observations only. The decision core has two layers. The closed layer is a self-tuning fuzzy doctrine in which bounded observation features aggregate into decision concepts, a sparse rule base maps the concepts to intervention families, and an evolving mechanism tunes consequents and term boundaries while the vocabulary stays fixed. The open layer sits on top of it. When the standing decision falls short, a pretrained large language model (LLM) proposes a new rule, an intermediate concept, a blend of existing interventions, or an intervention composed from physical effects the simulator can execute. Nothing enters the doctrine unless it passes a chain of algebraic gates and a paired forecast test on independently seeded futures.""",
"P5": """This article asks what the open layer is worth to a commander and answers with paired measurement. The same fires are replayed with no decision support, with the closed layer alone, and with the two layers together, so the benefit of opening the decision space is read from the difference it makes on identical worlds. Three contributions follow. First, the open decision space is defined as an architectural object, the set of orders a system may issue together with the rules under which that set may be extended, and the invariants this object preserves are stated and proved for the gate chain of Section IV. Second, the extension mechanism is specified to the level of its grammar and geometry. A proposal is a tuple over a closed basis of verified effects, sectors, and distance bands, interpreted by the simulator through a fixed operator algebra, so the space is closed at effects and open at interventions. Third, the architecture is measured end to end on an independently validated hazard model, over paired worlds drawn from several scenario families, with every proposal, gate verdict, and admitted object logged, so that the question of what was learned can be answered object by object rather than only in aggregate. The contribution is architectural rather than algorithmic. The concept-driven inference and the generative proposer are replaceable components, and the invariants are properties of the gates. Section II reviews related work. Section III presents the architecture and Section IV the decision core. Section V reports the results, Section VI discusses them, and Section VII concludes.""",
}
def find_p(prefix):
    for p in body.iter(w("p")):
        if ptext(p).startswith(prefix): return p
    raise KeyError(prefix)
WORD = re.compile(r"\s+|[^\s]+")
def old_tokens(p):
    """[(key, kind, payload, rpr)] kind: 'w' word/space text, 'c' citation sdt, 'f' other field runs (atomic)"""
    toks = []; kids = [c for c in p if c.tag != w("pPr")]; i = 0
    while i < len(kids):
        c = kids[i]
        if c.tag == w("r"):
            fc = c.find(w("fldChar"))
            if fc is not None and fc.get(w("fldCharType")) == "begin":      # non-citation field: take runs up to end
                j = i
                while not (kids[j].tag == w("r") and kids[j].find(w("fldChar")) is not None and kids[j].find(w("fldChar")).get(w("fldCharType")) == "end"): j += 1
                toks.append((("f", id(c)), "f", kids[i:j + 1], None)); i = j + 1; continue
            t = c.find(w("t"))
            if t is not None and t.text:
                txt = t.text; rp = c.find(w("rPr")); key = etree.tostring(rp) if rp is not None else b""
                while i + 1 < len(kids) and kids[i + 1].tag == w("r") and kids[i + 1].find(w("t")) is not None and kids[i + 1].find(w("fldChar")) is None \
                        and (etree.tostring(kids[i + 1].find(w("rPr"))) if kids[i + 1].find(w("rPr")) is not None else b"") == key:
                    i += 1; txt += kids[i].find(w("t")).text or ""
                for m in WORD.findall(txt): toks.append((m, "w", m, rp))
            i += 1
        elif c.tag == w("sdt"): toks.append((("c", tuple(sdt_tags(c))), "c", c, None)); i += 1
        else: i += 1                                                           # proofErr, bookmarks: dropped
    return toks
def new_tokens(text):
    toks = []
    for seg in re.split(r"(\{[^}]+\})", text):
        if not seg: continue
        if seg.startswith("{"): tags = tuple(seg[1:-1].split(";")); toks.append((("c", tags), "c", tags, None))
        else:
            for m in WORD.findall(seg): toks.append((m, "w", m, None))
    return toks
def emit(toks, rpr_default, deleted=False, inserted=False):
    """tokens -> list of elements; consecutive words merge into one run"""
    out = []; buf = ""; brpr = None
    def flush():
        nonlocal buf, brpr
        if buf:
            r = run(buf, brpr if brpr is not None else rpr_default)
            out.append(del_el(r) if deleted else r); buf = ""
    for key, kind, payload, rpr in toks:
        if kind == "w":
            if buf and (rpr is not brpr): flush()
            buf += payload; brpr = rpr
        else:
            flush()
            if kind == "c":
                el = payload if isinstance(payload, etree._Element) else cite_sdt(list(payload), rpr_default)
                out.append(del_el(el) if deleted else el)
            else:
                for r in payload: out.append(del_el(r) if deleted else r)
    flush(); return out
def diff_par(p, text):
    old = old_tokens(p); new = new_tokens(text)
    rpr_default = next((t[3] for t in old if t[1] == "w" and t[3] is not None), None)
    for c in [c for c in p if c.tag != w("pPr")]: p.remove(c)
    sm = difflib.SequenceMatcher(a=[t[0] for t in old], b=[t[0] for t in new], autojunk=False)
    ops = sm.get_opcodes(); merged = []
    for op in ops:
        if op[0] == "equal" and (op[2] - op[1]) < 7 and merged and merged[-1][0] != "equal" and ops.index(op) < len(ops) - 1:
            op = ("replace", op[1], op[2], op[3], op[4])
        if merged and merged[-1][0] != "equal" and op[0] != "equal":
            merged[-1] = ("replace", merged[-1][1], op[2], merged[-1][3], op[4])
        else: merged.append(op)
    for op, a0, a1, b0, b1 in merged:
        if op == "equal":
            for el in emit(old[a0:a1], rpr_default): p.append(el)
        else:
            if a1 > a0: p.append(dele(*emit(old[a0:a1], rpr_default, deleted=True)))
            if b1 > b0: p.append(ins(*emit(new[b0:b1], rpr_default)))
def mark_par(p, kind):
    ppr = p.find(w("pPr"))
    if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
    rp = ppr.find(w("rPr"))
    if rp is None: rp = E("rPr"); ppr.append(rp)
    rp.insert(0, E(kind, {"id": nid(), "author": AUTHOR, "date": DATE}))
def delete_par(p, mark=True):
    for c in [c for c in p if c.tag != w("pPr")]:
        if c.tag in (w("r"), w("sdt")): d = dele(); p.replace(c, d); d.append(del_el(c))
        elif c.tag in (w("proofErr"),): p.remove(c)
    if mark: mark_par(p, "del")

p1 = find_p("ILDFIRE"); diff_par(p1, NEW["P1"])
pda = find_p("DisasterAware answers these gaps"); base_ppr = copy.deepcopy(pda.find(w("pPr")))
anchor = p1
for key in ("P2", "P3"):
    p = E("p", None, copy.deepcopy(base_ppr)); mark_par(p, "ins"); p.append(ins(*emit(new_tokens(NEW[key]), None))); anchor.addnext(p); anchor = p
diff_par(pda, NEW["P4"])
diff_par(find_p("This article asks what the open layer"), NEW["P5"])
delete_par(find_p("This is a systems contribution"))
cap1 = find_p("TABLE "); cap2 = cap1.getnext(); tbl = cap2.getnext(); spacer = tbl.getnext()
assert cap2.tag == w("p") and tbl.tag == w("tbl") and spacer.tag == w("p") and ptext(spacer) == "", "Table I structure"
for p in (cap1, cap2, spacer): delete_par(p)
for tr in tbl.iter(w("tr")):
    trpr = tr.find(w("trPr"))
    if trpr is None: trpr = E("trPr"); tr.insert(0, trpr)
    trpr.append(E("del", {"id": nid(), "author": AUTHOR, "date": DATE}))
    for p in tr.iter(w("p")): delete_par(p, mark=False)

# ============================================================ 3. renumber citations (live fields only) and table captions
order = []; fields = []
for it in body.iter(w("instrText")):
    if "CITATION" not in (it.text or ""): continue
    tags = [a or b for a, b in re.findall(r"CITATION (\S+)|\\m (\S+)", it.text)]
    for t in tags:
        if t not in order: order.append(t)
    fields.append((it, tags))
assert set(order) == set(TAGS.values()), set(TAGS.values()) - set(order)
num = {t: i + 1 for i, t in enumerate(order)}
for it, tags in fields:
    tags = sorted(tags, key=lambda t: num[t])
    it.text = f" CITATION {tags[0]} \\l 1033" + "".join(f" \\m {t}" for t in tags[1:]) + " "
    for sib in it.getparent().itersiblings():
        t = sib.find(w("t"))
        if t is not None: t.text = "[" + ", ".join(str(num[x]) for x in tags) + "]"; break
ROMAN = ["I", "II", "III", "IV", "V", "VI"]; k = 0; ref_map = {}
for it in body.iter(w("instrText")):
    if "SEQ Table" in (it.text or ""):
        k += 1; it.text = " SEQ Table \\* ROMAN "
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None: t.text = ROMAN[k - 1]; break
        for bm in it.getparent().getparent().iter(w("bookmarkStart")):
            if bm.get(w("name"), "").startswith("_RefTab"): ref_map[bm.get(w("name"))] = ROMAN[k - 1]
for it in body.iter(w("instrText")):
    m = re.search(r"REF (_RefTab\w+)", it.text or "")
    if m and m.group(1) in ref_map:
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None: t.text = ref_map[m.group(1)]; break

# ============================================================ 4. reference list -> BIBLIOGRAPHY field (tracked)
refps = []
for p in body.iter(w("p")):
    bs = p.find(w("bookmarkStart"))
    if bs is not None and bs.get(w("name"), "").startswith("_RefB"): refps.append(p)
assert len(refps) == 52, len(refps)
for p in refps:
    for s in p.findall(w("sdt")): p.remove(s)        # leftover empty bibliography placeholder
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
def bib_ppr(inserted=True, old=None):
    ppr = E("pPr", None, E("pStyle", {"val": "Bibliography"})); rp = E("rPr")
    if inserted: rp.append(E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}))
    ppr.append(rp)
    if old is not None:
        ch = E("pPrChange", {"id": nid(), "author": AUTHOR, "date": DATE}); o = copy.deepcopy(old)
        for x in o.findall(w("rPr")): o.remove(x)
        ch.append(o); ppr.append(ch)
    return ppr
for p in refps[:-1]: delete_par(p)
last = refps[-1]; old_ppr = copy.deepcopy(last.find(w("pPr"))); delete_par(last, mark=False)
last.replace(last.find(w("pPr")), bib_ppr(inserted=False, old=old_ppr))
for i in range(1, 52):
    tag = order[i - 1]; p = E("p", None, bib_ppr())
    body_runs = [run(f"[{i}]\t", rpr16)] + bibfmt.entry_runs(RECS, N_OF_TAG[tag], rpr16, run)
    if i == 1: body_runs = fld_runs("BIBLIOGRAPHY", [], rpr16)[:3] + body_runs
    p.append(ins(*body_runs)); last.addprevious(p)
endr = E("r", None, copy.deepcopy(rpr16), E("fldChar", {"fldCharType": "end"}))
last.append(ins(run("[52]\t", rpr16), *bibfmt.entry_runs(RECS, N_OF_TAG[order[51]], rpr16, run), endr))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ============================================================ 5. sources part, styles, settings
def b(tag, text=None, *kids):
    e = etree.Element("{%s}%s" % (B, tag))
    if text is not None: e.text = text
    for c in kids: e.append(c)
    return e
def namelist(names):
    nl = b("NameList")
    for nm in names:
        last_, first, mid = bibfmt.person(nm); pe = b("Person", None, b("Last", last_))
        if first: pe.append(b("First", first))
        if mid: pe.append(b("Middle", mid))
        nl.append(pe)
    return nl
root = etree.Element("{%s}Sources" % B, nsmap={"b": B, None: B})
root.set("SelectedStyle", "\\IEEE2006OfficeOnline.xsl"); root.set("StyleName", "IEEE"); root.set("Version", "2006")
for n in sorted(RECS):
    r = RECS[n]; s = b("Source", None, b("Tag", TAGS[n]), b("SourceType", r["type"]), b("Guid", "{%s}" % str(uuid.uuid4()).upper()))
    au = b("Author", None, b("Author", None, namelist(r["authors"])))
    if r.get("editors"): au.append(b("Editor", None, namelist(r["editors"])))
    s.append(au); s.append(b("Title", r["title"]))
    for key, tag in (("journal", "JournalName"), ("book", "BookTitle"), ("conference", "ConferenceName"), ("pubtitle", "PublicationTitle")):
        if r.get(key): s.append(b(tag, r[key]))
    s.append(b("Year", r["year"]))
    for key, tag in (("volume", "Volume"), ("issue", "Issue"), ("pages", "Pages"), ("publisher", "Publisher"), ("city", "City"), ("institution", "Institution"), ("reporttype", "ThesisType")):
        if r.get(key): s.append(b(tag, r[key]))
    s.append(b("RefOrder", str(num[TAGS[n]]))); root.append(s)
open(f"{D}/customXml/item2.xml", "wb").write(etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True))
s = open(f"{D}/word/styles.xml", encoding="utf-8").read()
new_bib = ('<w:style w:type="paragraph" w:styleId="Bibliography"><w:name w:val="Bibliography"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>'
           '<w:uiPriority w:val="37"/><w:unhideWhenUsed/><w:pPr><w:tabs><w:tab w:val="left" w:pos="270"/></w:tabs><w:spacing w:after="20"/>'
           '<w:ind w:left="270" w:hanging="270"/><w:jc w:val="both"/></w:pPr><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr></w:style>')
s, kk = re.subn(r'<w:style w:type="paragraph" w:styleId="Bibliography">.*?</w:style>', new_bib, s, flags=re.S); assert kk == 1
open(f"{D}/word/styles.xml", "w", encoding="utf-8").write(s)
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
if "<w:captions>" not in s:
    s = s.replace("<w:shapeDefaults>", '<w:captions><w:caption w:name="Fig." w:pos="below" w:numFmt="decimal"/><w:caption w:name="Table" w:pos="above" w:numFmt="upperRoman"/></w:captions><w:shapeDefaults>', 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("citation controls:", converted, "| live fields:", len(fields), "| renumbered:", sum(1 for t in order if num[t] != N_OF_TAG[t]), "| tables:", k, ref_map)
