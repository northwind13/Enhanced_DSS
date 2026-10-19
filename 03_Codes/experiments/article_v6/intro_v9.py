"""v9_v1 -> v9_v2: Introduction rewritten (stronger problem statement, no design numbers, Table I folded into prose,
scope sentence merged into the contribution paragraph), all citations renumbered by first appearance, tracked changes.
usage: python3 intro_v9.py <unzipped v9_v1 dir>"""
import re, sys, copy, datetime
from lxml import etree
import sources_v9

D = sys.argv[1]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"; NS = {"w": W, "b": B}
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [12000]
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
def cite_sdt(tags, rpr):
    instr = f"CITATION {tags[0]} \\l 1033" + "".join(f" \\m {t}" for t in tags[1:])
    rp = copy.deepcopy(rpr) if rpr is not None else E("rPr"); rp.append(E("noProof"))
    runs = fld_runs(instr, [run("[?]", rp)], rpr)
    return E("sdt", None, E("sdtPr", None, E("id", {"val": str(-(1990000000 - int(nid())))}), E("citation")), E("sdtContent", None, *runs))

# ---------------------------------------------------------------- the new Introduction ({Tag;Tag} = one citation field)
NEW = {
"ILDFIRE response is carried out": """ILDFIRE response is carried out by a system of systems {Turoff04}. Satellites and weather services deliver observations, a hazard model turns them into an expected fire, regional teams decide where to commit crews, aircraft, and evacuation orders, and an operator supervises the whole under a clock set by the fire {Bowman20;Thompson11}. Fire seasons are becoming longer and more severe {Bowman20}, and the decisions that matter most are taken in the first hours, when observations are sparse, the forecast is uncertain, and a wrong commitment is most costly {Thompson11;Calkin21}. A recommendation that arrives after the fire has moved is of no use, so the support has to keep pace with the incident.""",

"__P2": """Decision support systems (DSSs) exist for parts of this task {Power02;Vasquez21}. Spread simulators forecast fire behaviour from fuel, terrain, and weather {Rothermel72;Finney06}. Allocation tools assign crews and aircraft against a forecast {Pais21;Orphanoudakis25}. Integrated platforms such as the Wildland Fire Decision Support System (WFDSS) bring these parts together for use in the field {NoonanWright11;Fillmore23;Calkin21}. Field studies of these tools report their adoption more often than a measured improvement of the decisions taken with them {Rapp20;Rapp26}, and a critical review of DSS research finds the same distance between what is built and what is shown to help {Arnott05}.""",

"__P3": """Three structural limits repeat across these systems. First, the alternatives are predefined. A rule base, an optimisation model, or a menu of tactics fixes the set of orders before the incident, so the system can rank the alternatives it is given but cannot propose what the incident needs {NoonanWright11;Malczewski99;Levin14}. Decision analysis has long identified the generation of alternatives as the weakest step of the process {Arbel82;Colorni18;Hofstede95}, and the methods proposed for it either enumerate variations of a fixed model {Brill90;Fazlollahi01} or depend on the recall of the operator under time pressure {Klein17}. Second, uncertainty is representable only through prior distributions or belief intervals that real-time sensing cannot supply {Morgan90;Shafer76}, and the reasoning behind a recommendation is exposed, when it is exposed at all, through post-hoc attribution {Ribeiro16}. Third, integrated platforms embed the forecast and the decision logic in one tool {NoonanWright11;Calkin21}, so neither is validated on its own and the benefit of a decision mechanism cannot be separated from the quality of the forecast behind it. Taken together, these limits mean that the decision space of current systems is closed. It is fixed before the incident and cannot be verified or extended during it.""",

"DisasterAware answers these gaps": """DisasterAware answers these limits with an architecture whose central contribution is an open decision space. The set of orders the system may issue is not fixed in advance but grows during the incident, under verification by the hazard model. The architecture separates a simulation core, which owns the hazard state and is validated against satellite records, from a decision core that reads the state through confidence-rated observations only. The decision core has two layers. The closed layer is a self-tuning fuzzy doctrine in which bounded observation features aggregate into decision concepts, a sparse rule base maps the concepts to intervention families, and an evolving mechanism tunes consequents and term boundaries while the vocabulary stays fixed. The open layer sits on top of it. When the standing decision falls short, a pretrained large language model (LLM) proposes a new rule, an intermediate concept, a blend of existing interventions, or an intervention composed from physical effects the simulator can execute. Nothing enters the doctrine unless it passes a chain of algebraic gates and a paired forecast test on independently seeded futures.""",

"This article asks what the open layer": """This article asks what the open layer is worth to a commander and answers with paired measurement. The same fires are replayed with no decision support, with the closed layer alone, and with the two layers together, so the benefit of opening the decision space is read from the difference it makes on identical worlds. Three contributions follow. First, the open decision space is defined as an architectural object, the set of orders a system may issue together with the rules under which that set may be extended, and the invariants this object preserves are stated and proved for the gate chain of Section IV. Second, the extension mechanism is specified to the level of its grammar and geometry. A proposal is a tuple over a closed basis of verified effects, sectors, and distance bands, interpreted by the simulator through a fixed operator algebra, so the space is closed at effects and open at interventions. Third, the architecture is measured end to end on an independently validated hazard model, over paired worlds drawn from several scenario families, with every proposal, gate verdict, and admitted object logged, so that the question of what was learned can be answered object by object rather than only in aggregate. The contribution is architectural rather than algorithmic. The concept-driven inference and the generative proposer are replaceable components, and the invariants are properties of the gates. Section II reviews related work. Section III presents the architecture and Section IV the decision core. Section V reports the results, Section VI discusses them, and Section VII concludes.""",
}

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
def find_p(prefix):
    for p in body.iter(w("p")):
        if ptext(p).startswith(prefix) or ("ILDFIRE" in prefix and ptext(p).startswith("ILDFIRE")): return p
    raise KeyError(prefix)
def new_runs(text, rpr):
    out = []
    for seg in re.split(r"(\{[^}]+\})", text):
        if not seg: continue
        if seg.startswith("{"): out.append(cite_sdt(seg[1:-1].split(";"), rpr))
        else: out.append(run(seg, rpr))
    return out
def replace_par(p, text):
    """delete every run of p (tracked) and insert the new text after them"""
    rpr = None
    for r in p.iter(w("r")):
        t = r.find(w("t"))
        if t is not None and (t.text or "").strip(): rpr = r.find(w("rPr")); break
    kids = [c for c in p if c.tag != w("pPr")]
    d = dele()
    for c in kids:
        if c.tag == w("r"): p.remove(c); d.append(del_run(c))
        elif c.tag in (w("sdt"), w("ins")):           # citation controls from v9_v1: delete their runs
            for r in list(c.iter(w("r"))):
                r.getparent().remove(r); d.append(del_run(r))
            p.remove(c)
        elif c.tag == w("del"): p.remove(c); d.extend(list(c))
        elif c.tag in (w("bookmarkStart"), w("bookmarkEnd"), w("proofErr")): p.remove(c)
    p.append(d); p.append(ins(*new_runs(text, rpr)))
def mark_par_inserted(p):
    ppr = p.find(w("pPr")); rp = ppr.find(w("rPr"))
    if rp is None: rp = E("rPr"); ppr.append(rp)
    rp.insert(0, E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}))
def delete_par(p, mark=True):
    for r in list(p.iter(w("r"))):
        par = r.getparent()
        if par.tag == w("del"): continue
        d = dele(); par.replace(r, d); d.append(del_run(r))
    for c in list(p):
        if c.tag == w("ins"):
            i = p.index(c)
            for k in list(c): p.insert(i, k); i += 1
            p.remove(c)
    if mark:
        ppr = p.find(w("pPr"))
        if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
        rp = ppr.find(w("rPr"))
        if rp is None: rp = E("rPr"); ppr.append(rp)
        rp.insert(0, E("del", {"id": nid(), "author": AUTHOR, "date": DATE}))

p1 = find_p("ILDFIRE"); replace_par(p1, NEW["ILDFIRE response is carried out"])
# two new paragraphs after the first one (same pPr as the DisasterAware paragraph: Text style, first-line indent)
pda = find_p("DisasterAware answers these gaps"); base_ppr = copy.deepcopy(pda.find(w("pPr")))
anchor = p1
for key in ("__P2", "__P3"):
    p = E("p", None, copy.deepcopy(base_ppr)); mark_par_inserted(p)
    p.append(ins(*new_runs(NEW[key], None))); anchor.addnext(p); anchor = p
replace_par(pda, NEW["DisasterAware answers these gaps"])
pc = find_p("This article asks what the open layer"); replace_par(pc, NEW["This article asks what the open layer"])
# the scope paragraph is merged into the contribution paragraph
delete_par(find_p("This is a systems contribution"))
# Table I (caption, table, spacer) folded into the prose
cap1 = find_p("TABLE "); cap2 = cap1.getnext(); tbl = cap2.getnext(); spacer = tbl.getnext()
assert cap2.tag == w("p") and tbl.tag == w("tbl") and spacer.tag == w("p") and ptext(spacer) == "", "Table I structure"
for p in (cap1, cap2, spacer): delete_par(p)
for tr in tbl.iter(w("tr")):
    trpr = tr.find(w("trPr"))
    if trpr is None: trpr = E("trPr"); tr.insert(0, trpr)
    trpr.append(E("del", {"id": nid(), "author": AUTHOR, "date": DATE}))
    for p in tr.iter(w("p")): delete_par(p, mark=False)

# ---------------------------------------------------------------- renumber every live CITATION field by first appearance
order = []; fields = []
for it in body.iter(w("instrText")):
    if "CITATION" not in (it.text or ""): continue
    tags = re.findall(r"CITATION (\S+)|\\m (\S+)", it.text); tags = [a or b for a, b in tags]
    for t in tags:
        if t not in order: order.append(t)
    fields.append((it, tags))
num = {t: i + 1 for i, t in enumerate(order)}
for it, tags in fields:
    r = it.getparent(); cached = None
    for sib in r.itersiblings():
        t = sib.find(w("t"))
        if t is not None: cached = t; break
        fc = sib.find(w("fldChar"))
        if fc is not None and fc.get(w("fldCharType")) == "end": break
    cached.text = ", ".join(f"[{num[t]}]" for t in sorted(tags, key=lambda t: num[t]))
    it.text = f" CITATION {sorted(tags, key=lambda t: num[t])[0]} \\l 1033" + "".join(f" \\m {t}" for t in sorted(tags, key=lambda t: num[t])[1:]) + " "

# ---------------------------------------------------------------- bibliography: reorder the cached entries and RefOrder
RECS = sources_v9.load()
sys.argv = [sys.argv[0], D]; import importlib.util
spec = importlib.util.spec_from_file_location("bibfmt", "bibfmt.py"); bibfmt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bibfmt)
TAGS = bibfmt.make_tags(RECS); n_of_tag = {v: k for k, v in TAGS.items()}
assert set(order) == set(TAGS.values()), set(TAGS.values()) - set(order)
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
bibps = [p for p in body.iter(w("p")) if p.find("w:pPr/w:pStyle", NS) is not None and p.find("w:pPr/w:pStyle", NS).get(w("val")) == "Bibliography"]
assert len(bibps) == 52, len(bibps)
for i, p in enumerate(bibps):
    tag = order[i]; old_n = n_of_tag[tag]
    i_ = p.find(w("ins")); keep_head = [r for r in i_ if r.find(w("fldChar")) is not None and r.find(w("fldChar")).get(w("fldCharType")) in ("begin", "separate") or r.find(w("instrText")) is not None]
    keep_tail = [r for r in i_ if r.find(w("fldChar")) is not None and r.find(w("fldChar")).get(w("fldCharType")) == "end"]
    for r in list(i_): i_.remove(r)
    for r in keep_head: i_.append(r)
    i_.append(run(f"[{i + 1}]\t", rpr16))
    for r in bibfmt.entry_runs(RECS, old_n, rpr16, run): i_.append(r)
    for r in keep_tail: i_.append(r)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
src = etree.parse(f"{D}/customXml/item2.xml")
for s in src.getroot().findall("b:Source", NS):
    tag = s.find("b:Tag", NS).text; s.find("b:RefOrder", NS).text = str(num[tag])
src.write(f"{D}/customXml/item2.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("order changes:", sum(1 for t in order if num[t] != n_of_tag[t]), "| first 12:", [(num[t], n_of_tag[t]) for t in order[:12]])

# ---------------------------------------------------------------- table captions: Roman numerals, renumbered after Table I is removed
doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))
ROMAN = ["I", "II", "III", "IV", "V", "VI"]
k = 0; ref_map = {}
for it in body.iter(w("instrText")):
    if "SEQ Table" in (it.text or ""):
        k += 1; it.text = " SEQ Table \\* ROMAN "
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None: t.text = ROMAN[k - 1]; break
        # bookmark name of this caption -> new number
        for bm in it.getparent().getparent().iter(w("bookmarkStart")):
            if bm.get(w("name"), "").startswith("_RefTab"): ref_map[bm.get(w("name"))] = ROMAN[k - 1]
for it in body.iter(w("instrText")):
    m = re.search(r"REF (_RefTab\w+)", it.text or "")
    if m and m.group(1) in ref_map:
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None: t.text = ref_map[m.group(1)]; break
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:captions>" not in s:
    s = s.replace("<w:shapeDefaults>", '<w:captions><w:caption w:name="Fig." w:pos="below" w:numFmt="decimal"/><w:caption w:name="Table" w:pos="above" w:numFmt="upperRoman"/></w:captions><w:shapeDefaults>', 1)
    open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("tables renumbered:", k, ref_map)
