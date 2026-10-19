"""v7 -> v7_v1: reference list with its own numbering (starts at [1]), bookmarks for every REF _RefBn,
cached citation numbers corrected, six uncited references cited, Track Changes on. All edits tracked."""
import re, sys, copy, datetime
from lxml import etree
D = sys.argv[1]; REFS = [re.sub(r"^\[\d+\]\s*", "", l.strip()) for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"; NS = {"w": W}
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [7000]
def nid(): _id[0] += 1; return str(_id[0])
_bm = [2000]
def bmid(): _bm[0] += 1; return str(_bm[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def ins(*kids): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def dele(*kids): return E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def run(text, rpr=None):
    r = E("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
def del_run(r):
    r2 = copy.deepcopy(r)
    for t in r2.iter(w("t")): t.tag = w("delText")
    for t in r2.iter(w("instrText")): t.tag = w("delInstrText")
    return r2
def fld(instr, cached, rpr=None):
    out = []
    for kind in ("begin", None, "separate", "cached", "end"):
        r = E("r")
        if rpr is not None: r.append(copy.deepcopy(rpr))
        if kind == "cached":
            rp = r.find(w("rPr"))
            if rp is None: rp = E("rPr"); r.append(rp)
            rp.append(E("noProof")); t = E("t"); t.text = cached; t.set("{%s}space" % XML, "preserve"); r.append(t)
        elif kind is None:
            it = E("instrText"); it.text = f" {instr} "; it.set("{%s}space" % XML, "preserve"); r.append(it)
        else: r.append(E("fldChar", {"fldCharType": kind}))
        out.append(r)
    return out
def cite(n, rpr): return fld(f"REF _RefB{n} \\r \\h", f"[{n}]", rpr)
def mark_par_inserted(p):
    ppr = p.find(w("pPr"))
    if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
    rpr = ppr.find(w("rPr"))
    if rpr is None: rpr = E("rPr"); ppr.append(rpr)
    rpr.insert(0, E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}))
def delete_par_content(p):
    for r in list(p.findall(w("r"))): d = dele(); p.replace(r, d); d.append(del_run(r))
    for r in list(p.iter(w("r"))):   # runs inside sdt/hyperlink children
        if r.getparent().tag != w("del"): d = dele(); r.getparent().replace(r, d); d.append(del_run(r))
    ppr = p.find(w("pPr"))
    if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
    rpr = ppr.find(w("rPr"))
    if rpr is None: rpr = E("rPr"); ppr.append(rpr)
    rpr.insert(0, E("del", {"id": nid(), "author": AUTHOR, "date": DATE}))
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))

# ---------------------------------------------------------------- 1. dedicated numbering for the reference list
num = etree.parse(f"{D}/word/numbering.xml"); nroot = num.getroot()
aid = str(max(int(a.get(w("abstractNumId"))) for a in nroot.findall(w("abstractNum"))) + 1)
NUMID = str(max(int(a.get(w("numId"))) for a in nroot.findall(w("num"))) + 1)
absn = E("abstractNum", {"abstractNumId": aid}, E("multiLevelType", {"val": "singleLevel"}),
         E("lvl", {"ilvl": "0"}, E("start", {"val": "1"}), E("numFmt", {"val": "decimal"}), E("lvlText", {"val": "[%1]"}), E("lvlJc", {"val": "left"}),
           E("pPr", None, E("ind", {"left": "270", "hanging": "270"})), E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))))
last_abs = nroot.findall(w("abstractNum"))[-1]; last_abs.addnext(absn)
nroot.append(E("num", {"numId": NUMID}, E("abstractNumId", {"val": aid})))
num.write(f"{D}/word/numbering.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ---------------------------------------------------------------- 2. reference list after the References heading, replacing the bibliography control
h = [k for k in body.iter(w("p")) if ptext(k).strip() == "References" and k.find("w:pPr/w:pStyle", NS) is not None]
assert h, "References heading not found"; h = h[-1]
removed = 0; prev = h
nxt = h.getnext()
if nxt is not None and nxt.tag == w("sdt"):          # the empty Word bibliography control
    for p in nxt.iter(w("p")): delete_par_content(p)
    removed += 1; prev = nxt
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
for n, rtxt in enumerate(REFS, 1):
    p = E("p", None, E("pPr", None, E("numPr", None, E("ilvl", {"val": "0"}), E("numId", {"val": NUMID})), E("spacing", {"after": "20"}), E("ind", {"left": "270", "hanging": "270"}), E("jc", {"val": "both"}), copy.deepcopy(rpr16)))
    i_ = ins()
    for seg in re.split(r"(\*[^*]+\*)", rtxt):
        if not seg: continue
        rp = copy.deepcopy(rpr16)
        if seg.startswith("*"): rp.insert(0, E("i")); seg = seg[1:-1]
        i_.append(run(seg, rp))
    b = bmid(); p.append(E("bookmarkStart", {"id": b, "name": f"_RefB{n}"})); p.append(i_); p.append(E("bookmarkEnd", {"id": b}))
    mark_par_inserted(p); prev.addnext(p); prev = p

# ---------------------------------------------------------------- 3. cached citation numbers -> [n] (the field code is the truth)
fixed = 0
for p in body.iter(w("p")):
    runs = list(p.iter(w("r")))
    for k, r in enumerate(runs):
        it = r.find(w("instrText"))
        if it is not None and it.text and "REF _RefB" in it.text:
            n = re.search(r"_RefB(\d+)", it.text).group(1)
            for r2 in runs[k + 1:k + 4]:
                t = r2.find(w("t"))
                if t is not None and t.text and t.text.startswith("["):
                    if t.text != f"[{n}]": t.text = f"[{n}]"; fixed += 1
                    break

# ---------------------------------------------------------------- 4. cite the six references that no sentence cites yet
def find_run(substr):
    for p in body.iter(w("p")):
        for r in p.findall(w("r")):
            t = r.find(w("t"))
            if t is not None and t.text and substr in t.text: return p, r, t
    return None, None, None

def add_after_text(substr, before_text, nodes):
    """split the run at the end of before_text (inside substr match) and insert nodes there (tracked)"""
    p, r, t = find_run(substr); assert r is not None, substr
    txt = t.text; cut = txt.index(before_text) + len(before_text)
    rpr = r.find(w("rPr")); t.text = txt[:cut]
    i_ = ins(*nodes); r.addnext(i_)
    if txt[cut:]: i_.addnext(run(txt[cut:], rpr))
    return rpr

# [5] Bowman, before the existing [6]: "clock set by the fire [5], [6]"
p, r, t = find_run("clock set by the fire "); rpr = r.find(w("rPr"))
r.addnext(ins(*cite(5, rpr), run(", ", rpr)))
# [18] Turoff: "carried out by a system of systems [18]."
add_after_text("carried out by a system of systems", "carried out by a system of systems", [run(" ", None)] + cite(18, None))
# [1] Power, [10] Vásquez: "exist for parts of this task [1], [10], from spread simulators"
add_after_text("Decision support systems exist for parts of this task", "exist for parts of this task", [run(" ", None)] + cite(1, None) + [run(", ", None)] + cite(10, None))
# [22] Comes: "multi-criteria allocation [15], [22], [24]" -> insert after the first field of that sentence
p, r, t = find_run("and multi-criteria allocation "); rpr = r.find(w("rPr"))
runs = list(p.iter(w("r"))); k = runs.index(r)
end_run = next(x for x in runs[k:] if x.find(w("fldChar")) is not None and x.find(w("fldChar")).get(w("fldCharType")) == "end")
end_run.addnext(ins(run(", ", rpr), *cite(22, rpr)))
# [4] Ribeiro (LIME): Table I, "post-hoc attribution [4]"
p, r, t = find_run("post-hoc attribution"); rpr = r.find(w("rPr"))
add_after_text("post-hoc attribution", "post-hoc attribution", [run(" ", rpr)] + cite(4, rpr))

doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ---------------------------------------------------------------- 5. settings
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("refs:", len(REFS), "numId:", NUMID, "cached fixed:", fixed, "sdt removed:", removed)
