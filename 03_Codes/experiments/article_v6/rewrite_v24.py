"""v23_v1 -> v23_v2: rewrite Sections III and IV from content_v23.BLOCKS.
New text paragraphs and headings are tracked insertions; old paragraphs are tracked deletions placed after the new text;
equation tables and algorithm boxes are new tables whose rows are marked as inserted (OMML itself cannot sit inside w:ins).
Existing figure blocks (drawing + caption) are moved into their new positions unchanged.
usage: python3 rewrite_v23.py <unzipped dir> content_v23.py"""
import re, sys, copy, datetime, subprocess, os, importlib.util
from lxml import etree

D = sys.argv[1]
spec = importlib.util.spec_from_file_location("content", sys.argv[2]); content = importlib.util.module_from_spec(spec); spec.loader.exec_module(content)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
def w(t): return "{%s}%s" % (W, t)
def m(t): return "{%s}%s" % (M, t)
AUTHOR = "Claude"; DATE = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [30000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
def mark(tag): return E(tag, {"id": nid(), "author": AUTHOR, "date": DATE})
def ins(*kids): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def dele(*kids): return E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def run(text, rpr=None):
    r = E("r")
    if rpr is not None and len(rpr): r.append(copy.deepcopy(rpr))
    t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r
def del_el(el):
    e2 = copy.deepcopy(el)
    for t in e2.iter(w("t")): t.tag = w("delText")
    for t in e2.iter(w("instrText")): t.tag = w("delInstrText")
    return e2
def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))
def pstyle(p):
    ppr = p.find(w("pPr"))
    if ppr is None: return ""
    ps = ppr.find(w("pStyle")); return ps.get(w("val")) if ps is not None else ""

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))

# ---------------------------------------------------------------- citations: number -> tag
NUM2TAG = {}
for it in body.iter(w("instrText")):
    if "CITATION" in (it.text or ""):
        tags = [a or b for a, b in re.findall(r"CITATION (\S+)|\\m (\S+)", it.text)]
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None:
                nums = [int(x) for x in re.findall(r"\d+", t.text)]; break
        for n, tg in zip(nums, tags): NUM2TAG[n] = tg
def cite_sdt(nums):
    nums = sorted(nums); tags = [NUM2TAG[n] for n in nums]
    instr = f"CITATION {tags[0]} \\l 1033" + "".join(f" \\m {t}" for t in tags[1:])
    runs = []
    for kind in ("begin", "instr", "separate"):
        r = E("r")
        if kind == "instr":
            x = E("instrText"); x.text = f" {instr} "; x.set("{%s}space" % XML, "preserve"); r.append(x)
        else:
            fc = E("fldChar", {"fldCharType": kind})
            if kind == "begin": fc.set(w("fldLock"), "1")
            r.append(fc)
        runs.append(r)
    rp = E("rPr", None, E("noProof")); runs.append(run("[" + ", ".join(map(str, nums)) + "]", rp))
    r = E("r"); r.append(E("fldChar", {"fldCharType": "end"})); runs.append(r)
    return E("sdt", None, E("sdtPr", None, E("id", {"val": str(-(1980000000 - int(nid())))}), E("citation")), E("sdtContent", None, *runs))

# ---------------------------------------------------------------- equation numbering
EQ_ORDER = []
for b in content.BLOCKS:
    if b[0] == "EQ":
        for key, _ in b[1]: EQ_ORDER.append(key)
EQNUM = {k: i + 1 for i, k in enumerate(EQ_ORDER)}
def bmname(key): return "_RefEqN_" + key
_bm = [900]
def ref_field(key):
    """REF field runs with cached equation number"""
    out = []
    r = E("r"); r.append(E("fldChar", {"fldCharType": "begin"})); out.append(r)
    r = E("r"); x = E("instrText"); x.text = f" REF {bmname(key)} \\h "; x.set("{%s}space" % XML, "preserve"); r.append(x); out.append(r)
    r = E("r"); r.append(E("fldChar", {"fldCharType": "separate"})); out.append(r)
    out.append(run(str(EQNUM[key]), E("rPr", None, E("noProof"))))
    r = E("r"); r.append(E("fldChar", {"fldCharType": "end"})); out.append(r)
    return out

# ---------------------------------------------------------------- inline text -> runs (citations {n;m}, equation refs (EQ:key))
INLINE = []          # latex strings of inline math, in document order
def text_runs(text, rpr=None):
    """list of run-level elements; an inline $...$ becomes ("MATH", latex) and is resolved by para()"""
    out = []
    for seg in re.split(r"(\{\d+(?:;\d+)*\}|\(EQ:\w+\)|\$[^$]+\$)", text):
        if not seg: continue
        if seg.startswith("{"): out.append(cite_sdt([int(x) for x in seg[1:-1].split(";")]))
        elif seg.startswith("(EQ:"):
            out.append(run("(", rpr)); out.extend(ref_field(seg[4:-1])); out.append(run(")", rpr))
        elif seg.startswith("$"): out.append(("MATH", seg[1:-1]))
        else: out.append(run(seg, rpr))
    return out
MATH_POOL = {}       # filled after the pandoc pass: index -> oMath
_mi = [0]
def resolve(items):
    """group run-level items into w:ins blocks, with bare inline oMath between them"""
    out = []; buf = []
    def flush():
        if buf: out.append(ins(*buf)); buf.clear()
    for it in items:
        if isinstance(it, tuple) and it[0] == "MATH":
            flush(); om = copy.deepcopy(MATH_POOL[_mi[0]]); _mi[0] += 1
            for r in om.iter(m("r")):          # italic math font as in the rest of the document
                if r.find("{%s}rPr" % W) is None:
                    rp = E("rPr", None, E("rFonts", {"ascii": "Cambria Math", "hAnsi": "Cambria Math"})); r.append(rp)
            out.append(om)
        else: buf.append(it)
    flush(); return out

# ---------------------------------------------------------------- paragraph builders (tracked insertions)
def para(style, text, extra_ppr=None):
    ppr = E("pPr")
    if style: ppr.append(E("pStyle", {"val": style}))
    if extra_ppr is not None:
        for e in extra_ppr: ppr.append(e)
    rp = E("rPr"); rp.append(mark("ins")); ppr.append(rp)
    p = E("p", None, ppr)
    for el in resolve(text_runs(text)): p.append(el)
    return p

# ---------------------------------------------------------------- OMML via pandoc
def omml_all(latex_list):
    os.makedirs("pd", exist_ok=True)
    src = "\n\n".join(f"$${x}$$" for x in latex_list)
    open("pd/eqs.md", "w", encoding="utf-8").write(src + "\n")
    subprocess.run(["pandoc", "pd/eqs.md", "-o", "pd/eqs.docx"], check=True)
    subprocess.run(["unzip", "-o", "-q", "pd/eqs.docx", "word/document.xml", "-d", "pd/eqs"], check=True)
    t = etree.parse("pd/eqs/word/document.xml")
    maths = list(t.getroot().iter(m("oMath")))
    assert len(maths) == len(latex_list), (len(maths), len(latex_list))
    out = []
    for x in maths:
        x = copy.deepcopy(x)
        for nary in x.iter(m("nary")):          # hide empty limits instead of showing placeholder boxes
            pr = nary.find(m("naryPr"))
            if pr is None: pr = etree.Element(m("naryPr")); nary.insert(0, pr)
            for lim, tag in (("sub", "subHide"), ("sup", "supHide")):
                e = nary.find(m(lim))
                if e is not None and len(e) == 0 and pr.find(m(tag)) is None:
                    h = etree.SubElement(pr, m(tag)); h.set(m("val"), "1")
        for r in list(x.iter(m("r"))):
            t_ = r.find(m("t"))
            if t_ is not None and (t_.text or "") == "": r.getparent().remove(r)
        out.append(x)
    return out

# equation table template from the document
tmpl_tbl = next(t for t in body.iter(w("tbl")) if t.find(".//" + m("oMath")) is not None)
TBLPR = copy.deepcopy(tmpl_tbl.find(w("tblPr"))); TBLGRID = copy.deepcopy(tmpl_tbl.find(w("tblGrid")))
def eq_table(rows):
    """rows: list of (key, oMath element)"""
    tbl = E("tbl", None, copy.deepcopy(TBLPR), copy.deepcopy(TBLGRID))
    for key, om in rows:
        tr = E("tr", None, E("trPr", None, E("cantSplit"), mark("ins")))
        # math cell
        tc1 = E("tc", None, E("tcPr", None, E("tcW", {"w": "4248", "type": "dxa"}), E("vAlign", {"val": "center"})))
        p1 = E("p", None, E("pPr", None, E("keepNext"), E("spacing", {"before": "60"}), E("jc", {"val": "center"}), E("rPr", None, mark("ins"))))
        mp = etree.SubElement(p1, m("oMathPara"))
        for line in (om if isinstance(om, list) else [om]): mp.append(line)
        tc1.append(p1)
        # number cell
        tc2 = E("tc", None, E("tcPr", None, E("tcW", {"w": "720", "type": "dxa"}), E("vAlign", {"val": "center"})))
        p2 = E("p", None, E("pPr", None, E("keepNext"), E("jc", {"val": "right"}), E("rPr", None, mark("ins"))))
        _bm[0] += 1; bid = str(_bm[0])
        p2.append(run("(")); p2.append(E("bookmarkStart", {"id": bid, "name": bmname(key)}))
        r = E("r"); r.append(E("fldChar", {"fldCharType": "begin"})); p2.append(r)
        r = E("r"); x = E("instrText"); x.text = " SEQ ( \\* ARABIC "; x.set("{%s}space" % XML, "preserve"); r.append(x); p2.append(r)
        r = E("r"); r.append(E("fldChar", {"fldCharType": "separate"})); p2.append(r)
        p2.append(run(str(EQNUM[key]), E("rPr", None, E("noProof"))))
        r = E("r"); r.append(E("fldChar", {"fldCharType": "end"})); p2.append(r)
        p2.append(E("bookmarkEnd", {"id": bid})); p2.append(run(")"))
        tc2.append(p2); tr.append(tc1); tr.append(tc2); tbl.append(tr)
    return tbl

# ---------------------------------------------------------------- algorithm box (IEEE style: rules above and below, header rule)
def alg_table(name, title, lines):
    tblpr = E("tblPr", None, E("tblW", {"w": "4968", "type": "dxa"}),
              E("tblBorders", None, E("top", {"val": "single", "sz": "8", "space": "0", "color": "auto"}), E("left", {"val": "nil"}),
                E("bottom", {"val": "single", "sz": "8", "space": "0", "color": "auto"}), E("right", {"val": "nil"}),
                E("insideH", {"val": "single", "sz": "4", "space": "0", "color": "auto"}), E("insideV", {"val": "nil"})),
              E("tblLayout", {"type": "fixed"}), E("tblCellMar", None, E("left", {"w": "40", "type": "dxa"}), E("right", {"w": "40", "type": "dxa"})))
    tbl = E("tbl", None, tblpr, E("tblGrid", None, E("gridCol", {"w": "4968"})))
    sz = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
    def cellp(runs, indent=0, bold_prefix=None):
        ppr = E("pPr", None, E("keepNext"), E("spacing", {"before": "0", "after": "0", "line": "240", "lineRule": "auto"}), E("ind", {"left": str(indent)}), E("rPr", None, mark("ins")))
        p = E("p", None, ppr)
        for el in resolve(runs): p.append(el)
        return p
    def row(paras):
        tr = E("tr", None, E("trPr", None, E("cantSplit"), mark("ins")))
        tc = E("tc", None, E("tcPr", None, E("tcW", {"w": "4968", "type": "dxa"})))
        for p in paras: tc.append(p)
        tr.append(tc); return tr
    b = E("rPr", None, E("b"), E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
    head = cellp([run(name + " ", b)] + text_runs(title, sz))
    tbl.append(row([head]))
    paras = []
    for i, (lvl, text) in enumerate(lines, 1):
        rs = [run(f"{i}: ", sz)] + text_runs(text, sz)
        paras.append(cellp(rs, indent=120 * lvl))
    tbl.append(row(paras)); return tbl

# ---------------------------------------------------------------- locate the old region and figure blocks
kids = list(body)
def h1_index(prefix):
    for i, k in enumerate(kids):
        if k.tag == w("p") and pstyle(k) == "Heading1" and ptext(k).startswith(prefix): return i
    raise KeyError(prefix)
i3, i5 = h1_index("III."), h1_index("V.")
region = kids[i3:i5]
drawpars = []
for p in body.iter(w("p")):
    if p.find(".//" + w("drawing")) is not None and p.getparent() is body and p not in drawpars: drawpars.append(p)
FIG = {}; FLOAT = []
for n, p in enumerate(drawpars, 1):
    if p in region:
        cap = p.getnext()
        if cap is not None and pstyle(cap) == "FigureCaption": FIG[n] = [p, cap]
        else: FLOAT.append(p)
figpars = {id(x) for v in FIG.values() for x in v} | {id(x) for x in FLOAT}
old_pars = [k for k in region if k.tag == w("p") and id(k) not in figpars]
old_tbls = [k for k in region if k.tag == w("tbl")]
def norm_old(p):
    t = "".join(x.text or "" for x in p.iter(w("t")))
    return re.sub(r"[^a-z]", "", t.lower())
def norm_new(text):
    t = re.sub(r"\$[^$]+\$|\(EQ:\w+\)|\{\d+(?:;\d+)*\}", "", text)
    return re.sub(r"[^a-z]", "", t.lower())
OLD_BY_NORM = {}
for p in old_pars:
    if pstyle(p) in ("Text", "Heading1", "Heading2"): OLD_BY_NORM.setdefault(norm_old(p), []).append(p)
reused = []
# ---------------------------------------------------------------- build new elements
def inline_of(text): return [seg[1:-1] for seg in re.findall(r"\$[^$]+\$", text)]
latex = []; disp_idx = {}
for b in content.BLOCKS:
    if b[0] == "EQ":
        for key, x in b[1]:
            lines = x if isinstance(x, list) else [x]
            disp_idx[key] = list(range(len(latex), len(latex) + len(lines))); latex.extend(lines)
    elif b[0] in ("P", "H1", "H2"): latex.extend(inline_of(b[1]))
    elif b[0] == "FIG": latex.extend(inline_of(b[2]))
    elif b[0] == "ALG":
        latex.extend(inline_of(b[1][1]))
        for _, line in b[1][2]: latex.extend(inline_of(line))
ALL = omml_all(latex)
disp_set = {i for v in disp_idx.values() for i in v}
MATH_POOL.update({n: ALL[i] for n, i in enumerate(i for i in range(len(ALL)) if i not in disp_set)})
new = []
for b in content.BLOCKS:
    kind = b[0]
    if kind in ("H1", "H2", "P"):
        style = {"H1": "Heading1", "H2": "Heading2", "P": "Text"}[kind]
        cands = OLD_BY_NORM.get(norm_new(b[1]), [])
        cands = [c for c in cands if pstyle(c) == style and c not in reused]
        if cands and norm_new(b[1]):
            p = cands[0]; reused.append(p); body.remove(p); new.append(p)
            _mi[0] += len(inline_of(b[1]))          # keep the inline-math pool aligned
        else: new.append(para(style, b[1]))
        if kind == "H1" and b[1].startswith("III.") and FLOAT:
            for fp in FLOAT: body.remove(fp); new.append(fp)
    elif kind == "EQ":
        rows = [(key, [ALL[i] for i in disp_idx[key]]) for key, _ in b[1]]
        new.append(eq_table(rows))
    elif kind == "ALG":
        name, title, lines = b[1]; new.append(alg_table(name, title, lines))
    elif kind == "FIG":
        drw, cap = FIG[b[1]]; body.remove(drw); body.remove(cap); new.append(drw)
        # caption: keep "Fig. " + bookmark + SEQ field, delete the old text (tracked), insert the new text
        kids_c = [c for c in cap if c.tag != w("pPr")]
        cut = next(i for i, c in enumerate(kids_c) if c.tag == w("bookmarkEnd")) + 1
        for c in kids_c[cut:]:
            cap.remove(c)
            if c.tag in (w("r"), w("sdt")):
                dd = dele(); dd.append(del_el(c)); cap.append(dd)
        for el in resolve([run(". ")] + text_runs(b[2])): cap.append(el)
        new.append(cap)
    else: raise ValueError(kind)

# ---------------------------------------------------------------- tracked deletion of the old paragraphs; drop old equation tables
def delete_par(p):
    for c in [c for c in p if c.tag != w("pPr")]:
        if c.tag in (w("r"), w("sdt")): d = dele(); p.replace(c, d); d.append(del_el(c))
        elif c.tag == w("proofErr"): p.remove(c)
    ppr = p.find(w("pPr"))
    if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
    rp = ppr.find(w("rPr"))
    if rp is None: rp = E("rPr"); ppr.append(rp)
    rp.insert(0, mark("del"))
for t in old_tbls: body.remove(t)
old_pars = [p for p in old_pars if p not in reused]
for p in old_pars: body.remove(p); delete_par(p)
anchor = kids[i5]
for el in new + old_pars: anchor.addprevious(el)

doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("new elements:", len(new), "| equations:", len(EQ_ORDER), "| old paragraphs deleted:", len(old_pars), "| old equation tables removed:", len(old_tbls), "| figures moved:", sorted(FIG))

# ---------------------------------------------------------------- figure image of Fig. 2 (decision core): regenerated PNG
import shutil
from PIL import Image
A = "http://schemas.openxmlformats.org/drawingml/2006/main"; R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"; WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
rels = etree.parse(f"{D}/word/_rels/document.xml.rels"); rmap = {e.get("Id"): e.get("Target") for e in rels.getroot()}
dr = FIG[2][0].find(".//" + w("drawing"))
blip = next(dr.iter("{%s}blip" % A)); target = rmap[blip.get("{%s}embed" % R)]
src = "/home/claude/v23/fig/fig2_decisioncore.png"; shutil.copy(src, f"{D}/word/{target}")
iw, ih = Image.open(src).size
for ext in list(dr.iter("{%s}extent" % WP)) + list(dr.iter("{%s}ext" % A)):
    cx = 3017520; ext.set("cx", str(cx)); ext.set("cy", str(int(cx * ih / iw)))
# ---------------------------------------------------------------- equation numbers: cached REF values follow the SEQ order
seq = {}
for p in body.iter(w("p")):
    for it in p.iter(w("instrText")):
        if "SEQ (" in (it.text or ""):
            n = len(seq) + 1
            bm = p.find(w("bookmarkStart"))
            if bm is not None: seq[bm.get(w("name"))] = n
            for sib in it.getparent().itersiblings():
                t = sib.find(w("t"))
                if t is not None: t.text = str(n); break
fixed = 0
for it in body.iter(w("instrText")):
    m_ = re.match(r"\s*REF (_RefEqN_\w+)", it.text or "")
    if m_ and m_.group(1) in seq:
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None:
                if t.text != str(seq[m_.group(1)]): fixed += 1
                t.text = str(seq[m_.group(1)]); break
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("Fig. 2 image refreshed:", target, "| equations numbered:", len(seq), "| REF values corrected:", fixed, "| paragraphs reused:", len(reused))
