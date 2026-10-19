"""Turn main_article.docx into a cross-referenced IEEE manuscript, as tracked changes.
- author line + first-page affiliation footnote
- citations [n] -> REF fields to bookmarked reference entries; reference list from refs.txt (IEEE style, numbered [n])
- figure captions: "Fig. " + SEQ Figure; table titles: "TABLE " + SEQ Table (roman); in-text Fig./Table -> REF fields
- equations: Equation paragraphs -> borderless 2-column table (OMML left, "(" SEQ ( ")" right); in-text (n) -> REF
Usage: python xref_main.py <unpacked dir> <refs.txt>
"""
import re, sys, copy, datetime
from lxml import etree

D = sys.argv[1]; REFS = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W, "m": M}
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [5000]
def nid(): _id[0] += 1; return str(_id[0])
_bm = [1000]
def bmid(): _bm[0] += 1; return str(_bm[0])

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))


def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag) if ":" not in tag else "{%s}%s" % (NS[tag.split(":")[0]], tag.split(":")[1]))
    for k, v in (attrib or {}).items(): e.set(w(k) if ":" not in k else "{%s}%s" % (NS[k.split(":")[0]], k.split(":")[1]), v)
    for c in kids: e.append(c)
    return e


def ins(*kids): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def dele(*kids): return E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)


def run(text, rpr=None, space=True):
    r = E("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t = E("t"); t.text = text; t.set("{%s}space" % XML, "preserve"); r.append(t); return r


def del_run(r):
    """copy of run r converted to deleted form"""
    r2 = copy.deepcopy(r)
    for t in r2.iter(w("t")): t.tag = w("delText")
    for t in r2.iter(w("instrText")): t.tag = w("delInstrText")
    return r2


def fld(instr, cached, rpr=None):
    """complex field as a list of runs"""
    out = []
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); r.append(E("fldChar", {"fldCharType": "begin"})); out.append(r)
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); it = E("instrText"); it.text = f" {instr} "; it.set("{%s}space" % XML, "preserve"); r.append(it); out.append(r)
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); r.append(E("fldChar", {"fldCharType": "separate"})); out.append(r)
    r = E("r"); rp = copy.deepcopy(rpr) if rpr is not None else E("rPr"); rp.append(E("noProof")); r.append(rp); t = E("t"); t.text = cached; t.set("{%s}space" % XML, "preserve"); r.append(t); out.append(r)
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); r.append(E("fldChar", {"fldCharType": "end"})); out.append(r)
    return out


def bookmark(name, *kids):
    i = bmid(); return [E("bookmarkStart", {"id": i, "name": name})] + list(kids) + [E("bookmarkEnd", {"id": i})]


def ref(name, cached, rpr=None, para_num=False):
    instr = "REF " + name + (" \\r" if para_num else "") + " \\h"
    return fld(instr, cached, rpr)


def mark_math(om, kind):
    """Word's own way of tracking math: the w:ins / w:del marker sits in the w:rPr of every m:r"""
    for r in om.iter("{%s}r" % M):
        rpr = r.find(w("rPr"))
        if rpr is None:
            rpr = E("rPr"); mpr = r.find("{%s}rPr" % M)
            r.insert(r.index(mpr) + 1 if mpr is not None else 0, rpr)
        rpr.insert(0, E(kind, {"id": nid(), "author": AUTHOR, "date": DATE}))


def mark_par_inserted(p):
    ppr = p.find(w("pPr"))
    if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
    rpr = ppr.find(w("rPr"))
    if rpr is None: rpr = E("rPr"); ppr.append(rpr)
    rpr.insert(0, E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}))


def mark_par_deleted(p):
    ppr = p.find(w("pPr"))
    if ppr is None: ppr = E("pPr"); p.insert(0, ppr)
    rpr = ppr.find(w("rPr"))
    if rpr is None: rpr = E("rPr"); ppr.append(rpr)
    rpr.insert(0, E("del", {"id": nid(), "author": AUTHOR, "date": DATE}))


def delete_par_content(p):
    """wrap every run / math of paragraph p in w:del and mark its paragraph mark deleted"""
    for ch in list(p):
        if ch.tag == w("r"):
            d = dele(); p.replace(ch, d); d.append(del_run(ch))
        elif ch.tag in ("{%s}oMath" % M, "{%s}oMathPara" % M):
            p.remove(ch)   # math itself is moved into the equation table untracked (Word has no portable markup for tracked math)
        elif ch.tag == w("hyperlink") or ch.tag == w("sdt"):
            for r in list(ch.iter(w("r"))):
                d = dele(); r.getparent().replace(r, d); d.append(del_run(r))
    mark_par_deleted(p)


def insert_par_content(p):
    for ch in list(p):
        if ch.tag in (w("r"), "{%s}oMath" % M, "{%s}oMathPara" % M):
            i = ins(); p.replace(ch, i); i.append(ch)
    mark_par_inserted(p)


def ptext(p): return "".join(t.text or "" for t in p.iter(w("t")))
def style(p):
    s = p.find("w:pPr/w:pStyle", NS); return s.get(w("val")) if s is not None else ""


def replace_in_run(r, pattern, make, p):
    """Split run r around regex matches of pattern in its text; for each match, emit del(old) + ins(make(match)).
    make(m) returns (deleted_text, [new nodes])."""
    t = r.find(w("t"))
    if t is None or not t.text: return 0
    txt = t.text; rpr = r.find(w("rPr"))
    ms = list(re.finditer(pattern, txt))
    if not ms: return 0
    parent = r.getparent(); idx = parent.index(r); pos = 0; new = []
    for m in ms:
        a, b = m.span()
        if a > pos: new.append(run(txt[pos:a], rpr))
        dtxt, nodes = make(m)
        keep_before = txt[a:b][: txt[a:b].index(dtxt)]; keep_after = txt[a:b][txt[a:b].index(dtxt) + len(dtxt):]
        if keep_before: new.append(run(keep_before, rpr))
        new.append(dele(del_run(run(dtxt, rpr))))
        cur = None
        for n in nodes:
            if n.tag in (w("bookmarkStart"), w("bookmarkEnd")): new.append(n); cur = None
            else:
                if cur is None: cur = ins(); new.append(cur)
                cur.append(n)
        if keep_after: new.append(run(keep_after, rpr))
        pos = b
    if pos < len(txt): new.append(run(txt[pos:], rpr))
    parent.remove(r)
    for k, n in enumerate(new): parent.insert(idx + k, n)
    return len(ms)


count = {"cite": 0, "fig": 0, "tab": 0, "eq": 0}
# normalise runs that carry several w:t children (merge_runs output): one w:t per run
multi = 0
for r in body.iter(w("r")):
    ts = r.findall(w("t"))
    if len(ts) > 1:
        if r.find(w("tab")) is not None or r.find(w("br")) is not None: raise SystemExit("run with tabs/breaks and several w:t: " + "".join(x.text or "" for x in ts))
        ts[0].text = "".join(x.text or "" for x in ts); ts[0].set("{%s}space" % XML, "preserve")
        for x in ts[1:]: r.remove(x)
        multi += 1
print("runs normalised:", multi)
pars = list(body.iter(w("p")))

# ------------------------------------------------------------------ 1. author line + footnote
for p in pars:
    if ptext(p).startswith("First A. Author"):
        rpr = p.find(".//w:r/w:rPr", NS)
        for r in list(p.findall(w("r"))): d = dele(); p.replace(r, d); d.append(del_run(r))
        i = ins(run("Çağlar Akman and M. Kemal Leblebicioğlu", rpr))
        if not any(fr.get(w("id")) == "1" for fr in body.iter(w("footnoteReference"))):   # the template already anchors footnote 1
            r = run(" ", rpr); r.insert(len(r) - 1, E("footnoteReference", {"customMarkFollows": "1", "id": "1"})); i.append(r)
        p.append(i); break
fn = etree.parse(f"{D}/word/footnotes.xml"); fnroot = fn.getroot()
f1 = [f for f in fnroot.findall(w("footnote")) if f.get(w("id")) == "1"][0]
p0 = f1.find(w("p")); rpr16 = copy.deepcopy(p0.find(".//w:r/w:rPr", NS))
# keep the user's affiliation text; only prepend the receipt line as an inserted run (no paragraph-mark changes)
if "Manuscript received" not in ptext(p0):
    first_run = p0.find(w("r")); r_new = run("Manuscript received [Month DD, YYYY]. (Corresponding author: Çağlar Akman.) ", rpr16)
    first_run.addprevious(ins(r_new))
fn.write(f"{D}/word/footnotes.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ------------------------------------------------------------------ 2. captions: figures and tables
fig_bm, tab_bm = {}, {}
for p in pars:
    st, tx = style(p), ptext(p)
    if st == "FigureCaption" and re.match(r"Fig\. \d+\.", tx):
        r = p.findall(w("r"))[0]; t = r.find(w("t")); rpr = r.find(w("rPr")); m = re.match(r"Fig\. (\d+)\.", t.text)
        n = m.group(1); name = f"_RefFig{n}"; fig_bm[n] = name
        def mk(m, n=n, name=name, rpr=rpr): return (f"Fig. {n}.", [run("Fig. ", rpr)] + bookmark(name, *fld("SEQ Figure \\* ARABIC", n, rpr)) + [run(".", rpr)])
        replace_in_run(r, r"^Fig\. \d+\.", mk, p); count["fig"] += 1
    if st == "TableTitle" and re.match(r"TABLE [IVX]+$", tx.strip()):
        r = p.findall(w("r"))[0]; rpr = r.find(w("rPr")); n = tx.strip().split()[1]; name = f"_RefTab{n}"; tab_bm[n] = name
        def mk(m, n=n, name=name, rpr=rpr): return (n, bookmark(name, *fld("SEQ Table \\* ROMAN", n, rpr)))
        replace_in_run(r, r"(?<=TABLE )[IVX]+", mk, p); count["tab"] += 1

# ------------------------------------------------------------------ 3. equations -> tables
eq_bm = {}
kids = list(body)
i = 0; groups = []
while i < len(kids):
    k = kids[i]
    if k.tag == w("p") and style(k) == "Equation":
        g = [k]; j = i + 1
        while j < len(kids) and kids[j].tag == w("p") and style(kids[j]) == "Equation" and not re.search(r"\(\d+\)\s*$", ptext(g[-1])):
            g.append(kids[j]); j += 1
        groups.append(g); i = j
    else: i += 1
TW = 4968; W1, W2 = 4248, 720
for g in groups:
    num = re.search(r"\((\d+)\)\s*$", ptext(g[-1])).group(1); name = f"_RefEq{num}"; eq_bm[num] = name
    tbl = E("tbl")
    tblPr = E("tblPr", None, E("tblW", {"w": str(TW), "type": "dxa"}),
              E("tblBorders", None, *[E(s, {"val": "nil"}) for s in ("top", "left", "bottom", "right", "insideH", "insideV")]),
              E("tblLayout", {"type": "fixed"}),
              E("tblCellMar", None, E("left", {"w": "0", "type": "dxa"}), E("right", {"w": "0", "type": "dxa"})),
              E("tblLook", {"val": "04A0", "firstRow": "1", "lastRow": "0", "firstColumn": "1", "lastColumn": "0", "noHBand": "0", "noVBand": "1"}))
    tbl.append(tblPr); tbl.append(E("tblGrid", None, E("gridCol", {"w": str(W1)}), E("gridCol", {"w": str(W2)})))
    tr = E("tr", None, E("trPr", None, E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}))); tbl.append(tr)
    tc1 = E("tc", None, E("tcPr", None, E("tcW", {"w": str(W1), "type": "dxa"}), E("vAlign", {"val": "center"}))); tr.append(tc1)
    for k, p in enumerate(g):
        om = p.find("{%s}oMath" % M)
        if om is None: om = p.find("{%s}oMathPara/{%s}oMath" % (M, M))
        np_ = E("p", None, E("pPr", None, E("keepNext"), E("spacing", {"before": "60" if k == 0 else "0", "after": "60" if k == len(g) - 1 else "0"}), E("jc", {"val": "center"})))
        if om is not None:
            omp = etree.Element("{%s}oMathPara" % M); omp.append(copy.deepcopy(om)); np_.append(omp)
        mark_par_inserted(np_); tc1.append(np_)
    tc2 = E("tc", None, E("tcPr", None, E("tcW", {"w": str(W2), "type": "dxa"}), E("vAlign", {"val": "center"}))); tr.append(tc2)
    rpr = E("rPr", None, E("sz", {"val": "20"}))
    np_ = E("p", None, E("pPr", None, E("keepNext"), E("spacing", {"before": "0", "after": "0"}), E("jc", {"val": "right"})))
    bs, be = bookmark(name)
    np_.append(ins(run("(", rpr))); np_.append(bs); np_.append(ins(*fld("SEQ ( \\* ARABIC", num, rpr))); np_.append(be); np_.append(ins(run(")", rpr)))
    mark_par_inserted(np_); tc2.append(np_)
    # place the table before the group and delete the old paragraphs completely (marks included)
    g[0].addprevious(tbl)
    for p in g: delete_par_content(p)
    count["eq"] += 1

# ------------------------------------------------------------------ 4. in-text references and citations
for p in list(body.iter(w("p"))):
    st = style(p)
    if st in ("Equation",) or p.getparent().tag == w("tc") and p.find(".//m:oMath", NS) is not None: continue
    for r in list(p.findall(w("r"))):
        t = r.find(w("t"))
        if t is None or not t.text: continue
        rpr = r.find(w("rPr"))
        if st not in ("FigureCaption", "TableTitle"):
            # citations [n]
            n = 0
            for rr in list(p.findall(w("r"))):  # re-scan since runs split
                tt = rr.find(w("t"))
                if tt is None or not tt.text or "[" not in tt.text: continue
                n += replace_in_run(rr, r"\[(\d{1,2})\]", lambda m, rpr=rr.find(w("rPr")): (m.group(0), ref(f"_RefB{m.group(1)}", f"[{m.group(1)}]", rpr, para_num=True)), p)
            count["cite"] += n
            for rr in list(p.findall(w("r"))):
                tt = rr.find(w("t"))
                if tt is None or not tt.text: continue
                rp2 = rr.find(w("rPr"))
                if "Fig. " in tt.text:
                    replace_in_run(rr, r"(?<=Fig\. )(\d{1,2})(?!\.)", lambda m, rp2=rp2: (m.group(1), ref(f"_RefFig{m.group(1)}", m.group(1), rp2)), p)
            for rr in list(p.findall(w("r"))):
                tt = rr.find(w("t"))
                if tt is None or not tt.text: continue
                rp2 = rr.find(w("rPr"))
                if "Table " in tt.text:
                    replace_in_run(rr, r"(?<=Table )([IVX]+)(?![a-z])", lambda m, rp2=rp2: (m.group(1), ref(f"_RefTab{m.group(1)}", m.group(1), rp2)), p)
            for rr in list(p.findall(w("r"))):
                tt = rr.find(w("t"))
                if tt is None or not tt.text: continue
                rp2 = rr.find(w("rPr"))
                if "(" in tt.text and eq_bm:
                    pat = r"(?<=\()(" + "|".join(sorted(eq_bm, key=int)) + r")(?=\))"
                    replace_in_run(rr, pat, lambda m, rp2=rp2: (m.group(1), ref(f"_RefEq{m.group(1)}", m.group(1), rp2)), p)

# ------------------------------------------------------------------ 5. reference list
sdt = [k for k in body if k.tag == w("sdt")]
anchor = sdt[-1] if sdt else list(body)[-2]
if sdt:
    for p in sdt[-1].iter(w("p")): delete_par_content(p)
h = E("p", None, E("pPr", None, E("pStyle", {"val": "Heading1"}))); h.append(ins(run("References"))); mark_par_inserted(h); anchor.addnext(h); prev = h
rpr16 = E("rPr", None, E("sz", {"val": "16"}), E("szCs", {"val": "16"}))
for n, rtxt in enumerate(REFS, 1):
    rtxt = re.sub(r"^\[\d+\]\s*", "", rtxt)
    p = E("p", None, E("pPr", None, E("numPr", None, E("ilvl", {"val": "0"}), E("numId", {"val": "1"})), E("ind", {"left": "270", "hanging": "270"}), E("jc", {"val": "both"}), copy.deepcopy(rpr16)))
    i_ = ins(); nodes = []
    for seg in re.split(r"(\*[^*]+\*)", rtxt):
        if not seg: continue
        rp = copy.deepcopy(rpr16)
        if seg.startswith("*"): rp.insert(0, E("i")); seg = seg[1:-1]
        nodes.append(run(seg, rp))
    bs, be = bookmark(f"_RefB{n}"); p.append(bs); [i_.append(x) for x in nodes]; p.append(i_); p.append(be)
    if n < len(REFS): mark_par_inserted(p)   # the final paragraph mark of the document stays unmarked
    prev.addnext(p); prev = p
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)

# ------------------------------------------------------------------ 6. settings: track revisions on, caption label "("
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
if "<w:captions" not in s:
    cap = '<w:captions><w:caption w:name="(" w:pos="below" w:numFmt="decimal"/></w:captions>'
    for tag in ("<w:shapeDefaults", "<w:decimalSymbol", "<w:listSeparator", "</w:settings>"):
        if tag in s: s = s.replace(tag, cap + tag, 1); break
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print(count, "fig", sorted(fig_bm), "tab", sorted(tab_bm), "eq", sorted(eq_bm, key=int))
