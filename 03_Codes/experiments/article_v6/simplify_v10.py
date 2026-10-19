"""v10 -> v10_v1: long sentences split into short ones with plain connectors, tracked at word level.
Citations {n;m} in the new text are rebuilt as locked CITATION controls with the tags already in the document;
cross-reference fields (Fig., Table, equation numbers) are kept where the visible text is unchanged.
usage: python3 simplify_v10.py <unzipped v10 dir> <texts.py>"""
import re, sys, copy, datetime, difflib, importlib.util
from lxml import etree

D = sys.argv[1]
spec = importlib.util.spec_from_file_location("texts", sys.argv[2]); texts = importlib.util.module_from_spec(spec); spec.loader.exec_module(texts)
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "http://www.w3.org/XML/1998/namespace"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [20000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag))
    for k, v in (attrib or {}).items(): e.set(w(k), v)
    for c in kids: e.append(c)
    return e
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

doc = etree.parse(f"{D}/word/document.xml"); body = doc.getroot().find(w("body"))

# number -> tag from the live citation controls
NUM2TAG = {}
for it in body.iter(w("instrText")):
    if "CITATION" in (it.text or ""):
        tags = [a or b for a, b in re.findall(r"CITATION (\S+)|\\m (\S+)", it.text)]
        for sib in it.getparent().itersiblings():
            t = sib.find(w("t"))
            if t is not None:
                nums = [int(x) for x in re.findall(r"\d+", t.text)]; break
        for n, tg in zip(nums, tags): NUM2TAG[n] = tg
assert len(NUM2TAG) == 52, len(NUM2TAG)
def cite_sdt(nums, rpr):
    nums = sorted(nums); tags = [NUM2TAG[n] for n in nums]
    instr = f"CITATION {tags[0]} \\l 1033" + "".join(f" \\m {t}" for t in tags[1:])
    rp = copy.deepcopy(rpr) if rpr is not None else E("rPr"); rp.append(E("noProof"))
    runs = []
    for kind in ("begin", "instr", "separate"):
        r = E("r")
        if rpr is not None and len(rpr): r.append(copy.deepcopy(rpr))
        if kind == "instr":
            x = E("instrText"); x.text = f" {instr} "; x.set("{%s}space" % XML, "preserve"); r.append(x)
        else:
            fc = E("fldChar", {"fldCharType": kind})
            if kind == "begin": fc.set(w("fldLock"), "1")
            r.append(fc)
        runs.append(r)
    runs.append(run("[" + ", ".join(str(n) for n in nums) + "]", rp))
    r = E("r")
    if rpr is not None and len(rpr): r.append(copy.deepcopy(rpr))
    r.append(E("fldChar", {"fldCharType": "end"})); runs.append(r)
    return E("sdt", None, E("sdtPr", None, E("id", {"val": str(-(1980000000 - int(nid())))}), E("citation")), E("sdtContent", None, *runs))
def cached_text(el):
    """visible text of a citation control or field run list"""
    return "".join(t.text or "" for x in (el if isinstance(el, list) else [el]) for t in x.iter(w("t")))

WORD = re.compile(r"\s+|\w+|[^\w\s]")
def old_tokens(p):
    toks = []; kids = [c for c in p if c.tag != w("pPr")]; i = 0
    while i < len(kids):
        c = kids[i]
        if c.tag == w("r"):
            fc = c.find(w("fldChar"))
            if fc is not None and fc.get(w("fldCharType")) == "begin":
                j = i
                while not (kids[j].tag == w("r") and kids[j].find(w("fldChar")) is not None and kids[j].find(w("fldChar")).get(w("fldCharType")) == "end"): j += 1
                grp = kids[i:j + 1]; toks.append((cached_text(grp), "f", grp, None)); i = j + 1; continue
            t = c.find(w("t"))
            if t is not None and t.text:
                txt = t.text; rp = c.find(w("rPr")); key = etree.tostring(rp) if rp is not None else b""
                while i + 1 < len(kids) and kids[i + 1].tag == w("r") and kids[i + 1].find(w("t")) is not None and kids[i + 1].find(w("fldChar")) is None \
                        and (etree.tostring(kids[i + 1].find(w("rPr"))) if kids[i + 1].find(w("rPr")) is not None else b"") == key:
                    i += 1; txt += kids[i].find(w("t")).text or ""
                for m in WORD.findall(txt): toks.append((m, "w", m, rp))
            i += 1
        elif c.tag == w("sdt"): toks.append((cached_text(c), "c", c, None)); i += 1
        else: i += 1
    return toks
def new_tokens(text):
    toks = []
    for seg in re.split(r"(\{[^}]+\})", text):
        if not seg: continue
        if seg.startswith("{"):
            nums = sorted(int(x) for x in seg[1:-1].split(";")); toks.append(("[" + ", ".join(map(str, nums)) + "]", "c", nums, None))
        else:
            for m in WORD.findall(seg): toks.append((m, "w", m, None))
    return toks
def emit(toks, rpr_default, deleted=False):
    out = []; buf = ""; brpr = None
    def flush():
        nonlocal buf, brpr
        if buf:
            r = run(buf, brpr if brpr is not None else rpr_default); out.append(del_el(r) if deleted else r); buf = ""
    for key, kind, payload, rpr in toks:
        if kind == "w":
            if buf and (rpr is not brpr): flush()
            buf += payload; brpr = rpr
        else:
            flush()
            if kind == "c":
                el = payload if isinstance(payload, etree._Element) else cite_sdt(payload, rpr_default)
                out.append(del_el(el) if deleted else el)
            else:
                for r in payload: out.append(del_el(r) if deleted else r)
    flush(); return out
def diff_par(p, text):
    old = old_tokens(p); new = new_tokens(text)
    nref_before = sum(1 for t in old if t[1] == "f")
    rpr_default = next((t[3] for t in old if t[1] == "w" and t[3] is not None), None)
    for c in [c for c in p if c.tag != w("pPr")]: p.remove(c)
    sm = difflib.SequenceMatcher(a=[t[0] for t in old], b=[t[0] for t in new], autojunk=False)
    ops = sm.get_opcodes(); merged = []
    for k, op in enumerate(ops):
        if op[0] == "equal" and (op[2] - op[1]) < 5 and merged and merged[-1][0] != "equal" and k < len(ops) - 1 \
                and not any(t[1] == "f" for t in old[op[1]:op[2]]):
            op = ("replace", op[1], op[2], op[3], op[4])
        if merged and merged[-1][0] != "equal" and op[0] != "equal":
            merged[-1] = ("replace", merged[-1][1], op[2], merged[-1][3], op[4])
        else: merged.append(op)
    kept = 0
    for op, a0, a1, b0, b1 in merged:
        if op == "equal":
            kept += sum(1 for t in old[a0:a1] if t[1] == "f")
            for el in emit(old[a0:a1], rpr_default): p.append(el)
        else:
            if a1 > a0: p.append(dele(*emit(old[a0:a1], rpr_default, deleted=True)))
            if b1 > b0: p.append(ins(*emit(new[b0:b1], rpr_default)))
    return nref_before, kept

def find_p(prefix):
    hits = [p for p in body.iter(w("p")) if ptext(p).startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits)); return hits[0]
changed = 0; lost = []
for prefix, text in texts.PARS:
    p = find_p(prefix)
    if re.sub(r"\s+", " ", ptext(p)).strip() == re.sub(r"\s+", " ", re.sub(r"\{[^}]+\}", "", text)).strip(): continue
    a, b = diff_par(p, text); changed += 1
    if b < a: lost.append((prefix[:30], a, b))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
s = open(f"{D}/word/settings.xml", encoding="utf-8").read()
if "<w:trackRevisions" not in s: s = s.replace("<w:defaultTabStop", "<w:trackRevisions/><w:defaultTabStop", 1)
open(f"{D}/word/settings.xml", "w", encoding="utf-8").write(s)
print("paragraphs changed:", changed, "| cross-reference fields lost:", lost)
