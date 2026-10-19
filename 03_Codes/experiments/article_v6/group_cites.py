"""Group adjacent citation fields: [1], [2] -> [1, 2]; three or more consecutive numbers -> [2-4] (en dash).
Fields keep REF \\r; grouped ones get \\t (number only) and literal brackets around the group (tracked insertions)."""
import re, sys, datetime, copy
from lxml import etree
D = sys.argv[1]; W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML = "{http://www.w3.org/XML/1998/namespace}"
def w(t): return "{%s}%s" % (W, t)
AUTHOR = "Claude"; DATE = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id = [9000]
def nid(): _id[0] += 1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e = etree.Element(w(tag)); [e.set(w(k), v) for k, v in (attrib or {}).items()]; [e.append(c) for c in kids]; return e
def ins(*kids): return E("ins", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def dele(*kids): return E("del", {"id": nid(), "author": AUTHOR, "date": DATE}, *kids)
def run(text, rpr=None):
    r = E("r"); (r.append(copy.deepcopy(rpr)) if rpr is not None else None); t = E("t"); t.text = text; t.set(XML + "space", "preserve"); r.append(t); return r
doc = etree.parse(f"{D}/word/document.xml"); root = doc.getroot()
groups = 0
for p in root.iter(w("p")):
    # sequence of top-level children: runs and ins/del wrappers; locate citation fields as (begin_idx, end_idx, number, cached_run)
    kids = []
    for c in p:
        if c.tag == w("ins"): kids.extend(list(c))      # runs inside tracked insertions take part too
        else: kids.append(c)
    def is_run(e): return e.tag == w("r")
    def put_before(e, r): (e.addprevious(r) if e.getparent().tag == w("ins") else e.addprevious(ins(r)))
    def put_after(e, r): (e.addnext(r) if e.getparent().tag == w("ins") else e.addnext(ins(r)))
    fields = []; i = 0
    while i < len(kids):
        e = kids[i]
        if is_run(e) and e.find(w("fldChar")) is not None and e.find(w("fldChar")).get(w("fldCharType")) == "begin":
            j = i + 1; instr = None; cached = None
            while j < len(kids):
                ej = kids[j]
                if is_run(ej):
                    it = ej.find(w("instrText"))
                    if it is not None: instr = (instr or "") + (it.text or "")
                    fc = ej.find(w("fldChar"))
                    if fc is not None and fc.get(w("fldCharType")) == "end": break
                    if ej.find(w("t")) is not None and instr: cached = ej
                j += 1
            if instr and "REF _RefB" in instr: fields.append((i, j, int(re.search(r"_RefB(\d+)", instr).group(1)), cached))
            i = j + 1
        else: i += 1
    # group fields separated only by a run whose text is ", "
    k = 0
    while k < len(fields):
        grp = [fields[k]]; m = k
        while m + 1 < len(fields):
            a_end = fields[m][1]; b_beg = fields[m + 1][0]
            between = kids[a_end + 1:b_beg]
            if len(between) == 1 and is_run(between[0]) and between[0].find(w("t")) is not None and between[0].find(w("t")).text == ", ":
                grp.append(fields[m + 1]); m += 1
            else: break
        if len(grp) >= 2:
            nums = sorted(g[2] for g in grp); rpr = kids[grp[0][0]].find(w("rPr"))
            # ascending order inside the group: reassign codes and cached numbers to the positions
            grp = [(g[0], g[1], n, g[3]) for g, n in zip(grp, nums)]
            for g in grp:
                for e in kids[g[0]:g[1] + 1]:
                    it = e.find(w("instrText"))
                    if it is not None: it.text = re.sub(r"_RefB\d+", f"_RefB{g[2]}", it.text)
            consecutive = False  # comma lists only, no dash ranges (author preference)
            # number-only fields
            for g in grp:
                for e in kids[g[0]:g[1] + 1]:
                    it = e.find(w("instrText"))
                    if it is not None: it.text = it.text.replace("\\r \\h", "\\r \\t \\h")
                if g[3] is not None: g[3].find(w("t")).text = str(g[2])
            # brackets around the group, and dash for a consecutive run
            put_before(kids[grp[0][0]], run("[", rpr))
            put_after(kids[grp[-1][1]], run("]", rpr))
            if consecutive:
                # delete the middle fields and the separators, replace the first separator by an en dash
                first_end = grp[0][1]; last_beg = grp[-1][0]
                for e in kids[first_end + 1:last_beg]:
                    d = dele(); e.addprevious(d); p.remove(e)
                    for t in e.iter(w("t")): t.tag = w("delText")
                    for t in e.iter(w("instrText")): t.tag = w("delInstrText")
                    d.append(e)
                kids[first_end].addnext(ins(run("–", rpr)))
            groups += 1
        k = m + 1
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
print("groups:", groups)
