"""Replace one paragraph (wholly inserted by this author, or plain) with new text containing {m:...} inline math and {c7;27} citations.
Text runs go into w:ins; math objects sit outside (OMML cannot be tracked). usage: python3 mathpar.py <dir> <prefix> <textfile>"""
import re, copy, datetime, sys
from lxml import etree
W="http://schemas.openxmlformats.org/wordprocessingml/2006/main"; M="http://schemas.openxmlformats.org/officeDocument/2006/math"; XML="http://www.w3.org/XML/1998/namespace"
def w(t): return "{%s}%s"%(W,t)
def m(t): return "{%s}%s"%(M,t)
AUTHOR="Claude"; DATE=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id=[41000]
def nid(): _id[0]+=1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e=etree.Element(w(tag))
    for k,v in (attrib or {}).items(): e.set(w(k),v)
    for c in kids: e.append(c)
    return e
def run(text, rpr=None):
    r=E("r")
    if rpr is not None and len(rpr): r.append(copy.deepcopy(rpr))
    t=E("t"); t.text=text; t.set("{%s}space"%XML,"preserve"); r.append(t); return r
def drun(text, rpr=None):
    r=run(text,rpr); r.find(w("t")).tag=w("delText"); return r
def ins(*kids): return E("ins",{"id":nid(),"author":AUTHOR,"date":DATE},*kids)
def dele(*kids): return E("del",{"id":nid(),"author":AUTHOR,"date":DATE},*kids)
def mt(text):
    t=etree.Element(m("t")); t.text=text; t.set("{%s}space"%XML,"preserve"); r=etree.Element(m("r")); r.append(t); return r
def math(expr):
    """'a^b rest' -> oMath(sSup(a,b) + rest); 'a_b' -> sSub; otherwise plain"""
    om=etree.Element(m("oMath"))
    mm=re.match(r"^(\w+)\^(\w+)(.*)$", expr)
    ms=re.match(r"^(\w+)_(\w+)(.*)$", expr)
    if mm:
        ss=etree.SubElement(om,m("sSup")); etree.SubElement(ss,m("e")).append(mt(mm.group(1))); etree.SubElement(ss,m("sup")).append(mt(mm.group(2)))
        if mm.group(3): om.append(mt(mm.group(3)))
    elif ms:
        ss=etree.SubElement(om,m("sSub")); etree.SubElement(ss,m("e")).append(mt(ms.group(1))); etree.SubElement(ss,m("sub")).append(mt(ms.group(2)))
        if ms.group(3): om.append(mt(ms.group(3)))
    else: om.append(mt(expr))
    return om
D, PREFIX, TXT = sys.argv[1], sys.argv[2], open(sys.argv[3], encoding="utf-8").read().strip()
doc=etree.parse(f"{D}/word/document.xml"); body=doc.getroot().find(w('body'))
NUM2TAG={}
for it in body.iter(w('instrText')):
    if 'CITATION' in (it.text or ''):
        tags=[a or b for a,b in re.findall(r"CITATION (\S+)|\\m (\S+)", it.text)]
        for sib in it.getparent().itersiblings():
            t=sib.find(w('t'))
            if t is not None: nums=[int(x) for x in re.findall(r'\d+',t.text)]; break
        for n,tg in zip(nums,tags): NUM2TAG[n]=tg
def cite(nums):
    tags=[NUM2TAG[n] for n in nums]; instr=f"CITATION {tags[0]} \\l 1033"+"".join(f" \\m {t}" for t in tags[1:])
    runs=[]
    for kind in ("begin","instr","separate"):
        r=E("r")
        if kind=="instr": x=E("instrText"); x.text=f" {instr} "; x.set("{%s}space"%XML,"preserve"); r.append(x)
        else:
            fc=E("fldChar",{"fldCharType":kind})
            if kind=="begin": fc.set(w("fldLock"),"1")
            r.append(fc)
        runs.append(r)
    rp=E("rPr"); rp.append(E("noProof")); runs.append(run("["+", ".join(map(str,nums))+"]", rp))
    r=E("r"); r.append(E("fldChar",{"fldCharType":"end"})); runs.append(r)
    return E("sdt",None,E("sdtPr",None,E("id",{"val":str(-(1960000000-int(nid())))}),E("citation")),E("sdtContent",None,*runs))
def ptext(p): return "".join(x.text or '' for x in p.iter(w('t')))
hits=[p for p in body.iter(w('p')) if ptext(p).startswith(PREFIX)]; assert len(hits)==1, (PREFIX, len(hits)); p=hits[0]
# delete the old content (tracked); content already inside w:ins by this author is simply dropped
for c in [c for c in p if c.tag!=w('pPr')]:
    if c.tag==w('ins'): p.remove(c)
    elif c.tag==w('del'): continue
    elif c.tag in (w('r'), w('sdt')):
        d=dele(); p.replace(c,d); cc=copy.deepcopy(c)
        for t in cc.iter(w('t')): t.tag=w('delText')
        for t in cc.iter(w('instrText')): t.tag=w('delInstrText')
        d.append(cc)
    elif c.tag==m('oMath') or c.tag==m('oMathPara'): p.remove(c)   # untracked math removal
    else: p.remove(c)
cur=[]
def flush():
    global cur
    if cur: p.append(ins(*cur)); cur=[]
for seg in re.split(r'(\{[^}]+\})', TXT):
    if not seg: continue
    if seg.startswith('{m:'): flush(); p.append(math(seg[3:-1]))
    elif seg.startswith('{c'): cur.append(cite([int(x) for x in seg[2:-1].split(';')]))
    else: cur.append(run(seg))
flush()
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding='UTF-8', standalone=True)
print("rewritten:", PREFIX)
