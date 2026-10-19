"""v19 -> v19_v1: small tracked text fixes in Sections I and II plus US spelling in the body (not in reference titles)."""
import re, sys, copy, datetime
from lxml import etree
D=sys.argv[1]
W="http://schemas.openxmlformats.org/wordprocessingml/2006/main"; XML="http://www.w3.org/XML/1998/namespace"; NS={"w":W}
def w(t): return "{%s}%s"%(W,t)
AUTHOR="Claude"; DATE=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"); _id=[50000]
def nid(): _id[0]+=1; return str(_id[0])
def E(tag, attrib=None, *kids):
    e=etree.Element(w(tag))
    for k,v in (attrib or {}).items(): e.set(w(k),v)
    for c in kids: e.append(c)
    return e
def run(text, rpr):
    r=E("r")
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t=E("t"); t.text=text; t.set("{%s}space"%XML,"preserve"); r.append(t); return r
def drun(text, rpr):
    r=run(text,rpr); r.find(w("t")).tag=w("delText"); return r
REPL=[
 ("in 5 scenario families","in five scenario families"),
 ("hence, it has to adapt to the incident as it moves.","hence, the support has to adapt to the incident as it moves."),
 ("A rule base, a menu of tactics or an optimisation model fixes the orders before the incident so the system","A rule base, a menu of tactics, or an optimization model fixes the orders before the incident, so the system"),
 (" and the reasoning behind a recommendation is exposed only via post-hoc attribution",", and the reasoning behind a recommendation is exposed only through post-hoc attribution"),
 ("VODA reads 10 features extracted from observations of heterogeneous sources, meaningful to an expert, and not redundant.","VODA reads ten features extracted from observations of heterogeneous sources. Each is bounded, meaningful to an expert, and not redundant with the others."),
 ("VODA uses Takagi-Sugeno (TS) form since its results are constants","VODA uses the Takagi-Sugeno (TS) form, whose rule consequents are constants"),
 ("nearly 10M","nearly ten million"),
 ("neighbourhood","neighborhood"), ("neighbours","neighbors"), ("neighbour","neighbor"), ("grey bars","gray bars"),
]
doc=etree.parse(f"{D}/word/document.xml"); body=doc.getroot().find(w("body"))
def is_bib(p):
    s=p.find("w:pPr/w:pStyle",NS); return s is not None and s.get(w("val"))=="Bibliography"
# merge adjacent plain text runs with identical formatting so phrases split by editing can be matched
merged=0
for p in body.iter(w("p")):
    if is_bib(p): continue
    kids=list(p)
    i=0
    while i < len(kids)-1:
        a,b=kids[i],kids[i+1]
        def plain(r): return r.tag==w("r") and all(c.tag in (w("rPr"),w("t")) for c in r) and r.find(w("t")) is not None
        if plain(a) and plain(b):
            ra,rb=a.find(w("rPr")),b.find(w("rPr"))
            ka=etree.tostring(ra) if ra is not None else b""; kb=etree.tostring(rb) if rb is not None else b""
            if ka==kb:
                ta,tb=a.find(w("t")),b.find(w("t")); ta.text=(ta.text or "")+(tb.text or ""); ta.set("{%s}space"%XML,"preserve"); p.remove(b); kids=list(p); merged+=1; continue
        i+=1
n=0
for old,new in REPL:
    for t in list(body.iter(w("t"))):
        if not t.text or old not in t.text: continue
        r=t.getparent(); par=r.getparent(); p=r
        while p.tag!=w("p"): p=p.getparent()
        if is_bib(p) or par.tag==w("del"): continue
        if par.tag==w("ins"): t.text=t.text.replace(old,new); n+=1; continue
        rpr=r.find(w("rPr")); a,b=t.text.split(old,1); pos=par.index(r); par.remove(r); els=[]
        if a: els.append(run(a,rpr))
        els.append(E("del",{"id":nid(),"author":AUTHOR,"date":DATE},drun(old,rpr)))
        els.append(E("ins",{"id":nid(),"author":AUTHOR,"date":DATE},run(new,rpr)))
        if b: els.append(run(b,rpr))
        for k,el in enumerate(els): par.insert(pos+k,el)
        n+=1
# II-A: tie the cost sentence back to VODA (append after the final citation of that paragraph)
p=[p for p in body.iter(w("p")) if "".join(x.text or '' for x in p.iter(w("t"))).startswith("The hazard model and the cost measure")][0]
last_t=[t for t in p.iter(w("t")) if t.text and t.text.strip()][-1]
assert last_t.text.strip()=="." , last_t.text
r=last_t.getparent(); r.addnext(E("ins",{"id":nid(),"author":AUTHOR,"date":DATE},run(" The decision cost of VODA has the same two parts.", r.find(w("rPr")))))
doc.write(f"{D}/word/document.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
s=open(f"{D}/word/settings.xml",encoding="utf-8").read()
if "<w:trackRevisions" not in s: s=s.replace("<w:defaultTabStop","<w:trackRevisions/><w:defaultTabStop",1); open(f"{D}/word/settings.xml","w",encoding="utf-8").write(s)
print("merged runs:", merged, "| replacements:", n)
