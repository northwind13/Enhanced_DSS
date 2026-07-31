"""A small builder for Office Math (OMML) inside python-docx.

python-docx has no notion of an equation, so a script that writes into
a thesis either produces mathematics or produces the strings "J_c" and
"eta" and hopes the reader forgives it. It should not: a symbol typed
as text sits in the wrong font, breaks the moment the style changes,
and tells a reader that the passage was pasted rather than written.

Everything here builds the same OMML that the equation editor writes,
so an inserted formula is indistinguishable from one the author typed
and can be edited in Word afterwards.

    from omml import math_para, sub, run, frac, txt
    math_para(doc, [sub("J", "c"), txt(" = "), num(0.2354)])
"""
from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def _e(tag, **attrs):
    el = OxmlElement(tag)
    for k, v in attrs.items():
        el.set(qn(k), v)
    return el


def _mrpr(italic=True):
    """Math run properties. Variables are italic, operator words are
    upright, which is the same convention the rest of the thesis uses."""
    rpr = _e("m:rPr")
    if not italic:
        rpr.append(_e("m:sty", **{"m:val": "p"}))
    wrpr = _e("w:rPr")
    f = _e("w:rFonts", **{"w:ascii": "Cambria Math",
                          "w:hAnsi": "Cambria Math"})
    wrpr.append(f)
    return rpr, wrpr


def run(text, italic=True):
    """One math run."""
    r = _e("m:r")
    rpr, wrpr = _mrpr(italic)
    r.append(rpr)
    r.append(wrpr)
    t = _e("m:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def txt(text):
    """Upright text inside a formula: operators, words, punctuation."""
    return run(text, italic=False)


def num(value, places=None):
    """A number, upright, formatted the way the thesis prints them."""
    if isinstance(value, str):
        return txt(value)
    if places is None:
        s = f"{value:g}"
    else:
        s = f"{value:.{places}f}"
    return txt(s)


def sub(base, subscript, base_italic=True, sub_italic=False):
    """base with a subscript."""
    s = _e("m:sSub")
    pr = _e("m:sSubPr")
    ctrl = _e("m:ctrlPr")
    _, wrpr = _mrpr()
    ctrl.append(wrpr)
    pr.append(ctrl)
    s.append(pr)
    e = _e("m:e")
    e.append(base if not isinstance(base, str)
             else run(base, italic=base_italic))
    s.append(e)
    sb = _e("m:sub")
    sb.append(subscript if not isinstance(subscript, str)
              else run(subscript, italic=sub_italic))
    s.append(sb)
    return s


def acc(base, char="̂"):
    """An accent over base; the default is a circumflex, the hat the
    chapter puts on a forecast quantity."""
    a = _e("m:acc")
    pr = _e("m:accPr")
    pr.append(_e("m:chr", **{"m:val": char}))
    ctrl = _e("m:ctrlPr")
    _, wrpr = _mrpr()
    ctrl.append(wrpr)
    pr.append(ctrl)
    a.append(pr)
    e = _e("m:e")
    e.append(base if not isinstance(base, str) else run(base))
    a.append(e)
    return a


def subsup(base, subscript, superscript):
    """base with a subscript and a superscript, as in a_n^eff."""
    s = _e("m:sSubSup")
    pr = _e("m:sSubSupPr")
    ctrl = _e("m:ctrlPr")
    _, wrpr = _mrpr()
    ctrl.append(wrpr)
    pr.append(ctrl)
    s.append(pr)
    for tag, val, ital in (("m:e", base, True),
                           ("m:sub", subscript, False),
                           ("m:sup", superscript, False)):
        el = _e(tag)
        el.append(val if not isinstance(val, str)
                  else run(val, italic=ital))
        s.append(el)
    return s


def frac(numer, denom):
    """A stacked fraction; numer and denom are lists of math elements."""
    f = _e("m:f")
    pr = _e("m:fPr")
    ctrl = _e("m:ctrlPr")
    _, wrpr = _mrpr()
    ctrl.append(wrpr)
    pr.append(ctrl)
    f.append(pr)
    n = _e("m:num")
    for x in (numer if isinstance(numer, (list, tuple)) else [numer]):
        n.append(x)
    f.append(n)
    d = _e("m:den")
    for x in (denom if isinstance(denom, (list, tuple)) else [denom]):
        d.append(x)
    f.append(d)
    return f


def oMath(parts):
    """An inline formula holding the given parts.

    A part may itself be a list, because a compound symbol such as a
    hatted J with an argument is several elements that a caller thinks
    of as one thing; flattening here keeps that convenience out of
    every call site.
    """
    m = _e("m:oMath")
    for p in parts:
        for q in (p if isinstance(p, (list, tuple)) else [p]):
            m.append(q)
    return m


def eq_table(doc, parts, bookmark, shown, bm_id, ins=None):
    """A numbered equation in the layout the thesis already uses.

    Every equation in this document sits in a borderless two column
    table: the formula on the left in the ParEq style, the number on
    the right as a SEQ field inside a bookmark, so that a REF field
    elsewhere can point at it and both renumber together. Writing a
    centred paragraph instead would look almost right and then fail the
    moment an equation is inserted ahead of it.
    """
    from docx.table import Table
    tbl = _e("w:tbl")
    pr = _e("w:tblPr")
    pr.append(_e("w:tblStyle", **{"w:val": "TableGrid"}))
    pr.append(_e("w:tblW", **{"w:w": "5000", "w:type": "pct"}))
    borders = _e("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_e(f"w:{side}", **{"w:val": "none", "w:sz": "0",
                                          "w:space": "0",
                                          "w:color": "auto"}))
    pr.append(borders)
    tbl.append(pr)
    grid = _e("w:tblGrid")
    for w in ("7000", "1047"):
        grid.append(_e("w:gridCol", **{"w:w": w}))
    tbl.append(grid)

    tr = _e("w:tr")
    for width, body in (("4350", "math"), ("650", "number")):
        tc = _e("w:tc")
        tcpr = _e("w:tcPr")
        tcpr.append(_e("w:tcW", **{"w:w": width, "w:type": "pct"}))
        tcpr.append(_e("w:vAlign", **{"w:val": "center"}))
        tc.append(tcpr)
        p = _e("w:p")
        ppr = _e("w:pPr")
        ppr.append(_e("w:pStyle", **{"w:val": "ParEq"}))
        p.append(ppr)
        if body == "math":
            m = oMath(parts)
            if ins is not None:
                w = _e("w:ins", **{"w:id": ins["id"],
                                   "w:author": ins["author"],
                                   "w:date": ins["date"]})
                w.append(m)
                p.append(w)
            else:
                p.append(m)
        else:
            p.append(_e("w:bookmarkStart", **{"w:id": str(bm_id),
                                              "w:name": bookmark}))
            runs = [_text_run("( ")] + _seq_runs(shown) + [_text_run(" )")]
            if ins is not None:
                w = _e("w:ins", **{"w:id": ins["id"] + "9",
                                   "w:author": ins["author"],
                                   "w:date": ins["date"]})
                for r in runs:
                    w.append(r)
                p.append(w)
            else:
                for r in runs:
                    p.append(r)
            p.append(_e("w:bookmarkEnd", **{"w:id": str(bm_id)}))
        tc.append(p)
        tr.append(tc)
    tbl.append(tr)
    return Table(tbl, doc)


def _text_run(text):
    r = _e("w:r")
    t = _e("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    return r


def _fld(instr, shown):
    out = []
    r = _e("w:r")
    r.append(_e("w:fldChar", **{"w:fldCharType": "begin"}))
    out.append(r)
    r = _e("w:r")
    t = _e("w:instrText")
    t.text = instr
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    out.append(r)
    r = _e("w:r")
    r.append(_e("w:fldChar", **{"w:fldCharType": "separate"}))
    out.append(r)
    r = _e("w:r")
    rpr = _e("w:rPr")
    rpr.append(_e("w:noProof"))
    r.append(rpr)
    t = _e("w:t")
    t.text = str(shown)
    r.append(t)
    out.append(r)
    r = _e("w:r")
    r.append(_e("w:fldChar", **{"w:fldCharType": "end"}))
    out.append(r)
    return out


def _seq_runs(shown):
    return _fld(" SEQ ( \\* ARABIC ", shown)


def ref_runs(bookmark, shown):
    """An inline cross reference to a numbered equation."""
    return _fld(f" REF {bookmark} \\h \\* MERGEFORMAT ", f"( {shown} )")


def math_para(doc, parts, ins=None, style=None):
    """A paragraph holding one centred formula, returned unattached.

    The formula is placed inline in a centred paragraph rather than in
    an oMathPara block: a tracked insertion may wrap runs, and wrapping
    a block-level math paragraph is not something Word will read back.
    """
    from docx.text.paragraph import Paragraph
    par = Paragraph(_e("w:p"), doc)
    ppr = _e("w:pPr")
    ppr.append(_e("w:jc", **{"w:val": "center"}))
    par._p.append(ppr)
    body = oMath(parts)
    if ins is not None:
        wrap = _e("w:ins", **{"w:id": ins["id"], "w:author": ins["author"],
                              "w:date": ins["date"]})
        wrap.append(body)
        par._p.append(wrap)
    else:
        par._p.append(body)
    if style:
        try:
            par.style = doc.styles[style]
        except KeyError:
            pass
    return par
