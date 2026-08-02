"""Correct the attention threshold in Chapter 4.

Three faults are fixed, all in the coordination part of Chapter 4.

  the dead branch    Equation 67 attends a region whose priority is
                     "at least a fraction of the leader OR above an
                     absolute threshold". With priorities and the
                     threshold both in the unit interval the second
                     arm can never bind, so the rule is relative and
                     only relative. The equation and the sentence that
                     paraphrases it are corrected to say so.
  the missing count  What the coordinator does is fund k of its N
                     regions. The threshold is the coordinate that
                     count is expressed in. k had no symbol, no
                     equation and no log entry, so the chapter could
                     not report the thing that acts.
  the free setting   The threshold was presented as a preference. It
                     is the shadow price of a capacity constraint: a
                     new paragraph derives it from the pool.

Usage: python experiments/fill_ch4_attention.py IN.docx OUT.docx
"""
from __future__ import annotations

import os
import sys

import docx
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from fill_sensitivity import DATE, _el, _nid, ins_run   # noqa: E402
import omml as M                                        # noqa: E402

AUTHOR = "Claude"
EQ_SHARE = "_RefEqShare"


def _ins():
    return dict(id=_nid(), author=AUTHOR, date=DATE)


def _mark_para_inserted(par):
    p = par._p
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = _el("w:pPr")
        p.insert(0, ppr)
    rpr = ppr.find(qn("w:rPr"))
    if rpr is None:
        rpr = _el("w:rPr")
        ppr.append(rpr)
    rpr.insert(0, _el("w:ins", **{"w:id": _nid(), "w:author": AUTHOR,
                                  "w:date": DATE}))


def _del_math(p):
    """Tracked-delete every formula and run of a paragraph.

    Word represents a deleted formula as the formula wrapped in
    <w:del>, the same way it treats a run, so the reader sees the old
    equation struck through beside the new one instead of finding it
    gone without trace.

    A DISPLAY EQUATION IS NOT A DIRECT CHILD. Word wraps a centred
    formula in <m:oMathPara>, so a sweep over the paragraph's direct
    children finds the inline ones and silently leaves the display ones
    standing. That is how the old share definition survived a delete
    that reported success.
    """
    for el in list(p.iter(qn("m:oMath"))) + list(p.findall(qn("w:r"))):
        parent = el.getparent()
        if parent is None or parent.tag == qn("w:del"):
            continue
        if el.tag == qn("m:oMath"):
            d = _el("w:del", **{"w:id": _nid(), "w:author": AUTHOR,
                                "w:date": DATE})
            parent.replace(el, d)
            d.append(el)
        else:
            d = _el("w:del", **{"w:id": _nid(), "w:author": AUTHOR,
                                "w:date": DATE})
            parent.replace(el, d)
            d.append(el)
            for t in el.findall(qn("w:t")):
                t.tag = qn("w:delText")


def _nfocus():
    """The chapter's symbol for the attended count."""
    return M.subsup("N", "Local_DSS", "focus")


def _pmax():
    """The leading priority of the cycle, in the chapter's notation."""
    return M.sub("p", "max")


def _p_rank(n):
    """The n-th largest priority. Used only where a rank is meant."""
    return M.sub("p", f"({n})")


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    doc = docx.Document(inp)
    from docx.text.paragraph import Paragraph

    body = doc.element.body

    # ---- locate the equation table that carries the share definition
    eq_tbl = None
    for tbl in body.iter(qn("w:tbl")):
        for bm in tbl.iter(qn("w:bookmarkStart")):
            if bm.get(qn("w:name")) == EQ_SHARE:
                eq_tbl = tbl
                break
        if eq_tbl is not None:
            break
    if eq_tbl is None:
        raise SystemExit(f"equation bookmark {EQ_SHARE} not found")

    # the formula lives in the first cell of the single row
    cell_p = None
    for tc in eq_tbl.iter(qn("w:tc")):
        if any(x.tag == qn("m:oMath") for x in tc.iter()):
            # EVERY paragraph of the cell, not the first: the author may
            # have added lines to the equation, and a delete that only
            # reaches the first one leaves a contradiction behind
            for par in tc.findall(qn("w:p")):
                _del_math(par)
                cell_p = par
            break
    if cell_p is None:
        raise SystemExit("the equation cell holds no formula")

    # ---- 1. the attended set, replacing the old share definition
    new_math = M.oMath([
        M.run("A"), M.txt("("), M.sub(M.run("τ"), "att"), M.txt(") = {"),
        M.run("i"), M.txt(" : "), M.sub("p", "i"), M.txt(" ≥ "),
        M.sub(M.run("τ"), "att"), _pmax(), M.txt("},   "), _nfocus(),
        M.txt(" = |"), M.run("A"), M.txt("("), M.sub(M.run("τ"), "att"),
        M.txt(")|")])
    wrap = _el("w:ins", **{"w:id": _nid(), "w:author": AUTHOR,
                           "w:date": DATE})
    wrap.append(new_math)
    cell_p.append(wrap)

    anchor = _Anchor(eq_tbl)

    def add_par(text, after=None):
        par = Paragraph(_el("w:p"), doc)
        par._p.append(ins_run(text))
        _mark_para_inserted(par)
        (after or anchor)._p.addnext(par._p)
        return par

    def add_mixed(head, tail, after=None):
        """A paragraph that mixes prose and inline formulas."""
        par = Paragraph(_el("w:p"), doc)
        for text, math in head:
            if text is not None:
                par._p.append(ins_run(text))
            if math is not None:
                w = _el("w:ins", **{"w:id": _nid(), "w:author": AUTHOR,
                                    "w:date": DATE})
                w.append(M.oMath([math]))
                par._p.append(w)
        par._p.append(ins_run(tail))
        _mark_para_inserted(par)
        (after or anchor)._p.addnext(par._p)
        return par

    def add_eq(parts, bookmark, shown, after):
        tb = M.eq_table(doc, parts, bookmark, shown, _nid(), ins=_ins())
        after._p.addnext(tb._tbl)
        return _Anchor(tb._tbl)

    # ---- 2. the share, as its own numbered equation
    a = add_par("The coordinator then assigns an attention share. An "
                "attended region keeps the whole of it; a monitored "
                "region keeps a floor plus a part that falls with its "
                "relative priority.")
    a = _Anchor(a._p)
    a = add_eq([M.sub("s", "i"), M.txt(" = 1 for "), M.run("i"),
                M.txt(" ∈ "), M.run("A"), M.txt("("), M.sub(M.run("τ"), "att"),
                M.txt("),   "), M.sub("s", "i"), M.txt(" = "),
                M.sub("s", "min"), M.txt(" + (1 − "),
                M.sub("s", "min"), M.txt(") "),
                M.frac(M.sub("p", "i"), _pmax()),
                M.txt(" otherwise")], "_RefEqAttShare", 68, a)

    a = _Anchor(add_par(
        "with a floor of 0.50. The share scales the offensive "
        "intensities, sets the regional acceptance threshold, and "
        "weights the region in the ranking that spends the shared "
        "budget. The attended count is the operational quantity, and "
        "the threshold is the coordinate in which it is expressed.",
        after=a)._p)

    # ---- 3. what the threshold is
    a = _Anchor(add_par(
        "The attention threshold is not a free setting. The shared "
        "pool is a capacity flow, and suppression returns on that flow "
        "are not linear. Control is produced only once the line "
        "construction rate exceeds the rate at which the fire "
        "perimeter grows, so effort placed below that rate buys almost "
        "no containment (Parks, 1964; Fried and Fried, 1996; Hirsch "
        "and Martell, 1996). The return of a region is therefore "
        "convex near the origin, and the allocation that maximises the "
        "total return under a fixed pool is a corner solution: a "
        "subset of regions funded at full strength rather than every "
        "region funded in part. The size of that subset follows from "
        "the pool, rounded down to a whole number of regions.",
        after=a)._p)

    # THE FLOOR BRACKETS ARE SPELLED OUT RATHER THAN DRAWN. The glyphs
    # do not survive every renderer, and an equation that reads "l B /
    # b J" in one viewer is worse than one that says what it means in
    # the sentence beside it.
    a = add_eq([_nfocus(), M.txt("* = min("), M.sub("n", "fire"),
                M.txt(", "), M.frac(M.run("B"), M.sub("b", "min")),
                M.txt(")")], "_RefEqKstar", 69, a)

    a = _Anchor(add_mixed(
        [("Here ", None), (None, M.run("B")),
         (" is the shared pool, ", None), (None, M.sub("b", "min")),
         (" the smallest force that changes the outcome in a region, "
          "and ", None), (None, M.sub("n", "fire")),
         (" the number of regions holding a fire. ", None)],
        "Ranking by an index and serving above "
        "a cut is the standard form of an optimal policy under a "
        "capacity constraint, and the cut is the Lagrange multiplier "
        "of that constraint (Cox and Smith, 1961; Whittle, 1988). The "
        "attention threshold is that cut, expressed relative to the "
        "leading priority.", after=a)._p)

    a = add_eq([M.sub(M.run("τ"), "att"), M.txt("* = "),
                M.frac(_p_rank("N*"), _pmax())], "_RefEqTauStar", 70, a)

    add_par(
        "Read this way the threshold carries a price rather than a "
        "preference. It states how much a region must be worth before "
        "it may draw on capacity that the leading region would "
        "otherwise use. The derivation predicts that the threshold "
        "falls as the pool grows, because a larger pool supports more "
        "simultaneous fronts at full strength, and that it approaches "
        "one when the pool supports a single front.", after=a)

    # ---- 4. the sentence that paraphrases the old rule
    fixed = 0
    for par in doc.paragraphs:
        if "or above an absolute threshold" not in par.text:
            continue
        for r in list(par._p.findall(qn("w:r"))):
            ts = r.findall(qn("w:t"))
            txt = "".join(t.text or "" for t in ts)
            if "or above an absolute threshold" not in txt:
                continue
            d = _el("w:del", **{"w:id": _nid(), "w:author": AUTHOR,
                                "w:date": DATE})
            par._p.replace(r, d)
            d.append(r)
            for t in ts:
                t.tag = qn("w:delText")
            d.addnext(ins_run(
                txt.replace(", or above an absolute threshold", "")))
            fixed += 1
            break
        par._p.append(ins_run(
            " The fraction is the attention threshold, and the number "
            "of attended regions k that it produces is the quantity "
            "that governs the outcome."))
        break

    doc.save(outp)
    print("written:", outp)
    print(f"   equation {EQ_SHARE} replaced by the attended set")
    print("   three equations inserted: share, k*, tau*")
    print(f"   absolute-threshold clause removed in {fixed} run(s)")


class _Anchor:
    def __init__(self, element):
        self._p = element


if __name__ == "__main__":
    main()
