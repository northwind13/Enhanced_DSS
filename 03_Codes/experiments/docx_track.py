"""Tracked-changes primitives for the scripts that write into the thesis.

Word records an edit rather than applying it: an inserted run is wrapped
in <w:ins>, a deleted one in <w:del> with its <w:t> retagged <w:delText>,
and a whole table row is marked through <w:trPr>. A script that writes
into a chapter must produce that form, otherwise the supervisor receives
a document in which the changes cannot be told from the original text and
cannot be accepted or rejected one by one.

These four helpers are the whole of it. They used to live inside the
Section 5.5 writer, which meant the Chapter 4 writers imported a module
they had nothing to do with in order to reach them, and could not
outlive it.

  _nid          a revision id that is unique within one run
  _el           an OxmlElement with namespaced attributes already set
  ins_run       a run of text marked as an insertion
  _ins_row      marks an existing table row as inserted
  del_runs_of   marks every live run of a paragraph as deleted
  fill_cell     replaces the contents of a table cell, both marks kept

AUTHOR and DATE are shared on purpose: every mark a run leaves carries
one author and one timestamp, so Word groups them as a single review
pass instead of scattering them.
"""
from __future__ import annotations

from datetime import datetime, timezone

from docx.oxml.ns import qn

AUTHOR = "Claude"
DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

#: Revision ids start well above the ones Word tends to allocate, so a
#: document that already carries review marks does not collide with the
#: ones written here.
_ID = [9000]


def _nid():
    """The next revision id, as the string the attribute expects."""
    _ID[0] += 1
    return str(_ID[0])


def _el(tag, **attrs):
    """An OOXML element, with every attribute name resolved to its URI.

    Attributes are passed in prefixed form, w:id rather than id, because
    that is how they read in the specification and in the XML itself.
    """
    from docx.oxml import OxmlElement
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), v)
    return e


def ins_run(text, bold=False, italic=False):
    """A run of text wrapped as a tracked insertion.

    Whitespace is preserved explicitly. Without xml:space Word collapses
    a leading or trailing space, which silently welds an inserted
    sentence onto the word before it.
    """
    ins = _el("w:ins", **{"w:id": _nid(), "w:author": AUTHOR,
                          "w:date": DATE})
    r = _el("w:r")
    if bold or italic:
        rpr = _el("w:rPr")
        if bold:
            rpr.append(_el("w:b"))
        if italic:
            rpr.append(_el("w:i"))
        r.append(rpr)
    t = _el("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    ins.append(r)
    return ins


def del_runs_of(p):
    """Wrap every live run of a paragraph in <w:del>, insertions included.

    A run that is itself already inside a <w:del> is left alone, and a
    run inside a <w:ins> is still live text and is deleted like any
    other. Skipping the second case is how a script ends up striking
    through the original wording and leaving an earlier draft of its own
    standing beside it.
    """
    todo = [r for r in p.iter(qn("w:r"))
            if r.getparent().tag != qn("w:del")]
    for r in todo:
        parent = r.getparent()
        d = _el("w:del", **{"w:id": _nid(), "w:author": AUTHOR,
                            "w:date": DATE})
        parent.replace(r, d)
        d.append(r)
        for t in r.findall(qn("w:t")):
            t.tag = qn("w:delText")


def fill_cell(cell, value):
    """Replace a cell's text, showing both the old and the new wording."""
    p = cell.paragraphs[0]._p
    del_runs_of(p)
    p.append(ins_run(str(value)))


def _ins_row(row):
    """Mark an existing table row as inserted.

    The mark belongs on the row properties, not on the runs inside it. A
    row whose cells are all marked inserted but whose <w:trPr> is not
    leaves the row itself in the document when the changes are rejected,
    so the table comes back one empty row longer than it started.
    """
    tr = row._tr
    trpr = tr.find(qn("w:trPr"))
    if trpr is None:
        trpr = _el("w:trPr")
        tr.insert(0, trpr)
    trpr.append(_el("w:ins", **{"w:id": _nid(), "w:author": AUTHOR,
                                "w:date": DATE}))
