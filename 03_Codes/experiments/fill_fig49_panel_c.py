"""Bring the text of Figure 4.9 into step with the drawing.

The drawing now carries a third panel, the clause actuator, added by
experiments/update_drawio_gates.py. The caption still promised two
panels and the paragraph that introduces the three kinds of object
pointed at (a) and (b) and left the third kind without a panel to look
at. Both are corrected here as tracked changes.

The image itself is NOT replaced. A drawing exported from draw.io is
produced by the tool that owns it, and this run has no draw.io to run,
so the picture is left for the user to re-export and drop in. The text
says what the picture must show, and this file records that dependency
rather than hiding it.

Usage: python experiments/fill_fig49_panel_c.py IN.docx OUT.docx
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import docx                                                # noqa: E402

from fill_ch4_mechanism import find, retext                # noqa: E402

#: (phrase, replacement, description, style the paragraph must have).
#: THE LIST OF FIGURES IS NOT EDITED. It is generated from the captions
#: by a field and rebuilt whenever the document updates, so a tracked
#: change written into it is a change the reader has to accept and Word
#: then throws away. Only the caption itself is touched.
EDITS = [
    ("Package objects: (a) macro intervention and (b) intermediate "
     "concept",
     "Package objects: (a) macro intervention, (b) intermediate "
     "concept and (c) clause actuator",
     "Figure 4.9 caption", "Caption"),
    # the sentence that introduces the third kind and shows no panel
    ("A third kind, the clause actuator, carries its own geometry "
     "rather than a weight vector",
     "As illustrated in Figure 4.9 (c), a third kind, the clause "
     "actuator, carries its own geometry rather than a weight vector",
     "the sentence introducing the clause actuator", None),
    # the panel reference for the intermediate concept says (b) already
    ("Every proposal, whether a rule or a package, enters the gate "
     "sequence as a whole.",
     "Every proposal, whether a rule or a package, enters the gate "
     "sequence as a whole, and Figure 4.8 shows that sequence with the "
     "three kinds of object entering it side by side.",
     "the pointer from the object kinds to the gate figure", None),
]


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    doc = docx.Document(inp)
    done = []
    for old, new, what, style in EDITS:
        hits = 0
        for p in doc.paragraphs:
            if style is not None and p.style.name != style:
                continue
            if old in "".join(t.text or "" for t in p._p.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml"
                    "/2006/main}t")):
                if retext(p, old, new):
                    hits += 1
        if hits:
            done.append(f"{what}: {hits} place(s)")
        else:
            print(f"  ! not matched, left unchanged: {what}")
    doc.save(outp)
    print("written:", outp)
    for d in done:
        print("  -", d)
    print()
    print("STILL TO DO BY HAND: export Figure 4.8 and Figure 4.9 from "
          "BlockDiagram_DISASTERAWARE.drawio and replace the two "
          "pictures in the document.")


if __name__ == "__main__":
    main()
