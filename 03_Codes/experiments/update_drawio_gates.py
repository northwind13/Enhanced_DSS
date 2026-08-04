"""Bring Figures 4.8 and 4.9 of the block diagram up to the text.

WHAT WAS OUT OF STEP. Chapter 4 walks nine gates, G1, G2, G2b, G2c, G2d,
G3, G4, G5 and G5b, and says a package may carry three kinds of object.
Figure 4.8 drew seven gates and two kinds of object, and its G5 box had
no rejection edge, so the drawing said a package always passes G5. Figure
4.9 drew the macro intervention and the intermediate concept and left out
the clause actuator, which is the only object that carries geometry and
the one Section 5.5.3 now reports on.

WHAT THIS DOES
  Figure 4.8  inserts G2c (relevance) and G2d (pool economy) in their
              place in the chain, moves G3, G4, G5 and G5b down to make
              room, gives G5, G2c and G2d their rejection edges, draws
              the admission a plain rule takes after G4, and adds the
              clause actuator to the row of package objects
  Figure 4.9  adds panel (c), the clause actuator, with a definition
              taken from the campaign ledger

THE FILE IS EDITED IN PLACE ON A COPY. The original is left beside it
with a .bak suffix, because a hand-drawn diagram is not reproducible and
a script that touches one has to be reversible.

    python experiments/update_drawio_gates.py [PATH.drawio]
"""
from __future__ import annotations

import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                       "01_Thesis", "BlockDiagram_DISASTERAWARE.drawio")

F48 = "F48gatesCLR"
F49 = "F49pkgobj"
P = "8fn3158Nofi3qeclke3r-"          # the id prefix of Figure 4.8
Q = "bxTQW_NBIBj9sRsA5LDN-"          # the id prefix of Figure 4.9

BOX = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fontSize=14;"
       "align=center;verticalAlign=middle;spacingLeft=6;spacingRight=6;")
PASS_E = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
          "strokeColor=#555555;strokeWidth=1.6;dashed=0;endArrow=block;"
          "fontSize=14;")
REJ_E = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
         "strokeColor=#B03A2E;strokeWidth=1.6;dashed=0;endArrow=block;"
         "exitX=1;exitY=0.5;exitDx=0;exitDy=0;fontSize=14;")
OK_E = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;"
        "strokeColor=#2E7D32;strokeWidth=1.6;dashed=0;endArrow=block;"
        "exitX=0;exitY=0.5;exitDx=0;exitDy=0;fontSize=14;")
LBL = ("edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0;"
       "points=[];fontSize=14;")

#: THE GATE COLUMN AFTER THE INSERTION. Two boxes join the chain, so
#: everything below G2b moves by two slots of 83 points.
MOVE = {f"{P}87": 792, f"{P}88": 875, f"{P}89": 958, f"{P}200": 1041}
#: and the two side boxes re-centre on the longer column
SIDE = {f"{P}90": 676, f"{P}91": 1000}
#: three package objects share the row, so all three narrow to 230
ROW = {f"{P}82": 2791, f"{P}83": 3047}


def vertex(cid, value, x, y, w, h, style=BOX):
    return (f'<mxCell id="{cid}" value="{value}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" '
            f'width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(cid, src, tgt, style=PASS_E, label=None, extra=""):
    out = (f'<mxCell id="{cid}" edge="1" parent="1" source="{src}" '
           f'target="{tgt}" style="{style}{extra}" value="">'
           f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    if label:
        out += (f'<mxCell id="{cid}L" value="{label}" style="{LBL}" '
                f'vertex="1" connectable="0" parent="{cid}">'
                f'<mxGeometry relative="1" as="geometry">'
                f'<mxPoint as="offset"/></mxGeometry></mxCell>')
    return out


def cell(body, cid):
    m = re.search(r'<mxCell\b[^>]*id="%s".*?(?:/>|</mxCell>)'
                  % re.escape(cid), body, re.S)
    if m is None:
        raise SystemExit(f"cell not found: {cid}")
    return m


def set_geo(body, cid, **kw):
    """Rewrite named attributes of one cell's geometry."""
    m = cell(body, cid)
    blk = m.group(0)
    g = re.search(r'<mxGeometry\b[^>]*/>', blk)
    if g is None:
        raise SystemExit(f"{cid} has no simple geometry")
    geo = g.group(0)
    for k, v in kw.items():
        if re.search(r'\b%s="[^"]*"' % k, geo):
            geo = re.sub(r'\b%s="[^"]*"' % k, f'{k}="{v}"', geo)
        else:
            geo = geo.replace("/>", f' {k}="{v}"/>')
    return body[:m.start()] + blk.replace(g.group(0), geo) + body[m.end():]


def repoint(body, cid, **kw):
    m = cell(body, cid)
    blk = m.group(0)
    for k, v in kw.items():
        blk = re.sub(r'\b%s="[^"]*"' % k, f'{k}="{v}"', blk)
    return body[:m.start()] + blk + body[m.end():]


def fix48(body):
    n = []
    for cid, y in MOVE.items():
        body = set_geo(body, cid, y=y)
    n.append("moved G3, G4, G5 and G5b down two slots")
    for cid, y in SIDE.items():
        body = set_geo(body, cid, y=y)
    n.append("re-centred the rejected and admitted boxes")

    # ---- the row of package objects ---------------------------------
    for cid, x in ROW.items():
        body = set_geo(body, cid, x=x, width=230)
    clause = f"{P}210"
    body = body.replace(
        "</root>",
        vertex(clause,
               "&lt;span&gt;&lt;b&gt;Clause actuator&lt;/b&gt;&lt;/span&gt;"
               "&lt;br&gt;&lt;span&gt;up to three verified effects, each "
               "on a sector over a range of cells&lt;/span&gt;",
               3303, 245, 230, 54)
        + edge(f"{P}211", f"{P}81", clause, PASS_E,
               "&lt;span style=&quot;color: rgb(85, 85, 85);&quot;&gt;"
               "new&lt;/span&gt;&lt;div&gt;&lt;span style=&quot;color: "
               "rgb(85, 85, 85);&quot;&gt;actuator&lt;/span&gt;&lt;/div&gt;")
        + edge(f"{P}212", clause, f"{P}84", PASS_E, None,
               "entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
               "exitX=0.5;exitY=1;exitDx=0;exitDy=0;")
        + "</root>")
    n.append("added the clause actuator to the package objects")

    # ---- the two gates the figure was missing -----------------------
    g2c, g2d = f"{P}202", f"{P}203"
    body = body.replace(
        "</root>",
        vertex(g2c,
               "&lt;span&gt;&lt;b&gt;G2c Relevance: &lt;/b&gt;every "
               "antecedent of the proposal fires in the present "
               "situation&lt;/span&gt;", 2970, 626, 350, 50)
        + vertex(g2d,
                 "&lt;span&gt;&lt;b&gt;G2d Pool economy: &lt;/b&gt;a "
                 "proposal that adds physical work is refused when the "
                 "standing orders already claim the funded budget"
                 "&lt;/span&gt;", 2970, 709, 350, 60)
        + edge(f"{P}204", f"{P}86", g2c, PASS_E, "pass")
        + edge(f"{P}205", g2c, g2d, PASS_E, "pass")
        + edge(f"{P}206", g2c, f"{P}90", REJ_E)
        + edge(f"{P}207", g2d, f"{P}90", REJ_E)
        + "</root>")
    # G2b used to hand straight to G3; it now hands to G2c
    body = repoint(body, f"{P}108", source=g2d)
    n.append("inserted G2c and G2d and rewired the pass chain")

    # ---- the rejection G5 never had, and the plain-rule admission ---
    body = body.replace(
        "</root>",
        edge(f"{P}208", f"{P}89", f"{P}90", REJ_E)
        + edge(f"{P}209", f"{P}88", f"{P}91", OK_E,
               "&lt;span style=&quot;color: rgb(46, 125, 50);&quot;&gt;"
               "a plain rule is admitted here&lt;/span&gt;")
        + "</root>")
    n.append("gave G5 its rejection edge and drew the plain-rule "
             "admission after G4")
    return body, n


def fix49(body):
    y0 = 720
    add = [
        # the panel title copies the style of (a) and (b) exactly, so
        # the three headings of the figure are set the same way
        vertex(f"{Q}40",
               "(c)  Clause actuator: a new intervention that carries "
               "its own geometry", 1268, y0, 607, 24,
               "text;html=1;fontSize=12;fontColor=#333333;align=left;"
               "verticalAlign=middle;fontStyle=1;"),
        vertex(f"{Q}41",
               "&lt;span&gt;&lt;b&gt;New clause actuator&lt;/b&gt;"
               "&lt;/span&gt;&lt;br&gt;&lt;span&gt;lakeside_capacity_"
               "bridge&lt;/span&gt;", 1288, y0 + 70, 250, 54),
        vertex(f"{Q}42", "draft on at_fire&lt;br&gt;cells [0, 3], amount "
                         "1.0", 1655, y0 + 25, 220, 44),
        vertex(f"{Q}43", "draft on head&lt;br&gt;cells [0, 8], amount "
                         "1.0", 1655, y0 + 79, 220, 44),
        vertex(f"{Q}44", "evacuate on populated&lt;br&gt;cells [0, 6], "
                         "amount 1.0", 1655, y0 + 133, 220, 44),
        edge(f"{Q}45", f"{Q}41", f"{Q}42", PASS_E, "compiles into"),
        edge(f"{Q}46", f"{Q}41", f"{Q}43", PASS_E,
             "(&amp;#8804; 3 clauses)"),
        edge(f"{Q}47", f"{Q}41", f"{Q}44", PASS_E),
        vertex(f"{Q}48",
               "each clause is one verified effect, on one sector, over "
               "a range of cells from the front", 1288, y0 + 135, 250,
               54),
        vertex(f"{Q}49",
               "Rule that orders the actuator: IF fire threat level is "
               "VH AND evacuation pressure is VH AND suppression "
               "feasibility is L THEN lakeside_capacity_bridge 1.00",
               1287, y0 + 205, 588, 50),
        vertex(f"{Q}50",
               "G5 asks whether the growth pays. G5b asks whether the "
               "payoff belongs to the new object, by repeating both "
               "rollouts with it struck out.", 1287, y0 + 268, 588, 50),
        edge(f"{Q}51", f"{Q}41", f"{Q}49", PASS_E, "ordered by"),
    ]
    body = body.replace("</root>", "".join(add) + "</root>")
    return body, ["added panel (c), the clause actuator"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    src = open(path, encoding="utf-8").read()
    shutil.copyfile(path, path + ".bak")

    notes = []
    for did, fixer in ((F48, fix48), (F49, fix49)):
        m = re.search(r'(<diagram[^>]*id="%s"[^>]*>)(.*?)(</diagram>)'
                      % did, src, re.S)
        if m is None:
            raise SystemExit(f"page not found: {did}")
        body, n = fixer(m.group(2))
        src = src[:m.start()] + m.group(1) + body + m.group(3) \
            + src[m.end():]
        notes += [f"{did}: {x}" for x in n]

    open(path, "w", encoding="utf-8").write(src)
    print("written:", path)
    print("backup :", path + ".bak")
    for x in notes:
        print("  -", x)


if __name__ == "__main__":
    main()
