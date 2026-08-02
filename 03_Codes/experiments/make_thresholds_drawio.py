"""Generate a draw.io (.drawio) file holding the two threshold panels.

Everything is native draw.io geometry, so the figure can be edited by
hand: axes, ticks, grid, shaded inactive bands, the data polyline and
its markers, and all the labels.
"""
import os
import xml.sax.saxutils as sx

# PATHS ARE RESOLVED FROM THIS FILE, never from the machine that first
# ran it. The script was written in a sandbox whose absolute paths do
# not exist anywhere else, so it failed on the first line the moment it
# was run from the repository. Everything is now relative to the script
# itself, which is the convention the rest of the experiments follow.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

JTH = [(0.05, 100.0), (0.15, 89.5), (0.25, 67.3), (0.35, 42.0),
       (0.45, 30.8), (0.55, 29.5), (0.65, 29.5), (0.75, 29.5),
       (0.85, 29.5), (0.95, 29.5)]
ETA = [(0.05, 0.0), (0.15, 0.0), (0.25, 0.0), (0.35, 0.0), (0.45, 0.2),
       (0.55, 5.8), (0.65, 39.7), (0.75, 86.8), (0.85, 97.6),
       (0.95, 100.0)]

C_A = "#1F5F9E"
C_B = "#7A5AA8"
INK = "#33322E"
MUTED = "#77746C"
GRID = "#E3E1DC"
DEAD = "#F2EFE9"

PW, PH = 430, 300          # plot box
GAP = 110                  # gap between panels
X0A, Y0 = 90, 90           # top-left of panel a plot box
X0B = X0A + PW + GAP

cells = []
_id = [10]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def box(x, y, w, h, style, value=""):
    i = nid()
    cells.append(
        f'<mxCell id="{i}" value="{sx.escape(value)}" style="{style}" '
        f'vertex="1" parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
    return i


def line(x1, y1, x2, y2, style):
    i = nid()
    cells.append(
        f'<mxCell id="{i}" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
        f'</mxGeometry></mxCell>')
    return i


def polyline(pts, color):
    i = nid()
    mid = "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts[1:-1])
    cells.append(
        f'<mxCell id="{i}" style="endArrow=none;html=1;rounded=0;'
        f'strokeColor={color};strokeWidth=2.5;" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]:.1f}" y="{pts[-1][1]:.1f}" as="targetPoint"/>'
        f'<Array as="points">{mid}</Array></mxGeometry></mxCell>')


def text(x, y, w, h, s, size=11, color=INK, align="center", bold=0,
         valign="middle"):
    box(x, y, w, h,
        f"text;html=1;strokeColor=none;fillColor=none;align={align};"
        f"verticalAlign={valign};fontSize={size};fontColor={color};"
        f"fontStyle={bold};whiteSpace=wrap;", s)


def vtext(x, y, w, h, s, size=11, color=MUTED):
    box(x, y, w, h,
        f"text;html=1;strokeColor=none;fillColor=none;align=center;"
        f"verticalAlign=middle;fontSize={size};fontColor={color};"
        f"horizontal=0;whiteSpace=wrap;", s)


def panel(x0, data, color, marker, title, xlab, ylab, cfg, cfg_txt,
          dead_bands, notes, oper, oper_y=-4):
    def px(v):
        return x0 + v * PW
    def py(v):
        return Y0 + PH - (v / 110.0) * PH

    for lo, hi in dead_bands:
        box(px(lo), Y0, (hi - lo) * PW, PH,
            f"rounded=0;fillColor={DEAD};strokeColor=none;")
    for v in (0, 20, 40, 60, 80, 100):
        line(x0, py(v), x0 + PW, py(v),
             f"endArrow=none;html=1;strokeColor={GRID};strokeWidth=1;")
        text(x0 - 46, py(v) - 10, 38, 20, str(v), 11, MUTED, "right")
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        line(px(v), Y0 + PH, px(v), Y0 + PH + 6,
             f"endArrow=none;html=1;strokeColor={GRID};strokeWidth=1;")
        text(px(v) - 22, Y0 + PH + 8, 44, 18, f"{v:.1f}", 11, MUTED)
    line(x0, Y0 + PH, x0 + PW, Y0 + PH,
         f"endArrow=none;html=1;strokeColor={GRID};strokeWidth=1.5;")
    line(x0, Y0, x0, Y0 + PH,
         f"endArrow=none;html=1;strokeColor={GRID};strokeWidth=1.5;")

    line(px(cfg), Y0, px(cfg), Y0 + PH,
         f"endArrow=none;html=1;strokeColor={MUTED};strokeWidth=1;dashed=1;"
         "dashPattern=2 4;")
    text(px(cfg) + 6, Y0 - 4, 150, 18, cfg_txt, 11, MUTED, "left")

    pts = [(px(a), py(b)) for a, b in data]
    polyline(pts, color)
    for (a, b) in data:
        cx, cy = px(a), py(b)
        box(cx - 5.5, cy - 5.5, 11, 11,
            f"{marker};fillColor={color};strokeColor=#FFFFFF;strokeWidth=1.5;")

    lo, hi = oper
    line(px(lo), py(oper_y), px(hi), py(oper_y),
         f"endArrow=classic;startArrow=classic;html=1;strokeColor={color};"
         "strokeWidth=1.5;endSize=4;startSize=4;")
    _dy = -24 if oper_y < 50 else 4
    text(px(lo), py(oper_y) + _dy, (hi - lo) * PW, 18,
         f"Operable range {lo:.2f} to {hi:.2f}", 11, color)

    for nx, ny, nw, s in notes:
        text(px(nx), py(ny), nw, 60, s, 11, MUTED)

    text(x0 - 4, Y0 - 44, PW + 4, 24, title, 14, INK, "left", 1)
    text(x0, Y0 + PH + 32, PW, 22, xlab, 12, MUTED)
    vtext(x0 - 92, Y0, 24, PH, ylab, 12, MUTED)


panel(
    X0A, JTH, C_A, "ellipse",
    "(a) A lower satisficing bound makes the system adapt more often",
    "Satisficing bound J&lt;sub&gt;TH&lt;/sub&gt; &#8212; the forecast cost at which "
    "the standing decision is accepted",
    "Decision cycles in which the adaptation ladder is engaged (%)",
    0.35, "Configured value 0.35",
    [(0.45, 1.0)],
    [(0.60, 72, 190, "Above 0.45 the bound never binds:<br/>the relative margin "
                     "against no action<br/>is always the smaller of the two")],
    (0.05, 0.45))

panel(
    X0B, ETA, C_B, "triangle;direction=north",
    "(b) A higher quality gate derates offensive orders more often",
    "Fail-safe quality gate &#951; &#8212; the minimum decision quality an "
    "offensive order must reach",
    "Region decisions in which the graduated fail-safe derates the orders (%)",
    0.60, "Configured value 0.60",
    [(0.0, 0.55), (0.85, 1.0)],
    [(0.245, 72, 175, "Below 0.55 the decision quality<br/>never falls that "
                      "low, so the<br/>fail-safe never engages"),
     (0.905, 72, 150, "Above 0.85 every decision<br/>is derated: a permanent<br/>"
                      "reduction, not a fail-safe")],
    (0.55, 0.85), oper_y=108)

W = X0B + PW + 60
text(60, 22, W - 120, 44,
     "What the two acceptance thresholds govern. Ten worlds, four simultaneous "
     "ignitions, resource pool 0.25, all three adaptation stages active. "
     "A decision cycle is one pass of the decision loop; a region decision is "
     "one region within one cycle.",
     12, INK, "left")

xml = ('<mxfile host="app.diagrams.net">'
       '<diagram id="thresholds" name="thresholds">'
       f'<mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" '
       f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
       f'pageWidth="{W}" pageHeight="560" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
os.makedirs(FIGDIR, exist_ok=True)
open(os.path.join(FIGDIR, "fig_thresholds.drawio"), "w", encoding="utf8").write(xml)
print("written", len(xml), "bytes,", len(cells), "cells")
