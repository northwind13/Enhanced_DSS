"""draw.io figure for Section 5.5.4: decision latency and computational cost.

Two panels on a shared logarithmic time axis, so the configured decision
interval and the measured cycle time appear in the same picture.

  (a) cycle time against the number of local regions
  (b) cycle time against the size of the grid
"""
import os
import math
import xml.sax.saxutils as sx

# PATHS ARE RESOLVED FROM THIS FILE, never from the machine that first
# ran it. The script was written in a sandbox whose absolute paths do
# not exist anywhere else, so it failed on the first line the moment it
# was run from the repository. Everything is now relative to the script
# itself, which is the convention the rest of the experiments follow.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FIGDIR = os.path.join(HERE, "..", "..", "01_Thesis", "figures")

# median over three worlds, with the range across those worlds
REG = [(1, 249.3, 246, 895), (2, 257.4, 252, 689), (4, 265.7, 258, 744),
       (8, 277.3, 274, 638), (16, 318.9, 306, 883)]
GRID = [("80 &#215; 60<br/>4 800 cells", 264.3, 257, 698),
        ("120 &#215; 90<br/>10 800 cells", 461.3, 368, 1154),
        ("160 &#215; 120<br/>19 200 cells", 1823.2, 759, 1905)]
INTERVAL = 720000.0          # 12 minutes in milliseconds

C_A = "#1F5F9E"
C_B = "#7A5AA8"
C_REF = "#B3492D"
INK, MUTED, GRIDC, BAND = "#33322E", "#77746C", "#E3E1DC", "#DCE6F1"

PW, PH = 400, 330
GAP = 130
X0A, Y0 = 110, 100
X0B = X0A + PW + GAP
LO, HI = 100.0, 2_000_000.0      # y range in ms

cells = []
_id = [10]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def box(x, y, w, h, style, value=""):
    cells.append(
        f'<mxCell id="{nid()}" value="{sx.escape(value)}" style="{style}" '
        f'vertex="1" parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def line(x1, y1, x2, y2, style):
    cells.append(
        f'<mxCell id="{nid()}" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
        f'</mxGeometry></mxCell>')


def polyline(pts, color):
    mid = "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts[1:-1])
    cells.append(
        f'<mxCell id="{nid()}" style="endArrow=none;html=1;rounded=0;'
        f'strokeColor={color};strokeWidth=2.5;" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{pts[-1][0]:.1f}" y="{pts[-1][1]:.1f}" as="targetPoint"/>'
        f'<Array as="points">{mid}</Array></mxGeometry></mxCell>')


def text(x, y, w, h, s, size=11, color=INK, align="center", bold=0):
    box(x, y, w, h,
        f"text;html=1;strokeColor=none;fillColor=none;align={align};"
        f"verticalAlign=middle;fontSize={size};fontColor={color};"
        f"fontStyle={bold};whiteSpace=wrap;", s)


def vtext(x, y, w, h, s, size=12, color=MUTED):
    box(x, y, w, h,
        f"text;html=1;strokeColor=none;fillColor=none;align=center;"
        f"verticalAlign=middle;fontSize={size};fontColor={color};"
        f"horizontal=0;whiteSpace=wrap;", s)


def py(v):
    """log scale, ms to pixels"""
    f = (math.log10(v) - math.log10(LO)) / (math.log10(HI) - math.log10(LO))
    return Y0 + PH - f * PH


def panel(x0, data, color, marker, title, xlab, ylab_on):
    n = len(data)
    step = PW / (n + 1)

    def px(i):
        return x0 + (i + 1) * step

    for v, lab in ((100, "100 ms"), (1000, "1 s"), (10000, "10 s"),
                   (100000, "100 s"), (1000000, "1000 s")):
        line(x0, py(v), x0 + PW, py(v),
             f"endArrow=none;html=1;strokeColor={GRIDC};strokeWidth=1;")
        if ylab_on:
            text(x0 - 62, py(v) - 10, 54, 20, lab, 11, MUTED, "right")
    line(x0, Y0, x0, Y0 + PH,
         f"endArrow=none;html=1;strokeColor={GRIDC};strokeWidth=1.5;")
    line(x0, Y0 + PH, x0 + PW, Y0 + PH,
         f"endArrow=none;html=1;strokeColor={GRIDC};strokeWidth=1.5;")

    # the budget
    line(x0, py(INTERVAL), x0 + PW, py(INTERVAL),
         f"endArrow=none;html=1;strokeColor={C_REF};strokeWidth=2;dashed=1;"
         "dashPattern=6 3;")
    text(x0 + 6, py(INTERVAL) - 22, 330, 18,
         "Configured decision interval, 12 minutes", 11, C_REF, "left")

    if ylab_on:
        line(x0 + PW * 0.60, py(INTERVAL) * 1.0 + 6, x0 + PW * 0.60,
             py(320) - 8,
             f"endArrow=classic;startArrow=classic;html=1;strokeColor={MUTED};"
             "strokeWidth=1;endSize=4;startSize=4;")
        text(x0 + PW * 0.62, py(6000) - 24, 150, 48,
             "the median cycle is about<br/>2 600 times shorter than<br/>"
             "the interval it is given", 11, MUTED, "left")

    # spread across the three worlds
    for i, row in enumerate(data):
        _, med, lo, hi = row
        box(px(i) - 7, py(hi), 14, py(lo) - py(hi),
            f"rounded=0;fillColor={BAND};strokeColor=none;opacity=70;")

    pts = [(px(i), py(r[1])) for i, r in enumerate(data)]
    polyline(pts, color)
    for i, r in enumerate(data):
        box(px(i) - 5.5, py(r[1]) - 5.5, 11, 11,
            f"{marker};fillColor={color};strokeColor=#FFFFFF;strokeWidth=1.5;")
        text(px(i) - 46, py(r[1]) - 30, 92, 18, f"{r[1]:.0f} ms", 11, color)
        text(px(i) - step / 2, Y0 + PH + 8, step, 34, str(r[0]), 11, MUTED)

    text(x0 - 4, Y0 - 46, PW + 4, 24, title, 14, INK, "left", 1)
    text(x0, Y0 + PH + 48, PW, 22, xlab, 12, MUTED)
    if ylab_on:
        vtext(x0 - 108, Y0, 24, PH,
              "Wall time of one decision cycle (logarithmic)")


panel(X0A, REG, C_A, "ellipse",
      "(a) The cost is nearly flat in the number of local regions",
      "Number of local regions", True)
panel(X0B, GRID, C_B, "triangle;direction=north",
      "(b) The cost grows with the size of the incident",
      "Grid of the simulated domain", False)

W = X0B + PW + 60
text(70, 26, W - 130, 48,
     "Decision latency and computational cost. Median over three worlds of "
     "the wall time of one pass of the decision loop, which covers the "
     "features and their confidences, the concept gates, the rule base, the "
     "global coordination, the two shadow forecasts of the acceptance test "
     "and any adaptation stage that runs. The shaded strip is the range over "
     "the three worlds. Four simultaneous ignitions, resource pool 0.25.",
     12, INK, "left")

xml = ('<mxfile host="app.diagrams.net">'
       '<diagram id="cost" name="cost">'
       f'<mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" '
       f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
       f'pageWidth="{W}" pageHeight="600" math="1" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
os.makedirs(FIGDIR, exist_ok=True)
open(os.path.join(FIGDIR, "fig_cost.drawio"), "w", encoding="utf8").write(xml)
print("written", len(xml), "bytes,", len(cells), "cells")
