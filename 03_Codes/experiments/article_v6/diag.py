"""Tiny block-diagram kit: one spec -> matplotlib PNG + draw.io XML.
Coordinates in points of a page-like canvas (1 unit = 1 px at 96 dpi for draw.io).
"""
from __future__ import annotations
import html, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

PAL = {   # Microsoft Office theme colours (Word palette) and their tints/shades
    "core":  ("#D9E2F3", "#2F5597"),   # Blue, Accent 1 (lighter 80 % / darker 25 %)
    "agent": ("#E2F0D9", "#548235"),   # Green, Accent 6 (lighter 80 % / darker 25 %)
    "closed": ("#B4C7E7", "#1F3864"),  # Blue, Accent 1 (lighter 60 % / darker 50 %)
    "open":  ("#F2CCCC", "#C00000"),   # Dark Red (standard colour) and its tint
    "gate":  ("#EDEDED", "#7F7F7F"),   # White darker 5 % / 50 %
    "ext":   ("#F2F2F2", "#595959"),   # White darker 5 % / Black lighter 35 %
    "cmd":   ("#DBDBDB", "#262626"),   # White darker 15 % / Black lighter 15 %
    "plain": ("#FFFFFF", "#404040"),
    "none":  ("none", "none"),
}


def mp_text(s):
    """text for matplotlib: $...$ segments are mathtext already."""
    return s


def dio_text(s):
    """text for draw.io (html labels, math=1): $...$ -> \(...\) for MathJax, rest html-escaped, newlines -> <br>."""
    out = []
    for i, seg in enumerate(s.split("$")):
        if i % 2 == 0: out.append(html.escape(seg).replace("\n", "&lt;br&gt;"))
        else: out.append("\\(" + html.escape(seg, quote=False) + "\\)")
    return "".join(out)


class Diagram:
    def __init__(self, w, h, font=7.0):
        self.w, self.h, self.font = w, h, font
        self.boxes, self.edges, self.texts = [], [], []
        self.shapes = []   # primitive rects / lines / ellipses / images for charts

    # ------------------------------------------------------------ chart primitives
    def rect(self, x, y, w, h, fill="#cccccc", stroke="none", lw=0.6, dashed=False, alpha=1.0, hatch=None):
        self.shapes.append(dict(t="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, lw=lw, dashed=dashed, alpha=alpha, hatch=hatch))

    def ellipse(self, x, y, w, h, fill="#cccccc", stroke="none", lw=0.6, alpha=1.0):
        self.shapes.append(dict(t="ellipse", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, lw=lw, alpha=alpha))

    def line(self, pts, color="#444444", lw=0.8, dashed=False, dotted=False, arrow=False):
        self.shapes.append(dict(t="line", pts=list(pts), color=color, lw=lw, dashed=dashed, dotted=dotted, arrow=arrow))

    def image(self, x, y, w, h, path):
        self.shapes.append(dict(t="image", x=x, y=y, w=w, h=h, path=path))

    def box(self, x, y, w, h, text, kind="plain", bold=False, fs=None, dashed=False, rounded=False, align="center", name=None):
        self.boxes.append(dict(x=x, y=y, w=w, h=h, text=text, kind=kind, bold=bold, fs=fs or self.font, dashed=dashed, rounded=rounded, align=align, id=name or f"b{len(self.boxes)+2}"))
        return self.boxes[-1]

    def text(self, x, y, text, fs=None, bold=False, color="#333", ha="left", rot=0, italic=False):
        self.texts.append(dict(x=x, y=y, text=text, fs=fs or self.font, bold=bold, color=color, ha=ha, rot=rot, italic=italic))

    def edge(self, a, b, label="", kind="plain", style="-|>", dashed=False, src="auto", dst="auto", via=None, lw=0.9, lpos=0.5, loff=(0, 0), p0=None, p1=None):
        self.edges.append(dict(a=a, b=b, label=label, kind=kind, style=style, dashed=dashed, src=src, dst=dst, via=via or [], lw=lw, lpos=lpos, loff=loff, p0=p0, p1=p1))

    # ------------------------------------------------------------ geometry
    @staticmethod
    def _port(b, side):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        return {"n": (x + w / 2, y), "s": (x + w / 2, y + h), "w": (x, y + h / 2), "e": (x + w, y + h / 2), "c": (x + w / 2, y + h / 2)}[side]

    def _auto(self, a, b):
        ax, ay = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
        bx, by = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        dx, dy = bx - ax, by - ay
        if abs(dx) * 0.8 > abs(dy):
            return ("e", "w") if dx > 0 else ("w", "e")
        return ("s", "n") if dy > 0 else ("n", "s")

    # ------------------------------------------------------------ render
    def render(self, path, dpi=300, scale=1 / 78.0):
        fig = plt.figure(figsize=(self.w * scale, self.h * scale))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, self.w); ax.set_ylim(self.h, 0); ax.axis("off")
        for s in self.shapes:
            if s["t"] == "rect":
                ax.add_patch(Rectangle((s["x"], s["y"]), s["w"], s["h"], fc=s["fill"] if s["fill"] != "none" else "none", ec=s["stroke"] if s["stroke"] != "none" else "none",
                                       lw=s["lw"], ls="--" if s["dashed"] else "-", alpha=s["alpha"], hatch=s["hatch"]))
            elif s["t"] == "ellipse":
                from matplotlib.patches import Ellipse
                ax.add_patch(Ellipse((s["x"] + s["w"] / 2, s["y"] + s["h"] / 2), s["w"], s["h"], fc=s["fill"] if s["fill"] != "none" else "none", ec=s["stroke"] if s["stroke"] != "none" else "none", lw=s["lw"], alpha=s["alpha"]))
            elif s["t"] == "line":
                xs = [p[0] for p in s["pts"]]; ys = [p[1] for p in s["pts"]]
                ax.plot(xs, ys, color=s["color"], lw=s["lw"], ls=":" if s["dotted"] else ("--" if s["dashed"] else "-"), solid_capstyle="butt")
                if s["arrow"]:
                    ax.add_patch(FancyArrowPatch(s["pts"][-2], s["pts"][-1], arrowstyle="-|>", mutation_scale=6, lw=s["lw"], color=s["color"], shrinkA=0, shrinkB=0))
            elif s["t"] == "image":
                import matplotlib.image as mpimg
                ax.imshow(mpimg.imread(s["path"]), extent=(s["x"], s["x"] + s["w"], s["y"] + s["h"], s["y"]), aspect="auto", zorder=0)
        for b in self.boxes:
            fill, edge = PAL[b["kind"]]
            if b["kind"] != "none":
                patch = FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle=f"round,pad=0,rounding_size={4 if b['rounded'] else 0}",
                                       fc=fill, ec=edge, lw=0.8, ls="--" if b["dashed"] else "-")
                ax.add_patch(patch)
            if b["text"]:
                tx = b["x"] + (b["w"] / 2 if b["align"] == "center" else 6)
                ax.text(tx, b["y"] + b["h"] / 2, b["text"], ha=b["align"], va="center", fontsize=b["fs"], fontweight="bold" if b["bold"] else "normal",
                        color=PAL[b["kind"]][1] if b["kind"] != "none" else "#333", linespacing=1.15, wrap=False)
        for t in self.texts:
            ax.text(t["x"], t["y"], t["text"], fontsize=t["fs"], fontweight="bold" if t["bold"] else "normal", color=t["color"], ha=t["ha"], va="center", rotation=t["rot"], fontstyle="italic" if t["italic"] else "normal")
        for e in self.edges:
            a, b = e["a"], e["b"]
            sa, sb = (e["src"], e["dst"]) if e["src"] != "auto" else self._auto(a, b)
            if e["dst"] == "auto" and e["src"] != "auto":
                sb = self._auto(a, b)[1]
            p0, p1 = e["p0"] or self._port(a, sa), e["p1"] or self._port(b, sb)
            pts = [p0] + e["via"] + [p1]
            col = PAL[e["kind"]][1] if e["kind"] != "plain" else "#444"
            for i in range(len(pts) - 1):
                last = i == len(pts) - 2
                arr = FancyArrowPatch(pts[i], pts[i + 1], arrowstyle=e["style"] if last else "-", mutation_scale=7, lw=e["lw"], color=col,
                                      ls="--" if e["dashed"] else "-", shrinkA=0, shrinkB=0)
                ax.add_patch(arr)
            if e["label"]:
                # label at lpos along the polyline
                L = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]; tot = sum(L); d = e["lpos"] * tot
                for i, l in enumerate(L):
                    if d <= l or i == len(L) - 1:
                        f = d / l if l else 0; x = pts[i][0] + f * (pts[i + 1][0] - pts[i][0]); y = pts[i][1] + f * (pts[i + 1][1] - pts[i][1]); break
                    d -= l
                ax.text(x + e["loff"][0], y + e["loff"][1], e["label"], fontsize=self.font - 0.8, color=col, ha="center", va="center")
        fig.savefig(path, dpi=dpi); plt.close(fig)

    # ------------------------------------------------------------ draw.io
    def drawio_cells(self):
        cells = []
        for i, s in enumerate(self.shapes):
            sid = f"s{i}"
            if s["t"] in ("rect", "ellipse"):
                fill = s["fill"] if s["fill"] != "none" else "none"; stroke = s["stroke"] if s["stroke"] != "none" else "none"
                style = (f"{'ellipse;' if s['t']=='ellipse' else 'rounded=0;'}whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={s['lw']:.1f};"
                         f"{'dashed=1;' if s.get('dashed') else ''}{'opacity=%d;' % int(s['alpha']*100) if s['alpha'] < 1 else ''}{'fillStyle=hatch;' if s.get('hatch') else ''}")
                cells.append(f'<mxCell id="{sid}" value="" style="{style}" vertex="1" parent="1"><mxGeometry x="{s["x"]:.1f}" y="{s["y"]:.1f}" width="{max(s["w"],0.5):.1f}" height="{max(s["h"],0.5):.1f}" as="geometry"/></mxCell>')
            elif s["t"] == "line":
                pts = s["pts"]
                style = f"endArrow={'block' if s['arrow'] else 'none'};endFill=1;html=1;strokeColor={s['color']};strokeWidth={s['lw']:.1f};{'dashed=1;' if s['dashed'] else ''}{'dashed=1;dashPattern=1 2;' if s['dotted'] else ''}"
                mid = "".join(f'<mxPoint x="{p[0]:.1f}" y="{p[1]:.1f}"/>' for p in pts[1:-1])
                arr = f'<Array as="points">{mid}</Array>' if mid else ""
                cells.append(f'<mxCell id="{sid}" value="" style="{style}" edge="1" parent="1"><mxGeometry width="50" height="50" relative="1" as="geometry"><mxPoint x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}" as="sourcePoint"/><mxPoint x="{pts[-1][0]:.1f}" y="{pts[-1][1]:.1f}" as="targetPoint"/>{arr}</mxGeometry></mxCell>')
            elif s["t"] == "image":
                import base64, mimetypes
                mt = mimetypes.guess_type(s["path"])[0] or "image/png"
                b64 = base64.b64encode(open(s["path"], "rb").read()).decode()
                style = f"shape=image;verticalLabelPosition=bottom;labelBackgroundColor=default;verticalAlign=top;aspect=fixed;imageAspect=0;image=data:{mt},{b64};"
                cells.append(f'<mxCell id="{sid}" value="" style="{style}" vertex="1" parent="1"><mxGeometry x="{s["x"]:.1f}" y="{s["y"]:.1f}" width="{s["w"]:.1f}" height="{s["h"]:.1f}" as="geometry"/></mxCell>')
        for b in self.boxes:
            fill, edge = PAL[b["kind"]]
            style = (f"rounded={1 if b['rounded'] else 0};whiteSpace=wrap;html=1;fillColor={fill if fill!='none' else 'none'};strokeColor={edge if edge!='none' else 'none'};"
                     f"fontSize={int(round(b['fs']*1.3))};fontFamily=Helvetica;fontColor={edge if edge!='none' else '#333333'};{'fontStyle=1;' if b['bold'] else ''}{'dashed=1;' if b['dashed'] else ''}"
                     f"align={b['align']};verticalAlign=middle;spacingLeft={0 if b['align']=='center' else 6};")
            txt = dio_text(b["text"])
            cells.append(f'<mxCell id="{b["id"]}" value="{txt}" style="{style}" vertex="1" parent="1"><mxGeometry x="{b["x"]:.0f}" y="{b["y"]:.0f}" width="{b["w"]:.0f}" height="{b["h"]:.0f}" as="geometry"/></mxCell>')
        for i, t in enumerate(self.texts):
            txt = dio_text(t["text"])
            style = f"text;html=1;align={'left' if t['ha']=='left' else 'center'};verticalAlign=middle;fontSize={int(round(t['fs']*1.3))};fontColor={t['color']};{'fontStyle=1;' if t['bold'] else ''}{'rotation=%d;' % t['rot'] if t['rot'] else ''}"
            w = max(40, len(t["text"]) * t["fs"] * 0.75)
            x = t["x"] if t["ha"] == "left" else t["x"] - w / 2
            cells.append(f'<mxCell id="t{i}" value="{txt}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x:.0f}" y="{t["y"]-8:.0f}" width="{w:.0f}" height="16" as="geometry"/></mxCell>')
        for i, e in enumerate(self.edges):
            a, b = e["a"], e["b"]
            col = PAL[e["kind"]][1] if e["kind"] != "plain" else "#444444"
            style = f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={col};endArrow={'block' if '>' in e['style'] else 'none'};endFill=1;{'dashed=1;' if e['dashed'] else ''}fontSize={int(round((self.font-0.8)*1.3))};fontColor={col};labelBackgroundColor=none;"
            pts = "".join(f'<mxPoint x="{p[0]:.0f}" y="{p[1]:.0f}"/>' for p in e["via"])
            arr = f'<Array as="points">{pts}</Array>' if pts else ""
            cells.append(f'<mxCell id="e{i}" value="{dio_text(e["label"])}" style="{style}" edge="1" parent="1" source="{a["id"]}" target="{b["id"]}"><mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
        return "\n".join(cells)


def drawio_file(pages, path):
    """pages: list of (name, Diagram)."""
    out = ['<mxfile host="app.diagrams.net" modified="2026-10-10T00:00:00Z" agent="DisasterAware v6" version="24.0.0">']
    for i, (name, d) in enumerate(pages):
        out.append(f'<diagram id="p{i}" name="{html.escape(name)}"><mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{int(d.w)+40}" pageHeight="{int(d.h)+40}" math="1" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>')
        out.append(d.drawio_cells())
        out.append('</root></mxGraphModel></diagram>')
    out.append('</mxfile>')
    open(path, "w", encoding="utf-8").write("\n".join(out))
