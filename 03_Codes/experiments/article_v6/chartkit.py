"""Minimal chart kit on top of diag.Diagram: every mark is a primitive shape, so the same
spec renders to PNG (for the manuscript) and to editable draw.io cells (for the figure file)."""
from __future__ import annotations
import math
from diag import Diagram

INK, INK2, GRID = "#262626", "#595959", "#D9D9D9"


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0: return [lo]
    raw = span / n; mag = 10 ** math.floor(math.log10(raw)); r = raw / mag
    step = mag * (1 if r < 1.5 else 2 if r < 3 else 2.5 if r < 4 else 5 if r < 7.5 else 10)
    t = math.ceil(lo / step) * step; out = []
    while t <= hi + 1e-9: out.append(round(t, 10)); t += step
    return out


def fmt(v):
    return f"{v:g}"


class Ax:
    """A plotting area at canvas (x, y, w, h) mapping data limits to canvas coordinates (y grows downward)."""

    def __init__(self, d: Diagram, x, y, w, h, xlim=(0, 1), ylim=(0, 1), font=None):
        self.d, self.x, self.y, self.w, self.h = d, x, y, w, h
        self.xlim, self.ylim = xlim, ylim; self.font = font or d.font

    # mapping
    def px(self, v): return self.x + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w
    def py(self, v): return self.y + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    # frame
    def frame(self, left=True, bottom=True):
        if left: self.d.line([(self.x, self.y), (self.x, self.y + self.h)], INK2, 0.6)
        if bottom: self.d.line([(self.x, self.y + self.h), (self.x + self.w, self.y + self.h)], INK2, 0.6)

    def yticks(self, ticks=None, labels=None, grid=True, fs=None, fmt_=fmt):
        ticks = ticks if ticks is not None else nice_ticks(*self.ylim)
        for i, t in enumerate(ticks):
            if t < self.ylim[0] - 1e-9 or t > self.ylim[1] + 1e-9: continue
            yy = self.py(t)
            if grid and abs(t - self.ylim[0]) > 1e-9: self.d.line([(self.x, yy), (self.x + self.w, yy)], GRID, 0.5)
            self.d.line([(self.x - 2, yy), (self.x, yy)], INK2, 0.6)
            self.d.text(self.x - 3, yy, labels[i] if labels else fmt_(t), fs or self.font - 0.5, color=INK2, ha="right")
        return ticks

    def xticks(self, ticks, labels=None, fs=None, rot=0, grid=False):
        for i, t in enumerate(ticks):
            xx = self.px(t)
            if grid: self.d.line([(xx, self.y), (xx, self.y + self.h)], GRID, 0.5)
            self.d.line([(xx, self.y + self.h), (xx, self.y + self.h + 2)], INK2, 0.6)
            lab = labels[i] if labels else fmt(t)
            if rot:
                self.d.text(xx, self.y + self.h + 4 + 2.2 * len(lab), lab, fs or self.font - 0.5, color=INK2, ha="center", rot=rot)
            else:
                self.d.text(xx, self.y + self.h + 6 + 3.5 * lab.count("\n"), lab, fs or self.font - 0.5, color=INK2, ha="center")

    def xlabel(self, s, off=16, fs=None): self.d.text(self.x + self.w / 2, self.y + self.h + off, s, fs or self.font, color=INK, ha="center")
    def ylabel(self, s, off=24, fs=None): self.d.text(self.x - off, self.y + self.h / 2, s, fs or self.font, color=INK, ha="center", rot=90)
    def panel(self, s): self.d.text(self.x - 2, self.y - 6, s, self.font + 0.8, color=INK, ha="right")
    def title(self, s, fs=None): self.d.text(self.x + self.w / 2, self.y - 6, s, fs or self.font, color=INK, ha="center")

    # marks
    def bar(self, xc, h, bw, color, base=0, alpha=1.0, hatch=None, stroke="none"):
        y0, y1 = self.py(base), self.py(base + h)
        self.d.rect(self.px(xc) - bw / 2 * self.w / (self.xlim[1] - self.xlim[0]), min(y0, y1), bw * self.w / (self.xlim[1] - self.xlim[0]), abs(y0 - y1), color, stroke, 0.4, alpha=alpha, hatch=hatch)

    def barh(self, yc, wv, bh, color, base=0, alpha=1.0):
        x0, x1 = self.px(base), self.px(base + wv)
        hh = bh * self.h / (self.ylim[1] - self.ylim[0])
        self.d.rect(min(x0, x1), self.py(yc) - hh / 2, abs(x1 - x0), hh, color, "none", 0.4, alpha=alpha)

    def errbar(self, xc, v, e, color=INK2, cap=1.5):
        xx = self.px(xc); self.d.line([(xx, self.py(max(v - e, self.ylim[0]))), (xx, self.py(v + e))], color, 0.6)
        self.d.line([(xx - cap, self.py(v + e)), (xx + cap, self.py(v + e))], color, 0.6)
        if v - e > self.ylim[0]: self.d.line([(xx - cap, self.py(v - e)), (xx + cap, self.py(v - e))], color, 0.6)

    def errbarh(self, yc, v, e, color=INK2, cap=2):
        yy = self.py(yc); self.d.line([(self.px(v - e), yy), (self.px(v + e), yy)], color, 0.7)
        for xv in (v - e, v + e): self.d.line([(self.px(xv), yy - cap), (self.px(xv), yy + cap)], color, 0.7)

    def hline(self, v, color=INK2, lw=0.6, dashed=False, dotted=False): self.d.line([(self.x, self.py(v)), (self.x + self.w, self.py(v))], color, lw, dashed, dotted)
    def vline(self, v, color=INK2, lw=0.6, dashed=False, dotted=False): self.d.line([(self.px(v), self.y), (self.px(v), self.y + self.h)], color, lw, dashed, dotted)

    def plot(self, xs, ys, color, lw=1.2, dashed=False):
        pts = [(self.px(a), self.py(b)) for a, b in zip(xs, ys)]
        # draw.io edges keep every waypoint; thin the series so files stay small
        if len(pts) > 160:
            step = len(pts) / 160; pts = [pts[int(i * step)] for i in range(160)] + [pts[-1]]
        self.d.line(pts, color, lw, dashed)

    def step(self, xs, ys, color, lw=1.2):
        pts = []
        for i, (a, b) in enumerate(zip(xs, ys)):
            if i: pts.append((self.px(a), pts[-1][1]))
            pts.append((self.px(a), self.py(b)))
        if len(pts) > 220:
            step = len(pts) / 220; pts = [pts[int(i * step)] for i in range(220)] + [pts[-1]]
        self.d.line(pts, color, lw)

    def dot(self, xv, yv, r, color, alpha=1.0, stroke="none"):
        self.d.ellipse(self.px(xv) - r, self.py(yv) - r, 2 * r, 2 * r, color, stroke, 0.5, alpha)

    def text(self, xv, yv, s, fs=None, color=INK, ha="center", bold=False, dx=0, dy=0, rot=0):
        self.d.text(self.px(xv) + dx, self.py(yv) + dy, s, fs or self.font, bold=bold, color=color, ha=ha, rot=rot)

    def heatmap(self, M, rows, cols, cmap, fs=None, label=lambda v: fmt(v) if v else "", vmax=None, dark_from=0.55, rot=45, lab_fs=None):
        """M[i][j]; cells fill the whole area; rows/cols label the axes; cmap(f) -> hex for f in [0,1]."""
        n, m = len(rows), len(cols); cw, ch = self.w / m, self.h / n
        mx = vmax or max(max(r) for r in M) or 1
        for i in range(n):
            for j in range(m):
                f = M[i][j] / mx; self.d.rect(self.x + j * cw, self.y + i * ch, cw, ch, cmap(f), "#ffffff", 0.4)
                s = label(M[i][j])
                if s: self.d.text(self.x + (j + 0.5) * cw, self.y + (i + 0.5) * ch, s, lab_fs or fs or self.font - 1, color="#ffffff" if f > dark_from else INK, ha="center")
        for i, r in enumerate(rows): self.d.text(self.x - 3, self.y + (i + 0.5) * ch, r, fs or self.font - 0.6, color=INK2, ha="right")
        for j, c in enumerate(cols):
            if rot: self.d.text(self.x + (j + 0.5) * cw - 2, self.y + self.h + 3 + 2.0 * len(c), c, fs or self.font - 0.6, color=INK2, ha="center", rot=rot)
            else: self.d.text(self.x + (j + 0.5) * cw, self.y + self.h + 6, c, fs or self.font - 0.6, color=INK2, ha="center")

    def legend(self, items, x, y, fs=None, col=1, dx=58, dy=9, box=(7, 5)):
        """items: list of (color, label[, kind]) ; kind in {'rect','line','hatch'}; x,y in canvas px."""
        for k, it in enumerate(items):
            c, lab = it[0], it[1]; kind = it[2] if len(it) > 2 else "rect"
            cx, cy = x + (k // (math.ceil(len(items) / col)) if col > 1 else 0) * dx, y + (k % math.ceil(len(items) / col) if col > 1 else k) * dy
            if kind == "line": self.d.line([(cx, cy), (cx + box[0] + 2, cy)], c, 1.4)
            else: self.d.rect(cx, cy - box[1] / 2, box[0], box[1], c, "none", 0.4, hatch="////" if kind == "hatch" else None, alpha=0.5 if kind == "hatch" else 1)
            self.d.text(cx + box[0] + 3, cy, lab, fs or self.font - 0.6, color=INK)


def ramp(c0, c1):
    """linear hex ramp c0 (f=0) -> c1 (f=1)"""
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]; b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return lambda f: "#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * max(0, min(1, f)))) for k in range(3))


GREYS = ramp("#F2F2F2", "#404040"); GREENS = ramp("#E2F0D9", "#548235"); BLUES = ramp("#DEEBF7", "#2F5597")
