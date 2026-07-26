"""Representative-world figures for the evaluation scenarios (Chapter 5).

Table 5.12 defines five scenarios along two axes, landscape and resource
sufficiency, and the chapter shows ONE representative world of each so the
reader can see what a scenario actually looks like before reading fifty
seeds' worth of numbers about it. This script draws those worlds.

Each figure shows the four things the caption promises: the FUEL (land
cover), the ASSETS (settlements, buildings, critical facilities and where
the people are), the BASES (ground depots and helibases with their service
radius) and the IGNITION points. The fire is not run: this is the world at
t = 0, the input to the experiment rather than a result of it.

The worlds come from the same generator the dashboard uses, with the same
resource suggestion, so a figure cannot drift from what the experiments
actually ran on. Re-running the script reproduces the figures exactly: the
seed is part of the specification.

    python validation/make_scenario_figures.py            # all defined
    python validation/make_scenario_figures.py S1         # just one
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                            # noqa: E402

import dss                                                    # noqa: E402
from disaster_phyengine import terrain, viz                    # noqa: E402
from disaster_phyengine.config import (SimConfig,              # noqa: E402
                                       FUEL_NAME_TO_ID)
from disaster_phyengine.core import Simulator                  # noqa: E402


# ---------------------------------------------------------------- scenarios
# Table 5.12. `pool_scale` is the resource sufficiency axis: 1.0 is the
# suggested pool, below 1.0 is the capacity-limited arm.
SCENARIOS = {
    "S1": dict(
        label="Representative S1 world: remote forest, sufficient "
              "resources (fuel, assets, bases, ignition)",
        preset="Mountain forest", seed=1101,
        nx=120, ny=84, cell_m=30.0,
        settlements=2, pop_per_settlement=1200,
        pool_scale=1.0, n_ignitions=1, remote=True),
    "S2": dict(
        label="Representative S2 world: remote forest, capacity-limited "
              "pool, two remote ignitions",
        preset="Mountain forest", seed=1102,
        nx=120, ny=84, cell_m=30.0,
        settlements=2, pop_per_settlement=1200,
        pool_scale=0.6, n_ignitions=2, remote=True, min_sep_km=3.6),
}


def _remote_cells(world, min_dist_cells: int):
    """Burnable ground far from anything built.

    A remote ignition is the scenario's whole premise: it must start in the
    forest, not on the edge of a settlement, or the run measures a WUI
    problem under a remote label.
    """
    ft = np.asarray(world.fuel.ftype)
    fl0 = np.asarray(world.fuel.fload0)
    built = ((ft == FUEL_NAME_TO_ID["urban"])
             | (np.asarray(world.value.vbld) > 1e-6)
             | (np.asarray(world.value.vcrit) > 1e-6)
             | (np.asarray(world.value.vpop) > 1e-6))
    near = built.copy()
    for _ in range(max(1, int(min_dist_cells))):
        g = near.copy()
        g[1:, :] |= near[:-1, :]
        g[:-1, :] |= near[1:, :]
        g[:, 1:] |= near[:, :-1]
        g[:, :-1] |= near[:, 1:]
        near = g
    # AND NOT ON THE BORDER. An ignition against the frame is half drawn
    # and its fire leaves the domain immediately, which measures the edge
    # rather than the scenario.
    edge = 6
    inside = np.zeros_like(near)
    inside[edge:-edge, edge:-edge] = True
    ok = (fl0 > world.config.spread.eps_fuel) & ~near & inside
    # forest by preference: the scenario says remote FOREST
    forest = ok & np.isin(ft, [FUEL_NAME_TO_ID["pine_litter"],
                              FUEL_NAME_TO_ID["hardwood"],
                              FUEL_NAME_TO_ID["shrub"]])
    return forest if forest.any() else ok


def build_world(spec: dict):
    """One scenario-conformant world, plus its suggested resource pool."""
    cfg = SimConfig(nx=spec["nx"], ny=spec["ny"],
                    cell_size_m=spec["cell_m"])
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=spec["seed"], preset=spec["preset"],
        n_settlements=spec["settlements"],
        population_per_settlement=spec["pop_per_settlement"])

    # the suggested fleet, scaled by the sufficiency axis
    items, _why = dss.suggest_resource_items(w)
    scale = float(spec.get("pool_scale", 1.0))
    if scale != 1.0:
        for it in items:
            if "cap" in it:
                it["cap"] = float(it["cap"]) * scale
    pool = dss.build_resource_layer(w, items)

    # remote ignition(s), separated when the scenario asks for it
    rng = np.random.default_rng(spec["seed"])
    cand = _remote_cells(w, min_dist_cells=25 if spec.get("remote") else 0)
    ys, xs = np.where(cand)
    sep_cells = int(float(spec.get("min_sep_km", 0.0)) * 1000.0
                    / float(spec["cell_m"]))
    placed = []
    for _ in range(int(spec.get("n_ignitions", 1))):
        for _try in range(4000):
            k = int(rng.integers(0, len(xs)))
            x, y = int(xs[k]), int(ys[k])
            if all((x - px) ** 2 + (y - py) ** 2 >= sep_cells ** 2
                   for px, py in placed):
                placed.append((x, y))
                break
    for x, y in placed:
        w.add_ignition(x, y, step=0, radius=1)
    return w, items, pool, placed


def draw(name: str, out_dir: str, scale: int = 9) -> str:
    spec = SCENARIOS[name]
    w, items, pool, placed = build_world(spec)
    sim = Simulator(w)                       # t = 0, nothing burning yet

    depots = [(int(it["x"]), int(it["y"]), int(it.get("radius", 4)),
               float(it.get("cap", 0.8)),
               # SHORT LABELS. The renderer prints this beside the marker,
               # and a settlement holding both a depot and a helibase piled
               # two long names on top of each other and on the asset names.
               f"D{k + 1}" + ("h" if it.get("kind") == "helibase" else ""))
              for k, it in enumerate(items)
              if it.get("kind") in ("depot", "helibase")] or None

    img = viz.render_pil(
        w, sim=sim, scale=scale,
        show_fire=False,            # the world, not a run
        show_assets=True,
        # the caption promises FUEL, so the land cover is what the colours
        # must carry; the protection-value tint would wash the settlements
        # into pink blobs and hide the cover underneath them
        show_value=False,
        show_hillshade=True,
        show_roads=True, show_ignitions=True,
        # asset name labels collide with the base labels on a settlement
        # that holds both, and a figure is read at print size: the markers
        # carry the meaning and the legend below names them
        show_labels=False,
        show_grid=False, show_perimeter=False, show_wind=True,
        depots=depots)
    img = _with_legend(img, name)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_{name}_world.png")
    img.save(path, dpi=(300, 300))
    return path, w, items, placed


#: the groups a scenario figure shows, in the order the caption names them
_FIG_GROUPS = ("Land cover", "Assets", "Resources", "Markers")


def _fig_legend():
    """The symbol key, taken from the RENDERER's own legend.

    Writing the colours out by hand here would create a second description
    of the map that drifts from the first. legend_entries is what the
    dashboard shows and legend_icon_png draws each glyph with the same
    function the map uses, so the key on a thesis figure cannot claim a
    symbol the map does not draw.
    """
    out = []
    for grp, label, hexc, glyph in viz.legend_entries({}):
        if grp not in _FIG_GROUPS:
            continue
        if glyph in ("supp", "deploy", "cont", "prot", "evac", "warn"):
            continue                      # DSS orders: no run in this figure
        out.append((grp, label.split(" \u2014 ")[0].split(" (")[0],
                    hexc, glyph))
    return out


def _with_legend(img, name: str, pad: int = 10):
    """Put the symbol key under the map, drawn by the MAP's own code.

    The legend icons come from viz.legend_icon_png, the same functions the
    renderer uses, so the key on the figure cannot describe a glyph the map
    does not draw.
    """
    from PIL import Image, ImageDraw
    import io
    entries = _fig_legend()
    cols, line_h, icon = 4, 16, 12
    per_col = (len(entries) + cols - 1) // cols
    strip_h = pad * 2 + per_col * line_h
    out = Image.new("RGB", (img.width, img.height + strip_h),
                    (255, 255, 255))
    out.paste(img.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    d.line([0, img.height, img.width, img.height], fill=(170, 170, 170))
    col_w = img.width // cols
    for i, (_grp, label, hexc, glyph) in enumerate(entries):
        cx = pad + (i // per_col) * col_w
        cy = img.height + pad + (i % per_col) * line_h
        h = hexc.lstrip("#")
        rgb = tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
        ic = Image.open(io.BytesIO(
            viz.legend_icon_png(glyph, rgb, px=icon)))
        out.paste(ic, (cx, cy + 1), ic)
        d.text((cx + icon + 5, cy + 1), label[:30], fill=(25, 25, 25))
    return out


def _report(name, w, items, placed):
    ft = np.asarray(w.fuel.ftype)
    fl0 = np.asarray(w.fuel.fload0)
    vp = np.asarray(w.value.vpop)
    a = (np.clip(np.asarray(w.value.vbld), 0, 1)
         + np.clip(np.asarray(w.value.vcrit), 0, 1))
    ck = w.config.cell_area_ha / 100.0
    forest = np.isin(ft, [FUEL_NAME_TO_ID["pine_litter"],
                          FUEL_NAME_TO_ID["hardwood"],
                          FUEL_NAME_TO_ID["shrub"]])
    dep = [i for i in items if i.get("kind") in ("depot", "helibase")]
    print(f"  grid            {w.config.nx} x {w.config.ny} cells at "
          f"{w.config.cell_size_m:.0f} m "
          f"({w.config.nx * w.config.cell_size_m / 1000:.1f} x "
          f"{w.config.ny * w.config.cell_size_m / 1000:.1f} km)")
    print(f"  burnable        {int((fl0 > 0.03).sum())} cells, of which "
          f"forest/shrub {int((forest & (fl0 > 0.03)).sum())}")
    print(f"  built-up        {int((ft == FUEL_NAME_TO_ID['urban']).sum())} "
          f"cells | asset value {a.sum():.1f} | people "
          f"{float(vp.sum() * ck):.0f}")
    print(f"  bases           {len(dep)} "
          f"({', '.join(sorted({str(i['kind']) for i in dep}))})")
    for x, y in placed:
        print(f"  ignition        ({x}, {y}) on "
              f"{viz.FUEL_MODELS[int(ft[y, x])].name if hasattr(viz, 'FUEL_MODELS') else ft[y, x]}")


if __name__ == "__main__":
    want = sys.argv[1:] or list(SCENARIOS)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures")
    for nm in want:
        if nm not in SCENARIOS:
            print(f"unknown scenario {nm!r}; have {list(SCENARIOS)}")
            continue
        print(f"{nm}: {SCENARIOS[nm]['label']}")
        path, w, items, placed = draw(nm, out)
        _report(nm, w, items, placed)
        print(f"  -> {path}")
