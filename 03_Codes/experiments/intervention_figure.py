"""Thesis figure: ONE map with all six base interventions in action.

Builds a scenario where the doctrine has a reason to order every base
channel at once (a wind-driven fire bearing down on a populated town),
runs the real decision engine, and renders the map with the standard
symbol vocabulary plus a composed legend. The output is a single PNG
meant to be dropped into Chapter 4/5 as-is.

The figure is produced by the SAME renderer and the SAME legend code
the application uses, so the thesis figure cannot drift from the tool.

Run:  python experiments/intervention_figure.py
Out:  ../01_Thesis/figures/fig_base_interventions_map.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dss                                                # noqa: E402
from disaster_phyengine import terrain, viz               # noqa: E402
from disaster_phyengine.config import SimConfig           # noqa: E402
from disaster_phyengine.core import Simulator             # noqa: E402
from PIL import Image, ImageDraw                          # noqa: E402


def build_scene():
    cfg = SimConfig(nx=100, ny=70, cell_size_m=30.0)
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=42, relief_m=250,
                                   n_settlements=2)
    w.fuel.fmoist[:] = 0.09
    w.meteo.wws[:] = 7.0
    base, _ = dss.resource_suggestion(w)
    w.config.cost.capacity_reference = max(
        100.0, 1.2 * float((base.rcap * base.ravail).sum()))

    # the fire starts UPWIND of the main town so the doctrine has a
    # reason for every channel: offense on the front, defense at the
    # town, evacuation and warning for the people in the path
    towns = [a for a in w.assets if a.kind == "building"]
    t0 = max(towns, key=lambda a: getattr(a, "population", 0) or 0)
    ign_x = max(3, int(t0.x) - 13)
    ign_y = min(w.config.ny - 4, int(t0.y) + 5)
    sim = Simulator(w)
    sim.record_states = False
    w.add_ignition(ign_x, ign_y, step=0, radius=2)

    eng = dss.DecisionEngine(
        dss.partition_n(cfg.nx, cfg.ny, 1), base_pool=base,
        j_threshold=0.35, cycle_min=6.0, horizon_min=30.0,
        # the figure must show the seed doctrine acting, not whatever the
        # last session happened to learn (dss.isolated_store_path)
        state_path=dss.isolated_store_path("intervention_figure"),
        adapt_on=False)
    ov = None
    _peak = 0
    for step in range(45):                       # up to 90 minutes
        ov = eng.maybe_decide(sim) or ov
        sim.step(resource_override=ov)
        _b = int((sim.state.burning > 0.5).sum())
        _peak = max(_peak, _b)
        # freeze the frame at the height of the engagement: the fire
        # still burning hard, every order visibly in force. A nearly
        # extinguished fire makes a poor illustration of suppression.
        if step >= 15 and (_b == 0 or _b < 0.7 * _peak):
            break
    return w, sim, eng, base


def compose(w, sim, eng, base, out_path):
    scale = 12
    pil = viz.render_pil(
        w, sim=sim, scale=scale, show_labels=True, show_perimeter=True,
        alloc=getattr(sim, "last_applied_resource", None).rcap
        if getattr(sim, "last_applied_resource", None) is not None
        else None,
        actions=eng.last_actions,
        regions=[(r.x0, r.y0, r.x1, r.y1, r.name) for r in eng.regions],
        depots=None).convert("RGB")

    # ---- legend strip from the SAME single-source entries ----
    groups_wanted = ("DSS orders (base)", "Fire", "Markers")
    entries = [e for e in viz.legend_entries()
               if e[0] in groups_wanted]
    pad, row_h, sw = 14, 30, 22
    lw = pil.width
    lh = pad * 2 + row_h * (len(entries) + len(groups_wanted))
    leg = Image.new("RGB", (lw, lh), (250, 250, 248))
    d = ImageDraw.Draw(leg)
    d.font = viz._font(16)
    hfont = viz._font(17)
    y = pad
    for g in groups_wanted:
        d.text((pad, y), g, fill=(30, 30, 30), font=hfont)
        y += row_h
        for grp, label, colhex, glyph in entries:
            if grp != g:
                continue
            try:
                import io
                icon = Image.open(io.BytesIO(
                    viz.legend_icon_png(glyph, tuple(
                        int(colhex[i:i + 2], 16)
                        for i in (1, 3, 5)), px=sw))).convert("RGBA")
                leg.paste(icon, (pad + 10, y + 2), icon)
            except Exception:
                d.rectangle([pad + 10, y + 4, pad + 10 + sw - 4,
                             y + sw - 2], fill=colhex)
            d.text((pad + 14 + sw, y + 4), label, fill=(55, 55, 55))
            y += row_h
    fig = Image.new("RGB", (lw, pil.height + lh), (255, 255, 255))
    fig.paste(pil, (0, 0))
    fig.paste(leg, (0, pil.height))
    fig.save(out_path)
    return out_path


def main():
    w, sim, eng, base = build_scene()
    acts = eng.last_actions or {}
    u_all = {}
    for ro in acts.get("regions", []):
        for k, v in ro["u"].items():
            if isinstance(v, (int, float)):
                u_all[k] = max(u_all.get(k, 0.0), float(v))
    print("orders:", {k: round(v, 2) for k, v in sorted(u_all.items())
                      if v > 0.05})
    burning = int((sim.state.burning > 0.5).sum())
    print("burning:", burning, "burned:", int(sim.ever_burned.sum()),
          "evacuated:", round(sim.population_evacuated))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                           "01_Thesis", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = compose(w, sim, eng, base, os.path.join(
        out_dir, "fig_base_interventions_map.png"))
    print("written:", os.path.abspath(out))


if __name__ == "__main__":
    main()
