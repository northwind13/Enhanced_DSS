"""Representative-world figures for the evaluation scenarios (Chapter 5).

Table 5.12 defines five scenarios along three axes - landscape, resource
sufficiency and observation quality - and the chapter shows ONE
representative world of each, so the reader can see what a scenario is
before reading fifty seeds' worth of numbers about it.

    ID  Landscape                       Resources   Ignitions
    S1  remote forest, ~2 settlements   sufficient  1 remote
    S2  as S1                           limited     2 remote
    S3  WUI, ~15 settlements            sufficient  1 at the interface
    S4  as S3                           limited     2 at the interface
    S5  as S4                           limited     2, degraded observation

Each figure shows what the caption promises: the FUEL (land cover), the
ASSETS (settlements, facilities, where the people are), the SENSORS and the
RESOURCE BASES with their service radii, and the IGNITION points. The fire
is not run: this is the world at t = 0, the input to the experiment rather
than a result of it.

Nothing here is hand-placed. The sensors come from the Layer 1 network
suggestion (greedy maximum weighted coverage) and the bases from the Layer 1
resource suggestion, exactly as the dashboard stages them, so a figure
cannot drift from what the experiments ran on. The sufficiency axis is a
DENSITY on that suggestion, and what it buys is measured rather than
asserted: pool_efficiency is printed for every scenario and the script stops
if a "sufficient" world lands under 50% or a "limited" one over it.

    python validation/make_scenario_figures.py            # all five
    python validation/make_scenario_figures.py S1 S4      # just these
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


# ---------------------------------------------------------------- landscapes
#: the two worlds the five scenarios are drawn from. S1/S2 share one and
#: S3/S4/S5 share the other, which is the point of the table: the members of
#: a pair differ ONLY in what the response has to work with.
LANDSCAPES = {
    "remote": dict(
        seed=1101, nx=150, ny=110, cell_m=60.0,
        relief_m=520.0, forest_density=0.62, base_moisture=0.07,
        water_level=0.04, settlements=2, population=2400,
        building_scale=0.6, river=False, coast=False, farmland=False),
    "wui": dict(
        seed=2201, nx=150, ny=110, cell_m=60.0,
        relief_m=300.0, forest_density=0.46, base_moisture=0.07,
        water_level=0.05, settlements=15, population=90000,
        building_scale=1.3, river=False, coast=False, farmland=True,
        # fifteen towns carry sixty civic facilities between them: naming
        # each one buries the map in text, and the figure's job is to show
        # what the scenario IS, not to index it
        label_kinds=("building", "evac_route")),
}

#: The sufficiency axis. "Sufficient" is the suggestion as it stands
#: (density 1.0); "limited" is not a guessed number but a TARGET
#: effectiveness, and the density that reaches it is solved for on each
#: landscape (see _density_for). Hard-coding a density did not survive the
#: change of landscape: 0.18 left the WUI map at 63% effectiveness, because
#: fifteen settlements stage thirteen bases and the pool stayed adequate
#: however thin each one was.
SCENARIOS = {
    "S1": dict(landscape="remote", label="remote forest, sufficient "
                                         "resources, one remote ignition",
               density=1.0, ignitions=1, remote=True, observation="full"),
    "S2": dict(landscape="remote", label="remote forest, limited "
                                         "resources, two remote ignitions",
               target_eff=0.35, ignitions=2, remote=True,
               observation="full", min_sep_km=4.0),
    "S3": dict(landscape="wui", label="wildland-urban interface, sufficient "
                                      "resources, one ignition at the "
                                      "interface",
               density=1.0, ignitions=1, remote=False, observation="full"),
    "S4": dict(landscape="wui", label="wildland-urban interface, limited "
                                      "resources, two ignitions at the "
                                      "interface",
               target_eff=0.35, ignitions=2, remote=False,
               observation="full", min_sep_km=3.0),
    "S5": dict(landscape="wui", label="as S4, under degraded observation",
               target_eff=0.35, ignitions=2, remote=False,
               observation="degraded", min_sep_km=3.0),
}

#: what the planner is allowed to deploy. "degraded" is S5's whole point:
#: the satellite (always tasked) and the calls from the public remain, one
#: in-situ station survives, and the aerial recon and the lookout cameras
#: are gone - the sources that give the DSS a timely, well-located front.
SENSOR_BUDGETS = {
    "full": {"aerial": 2, "ground_camera": 3, "in_situ": 3,
             "field_report": 2},
    "degraded": {"aerial": 0, "ground_camera": 0, "in_situ": 1,
                 "field_report": 0},
}


def build_world(name: str):
    """The world, its staged sensors and its staged resource pool."""
    spec = SCENARIOS[name]
    land = LANDSCAPES[spec["landscape"]]
    cfg = SimConfig(nx=land["nx"], ny=land["ny"], cell_size_m=land["cell_m"])
    cfg.step_minutes = 2.0
    w = terrain.generate_landscape(
        cfg, seed=land["seed"], relief_m=land["relief_m"],
        forest_density=land["forest_density"],
        base_moisture=land["base_moisture"], water_level=land["water_level"],
        n_settlements=land["settlements"],
        population_per_settlement=land["population"],
        building_scale=land["building_scale"],
        with_assets=True, with_roads=True, accessibility=1.0,
        river=land["river"], coast=land["coast"],
        # NO FIELDS ON THE REMOTE SCENARIO. Cultivated parcels are a real
        # cover class, but they belong to inhabited valley floors; on a
        # world whose premise is that there is nothing out here they read
        # as an artefact and colour half the picture.
        farmland=bool(land.get("farmland", True)))

    # ---- Layer 1: the resource pool, at this scenario's sufficiency
    _dens = (float(spec["density"]) if "density" in spec
             else _density_for(w, float(spec["target_eff"])))
    items, res_why = dss.suggest_resource_items(w, density=_dens)
    base = dss.build_resource_layer(w, items)
    eff, parts = dss.pool_efficiency(w, base)
    spec = dict(spec, density=_dens)

    # ---- Layer 1: the sensor network
    placements, sens_why = dss.suggest_network(
        w, budget=dict(SENSOR_BUDGETS[spec["observation"]]))

    # ---- the ignitions the scenario asks for
    placed = _place_ignitions(w, spec)
    for x, y in placed:
        w.add_ignition(x, y, step=0, radius=1)
    return (w, items, base, eff, parts, placements, placed,
            (res_why, sens_why), _dens)


def _density_for(world, target: float, lo: float = 0.10, hi: float = 1.0,
                 iters: int = 12) -> float:
    """The staged-capacity density that puts effectiveness at `target`.

    pool_efficiency is the product of a REACH score (can the crews get to
    the ground that matters, which the density cannot change) and a
    CAPACITY score (is there enough of them, which is all the density
    changes), and the capacity score is monotone in the density and
    saturates at 1. So the density is solved by bisection rather than
    guessed, and a "limited" scenario means the same thing on a two-village
    map as on a fifteen-town one.
    """
    def _eff(d):
        _it, _ = dss.suggest_resource_items(world, density=float(d))
        return float(dss.pool_efficiency(
            world, dss.build_resource_layer(world, _it))[0])

    # THE FLOOR IS THE API'S FLOOR. suggest_resource_items clamps the
    # density to [0.1, 3.0], so bisecting below 0.1 converged onto a number
    # that was never used: the report said 0.020 while the pool was built
    # at 0.1. The search stays inside what the planner will honour.
    _hi_e = _eff(hi)
    if _hi_e <= target:            # even the full pool is under the target
        return float(hi)
    if _eff(lo) > target:          # even the thinnest pool is over it
        return float(lo)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        if _eff(mid) > target:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


def _place_ignitions(world, spec):
    """Remote ones deep in the fuel, interface ones just outside a town.

    A scenario is its ignition as much as its landscape: "remote" has to
    start in the forest or the run measures a WUI problem under a remote
    label, and "at the interface" has to start in the wildland NEXT TO the
    built-up edge rather than inside it, because a fire that starts in a
    street is an urban conflagration and not what this tests.
    """
    ft = np.asarray(world.fuel.ftype)
    fl0 = np.asarray(world.fuel.fload0)
    ny, nx = ft.shape
    built = ((ft == FUEL_NAME_TO_ID["urban"])
             | (np.asarray(world.value.vbld) > 1e-6)
             | (np.asarray(world.value.vcrit) > 1e-6)
             | (np.asarray(world.value.vpop) > 1e-6))
    burnable = fl0 > world.config.spread.eps_fuel
    edge = max(5, int(0.05 * min(nx, ny)))
    inside = np.zeros_like(built)
    inside[edge:-edge, edge:-edge] = True

    near = _grow(built, max(2, int(round(400.0 / world.config.cell_size_m))))
    far = _grow(built, max(8, int(round(1500.0 / world.config.cell_size_m))))
    if spec.get("remote"):
        ok = burnable & ~far & inside
    else:
        # the interface band: outside the built-up ground, within reach of it
        ok = burnable & ~built & near & inside
    forest = ok & np.isin(ft, [FUEL_NAME_TO_ID["pine_litter"],
                               FUEL_NAME_TO_ID["hardwood"],
                               FUEL_NAME_TO_ID["shrub"]])
    cand = forest if forest.any() else ok
    if not cand.any():
        cand = burnable & inside

    rng = np.random.default_rng(LANDSCAPES[spec["landscape"]]["seed"])
    ys, xs = np.where(cand)
    sep = int(float(spec.get("min_sep_km", 0.0)) * 1000.0
              / float(world.config.cell_size_m))
    placed = []
    for _ in range(int(spec.get("ignitions", 1))):
        for _try in range(6000):
            k = int(rng.integers(0, len(xs)))
            x, y = int(xs[k]), int(ys[k])
            if all((x - px) ** 2 + (y - py) ** 2 >= sep ** 2
                   for px, py in placed):
                placed.append((x, y))
                break
    return placed


def _grow(mask, k):
    out = np.asarray(mask).copy()
    for _ in range(max(0, int(k))):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


# ------------------------------------------------------------------ drawing
def _draw_lists(world, items, placements):
    """The sensor and depot tuples the renderer takes, built the same way
    the dashboard builds them, so a figure shows the dashboard's map."""
    cell = float(world.config.cell_size_m)
    sensors = []
    for i, d in enumerate(placements):
        _r_m = dss.SENSOR_CATALOG[d["kind"]]["radius_m"]
        sensors.append((d["x"], d["y"],
                        None if _r_m is None
                        else max(1, int(round(_r_m / cell))),
                        d["kind"], f"S{i + 1} {d['kind']}"))
    depots = [(int(it["x"]), int(it["y"]), int(it.get("radius", 4)),
               float(it.get("cap", 0.8)),
               f"D{k + 1} " + dss.RESOURCE_KINDS.get(
                   it.get("kind"), {}).get("short", str(it.get("kind", ""))))
              for k, it in enumerate(items)
              if it.get("kind") in ("depot", "helibase")]
    return sensors, (depots or None)


#: the groups a scenario figure shows, in the order the caption names them
_FIG_GROUPS = ("Land cover", "Assets", "Sensors (+ coverage fill)",
               "Resources", "Markers")


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
        out.append((grp, label.split(" — ")[0].split(" (")[0],
                    hexc, glyph))
    return out


def _with_legend(img, title, pad: int = 10):
    """Put the symbol key and the caption under the map, drawn by the MAP's
    own code (viz.legend_icon_png), so the key cannot describe a glyph the
    map does not draw."""
    from PIL import Image, ImageDraw
    import io
    entries = _fig_legend()
    cols, line_h, icon = 4, 17, 13
    per_col = (len(entries) + cols - 1) // cols
    strip_h = pad * 3 + 16 + per_col * line_h
    out = Image.new("RGB", (img.width, img.height + strip_h),
                    (255, 255, 255))
    out.paste(img.convert("RGB"), (0, 0))
    d = ImageDraw.Draw(out)
    d.line([0, img.height, img.width, img.height], fill=(170, 170, 170))
    d.text((pad, img.height + pad), title, fill=(15, 15, 15))
    col_w = img.width // cols
    for i, (_grp, label, hexc, glyph) in enumerate(entries):
        cx = pad + (i // per_col) * col_w
        cy = img.height + pad * 2 + 16 + (i % per_col) * line_h
        h = hexc.lstrip("#")
        rgb = tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
        ic = Image.open(io.BytesIO(viz.legend_icon_png(glyph, rgb, px=icon)))
        out.paste(ic, (cx, cy + 1), ic)
        d.text((cx + icon + 6, cy + 1), label[:34], fill=(25, 25, 25))
    return out


def draw(name: str, out_dir: str, scale: int = 9):
    spec = SCENARIOS[name]
    (w, items, base, eff, parts, placements, placed, _why,
     dens) = build_world(name)
    sensors, depots = _draw_lists(w, items, placements)
    sim = Simulator(w)                        # t = 0, nothing burning yet

    img = viz.render_pil(
        w, sim=sim, scale=scale,
        show_fire=False,            # the world, not a run
        show_assets=True,
        # the caption promises FUEL, so the land cover is what the colours
        # must carry; the protection-value tint would wash the settlements
        # into pink blobs and hide the cover underneath them
        show_value=False,
        show_hillshade=True, show_roads=True, show_ignitions=True,
        show_labels=True, show_grid=False, show_perimeter=False,
        show_wind=True, sensors=sensors, depots=depots,
        label_kinds=LANDSCAPES[spec["landscape"]].get("label_kinds"))
    img = _with_legend(
        img, f"{name} - {spec['label']}  |  resource effectiveness "
             f"{eff:.0%} (reach {parts['reach']:.0%}, capacity "
             f"{parts['capacity']:.0%})")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_{name}_world.png")
    img.save(path, dpi=(300, 300))
    return path, w, items, base, eff, parts, placements, placed, dens


def _dist_to_built(world, x, y):
    ft = np.asarray(world.fuel.ftype)
    built = ((ft == FUEL_NAME_TO_ID["urban"])
             | (np.asarray(world.value.vbld) > 1e-6))
    if not built.any():
        return float("inf")
    ys, xs = np.where(built)
    return float(np.sqrt((xs - x) ** 2 + (ys - y) ** 2).min()
                 * world.config.cell_size_m)


def _report(name, w, items, base, eff, parts, placements, placed, dens):
    spec = SCENARIOS[name]
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
    kinds = {}
    for d in placements:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    _names = {v: k for k, v in FUEL_NAME_TO_ID.items()}
    print(f"  grid            {w.config.nx} x {w.config.ny} at "
          f"{w.config.cell_size_m:.0f} m "
          f"({w.config.nx * w.config.cell_size_m / 1000:.1f} x "
          f"{w.config.ny * w.config.cell_size_m / 1000:.1f} km)")
    print(f"  burnable        {int((fl0 > 0.03).sum())} cells, "
          f"forest/shrub {int((forest & (fl0 > 0.03)).sum())}")
    print(f"  settlements     {len(terrain.settlements(w))} | built-up "
          f"{int((ft == FUEL_NAME_TO_ID['urban']).sum())} cells | asset "
          f"value {a.sum():.0f} | people {float(vp.sum() * ck):.0f}")
    print(f"  resources       density {dens:.3f} -> {len(dep)} "
          f"base(s) | effectiveness {eff:.1%} "
          f"(reach {parts['reach']:.0%}, capacity {parts['capacity']:.0%}, "
          f"air {parts.get('air', 0):.0%})")
    print(f"  sensors         {len(placements)} "
          f"({', '.join(f'{k} x{v}' for k, v in sorted(kinds.items()))})")
    for x, y in placed:
        print(f"  ignition        ({x}, {y}) on "
              f"{_names[int(ft[y, x])]}, "
              f"{_dist_to_built(w, x, y):.0f} m from built-up ground")

    # THE CLAIM IS CHECKED, NOT STATED. "Effectiveness exceeds 50% when
    # resources are sufficient and drops below 50% when they are limited"
    # is the definition of the axis, so a figure that violates it is a
    # broken scenario and must not go into the chapter.
    _suff = "density" in spec and float(spec["density"]) >= 1.0
    if _suff and eff <= 0.50:
        raise SystemExit(f"{name}: a sufficient pool measures {eff:.1%}")
    if (not _suff) and eff >= 0.50:
        raise SystemExit(f"{name}: a limited pool measures {eff:.1%}")


if __name__ == "__main__":
    want = sys.argv[1:] or list(SCENARIOS)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    for nm in want:
        if nm not in SCENARIOS:
            print(f"unknown scenario {nm!r}; have {list(SCENARIOS)}")
            continue
        print(f"{nm}: {SCENARIOS[nm]['label']}")
        path, *rest = draw(nm, out)
        _report(nm, *rest)
        print(f"  -> {path}")
