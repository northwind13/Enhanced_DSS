"""Layer 1 resource pool + Layer 4 decision execution.

Resource pool (Suggest resources): a baseline external resource layer
U_Res with the four fields of the resource data layer: capacity R_cap,
availability R_avail, efficiency R_eff and response time R_time. The
baseline reflects how real assets sit on a landscape: capacity is staged
at the settlements (fire stations) and along the road corridors,
efficiency follows the terrain access field, and the response time grows
with the travel distance from the road network.

Decision execution: each Local DSS turns its six intervention
intensities into the region's slice of the decision allocation U_DSS
(the same four fields), and the composed layer enters the simulation as
sim.step(resource_override=...). Nothing else touches the physics: the
suppression mapping converts the fields into a fuel reduction,
exactly as for a human-issued allocation.

    suppression_effort   -> R_cap on the active fire cells of the region
    containment_line     -> R_cap on a band ahead of the fire front
    asset_protection     -> R_cap around the protected assets
    resource_deployment  -> raises R_avail and cuts R_time (staging)
    evacuation, public_warning -> population-side orders; they do not
        move suppression resources (cost-side effect, next phase)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# Canonical resource-unit vocabulary: the SINGLE source of truth for the
# kinds of suppression resource that can be staged. The Add dropdown, the
# staged-pool list, the legend and the map glyph all key off this so the
# names and icons stay consistent.
RESOURCE_KINDS: Dict[str, dict] = {
    "depot":         dict(label="Ground depot", short="depot", aerial=False,
                          cap=0.8, radius=5, t_disp=10.0, addable=True),
    "helibase":      dict(label="Helibase (aerial)", short="helibase",
                          aerial=True, cap=0.6, radius=12, t_disp=6.0,
                          addable=True),
    "road_corridor": dict(label="Road corridor", short="road corridor",
                          aerial=False, cap=0.4, radius=0, t_disp=10.0,
                          addable=False),
}


def resource_kind_label(kind: str) -> str:
    return RESOURCE_KINDS.get(kind, {}).get("label", str(kind))


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(max(0, r))):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def suggest_resource_items(world, efficiency_target=None, density=1.0
                           ) -> Tuple[List[dict], List[str]]:
    """Itemized baseline pool: one editable row per staged asset.

    density: multiplies the staged capacity (R_cap) of every unit, i.e. how
    much suppression capacity sits on the map (the eta_cap = R_cap/R_cap_max
    term). 1.0 = nominal; <1 = a sparse, under-resourced pool that
    is ALLOWED to fail; >1 = a dense pool. This is separate from the
    effectiveness target (which adds aerial units to close reach gaps).

    Items: {"kind": "depot", x, y, radius, cap, label} for every
    settlement / critical facility site, plus one {"kind":
    "road_corridor", cap} item for the thin capacity along the roads.
    The rows are meant to be listed, edited and deleted in the UI, then
    rasterized by build_resource_layer."""
    items: List[dict] = []
    _d = float(np.clip(density, 0.1, 3.0))
    _cellm = float(getattr(world.config, "cell_size_m", 30.0))
    # a depot serves ~150 m of ground around its station regardless of
    # the grid: the radius is METER-based and converted to cells
    _rsrv = max(3, int(round(150.0 / max(_cellm, 1e-6))))
    # GROUND DEPOTS come ONLY from actual firefighting infrastructure (fire
    # stations), NOT from every building/facility. A hospital, school or
    # government office is a VALUE AT RISK to protect, not a fire brigade;
    # conflating the two gave every town a dozen fire depots. If the map has
    # no fire station, one fallback base is staged at the town centre so the
    # pool is not empty.
    _depot_cands = []
    _town_fallback = None
    for a in getattr(world, "assets", []):
        nm = str(getattr(a, "name", "")).lower()
        kind = getattr(a, "kind", "")
        is_station = ("fire" in nm and ("station" in nm or "brigade" in nm)) \
            or "fire_station" in nm
        if is_station:
            _depot_cands.append(dict(
                kind="depot", x=int(a.x), y=int(a.y),
                radius=max(_rsrv, int(getattr(a, "radius", 2)) + 2),
                cap=round(0.8 * _d, 3), avail=1.0, t_disp=10.0,
                label=str(getattr(a, "name", "fire station"))))
        elif kind == "building" and _town_fallback is None:
            _town_fallback = a         # the town centre (first building)
    if not _depot_cands and _town_fallback is not None:
        a = _town_fallback
        _depot_cands.append(dict(
            kind="depot", x=int(a.x), y=int(a.y),
            radius=max(_rsrv, int(getattr(a, "radius", 2)) + 2),
            cap=round(0.8 * _d, 3), avail=1.0, t_disp=10.0,
            label="town base (no fire station on the map)"))
    # SAFETY CAP: firefighting capacity is an operational resource; keep the
    # depot count bounded on asset-heavy maps.
    _MAX_DEPOTS = 12
    if len(_depot_cands) > _MAX_DEPOTS:
        _depot_cands.sort(key=lambda d: -float(d["radius"]))
        _depot_cands = _depot_cands[:_MAX_DEPOTS]
    items.extend(_depot_cands)
    if getattr(world, "roads", None) is not None \
            and np.asarray(world.roads).any():
        items.append(dict(kind="road_corridor", cap=round(0.4 * _d, 3),
                          avail=1.0, label="road corridor"))
    # one AERIAL unit by default: interventions are not road-bound.
    # The helibase serves a wide radius; within it the aerial share
    # substitutes for road access (derated by wind in the engine).
    cfg = world.config
    _hr = max(8, min(cfg.nx, cfg.ny) // 4)
    _hx, _hy = cfg.nx // 2, cfg.ny // 2
    for a in getattr(world, "assets", []):
        if getattr(a, "kind", "") == "building":
            _hx, _hy = int(a.x), int(a.y)
            break
    items.append(dict(kind="helibase", x=_hx, y=_hy, radius=_hr,
                      cap=round(0.6 * _d, 3), avail=1.0, t_disp=6.0,
                      label="helibase (aerial)"))
    lines = [
        f"R_cap: staged at {sum(1 for i in items if i['kind'] == 'depot')} "
        "fire-station depot(s) (0.8 rcap_max each; a town base is used only "
        "if no fire station exists) plus the road corridor (0.4) and "
        "helibase(s). Hospitals, schools, etc. are protected VALUES, not "
        "depots.",
        "R_avail: 1 wherever capacity is staged (nothing committed yet)",
        "R_eff: terrain access field G_access (workability of the ground)",
        "R_time: 10 min dispatch + 2 min per off-road cell from the "
        "road network",
        "Aerial: 1 helibase (0.6 rcap_max, 6 min dispatch); its share "
        "replaces road access in the reach product, derated by wind",
    ]
    if efficiency_target is not None:
        # GREEDY STAGING TO TARGET: add aerial units on the worst
        # risk-weighted reach gap until the expected intervention
        # effectiveness meets the requested target (or 10 additions)
        tgt = float(np.clip(efficiency_target, 0.05, 0.95))
        base0 = build_resource_layer(world, items)
        eff0, _c0 = pool_efficiency(world, base0)
        _scaled_down = False
        if eff0 > tgt + 0.005:
            # the target is EXACT in both directions: a target BELOW
            # the baseline scales every capacity DOWN until the pool
            # delivers roughly the requested effectiveness — this is
            # how under-resourced experiments ("the interventions
            # may FAIL") are staged.
            for _it2 in range(3):
                b_ = build_resource_layer(world, items)
                e_, c_ = pool_efficiency(world, b_)
                if e_ <= tgt + 0.01:
                    break
                _cap_t = tgt / max(c_["reach"], 1e-6)
                need_ = (0.05 * cfg.nx * cfg.ny
                         * max(cfg.suppression.rcap_max, 1e-6))
                have_ = 1.2 * float(
                    (b_.rcap * np.clip(b_.ravail, 0, 1)).sum())
                _sc = float(np.clip(_cap_t / max(have_ / need_,
                                                 1e-6), 0.02, 1.0))
                for it in items:
                    it["cap"] = max(0.02,
                                    round(float(it["cap"]) * _sc, 3))
            b_ = build_resource_layer(world, items)
            e_, c_ = pool_efficiency(world, b_)
            _scaled_down = True
            lines.append(
                f"Target {tgt:.0%} is EXACT: the baseline pool "
                f"delivered {eff0:.0%}, so every capacity was scaled "
                f"DOWN to land at {e_:.0%}. An under-resourced pool "
                "is a legitimate experiment: the interventions are "
                "ALLOWED to fail. Raise the target to re-arm.")
        # DO NOT RE-ARM WHAT WAS JUST DISARMED. The scale-down accepts
        # landing a hair below the target (tolerance +0.01), and the greedy
        # loop below then demanded eff >= tgt exactly, saw the shortfall and
        # staged a capacity-0.7 helibase, fourteen times the scaled-down
        # depots. The pool jumped from the requested 10% to 44% while the
        # message above still claimed it had landed at 10%.
        for _n in range(0 if _scaled_down else 25):
            base = build_resource_layer(world, items)
            eff, comp = pool_efficiency(world, base)
            if eff >= tgt:
                break
            beta = float(getattr(cfg.suppression, "beta_t", 0.03))
            acc = np.clip(world.topo.access, 0.0, 1.0)
            if getattr(base, "rair", None) is not None:
                acc = np.maximum(acc, 0.9 * np.clip(base.rair, 0, 1))
            reach = np.exp(-beta * base.rtime) * acc
            risk = _risk_field(world)
            gap = risk * (1.0 - reach)
            iy, ix = np.unravel_index(int(np.argmax(gap)), gap.shape)
            items.append(dict(kind="helibase", x=int(ix), y=int(iy),
                              radius=max(6, min(cfg.nx, cfg.ny) // 8),
                              cap=0.7, avail=1.0, t_disp=8.0,
                              label=f"helibase {_n + 2} (target "
                                    f"{tgt:.0%})"))
        base = build_resource_layer(world, items)
        eff, comp = pool_efficiency(world, base)
        lines.append(f"Target {tgt:.0%}: staged to expected "
                     f"effectiveness {eff:.0%} (reach "
                     f"{comp['reach']:.0%} x capacity "
                     f"{comp['capacity']:.0%})")
        if eff < tgt - 0.005:
            lines.append(
                "Note: the target was not fully reached. Reach is "
                "bounded by terrain access and flight time on this "
                "map: risk far from every base keeps a reach "
                "discount no matter how many units are staged. Add "
                "helibases NEAR the uncovered risk or raise depot "
                "capacities to close the gap.")
    return items, lines


def build_resource_layer(world, items: List[dict]):
    """Rasterize the item rows into U_Res = [R_cap, R_avail, R_eff,
    R_time]. R_eff and R_time always come from the map (terrain access
    and road-network distance); the items place the capacity."""
    from disaster_phyengine.layers import ResourceLayer
    cfg = world.config
    ny, nx = cfg.ny, cfg.nx
    rl = ResourceLayer.none(ny, nx)

    roads = getattr(world, "roads", None)
    roads = (np.asarray(roads, dtype=bool)
             if roads is not None else np.zeros((ny, nx), dtype=bool))
    # off-road travel: ~2 min per 30 m of ground, SCALED by the cell
    # size (on a 5 m grid the old per-cell constant meant 24 min per
    # 100 m and the whole pool looked unreachable)
    _tc = 2.0 * float(cfg.cell_size_m) / 30.0
    dist = np.full((ny, nx), 60.0)
    m = roads.copy()
    d = 0
    _dmax = max(30, int(round(30 * 30.0 / max(cfg.cell_size_m, 1e-6))))
    while m.any() and d < _dmax:
        dist[m & (dist > 60.0 - 1e-9)] = 10.0 + _tc * d
        m = _dilate(m, 1)
        d += 1
    dist = np.where(dist > 59.0, 10.0 + _tc * _dmax, dist)
    rl.rtime[:] = dist
    rl.reff[:] = np.clip(world.topo.access, 0.0, 1.0)

    cap = np.zeros((ny, nx))
    avail = np.zeros((ny, nx))
    yy, xx = np.ogrid[0:ny, 0:nx]
    for it in items:
        av = float(np.clip(it.get("avail", 1.0), 0.0, 1.0))
        if it.get("kind") == "road_corridor":
            band = _dilate(roads, 1)
            cap[band] = np.maximum(cap[band],
                                   float(np.clip(it.get("cap", 0.4),
                                                 0.0, 1.0)))
            avail[band] = np.maximum(avail[band], av)
        elif it.get("kind") == "helibase":
            # AERIAL UNITS ARE MOBILE: the aircraft flies to the fire
            # wherever it is, so the air cover spans the WHOLE map
            # and only the flight time grows with distance (reach
            # decays through exp(-beta_t R_time), wind derates in the
            # engine). The station disk still holds the STAGED
            # capacity; away from it the deployment order moves
            # capacity forward, as with any unit.
            r = max(1, int(it.get("radius", 8)))
            c = float(np.clip(it.get("cap", 0.6), 0.0, 1.0))
            disk = ((xx - int(it["x"])) ** 2
                    + (yy - int(it["y"])) ** 2 <= r * r)
            cap[disk] = np.maximum(cap[disk], c)
            avail[disk] = np.maximum(avail[disk], av)
            if rl.rair is None:
                rl.rair = np.zeros((ny, nx))
            rl.rair[:] = 1.0
            # flight time everywhere: dispatch + crow-line distance
            t0 = float(max(0.0, it.get("t_disp", 6.0)))
            dcell = np.sqrt((xx - int(it["x"])) ** 2
                            + (yy - int(it["y"])) ** 2)
            _fc = 0.5 * float(cfg.cell_size_m) / 30.0   # flight min/cell
            rl.rtime[:] = np.minimum(rl.rtime, t0 + _fc * dcell)
        elif it.get("kind") == "depot":
            r = max(1, int(it.get("radius", 4)))
            c = float(np.clip(it.get("cap", 0.8), 0.0, 1.0))
            disk = ((xx - int(it["x"])) ** 2
                    + (yy - int(it["y"])) ** 2 <= r * r)
            cap[disk] = np.maximum(cap[disk], c)
            avail[disk] = np.maximum(avail[disk], av)
            # inside the station area the response clock starts at the
            # depot's own dispatch time (crew on site), growing with the
            # distance from the depot
            t0 = float(max(0.0, it.get("t_disp", 10.0)))
            dcell = np.sqrt((xx - int(it["x"])) ** 2
                            + (yy - int(it["y"])) ** 2)
            _gc = 2.0 * float(cfg.cell_size_m) / 30.0   # ground min/cell
            rl.rtime[disk] = np.minimum(rl.rtime[disk],
                                        (t0 + _gc * dcell)[disk])
    rl.rcap[:] = cap * float(cfg.suppression.rcap_max)
    rl.ravail[:] = avail
    return rl


def resource_suggestion(world) -> Tuple[object, List[str]]:
    """Convenience wrapper: itemized suggestion rasterized to a layer."""
    items, lines = suggest_resource_items(world)
    return build_resource_layer(world, items), lines


def _risk_field(world):
    """Risk map shared by the network/pool planners: 0.45 ROS + 0.35
    value priority + 0.20 fuel load, water zeroed."""
    from disaster_phyengine.behavior import rate_of_spread_field
    from disaster_phyengine.config import FUEL_NAME_TO_ID
    ros = np.asarray(rate_of_spread_field(world), dtype=float)
    ros = ros / (ros.max() + 1e-9)
    pri = np.asarray(world.priority_field(), dtype=float)
    pri = pri / (pri.max() + 1e-9)
    fload = np.clip(np.asarray(world.fuel.fload, dtype=float), 0.0, 1.0)
    risk = 0.45 * ros + 0.35 * pri + 0.20 * fload
    risk[world.fuel.ftype == FUEL_NAME_TO_ID["water"]] = 0.0
    return risk


def pool_efficiency(world, base):
    """Expected intervention effectiveness of the STAGED pool in [0,1].

    Two independent limits multiply:
      reach score    = risk-weighted mean of exp(-beta_t R_time) x
                       G_access (can the crews physically work the
                       ground that matters?)
      capacity score = staged deployable capacity (with the 1.2 surge)
                       against the 5% coverage reference need used by
                       the z9 feature
    Aerial share substitutes for road access in the reach product at
    a 0.9 factor (average weather derating of the air ops).
    Returns (score, dict(reach=..., capacity=..., air=...))."""
    cfg = world.config
    beta = float(getattr(cfg.suppression, "beta_t", 0.03))
    acc = np.clip(world.topo.access, 0.0, 1.0)
    _ra = getattr(base, "rair", None)
    if _ra is not None:
        acc = np.maximum(acc, 0.9 * np.clip(_ra, 0.0, 1.0))
    reach = np.exp(-beta * base.rtime) * acc
    risk = _risk_field(world)
    r_score = float((risk * reach).sum() / max(risk.sum(), 1e-9))
    air_cov = (0.0 if _ra is None else
               float((risk * (np.clip(_ra, 0, 1) > 0.5)).sum()
                     / max(risk.sum(), 1e-9)))
    ncells = cfg.nx * cfg.ny
    need = 0.05 * ncells * max(cfg.suppression.rcap_max, 1e-6)
    have = 1.2 * float((base.rcap * np.clip(base.ravail, 0, 1)).sum())
    c_score = float(np.clip(have / need, 0.0, 1.0))
    return r_score * c_score, dict(reach=r_score, capacity=c_score,
                                   air=air_cov)


# ----------------------------------------------------------------------
# EFFECT GRAMMAR for runtime-defined actuators. The generative stage
# may DEFINE a new actuator as data: up to three clauses, each a
# verified physical effect applied to a named sector at a range from
# the front. The engine never runs model-written code; it interprets
# this closed grammar, and every definition still has to win the
# simulation gates before a rule may order it.
EFFECTS = ("wet", "clear", "ignite", "coat", "evacuate", "prime",
           "draft")
SECTORS = ("head", "flank", "rear", "ring", "at_fire", "assets",
           "populated")


def _sector_mask(sector, fire, world, rin, rout, cosang):
    ny, nx = world.config.ny, world.config.nx
    if sector == "populated":
        return np.asarray(world.value.vpop, dtype=float) > 1e-6
    if sector == "assets":
        yy, xx = np.ogrid[0:ny, 0:nx]
        m = np.zeros((ny, nx), dtype=bool)
        _thr = _dilate(fire, 15)
        for a in getattr(world, "assets", []):
            ax = min(max(int(a.x), 0), nx - 1)
            ay = min(max(int(a.y), 0), ny - 1)
            if _thr[ay, ax]:
                r = max(2, int(getattr(a, "radius", 2)))
                m |= (xx - a.x) ** 2 + (yy - a.y) ** 2 <= r * r
        return m
    if not fire.any():
        return np.zeros((ny, nx), dtype=bool)
    if sector == "at_fire":
        return _dilate(fire, 2)
    band = _dilate(fire, rout) & ~_dilate(fire, max(1, rin))
    if sector == "ring" or cosang is None:
        return band
    if sector == "head":
        return band & (cosang > 0.35)
    if sector == "flank":
        return band & (np.abs(cosang) <= 0.35)
    if sector == "rear":
        return band & (cosang < -0.35)
    return band


def _diggable_mask(world):
    """Ground a dozer can cut and a firing crew may light.

    Built-up cells and water are neither. A settlement in the path of a
    fire is DEFENDED, not levelled, and the DSS was able to order a
    containment line straight across one because the band was chosen on
    reachability alone.
    """
    import numpy as _np
    ft = _np.asarray(world.fuel.ftype)
    ok = (ft != 5) & (ft != 6)
    try:
        ok &= ~((_np.asarray(world.value.vbld) > 1e-6)
                | (_np.asarray(world.value.vcrit) > 1e-6))
    except Exception:
        pass
    return ok


def apply_actuator_clauses(out, world, fire, region_mask, clauses,
                           intensity, cap_max, workable, cosang,
                           cells_out=None):
    """Interpret one runtime-defined actuator at its firing intensity.
    cells_out: optional boolean array; the touched cells are OR-ed in
    so the map can draw the actuator's symbol exactly where it acts."""
    for cl in clauses[:3]:
        eff = str(cl.get("effect", ""))
        sec = str(cl.get("sector", "ring"))
        rr = cl.get("range") or [2, 6]
        rin, rout = int(rr[0]), int(rr[1])
        a = float(np.clip(float(cl.get("amount", 0.8))
                          * float(intensity), 0.0, 1.0))
        if a <= 0.05:
            continue
        m = _sector_mask(sec, fire, world, rin, rout, cosang)             & region_mask
        if eff in ("wet", "clear", "ignite", "coat"):
            m &= workable if eff in ("wet", "clear") else m
        # NOT THROUGH A SETTLEMENT, AND NOT ON WATER. Clearing digs ground
        # and lighting sets it alight; neither belongs on built-up cells,
        # and a generated actuator must not be able to order what the base
        # containment channel is forbidden to order.
        if eff in ("clear", "ignite"):
            m &= _diggable_mask(world)
        if not m.any():
            continue
        if cells_out is not None:
            cells_out |= m
        if eff == "wet":
            out.rcap[m] = np.maximum(out.rcap[m], a * cap_max)
            out.ravail[m] = np.maximum(out.ravail[m], 0.3 + 0.7 * a)
        elif eff == "clear":
            if out.rcut is None:
                out.rcut = np.zeros_like(out.rcap)
            out.rcut[m] = 1.0
            out.rcap[m] = np.maximum(out.rcap[m],
                                     max(a, 0.8) * cap_max)
            out.ravail[m] = np.maximum(out.ravail[m], 0.8)
        elif eff == "ignite":
            if out.rburn is None:
                out.rburn = np.zeros_like(out.rcap)
            out.rburn[m] = np.maximum(out.rburn[m], a)
        elif eff == "coat":
            if out.rret is None:
                out.rret = np.zeros_like(out.rcap)
            out.rret[m] = np.maximum(out.rret[m], a)
        elif eff == "evacuate":
            if out.revac is None:
                out.revac = np.zeros_like(out.rcap)
            _pm = m & (np.asarray(world.value.vpop) > 1e-6)
            out.revac[_pm] = np.maximum(out.revac[_pm], a)
        elif eff == "prime":
            if out.rwarn is None:
                out.rwarn = np.zeros_like(out.rcap)
            _pm = m & (np.asarray(world.value.vpop) > 1e-6)
            out.rwarn[_pm] = np.maximum(out.rwarn[_pm], a)
        elif eff == "draft":
            _wat = np.asarray(world.fuel.ftype == 5)
            if _wat.any():
                ny, nx = world.config.ny, world.config.nx
                _dw = np.full((ny, nx), 30.0)
                _fr = _wat.copy()
                for _d in range(30):
                    _nw = _fr & (_dw > _d)
                    _dw[_nw] = _d
                    _fr = _dilate(_fr, 1)
                _bo = 1.0 + 0.8 * a / (1.0 + _dw / 15.0)
                _mb = m & (out.rcap > 1e-6)
                out.rcap[_mb] = np.minimum(out.rcap[_mb] * _bo[_mb],
                                           1.5 * cap_max)


def decision_to_resources(world, burning, regions_intensities, base=None,
                          return_actions=False, macros=None):
    """Compose U_DSS from every region's intervention intensities.

    burning: boolean burning mask of the CURRENT state (drives where the
    suppression effort and the containment band are laid).
    regions_intensities: list of (region, {intervention: intensity}).
    base: the Layer-1 baseline pool; it bounds what can be ordered (the
    DSS allocates the staged pool, it does not create trucks).
    Returns a ResourceLayer for sim.step(resource_override=...)."""
    from disaster_phyengine.layers import ResourceLayer
    cfg = world.config
    ny, nx = cfg.ny, cfg.nx
    fire = np.asarray(burning, dtype=bool)
    cap_max = float(cfg.suppression.rcap_max)
    out = ResourceLayer.none(ny, nx)
    if base is not None:
        out.ravail[:] = base.ravail
        out.reff[:] = base.reff
        out.rtime[:] = base.rtime
        if getattr(base, "rair", None) is not None:
            out.rair = base.rair.copy()
    else:
        out.reff[:] = np.clip(world.topo.access, 0.0, 1.0)
        out.rtime[:] = 20.0
    if out.rcut is None:
        out.rcut = np.zeros((ny, nx))
    if out.revac is None:
        out.revac = np.zeros((ny, nx))
    # cells that no crew can physically work (beyond reach) take no
    # allocation: ordering there would only burn budget and money
    _beta = float(getattr(cfg.suppression, 'beta_t', 0.03))
    _acc0 = np.clip(world.topo.access, 0.0, 1.0)
    if getattr(out, "rair", None) is not None:
        _acc0 = np.maximum(_acc0, 0.9 * np.clip(out.rair, 0.0, 1.0))
    _reach0 = np.exp(-_beta * out.rtime) * _acc0
    _rb_w = np.ones((ny, nx))     # global share weights (per region)
    _macro_cells: dict = {}       # macro name -> cells it acted on
    m_supp = np.zeros((ny, nx), dtype=bool)
    m_cont = np.zeros((ny, nx), dtype=bool)
    m_prot = np.zeros((ny, nx), dtype=bool)
    region_orders = []
    at_fire_all = _dilate(fire, 2)
    # the containment band sits WELL ahead of the front: fuel must be
 # cleared before the fire arrives, and30 clears a cell over
    # multiple reference steps, so ordering at the last moment is useless
    band_all = _dilate(fire, 10) & ~_dilate(fire, 4)
    # ... and it is DUG IN THE SPREAD DIRECTION: the line goes where
    # the fire is heading (wind-toward head plus the flanks), not
    # behind it. Array-coordinate wind vector = (cos wwd, +sin wwd),
    # verified against free-run spread. The rear quarter is dropped
    # unless the fire is tiny (mop-up digs anywhere).
    if fire.any() and int(fire.sum()) >= 50:
        _fy, _fx = np.where(fire)
        _cx0, _cy0 = float(_fx.mean()), float(_fy.mean())
        _wd0 = float(np.mean(world.meteo.wwd[fire]))
        _wvx, _wvy = np.cos(_wd0), np.sin(_wd0)
        _yyb, _xxb = np.ogrid[0:ny, 0:nx]
        _ox = _xxb - _cx0
        _oy = _yyb - _cy0
        _nrm = np.sqrt(_ox * _ox + _oy * _oy) + 1e-9
        _down = (_ox * _wvx + _oy * _wvy) / _nrm > -0.3
        band_all &= _down
        _down_ok = True
        _cosang = (_ox * _wvx + _oy * _wvy) / _nrm
    if "_cosang" not in dir():
        _cosang = None
    yy, xx = np.ogrid[0:ny, 0:nx]
    for region, u in regions_intensities:
        sy, sx = region.slices()
        rb = np.zeros((ny, nx), dtype=bool)
        rb[sy, sx] = True
        u1 = float(u.get("suppression_effort", 0.0))
        u2 = float(u.get("resource_deployment", 0.0))
        u3 = float(u.get("containment_line", 0.0))
        u4 = float(u.get("asset_protection", 0.0))
        # MOP-UP: once the fire is down to scattered smolders the
        # crews hike in; the reachability filter only applies while
        # the fire is large enough that working beyond reach would
        # waste the budget of the active front
        _mopup = int(fire.sum()) < 50
        _workable = (_reach0 > 0.05) | _mopup
        # YOU DO NOT BULLDOZE A FUEL BREAK THROUGH A TOWN. A containment
        # line is dug ground: it needs terrain a dozer can cut. Built-up
        # cells and water are not that, and a settlement in the path of a
        # fire is DEFENDED (asset protection) rather than levelled. The
        # band was previously chosen on reachability alone, so the DSS
        # could and did order a line straight across a settlement.
        _diggable = _workable & _diggable_mask(world)
        m = at_fire_all & rb & _workable
        # DIRECT-ATTACK DOCTRINE: a burning cell that is worked at
        # all is worked at FULL strength; you do not send half a
        # crew to a flame front. The rule intensity u1 decides HOW
        # WIDELY the attack is funded (through the utility ranking
        # and the budget cut), not the per-cell strength. Without
        # this floor the rule combination (sum wv / max(sum w, 1))
        # kept every big-fire order below the commitment thresholds
        # and no open-terrain cell could ever be wetted or quenched.
        _u1_eff = (max(u1, 0.85)
                   if (u1 > 0.05 or (_mopup and u1 > 0.02)) else u1)
        out.rcap[m] = np.maximum(out.rcap[m], _u1_eff * cap_max)
        # committed crews are AVAILABLE crews: the attack cells get
        # the availability of their own commitment, not the region's
        # diluted deployment order
        out.ravail[m] = np.maximum(out.ravail[m],
                                   0.3 + 0.7 * _u1_eff)
        if u1 > 0.05:
            m_supp |= m
        m = band_all & rb & _diggable
        out.rcap[m] = np.maximum(out.rcap[m], u3 * cap_max)
        if u3 > 0.05:
            m_cont |= m
            out.rcut[m] = 1.0     # dig HERE and nowhere else
        if u4 > 0.0:
            # protect only assets the fire actually THREATENS (within
            # ~15 cells of the front): pre-wetting a town the fire
            # cannot reach wastes budget and paints the map blue for
            # no reason (user complaint), and it must NEVER dig
            _threat = _dilate(fire, 15)
            prot = np.zeros((ny, nx), dtype=bool)
            for a in getattr(world, "assets", []):
                if not (region.x0 <= a.x < region.x1
                        and region.y0 <= a.y < region.y1):
                    continue
                if not _threat[min(max(int(a.y), 0), ny - 1),
                               min(max(int(a.x), 0), nx - 1)]:
                    continue
                r = max(2, int(getattr(a, "radius", 2)))
                prot |= (xx - a.x) ** 2 + (yy - a.y) ** 2 <= r * r
            m = prot & rb & _workable
            out.rcap[m] = np.maximum(out.rcap[m], u4 * cap_max)
            if u4 > 0.05:
                m_prot |= m
        u5 = float(u.get("evacuation", 0.0))
        if u5 > 0.3:
            _pop_m = np.zeros((ny, nx), dtype=bool)
            _pop_m[sy, sx] = np.asarray(
                world.value.vpop[sy, sx], dtype=float) > 1e-6
            out.revac[_pop_m] = np.maximum(out.revac[_pop_m], u5)
        u7 = float(u.get("tactical_burn", 0.0))
        if u7 > 0.3:
            # COUNTER-FIRE: the firing crew lights a strip BETWEEN the
            # containment band and the front, on the downwind side, so
            # the counter fire eats the fuel the head fire is running
            # toward. Fire is fire: the engine will spread it, and a
            # badly judged order makes things worse, which is exactly
            # what the forecast gates are for.
            if out.rburn is None:
                out.rburn = np.zeros((ny, nx))
            _strip = (_dilate(fire, 6) & ~_dilate(fire, 3)
                      & rb & _workable)
            if locals().get("_down_ok"):
                _strip &= _down
            out.rburn[_strip] = np.maximum(out.rburn[_strip], u7)
            # the library actuator leaves the same per-cell trace a
            # runtime macro does: what acts on the map is drawn on it
            _mc7 = _macro_cells.setdefault(
                "tactical_burn", np.zeros((ny, nx), dtype=bool))
            _mc7 |= _strip
        u9 = float(u.get("retardant_drop", 0.0))
        if u9 > 0.3:
            # AERIAL RETARDANT/SOIL: coat the HEAD sector just ahead
            # of the front. Aerial delivery ignores road access; the
            # pass is narrow (2-5 cells out) because a wide carpet is
            # neither affordable nor needed: the head is what kills.
            # aircraft must draw the load from somewhere: a map
            # without any water body cannot support aerial drops
            if not np.asarray(world.fuel.ftype == 5).any():
                u9 = 0.0
        if u9 > 0.3:
            if out.rret is None:
                out.rret = np.zeros((ny, nx))
            _pass = _dilate(fire, 5) & ~_dilate(fire, 2) & rb
            if locals().get("_down_ok"):
                _pass &= _down
            out.rret[_pass] = np.maximum(out.rret[_pass], u9)
            _mc9 = _macro_cells.setdefault(
                "retardant_drop", np.zeros((ny, nx), dtype=bool))
            _mc9 |= _pass
        # RUNTIME-DEFINED actuators: any macro carrying clauses that
        # this region's rules fired is interpreted here
        for _mn, _md in (macros or {}).items():
            _ui = float(u.get(_mn, 0.0))
            if _ui <= 0.3:
                continue
            _mc = _macro_cells.setdefault(
                _mn, np.zeros((ny, nx), dtype=bool))
            _cls = _md.get("clauses")
            if _cls:
                apply_actuator_clauses(
                    out, world, fire, rb, _cls, _ui, cap_max,
                    _workable, locals().get("_cosang"),
                    cells_out=_mc)
            else:
                # composition macro: its footprint is where its
                # component channels act in this region, so the tag
                # lands on the worked cells, not on a region badge
                for _bi, _bw in _md.get("composition", []):
                    if float(_bw) * _ui <= 0.15:
                        continue
                    if _bi == "suppression_effort":
                        _mc |= at_fire_all & rb & _workable
                    elif _bi == "containment_line":
                        _mc |= band_all & rb & _diggable
                    elif _bi in ("evacuation", "public_warning"):
                        _mc |= rb & (np.asarray(world.value.vpop)
                                     > 1e-6)
                    elif _bi == "retardant_drop" \
                            and getattr(out, "rret", None) is not None:
                        _mc |= (out.rret > 0.3) & rb
        u6 = float(u.get("public_warning", 0.0))
        if u6 > 0.3:
            # WARNING PRIMES EVACUATION: no physical effect on its
            # own, but a warned population responds faster once an
            # evacuation order lands (readiness, not movement)
            if out.rwarn is None:
                out.rwarn = np.zeros((ny, nx))
            _pop_w = np.zeros((ny, nx), dtype=bool)
            _pop_w[sy, sx] = np.asarray(
                world.value.vpop[sy, sx], dtype=float) > 1e-6
            out.rwarn[_pop_w] = np.maximum(out.rwarn[_pop_w], u6)
        out.ravail[sy, sx] = np.clip(
            np.maximum(out.ravail[sy, sx], 0.3 + 0.7 * u2), 0.0, 1.0)
        out.rtime[sy, sx] = out.rtime[sy, sx] * (1.0 - 0.6 * u2)
        if base is not None:
            # the pool bounds the order, but DEPLOYMENT moves capacity
            # forward: a fully deployed region may stage up to its own
            # ordered tempo beyond the passive 0.2 floor
            floor = (0.2 + 0.5 * u2) * cap_max
            lim = 1.5 * np.maximum(base.rcap[sy, sx], floor)
            out.rcap[sy, sx] = np.minimum(out.rcap[sy, sx], lim)
        u8 = float(u.get("water_drafting", 0.0))
        if u8 > 0.3:
            # WATER DRAFTING: engines and helicopters refill from the
            # nearest water body (lake, river, sea) instead of driving
            # back to the depot, so the SUSTAINED capacity near water
            # rises above what the staged pool alone could hold. It is
            # applied AFTER the pool bound on purpose: the water is a
            # real extra source, not a reallocation. Far from water
            # the order does nothing; a map without water ignores it.
            _wat = np.asarray(world.fuel.ftype == 5)
            if _wat.any():
                # chebyshev distance-to-water by iterative dilation
                # (numpy only; capped at 30 cells = 900 m, beyond
                # which drafting saves nothing anyway)
                _dw = np.full((ny, nx), 60.0)
                _frontier = _wat.copy()
                for _d in range(60):
                    _new = _frontier & (_dw > _d)
                    _dw[_new] = _d
                    _frontier = _dilate(_frontier, 1)
                # GROUND shuttle: engines ferry water a few hundred
                # metres. AIR shuttle: a helicopter drafts from ANY
                # water body on the map, so wherever air cover exists
                # the boost barely decays with distance; that is what
                # lets one region's front drink another region's lake.
                _bo_g = 0.8 * u8 / (1.0 + _dw / 15.0)
                _bo_a = 0.8 * u8 / (1.0 + _dw / 45.0)
                _air = (np.clip(out.rair, 0.0, 1.0)
                        if getattr(out, "rair", None) is not None
                        else np.zeros((ny, nx)))
                _boost = 1.0 + np.maximum(_bo_g, _bo_a * _air)
                _mb = rb & (out.rcap > 1e-6)
                out.rcap[_mb] = np.minimum(
                    out.rcap[_mb] * _boost[_mb], 1.5 * cap_max)
                _mc8 = _macro_cells.setdefault(
                    "water_drafting", np.zeros((ny, nx), dtype=bool))
                _mc8 |= (_mb & (_boost > 1.05))
        _shr = float(u.get("_share", 1.0))
        if abs(_shr - 1.0) > 1e-6:
            # GLOBAL STEERING: the region's share scales its cells'
            # standing in the budget ranking, so capacity flows from
            # monitored regions toward the global focus
            _rb_w[sy, sx] = _shr
        region_orders.append(dict(
            name=getattr(region, "name", "?"),
            box=(int(region.x0), int(region.y0),
                 int(region.x1), int(region.y1)),
            u={k: float(v) for k, v in u.items()
               if k != "_share"}))
    # RESOURCE CONSERVATION BY CONCENTRATION: the DSS allocates the
    # staged pool (surge 1.2), and like a real incident commander it
    # CONCENTRATES: candidate cells are ranked by utility
    # (ordered intensity x reachability x value priority) and funded
    # AT FULL STRENGTH in that order until the budget runs out; the
    # rest gets nothing. Uniform down-scaling would butter the pool
    # over thousands of cells at homeopathic strength (no wetting,
    # no knockdown, full response cost) - measured and rejected.
    if base is not None:
        pool_total = float((base.rcap
                            * np.clip(base.ravail, 0, 1)).sum())
        budget = 1.2 * max(pool_total, 1e-6)
        av = np.clip(out.ravail, 0.0, 1.0)
        cost_cell = out.rcap * av
        committed = float(cost_cell.sum())
        if committed > budget:
            prio = np.asarray(world.priority_field(), dtype=float)
            pmax = float(prio.max())
            if pmax > 1e-9:
                prio = prio / pmax
            # extinguishing the ACTIVE fire is the mission, not a
            # competitor of asset prophylaxis: cells on or beside
            # flames outrank a same-capacity protective ring around
            # an unthreatened town, so a remote forest fire is never
            # defunded in favour of watching the cities
            util = (out.rcap * _reach0 * _rb_w
                    * (1.0 + 2.0 * prio + 1.5 * at_fire_all))
            ys_, xs_ = np.where(out.rcap > 1e-9)
            order = np.argsort(-util[ys_, xs_])
            cum = np.cumsum(cost_cell[ys_, xs_][order])
            cut = int(np.searchsorted(cum, budget, side="right"))
            drop = order[cut:]
            out.rcap[ys_[drop], xs_[drop]] = 0.0
        funded = out.rcap > 1e-9
        m_supp &= funded
        m_cont &= funded
        m_prot &= funded
        # WHAT THE ORDERS ASKED FOR AGAINST WHAT THERE WAS. Capacity here
        # is a FLOW, how much force can act per minute, so it does not
        # deplete; scarcity shows as demand exceeding the budget and cells
        # going unfunded. Without these two numbers the reader could not
        # tell a comfortable response from one that is already short.
        _demand = committed
        _budget = budget
    else:
        _demand = _budget = None
    if return_actions:
        return out, dict(supp=m_supp, cont=m_cont, prot=m_prot,
                         demand=_demand, budget=_budget,
                         regions=region_orders,
                         macro_cells={k: v for k, v in
                                      _macro_cells.items()
                                      if v.any()})
    return out
