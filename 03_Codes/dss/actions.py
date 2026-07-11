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
suppression mapping (Eq. 130) converts the fields into a fuel reduction,
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


def suggest_resource_items(world, efficiency_target=None
                           ) -> Tuple[List[dict], List[str]]:
    """Itemized baseline pool: one editable row per staged asset.

    Items: {"kind": "depot", x, y, radius, cap, label} for every
    settlement / critical facility site, plus one {"kind":
    "road_corridor", cap} item for the thin capacity along the roads.
    The rows are meant to be listed, edited and deleted in the UI, then
    rasterized by build_resource_layer."""
    items: List[dict] = []
    for a in getattr(world, "assets", []):
        if getattr(a, "kind", "") in ("building", "critical"):
            items.append(dict(kind="depot", x=int(a.x), y=int(a.y),
                              radius=max(3, int(getattr(a, "radius", 2))
                                         + 2),
                              cap=0.8, avail=1.0, t_disp=10.0,
                              label=str(getattr(a, "name", "site"))))
    if getattr(world, "roads", None) is not None \
            and np.asarray(world.roads).any():
        items.append(dict(kind="road_corridor", cap=0.4, avail=1.0,
                          label="road corridor"))
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
                      cap=0.6, avail=1.0, t_disp=6.0,
                      label="helibase (aerial)"))
    lines = [
        f"R_cap: staged at {sum(1 for i in items if i['kind'] == 'depot')} "
        "settlement/facility depots (0.8 rcap_max each) and along the "
        "road corridor (0.4 rcap_max)",
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
            lines.append(
                f"Target {tgt:.0%} is EXACT: the baseline pool "
                f"delivered {eff0:.0%}, so every capacity was scaled "
                f"DOWN to land at {e_:.0%}. An under-resourced pool "
                "is a legitimate experiment: the interventions are "
                "ALLOWED to fail. Raise the target to re-arm.")
        for _n in range(25):
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
    dist = np.full((ny, nx), 60.0)
    m = roads.copy()
    d = 0
    while m.any() and d < 30:
        dist[m & (dist > 60.0 - 1e-9)] = 10.0 + 2.0 * d
        m = _dilate(m, 1)
        d += 1
    dist = np.where(dist > 59.0, 10.0 + 2.0 * 30, dist)
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
            rl.rtime[:] = np.minimum(rl.rtime, t0 + 0.5 * dcell)
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
            rl.rtime[disk] = np.minimum(rl.rtime[disk],
                                        (t0 + 2.0 * dcell)[disk])
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


def decision_to_resources(world, burning, regions_intensities, base=None,
                          return_actions=False):
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
    m_supp = np.zeros((ny, nx), dtype=bool)
    m_cont = np.zeros((ny, nx), dtype=bool)
    m_prot = np.zeros((ny, nx), dtype=bool)
    region_orders = []
    at_fire_all = _dilate(fire, 2)
    # the containment band sits WELL ahead of the front: fuel must be
    # cleared before the fire arrives, and Eq. 130 clears a cell over
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
        m = band_all & rb & _workable
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
            u={k: float(u.get(k, 0.0)) for k in (
                "suppression_effort", "resource_deployment",
                "containment_line", "asset_protection",
                "evacuation", "public_warning")}))
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
    if return_actions:
        return out, dict(supp=m_supp, cont=m_cont, prot=m_prot,
                         regions=region_orders)
    return out
