"""The ten bounded observation features, per Local DSS region.

Each feature is a scalar in [0, 1] computed over the agent's cells from the
CURRENT observation (phase 1a: sensor-emulated, i.e. read directly from the
live simulation state; the sensor catalogue with partial coverage arrives in
a later phase). Every formula is explicit and hand-checkable; the grounding
of each feature in wildfire science is documented in the System Description page (DSS section).
"""

from __future__ import annotations

import numpy as np

FEATURE_ORDER = [
    "fire_intensity",         # Byram/Rothermel front strength
    "spread_potential",       # Rothermel steady-state ROS
    "weather_severity",       # FWI-style weather danger
    "ignition_proximity",     # closeness of active fire to the region
    "fuel_load",              # available fine fuel
    "asset_exposure",         # values at risk in the region
    "resource_accessibility", # how workable the terrain is (SDI-like)
    "access_road_status",     # road/egress availability
    "suppression_availability",  # deployable capacity present
    "temporal_urgency",       # time pressure to act
]

# display metadata: math symbol z_i, short name and WHAT it measures
FEATURE_META = [
    ("fire_intensity", "z_1", "fire intensity",
     "mean normalized fireline intensity I over the region's burning "
     "cells [0,1]"),
    ("spread_potential", "z_2", "spread potential",
     "mean rate of spread R_spread of the region, normalized by the "
     "calibration maximum [0,1]"),
    ("weather_severity", "z_3", "weather severity",
     "FWI-style danger from wind, temperature, humidity and dryness "
     "[0,1]"),
    ("ignition_proximity", "z_4", "ignition proximity",
     "closeness of the nearest active fire to the region (1 = inside) "
     "[0,1]"),
    ("fuel_load", "z_5", "fuel load",
     "mean available fine fuel F_load still standing in the region "
     "[0,1]"),
    ("asset_exposure", "z_6", "asset exposure",
     "protection-weighted values at risk (buildings, facilities, "
     "population) present in the region [0,1]"),
    ("resource_accessibility", "z_7", "resource accessibility",
     "how workable the ground is for crews: terrain access field "
     "G_access averaged over the region [0,1]"),
    ("access_road_status", "z_8", "access / road status",
     "road and egress availability of the region (road density, open "
     "routes) [0,1]"),
    ("suppression_availability", "z_9", "suppression availability",
     "deployable suppression capacity currently present in the region "
     "[0,1]"),
    ("temporal_urgency", "z_10", "temporal urgency",
     "time pressure to act: fire inside or bearing down on the region "
     "soon [0,1]"),
]
FEATURE_SYM = {k: sym for k, sym, _, _ in FEATURE_META}
FEATURE_NAME = {k: nm for k, _, nm, _ in FEATURE_META}
FEATURE_MEASURES = {k: ms for k, _, _, ms in FEATURE_META}


# which OBSERVED channels each feature's extraction map consumes; the
# features absent here read known priors / the weather service and carry
# full confidence (thesis Layer 2: the (value, confidence) pair is the
# complete interface between perception and reasoning)
FEATURE_CHANNELS = {
    "fire_intensity": ("burning", "intensity"),
    "ignition_proximity": ("burning",),
    "fuel_load": ("fload",),
    "temporal_urgency": ("burning",),
}


def feature_confidence(network, region) -> dict:
    """{feature: confidence in [0,1]}: each feature inherits the weakest
    confidence of the observation channels its extraction consumes;
    prior-driven features carry confidence one."""
    if network is None:
        return {k: 1.0 for k in FEATURE_ORDER}
    cc = network.region_conf_components(region)
    out = {}
    for k in FEATURE_ORDER:
        chs = FEATURE_CHANNELS.get(k)
        out[k] = (min(float(cc[c]) for c in chs) if chs else 1.0)
    return out


def ten_features(sim, region, network=None, pool=None) -> dict:
    """Return {feature_name: value in [0,1]} for one agent region.

    pool: the STAGED external resource layer (Layer 1). When given, z9
    reads the staged pool (resources exist and are deployable); the
    world's live resource field is the fallback.

    With a SensorNetwork the sensed state components (channels
    B, F_load, I, tau) come from the fused observation - stale or uncovered
    cells carry old values - while static prior maps (terrain, fuel type,
    values, own resources), fuel moisture and the weather field come from
    maps and the meteorological service. The matching confidence is
    network.region_conf(region) (see sensors.py confidence model)."""
    world = sim.world
    cfg = sim.cfg
    sy, sx = region.slices()

    if network is not None:
        B_full = network.obs["burning"] > 0.5
        B = B_full[sy, sx]
        I = network.obs["intensity"][sy, sx]
        fload = network.obs["fload"][sy, sx]
    else:
        B_full = sim.state.burning > 0.5
        B = sim.state.burning[sy, sx] > 0.5
        I = sim.state.intensity[sy, sx]
        fload = sim.state.fload[sy, sx]
    ftype = world.fuel.ftype[sy, sx]
    burnable = ftype > 0

    # --- z1 fire intensity: strongest front burning inside the region
    z1 = float(I.max()) if B.any() else 0.0

    # --- z2 spread potential: mean Rothermel ROS on burnable cells,
    #     normalized by a 30 m/min reference head-fire rate
    from disaster_phyengine.behavior import rate_of_spread_field
    ros = rate_of_spread_field(world)[sy, sx]
    z2 = float(np.clip(ros[burnable].mean() / 30.0, 0.0, 1.0)) \
        if burnable.any() else 0.0

    # --- z3 weather severity: wind + dryness + heat (FWI-flavoured mix)
    ws = float(world.meteo.wws[sy, sx].mean())
    rh = float(world.meteo.rh[sy, sx].mean())
    tt = float(world.meteo.temp[sy, sx].mean())
    z3 = float(np.clip(0.45 * min(ws / 20.0, 1.0)
                       + 0.35 * (1.0 - rh / 100.0)
                       + 0.20 * np.clip((tt - 15.0) / 25.0, 0.0, 1.0),
                       0.0, 1.0))

    # --- z4 ignition proximity: how close the nearest active fire is to
    #     this region (1 = burning inside, 0 = farther than the map diagonal)
    ny, nx = sim.state.burning.shape
    diag = float(np.hypot(nx, ny))
    if B.any():
        z4 = 1.0
    else:
        fy, fx = np.where(B_full)
        if fx.size == 0:
            z4 = 0.0
        else:
            cx = np.clip(fx, region.x0, region.x1 - 1)
            cy = np.clip(fy, region.y0, region.y1 - 1)
            d = float(np.hypot(fx - cx, fy - cy).min())
            z4 = float(np.clip(1.0 - d / (0.5 * diag), 0.0, 1.0))

    # --- z5 fuel load: mean remaining fuel on burnable cells
    z5 = float(fload[burnable].mean()) if burnable.any() else 0.0

    # --- z6 asset exposure: strongest protection priority in the region
    vprio = world.priority_field()[sy, sx]
    z6 = float(vprio.max()) if vprio.size else 0.0

    # --- z7 resource accessibility: mean terrain workability
    z7 = float(world.topo.access[sy, sx].mean())

    # --- z8 access and road status: road coverage vs a 5% reference density
    roads = getattr(world, "roads", None)
    if roads is None:
        z8 = 0.0
    else:
        frac = float(np.asarray(roads, dtype=bool)[sy, sx].mean())
        z8 = float(np.clip(frac / 0.05, 0.0, 1.0))

    # --- z9 suppression availability: deployable capacity present in the
    #     region vs the reference capacity over 20% of the cells
    # z9 measures SUPPLY, not average field strength: the staged
    # deployable capacity in the region against a reference need of
    # 5% coverage at full capacity (a depot serving a district reads
    # ~0.4-0.8, an empty wilderness region reads ~0). A regional MEAN
    # would dilute a full depot over thousands of cells to ~0.03 and
    # falsely starve suppression feasibility.
    _res = pool if pool is not None else world.resource
    supply = float((_res.rcap[sy, sx] * _res.ravail[sy, sx]).sum())
    need = max(cfg.suppression.rcap_max, 1e-6) * 0.05 * max(
        (sy.stop - sy.start) * (sx.stop - sx.start), 1)
    z9 = float(np.clip(supply / need, 0.0, 1.0))

    # --- z10 temporal urgency: fire at/near the region plus how much of the
    #     region is actively burning right now
    z10 = float(np.clip(0.6 * z4 + 0.4 * min(1.0, 10.0 * B.mean()),
                        0.0, 1.0))

    vals = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10]
    return dict(zip(FEATURE_ORDER, vals))
