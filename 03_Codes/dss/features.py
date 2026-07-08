"""The ten bounded observation features, per Local DSS region.

Each feature is a scalar in [0, 1] computed over the agent's cells from the
CURRENT observation (phase 1a: sensor-emulated, i.e. read directly from the
live simulation state; the sensor catalogue with partial coverage arrives in
a later phase). Every formula is explicit and hand-checkable; each feature
carries established wildfire-science grounding.
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


def ten_features(sim, region, network=None) -> dict:
    """Return {feature_name: value in [0,1]} for one agent region.

    With a SensorNetwork the sensed state components (channels
    B, F_load, I, tau) come from the fused observation - stale or uncovered
    cells carry old values - while static prior maps (terrain, fuel type,
    values, own resources), fuel moisture and the weather field come from
    maps and the meteorological service. The matching confidence is
    network.region_conf(region)."""
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
    rcap = world.resource.rcap[sy, sx] * world.resource.ravail[sy, sx]
    ref = max(cfg.suppression.rcap_max, 1e-6) * 0.2
    z9 = float(np.clip(rcap.mean() / ref, 0.0, 1.0))

    # --- z10 temporal urgency: fire at/near the region plus how much of the
    #     region is actively burning right now
    z10 = float(np.clip(0.6 * z4 + 0.4 * min(1.0, 10.0 * B.mean()),
                        0.0, 1.0))

    vals = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10]
    return dict(zip(FEATURE_ORDER, vals))
