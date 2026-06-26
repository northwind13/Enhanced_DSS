"""Randomised wildfire scenarios for Monte-Carlo evaluation."""
import numpy as np
from firesim import FireSim

def make_scenario(seed, H=60, W=60):
    rng=np.random.default_rng(1000+seed)
    sim=FireSim(H=H,W=W,seed=seed)
    iy=int(rng.uniform(0.45,0.75)*H); ix=int(rng.uniform(0.15,0.35)*W)
    wd=np.deg2rad(rng.uniform(20,70))            # generally toward NE (assets)
    ws=float(rng.uniform(0.45,0.7))
    sim.reset(ign=(iy,ix), wind_dir=wd, wind_speed=ws)
    return sim
