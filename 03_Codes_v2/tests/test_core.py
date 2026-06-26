"""Invariant tests for the DisasterAware simulation core and DSS."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
import numpy as np
import pytest
from firesim import FireSim
from dss import DisasterAwareDSS
from engine import Session

def run(sim, dss=None, steps=40, ignite=(40,30)):
    sim.ignite([ignite])
    for _ in range(steps):
        if dss is not None:
            d=dss.decide(); sim.step(U_supp=d['field'])
        else:
            sim.step()

def test_fuel_non_negative_and_water_never_burns():
    s=FireSim(H=60,W=60,seed=3); run(s)
    assert (s.F>=0).all()
    assert not (s.ever_burned & (s.cat==0)).any(), "water cells must never burn"

def test_burned_is_monotonic():
    s=FireSim(H=60,W=60,seed=3); s.ignite([(30,25)])
    prev=0
    for _ in range(40):
        s.step(); cur=int(s.ever_burned.sum())
        assert cur>=prev, "ever-burned area cannot shrink"
        prev=cur

def test_dss_reduces_impact_vs_baseline():
    base=FireSim(H=70,W=70,seed=5); run(base, None, 60, (45,30))
    man=FireSim(H=70,W=70,seed=5); d=DisasterAwareDSS(man,4,0.05,0.45,150,5); run(man, d, 60, (45,30))
    assert man.burned_fraction() <= base.burned_fraction()+1e-9
    assert man.asset_loss() <= base.asset_loss()+1e-9

def test_quality_bounded():
    s=FireSim(H=60,W=60,seed=2); d=DisasterAwareDSS(s,4,0.05,0.45,90,2); s.ignite([(30,25)])
    for _ in range(30):
        out=d.decide(); s.step(U_supp=out['field'])
        assert 0.0-1e-9 <= out['Q'] <= 1.0+1e-9
        for v in out['q'].values(): assert -1e-9 <= v <= 1+1e-9

def test_state_immutability_ui_params():
    """Changing wind/eta must not alter the already-realised baseline state."""
    S=Session(); S.reset({'H':60,'W':60,'seed':7}); S.ignite([(30,25)]); S.step(10)
    before=S.baseline.ever_burned.copy()
    S.set_params({'wind_speed':0.9,'eta':0.8,'humidity':0.1})
    assert (S.baseline.ever_burned==before).all(), "params must not retro-edit state"

def test_checkpoint_restore_roundtrip():
    S=Session(); S.reset({'H':60,'W':60,'seed':7}); S.ignite([(30,25)]); S.step(10)
    S.set_checkpoint(); burned_ck=S.managed.ever_burned.sum()
    S.step(8); assert S.managed.ever_burned.sum()>=burned_ck
    S.restore_checkpoint()
    assert abs(int(S.managed.ever_burned.sum())-int(burned_ck))<=2

def test_inspect_structure():
    S=Session(); S.reset({'H':60,'W':60,'seed':7}); S.ignite([(30,25)]); S.step(12)
    info=S.inspect(31,27)
    assert 'landuse' in info
    if 'features' in info:
        assert len(info['features'])==6 and len(info['concepts'])==4

def test_humidity_slows_fire():
    dry=FireSim(H=70,W=70,seed=8,humidity=0.1,spotting=0.0); run(dry,None,40,(45,30))
    wet=FireSim(H=70,W=70,seed=8,humidity=0.8,spotting=0.0); run(wet,None,40,(45,30))
    assert wet.burned_fraction() <= dry.burned_fraction()+1e-9
