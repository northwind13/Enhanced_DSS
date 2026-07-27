"""Sensitivity sweep for the DisasterAware decision layer.
One-at-a-time (OAT) sweep over the capacity-balance axes and the decision
thresholds on the compact capacity-limited synthetic testbed.
Writes sens.csv (resumable). Run:  python runner.py   then  python plot.py
"""
import importlib.util, os, sys, csv, time
HERE = os.path.dirname(os.path.abspath(__file__))
CODES = os.path.abspath(os.path.join(HERE, '..', '..', '03_Codes'))
sys.path.insert(0, CODES)
spec = importlib.util.spec_from_file_location('S', os.path.join(CODES, 'experiments', 'sensitivity.py'))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from disaster_phyengine.config import SimConfig
from disaster_phyengine import terrain
import dss
from disaster_phyengine.core import Simulator
from disaster_phyengine.costs import compute_costs

CSV   = os.path.join(HERE, 'sens.csv')
SEED  = 11
STEPS = 60          # steps per run (2 min each)
WIND  = 14.0        # m/s
MOIST = 0.05        # dead-fuel moisture fraction
# scenario / decision base values
BASE = dict(rcap=0.025, N=4, nign=3, J_TH=0.35, eta=0.6, tau=0.35,
            rho=0.9, horizon=15.0, cycle=1.0, wb=1.0, wa=1.0, wp=1.0)

def run(p):
    cfg = SimConfig(nx=80, ny=60, cell_size_m=30.0); cfg.step_minutes = 2.0
    w = terrain.generate_landscape(cfg, seed=SEED, preset="Rolling hills",
                                   n_settlements=5, population_per_settlement=15000)
    w.fuel.fmoist[:] = MOIST; w.meteo.wws[:] = WIND
    base, _ = dss.resource_suggestion(w)
    base.rcap[:] = base.rcap * p['rcap']            # real suppression capacity
    w.config.cost.capacity_reference = max(100.0, 1.2 * float((base.rcap * base.ravail).sum()))
    w.config.cost.w_burn = p['wb']; w.config.cost.w_asset = p['wa']; w.config.cost.w_pop = p['wp']
    for x, y in S.pick_ignitions(w, base, SEED, n=int(p['nign'])):
        w.add_ignition(x, y, step=0, radius=1)
    sim = Simulator(w); sim.record_states = False
    eng = dss.DecisionEngine(dss.partition_n(80, 60, int(p['N'])), base_pool=base,
        j_threshold=p['J_TH'], eta=p['eta'], attention_thr=p['tau'],
        cycle_min=p['cycle'], horizon_min=p['horizon'], adapt_on=False)
    for g in eng.gaters.values(): g.rho = p['rho']
    out = None
    for i in range(STEPS):
        sim.step(resource_override=eng.maybe_decide(sim))
        if int((sim.state.burning > 0.5).sum()) == 0 and i > 5:
            out = (i + 1) * 2.0; break
    fs  = sum(1 for c in eng.cycles for rd in (c.get('regions') or {}).values() if rd.get('failsafe'))
    allc = sum(1 for c in eng.cycles for rd in (c.get('regions') or {}).values())
    rep = compute_costs(sim)
    return dict(j=round(float(rep.j_physical), 5), b=int(sim.ever_burned.sum()),
                o=out if out is not None else -1, fs=round(fs / allc, 4) if allc else 0.0)

# (axis, value) sweep points; one axis varied, the rest at BASE
TODO = [('baseline', '-')]
for v in [0.015, 0.02, 0.05, 0.1]: TODO.append(('rcap', v))
for v in [1, 2, 8]:                TODO.append(('N', v))
for v in [2, 6, 12]:               TODO.append(('nign', v))
for v in [0.15, 0.60]:             TODO.append(('J_TH', v))
for v in [0.30, 0.90]:             TODO.append(('eta', v))
for v in [0.15, 0.70]:             TODO.append(('tau', v))
for v in [0.70, 0.99]:             TODO.append(('rho', v))
for v in [5.0, 45.0]:              TODO.append(('horizon', v))
for v in [8.0]:                    TODO.append(('cycle', v))
for v in [0.5, 2.0]:               TODO.append(('wb', v))
for v in [0.5, 2.0]:               TODO.append(('wa', v))
for v in [0.5, 2.0]:               TODO.append(('wp', v))

done = set()
if os.path.exists(CSV):
    for r in csv.reader(open(CSV)):
        if r and r[0] != 'axis': done.add((r[0], r[1]))
new = not os.path.exists(CSV)
f = open(CSV, 'a', newline=''); wtr = csv.writer(f)
if new:
    wtr.writerow(['axis', 'value', 'j_physical', 'burned', 'out_min', 'fs_frac', 'sec']); f.flush()
KEYS = {k: k for k in BASE}
for axis, val in TODO:
    if (axis, str(val)) in done:
        print(f"skip {axis}={val} (already done)"); continue
    p = dict(BASE)
    if axis != 'baseline': p[KEYS[axis]] = val
    t = time.perf_counter(); r = run(p)
    wtr.writerow([axis, val, r['j'], r['b'], r['o'], r['fs'], round(time.perf_counter() - t, 1)]); f.flush()
    print(f"{axis}={val}: j_phys={r['j']} burned={r['b']} out={r['o']} fs={r['fs']}")
f.close()
print("done ->", CSV)
