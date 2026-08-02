"""Observation decidable-band grid (fire load x sensor coverage) across
resource pools p in {0.25,0.5,0.75,1.0}. Free burn reused from the
calibration block. Parallel, incremental append to out/sens_runs.csv,
block='obs_calibration'; resumable on (pool,n_ign,n_sensors,seed).
"""
import argparse, csv, os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensitivity2 as S
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"out")
RUNS=os.path.join(OUT,"sens_runs.csv")
OBS_IGN=[1,2,4,8,12]; OBS_SENS=[1,2,3,5,9]; POOLS=[0.25,0.5,0.75,1.0]

def done_keys():
    k=set()
    for r in csv.DictReader(open(RUNS,encoding="utf-8")):
        if r.get("block")=="obs_calibration":
            k.add((round(float(r["pool"]),2),int(float(r["n_ign"])),
                   int(float(r["n_sensors"])),int(float(r["seed"]))))
    return k

def work(job):
    p,n,s,seed=job
    env=dict(S.ENV_BASE,n_ign=n,pool=p,n_sensors=s,n_regions=4)
    res=S.run_point(seed,"adaptive",env,S.TUNE_BASE,S.W_BASE)
    row=S._row("obs_calibration","grid",f"{p}|{n}|{s}","adaptive",seed,env,S.TUNE_BASE,S.W_BASE)
    row.update(res); return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=3); ap.add_argument("--workers",type=int,default=2)
    a=ap.parse_args(); seeds=S.SEEDS[:a.seeds]
    hdr=list(csv.DictReader(open(RUNS,encoding="utf-8")).fieldnames)
    done=done_keys()
    jobs=[(p,n,s,seed) for p in POOLS for seed in seeds for n in OBS_IGN for s in OBS_SENS
          if (round(p,2),n,s,seed) not in done]
    print(f"missing {len(jobs)} cells",flush=True)
    t0=time.time(); c=0
    with mp.Pool(a.workers) as pool:
        for row in pool.imap_unordered(work, jobs):
            with open(RUNS,"a",newline="",encoding="utf-8") as f:
                csv.DictWriter(f,fieldnames=hdr,extrasaction="ignore").writerow(row)
            c+=1
            print(f"[{c}/{len(jobs)}] {row['value']} seed={row['seed']} j={row['j_phys']} t={time.time()-t0:.0f}s",flush=True)
    print(f"appended {c} in {time.time()-t0:.0f}s")

if __name__=="__main__": main()
