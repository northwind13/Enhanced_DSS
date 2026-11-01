# K41: hand-written head retardant line vs recorded head coating, each alone in the closed layer, and with trigger and band swapped (referee M3c). CPU only.
# python k41_coating_tactics.py k41_coating_tactics/runs.csv written,recorded,w_trig_r_band,r_trig_w_band
import os,sys,csv,time
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path[:0]=[ROOT,ROOT+"/experiments"]
from open_space import campaign as C
def T(rule,cl): return dict(rule=rule,kind="actuator",clauses=cl,composition=None,concept=None,t_min=0.0,source="expert",accepted=True)
ARMS={
 "written":   T("IF fire threat level is H THEN head retardant line 1.0",[dict(effect="coat",sector="head",range=[2,6],amount=1.0)]),
 "recorded":  T("IF asset exposure risk is M THEN head coating 1.0",[dict(effect="coat",sector="head",range=[4,8],amount=1.0)]),
 "w_trig_r_band": T("IF fire threat level is H THEN head coating b 1.0",[dict(effect="coat",sector="head",range=[4,8],amount=1.0)]),
 "r_trig_w_band": T("IF asset exposure risk is M THEN head retardant line b 1.0",[dict(effect="coat",sector="head",range=[2,6],amount=1.0)]),
}
def job(a):
    arm,sc,sd=a
    row,_c,_a,_m=C.run_once("LOO",sc,"F5Ev",sd,script=[dict(ARMS[arm])])
    return dict(arm=arm,scenario=sc,seed=sd,burned_ha=row["burned_ha"],success=row["success"],coated_ha=row.get("coated_ha",""),t_coat=row.get("t_coat",""),redis_coat=row.get("redis_coat",""),j_total=row["j_total"],wall_s=row["wall_s"])
if __name__=="__main__":
    out=sys.argv[1]; arms=sys.argv[2].split(","); seeds=range(201,231) if len(sys.argv)<4 else [int(x) for x in sys.argv[3].split(",")]
    done=set()
    if os.path.exists(out): done={(r["arm"],r["scenario"],int(r["seed"])) for r in csv.DictReader(open(out))}
    jobs=[(a,s,d) for a in arms for s in ("S1","S2","S3","S4","S5") for d in seeds if (a,s,d) not in done]
    print(f"K41: {len(jobs)} runs to do ({len(done)} done)", flush=True)
    t0=time.time(); n=0
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(os.environ.get("W") or os.environ.get("W_CPU") or 12)) as p:
        for r in p.imap_unordered(job,jobs):
            new=not os.path.exists(out)
            with open(out,"a",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(r)); 
                if new: w.writeheader()
                w.writerow(r)
            n+=1
            print(f"[{n}/{len(jobs)}] {r['arm']} {r['scenario']} s{r['seed']}: burned {r['burned_ha']} ha, coated {r['coated_ha']} ha, success {r['success']}, {r['wall_s']}s (elapsed {time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)
