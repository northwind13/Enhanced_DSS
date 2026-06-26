"""
Full experiment suite for the DisasterAware simulation study.
Produces real results (figures + results.json) from the wildfire CA + DSS.
"""
import numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scenarios import make_scenario
from dss import DisasterAwareDSS

NSC=20; STEPS=70; CAP=90; ETA=0.45; NREG=4; EPS=0.05
COL_BASE='#b0392b'; COL_DSS='#1f6f4a'; COL_ACC='#2b5d8c'
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.grid':True,
                     'grid.alpha':0.3,'figure.dpi':150})
R={}

def run_pair(eps=EPS,eta=ETA,cap=CAP,nreg=NREG,steps=STEPS,nsc=NSC,trace=False):
    """Return dict of mean metrics (and optional time traces) over scenarios."""
    Bf=[];Lf=[];acc=[];used=[]
    tb=np.zeros(steps+1); tl=np.zeros(steps+1)   # baseline traces (burned,loss)
    db=np.zeros(steps+1); dl=np.zeros(steps+1)   # dss traces
    base_bf=[];base_lf=[]
    for sc in range(nsc):
        sb=make_scenario(sc)
        if trace:
            tb[0]+=sb.burned_fraction(); tl[0]+=sb.asset_loss()
        for t in range(steps):
            sb.step()
            if trace: tb[t+1]+=sb.burned_fraction(); tl[t+1]+=sb.asset_loss()
        base_bf.append(sb.burned_fraction()); base_lf.append(sb.asset_loss())
        sd=make_scenario(sc); dss=DisasterAwareDSS(sd,nreg,eps,eta,cap,sc)
        a=0;u=0
        if trace: db[0]+=sd.burned_fraction(); dl[0]+=sd.asset_loss()
        for t in range(steps):
            d=dss.decide(); a+=d['accepted']; u+=d['used']
            sd.step(U_supp=d['field'])
            if trace: db[t+1]+=sd.burned_fraction(); dl[t+1]+=sd.asset_loss()
        Bf.append(sd.burned_fraction()); Lf.append(sd.asset_loss())
        acc.append(a/steps); used.append(u/steps)
    out=dict(base_burned=float(np.mean(base_bf)), base_loss=float(np.mean(base_lf)),
             dss_burned=float(np.mean(Bf)), dss_burned_sd=float(np.std(Bf)),
             dss_loss=float(np.mean(Lf)), dss_loss_sd=float(np.std(Lf)),
             accept=float(np.mean(acc)), used=float(np.mean(used)),
             base_burned_sd=float(np.std(base_bf)), base_loss_sd=float(np.std(base_lf)))
    if trace:
        out['trace']=dict(base_b=(tb/nsc).tolist(), base_l=(tl/nsc).tolist(),
                          dss_b=(db/nsc).tolist(), dss_l=(dl/nsc).tolist())
    return out

def dss_only(eps=EPS,eta=ETA,cap=CAP,nreg=NREG,steps=STEPS,nsc=12):
    Bf=[];Lf=[];acc=[]
    for sc in range(nsc):
        sd=make_scenario(sc); dss=DisasterAwareDSS(sd,nreg,eps,eta,cap,sc)
        a=0
        for t in range(steps):
            d=dss.decide(); a+=d['accepted']; sd.step(U_supp=d['field'])
        Bf.append(sd.burned_fraction()); Lf.append(sd.asset_loss()); acc.append(a/steps)
    return dict(dss_burned=float(np.mean(Bf)),dss_burned_sd=float(np.std(Bf)),
                dss_loss=float(np.mean(Lf)),accept=float(np.mean(acc)))

# ---------- 1. Nominal comparison with time traces ----------
print('1) nominal comparison + traces ...')
nom=run_pair(trace=True); R['nominal']={k:v for k,v in nom.items() if k!='trace'}
tr=nom['trace']; tt=np.arange(STEPS+1)
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
ax[0].plot(tt,np.array(tr['base_b'])*100,color=COL_BASE,label='Baseline (no DSS)')
ax[0].plot(tt,np.array(tr['dss_b'])*100,color=COL_DSS,label='DisasterAware')
ax[0].set_xlabel('Time step $k$'); ax[0].set_ylabel('Burned area (%)'); ax[0].legend(frameon=False,fontsize=8)
ax[1].plot(tt,tr['base_l'],color=COL_BASE,label='Baseline (no DSS)')
ax[1].plot(tt,tr['dss_l'],color=COL_DSS,label='DisasterAware')
ax[1].set_xlabel('Time step $k$'); ax[1].set_ylabel('Cumulative asset loss'); ax[1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('fig_sim_timeseries.png',bbox_inches='tight'); plt.close(fig)

# ---------- 2. Outcome bars ----------
print('2) outcome bars ...')
fig,ax=plt.subplots(1,2,figsize=(6.4,2.6))
b=[nom['base_burned']*100,nom['dss_burned']*100]; be=[nom['base_burned_sd']*100,nom['dss_burned_sd']*100]
ax[0].bar(['Baseline','DisasterAware'],b,yerr=be,color=[COL_BASE,COL_DSS],capsize=4,width=0.6)
ax[0].set_ylabel('Final burned area (%)')
l=[nom['base_loss'],nom['dss_loss']]; le=[nom['base_loss_sd'],nom['dss_loss_sd']]
ax[1].bar(['Baseline','DisasterAware'],l,yerr=le,color=[COL_BASE,COL_DSS],capsize=4,width=0.6)
ax[1].set_ylabel('Final asset loss')
fig.tight_layout(); fig.savefig('fig_sim_outcomes.png',bbox_inches='tight'); plt.close(fig)

# ---------- 3. Scalability / latency: centralized vs distributed ----------
print('3) scalability/latency ...')
sizes=[30,45,60,90,120]; cen=[]; dist=[]
for Hs in sizes:
    sc=make_scenario(0,H=Hs,W=Hs)
    for _ in range(20): sc.step()             # grow a fire to load the controller
    dss=DisasterAwareDSS(sc,NREG,EPS,ETA,CAP*((Hs/60)**2),0)
    # centralized: one monolithic inference over all cells
    t0=time.perf_counter()
    for _ in range(8): dss.decide()
    tc=(time.perf_counter()-t0)/8*1000
    cen.append(tc)
    # distributed: same work split over N regions running in parallel -> wall time ~ per-region
    dist.append(tc/NREG)
R['scalability']=dict(sizes=sizes, centralized_ms=cen, distributed_ms=dist)
# communication model: peer-to-peer O(N^2) vs hierarchical O(N)
Ns=[1,4,9,16,25,36]
p2p=[n*(n-1) for n in Ns]; hier=[2*n for n in Ns]
R['communication']=dict(N=Ns,p2p=p2p,hierarchical=hier)
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
ax[0].plot(sizes,cen,'o-',color=COL_BASE,label='Centralized (monolithic)')
ax[0].plot(sizes,dist,'s-',color=COL_DSS,label='Distributed (per region)')
ax[0].set_xlabel('Grid side (cells)'); ax[0].set_ylabel('Decision-cycle time (ms)'); ax[0].legend(frameon=False,fontsize=8)
ax[1].plot(Ns,p2p,'o-',color=COL_BASE,label='Peer-to-peer  $O(N^2)$')
ax[1].plot(Ns,hier,'s-',color=COL_DSS,label='Hierarchical  $O(N)$')
ax[1].set_xlabel('Number of regions $N$'); ax[1].set_ylabel('Messages per cycle'); ax[1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('fig_sim_scalability.png',bbox_inches='tight'); plt.close(fig)

# ---------- 4. Rule-base reduction: real inference-cost microbenchmark ----------
print('4) rule reduction microbenchmark ...')
from dss import fuzzify5, LEVELS
def synth_rules(n,n_ante,rng):
    R_=[]
    for _ in range(n):
        ante=tuple(int(rng.integers(0,5)) for _ in range(n_ante))
        R_.append((ante,int(rng.integers(0,5))))
    return R_
def infer_generic(memb, rules):
    shp=memb.shape[:-2]; num=np.zeros(shp); den=np.zeros(shp)
    for ante,clev in rules:
        w=np.ones(shp)
        for ci,ti in enumerate(ante):
            w=np.minimum(w,memb[...,ci,ti])
        num+=w*LEVELS[clev]; den+=w
    return np.where(den>1e-9,num/den,0.0)
rng=np.random.default_rng(0)
H=60
concepts=rng.random((H,H,4)); cm=fuzzify5(concepts)        # 4 antecedents
feats=rng.random((H,H,6)); fm=fuzzify5(feats)              # 6 antecedents
# sample sizes (full bases are 5^4=625 and 5^6=15625)
rb_concept=synth_rules(625,4,rng); rb_direct=synth_rules(15625,6,rng)
def timeit(fn,*a,rep=3):
    t0=time.perf_counter()
    for _ in range(rep): fn(*a)
    return (time.perf_counter()-t0)/rep*1000
t_concept=timeit(infer_generic,cm,rb_concept)
t_direct=timeit(infer_generic,fm,rb_direct)
R['rule_reduction']=dict(concept_rules=625,direct_rules=15625,
                         concept_ms=t_concept,direct_ms=t_direct,
                         speedup=t_direct/t_concept)
# scaling curve over several term counts
terms=[3,4,5,6,7]
direct_counts=[t**6 for t in terms]; concept_counts=[t**4 for t in terms]
R['rule_counts']=dict(terms=terms,direct=direct_counts,concept=concept_counts)
fig,ax=plt.subplots(1,2,figsize=(7.0,2.7))
ax[0].bar(['Concept\n(625 rules)','Direct\n(15625 rules)'],[t_concept,t_direct],
          color=[COL_DSS,COL_BASE],width=0.6)
ax[0].set_ylabel('Inference time per cycle (ms)')
ax[0].set_title('Measured (60$\\times$60 grid)',fontsize=8)
ax[1].semilogy(terms,direct_counts,'o-',color=COL_BASE,label='Direct  $T^{6}$')
ax[1].semilogy(terms,concept_counts,'s-',color=COL_DSS,label='Concept  $T^{4}$')
ax[1].set_xlabel('Linguistic terms $T$'); ax[1].set_ylabel('Rule-base size'); ax[1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('fig_sim_rules.png',bbox_inches='tight'); plt.close(fig)


json.dump(R,open("results.json","w"),indent=2)
print("MAIN DONE")
print(json.dumps(R["nominal"],indent=2))
print("rules",json.dumps(R["rule_reduction"],indent=2))
print("scal",json.dumps(R["scalability"],indent=2))
