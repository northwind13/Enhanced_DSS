import numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scenarios import make_scenario
from dss import DisasterAwareDSS
STEPS=55; CAP=90; ETA=0.45; NREG=4; EPS=0.05
COL_BASE='#b0392b'; COL_DSS='#1f6f4a'; COL_ACC='#2b5d8c'
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3,'figure.dpi':150})
R=json.load(open('results.json'))
def dss_only(eps=EPS,eta=ETA,cap=CAP,nreg=NREG,steps=STEPS,nsc=8):
    Bf=[];Lf=[];acc=[]
    for sc in range(nsc):
        sd=make_scenario(sc); dss=DisasterAwareDSS(sd,nreg,eps,eta,cap,sc)
        a=0
        for t in range(steps):
            d=dss.decide(); a+=d['accepted']; sd.step(U_supp=d['field'])
        Bf.append(sd.burned_fraction()); Lf.append(sd.asset_loss()); acc.append(a/steps)
    return dict(dss_burned=float(np.mean(Bf)),dss_burned_sd=float(np.std(Bf)),
                dss_loss=float(np.mean(Lf)),accept=float(np.mean(acc)))
nom=R['nominal']
# ---------- 5. Sensitivity: eps, eta, capacity, N ----------
print('5) sensitivity sweeps ...')
eps_grid=[0.0,0.05,0.1,0.15,0.2,0.25,0.3]
eta_grid=[0.3,0.4,0.5,0.55,0.6,0.65,0.75]
cap_grid=[30,45,55,65,80,110,150]
n_grid=[1,4,9,16,25]
sens={'eps':{'x':eps_grid,'burned':[],'bsd':[],'loss':[]},
      'eta':{'x':eta_grid,'burned':[],'loss':[],'accept':[]},
      'cap':{'x':cap_grid,'burned':[],'loss':[]},
      'N':{'x':n_grid,'burned':[],'loss':[]}}
for e in eps_grid:
    r=dss_only(eps=e); sens['eps']['burned'].append(r['dss_burned']*100)
    sens['eps']['bsd'].append(r['dss_burned_sd']*100); sens['eps']['loss'].append(r['dss_loss'])
for e in eta_grid:
    r=dss_only(eta=e); sens['eta']['burned'].append(r['dss_burned']*100)
    sens['eta']['loss'].append(r['dss_loss']); sens['eta']['accept'].append(r['accept']*100)
for c in cap_grid:
    r=dss_only(cap=c); sens['cap']['burned'].append(r['dss_burned']*100); sens['cap']['loss'].append(r['dss_loss'])
for n in n_grid:
    r=dss_only(nreg=n); sens['N']['burned'].append(r['dss_burned']*100); sens['N']['loss'].append(r['dss_loss'])
R['sensitivity']=sens
base_b=nom['base_burned']*100
fig,ax=plt.subplots(2,2,figsize=(7.2,5.0))
ax[0,0].errorbar(eps_grid,sens['eps']['burned'],yerr=sens['eps']['bsd'],fmt='o-',color=COL_DSS,capsize=3)
ax[0,0].axhline(base_b,ls='--',color=COL_BASE,label='Baseline'); ax[0,0].set_xlabel('Observation noise $\\bar\\epsilon$')
ax[0,0].set_ylabel('Burned area (%)'); ax[0,0].legend(frameon=False,fontsize=8)
ax[0,1].plot(eta_grid,sens['eta']['burned'],'o-',color=COL_DSS,label='Burned area (%)')
axb=ax[0,1].twinx(); axb.plot(eta_grid,sens['eta']['accept'],'s--',color=COL_ACC,label='Acceptance (%)')
axb.set_ylabel('Acceptance rate (%)',color=COL_ACC); ax[0,1].set_xlabel('Acceptance threshold $\\eta$')
ax[0,1].set_ylabel('Burned area (%)'); ax[0,1].axhline(base_b,ls='--',color=COL_BASE)
ax[1,0].plot(cap_grid,sens['cap']['burned'],'o-',color=COL_DSS); ax[1,0].axhline(base_b,ls='--',color=COL_BASE,label='Baseline')
ax[1,0].set_xlabel('Resource capacity $C$'); ax[1,0].set_ylabel('Burned area (%)'); ax[1,0].legend(frameon=False,fontsize=8)
ax[1,1].plot(n_grid,sens['N']['burned'],'o-',color=COL_DSS); ax[1,1].axhline(base_b,ls='--',color=COL_BASE,label='Baseline')
ax[1,1].set_xlabel('Number of regions $N$'); ax[1,1].set_ylabel('Burned area (%)'); ax[1,1].set_ylim(0,base_b*1.15)
ax[1,1].legend(frameon=False,fontsize=8)
fig.tight_layout(); fig.savefig('fig_sim_sensitivity.png',bbox_inches='tight'); plt.close(fig)


json.dump(R,open('results.json','w'),indent=2)
print('SENS DONE')
print(json.dumps(R['sensitivity'],indent=2))
