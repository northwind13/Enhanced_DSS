import csv, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
rows=list(csv.DictReader(open(os.path.join(HERE,'sens.csv'))))
D={}
for r in rows:
    D.setdefault(r['axis'],[]).append((r['value'], float(r['j_physical']), int(r['burned']), r['out_min'], float(r['fs_frac'])))
base_j=[v for a,rs in D.items() if a=='baseline' for _,v,_,_,_ in rs][0]
base_b=[b for a,rs in D.items() if a=='baseline' for _,_,b,_,_ in rs][0]
# label map (thesis notation)
LAB={'nign':'Simultaneous ignitions','rcap':'Resource level','N':'Local regions N',
     'wb':'$w_{burn}$','wp':'$w_{pop}$','wa':'$w_{asset}$','tau':r'$\tau$ (attention)',
     'cycle':'Decision cycle','eta':r'$\eta$ (fail-safe gate)','horizon':'No-harm horizon',
     'rho':r'$\rho$ (persistence)','J_TH':'$J_{TH}$ (satisficing)'}
# --- Fig 1: tornado of physical-cost spread ---
spread={}
for a,rs in D.items():
    if a=='baseline': continue
    js=[v for _,v,_,_,_ in rs]+[base_j]
    spread[a]=(min(js),max(js))
order=sorted(spread, key=lambda a: spread[a][1]-spread[a][0])
fig,ax=plt.subplots(figsize=(7.2,4.2))
caps={'nign','rcap','N'}; wts={'wb','wp','wa'}
for i,a in enumerate(order):
    lo,hi=spread[a]
    col='#c0392b' if a in caps else ('#7f8c8d' if a in wts else '#2c7fb8')
    ax.barh(i, hi-lo, left=lo, color=col, edgecolor='black', height=0.6,
            hatch=('//' if a in wts else None))
    ax.text(hi+0.005, i, f"{hi-lo:.3f}", va='center', fontsize=8)
ax.axvline(base_j, color='k', ls='--', lw=0.8)
ax.text(base_j, len(order)-0.3, ' baseline', fontsize=8, rotation=90, va='top')
ax.set_yticks(range(len(order))); ax.set_yticklabels([LAB[a] for a in order])
ax.set_xlabel('Physical cost $J_{phys}$ (range over the parameter sweep)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#c0392b',label='Capacity balance'),
                   Patch(color='#2c7fb8',label='Decision thresholds (robust)'),
                   Patch(facecolor='#7f8c8d',hatch='//',label='Objective reweighting')],
          fontsize=8, loc='lower right')
ax.set_title('Sensitivity of physical cost (one-at-a-time)')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'fig_tornado.png'),dpi=150); plt.close()
# --- Fig 2: capacity response curves ---
fig,axs=plt.subplots(1,2,figsize=(8.4,3.6))
rc=sorted([(float(v),j) for v,j,_,_,_ in D['rcap']]+[(0.025,base_j)])
axs[0].plot([x for x,_ in rc],[y for _,y in rc],'o-',color='#c0392b')
axs[0].set_xlabel('Resource level (fraction of nominal)'); axs[0].set_ylabel('Physical cost $J_{phys}$')
axs[0].set_title('(a) Resource level'); axs[0].set_xscale('log'); axs[0].grid(alpha=.3)
ni=sorted([(int(v),j) for v,j,_,_,_ in D['nign']]+[(3,base_j)])
axs[1].plot([x for x,_ in ni],[y for _,y in ni],'s-',color='#c0392b')
axs[1].set_xlabel('Simultaneous ignitions'); axs[1].set_title('(b) Fire load'); axs[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE,'fig_capacity.png'),dpi=150); plt.close()
# --- Fig 3: robustness + N + fail-safe ---
fig,axs=plt.subplots(1,2,figsize=(8.4,3.6))
thr=['J_TH','eta','tau','rho','horizon','cycle']
for a in thr:
    pts=sorted([(float(v),j) for v,j,_,_,_ in D[a]])
    # normalize x to [0,1] over its span for overlay
    xs=[x for x,_ in pts]; span=(max(xs)-min(xs)) or 1
    axs[0].plot([(x-min(xs))/span for x,_ in pts],[y for _,y in pts],'o-',label=LAB[a],ms=4)
axs[0].axhline(base_j,color='k',ls='--',lw=.8)
axs[0].set_ylim(base_j-0.05,base_j+0.05)
axs[0].set_xlabel('Parameter value (normalized over its range)'); axs[0].set_ylabel('Physical cost $J_{phys}$')
axs[0].set_title('(a) Decision thresholds: robust'); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
# eta vs fail-safe fraction
et=sorted([(float(v),fs) for v,_,_,_,fs in D['eta']]+[(0.6,0.0)])
axs[1].plot([x for x,_ in et],[y for _,y in et],'D-',color='#2c7fb8')
axs[1].set_xlabel(r'$\eta$ (fail-safe quality gate)'); axs[1].set_ylabel('Fail-safe engagement fraction')
axs[1].set_title('(b) Fail-safe response to $\\eta$'); axs[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE,'fig_robust.png'),dpi=150); plt.close()
print('figures written'); import os
for f in ['fig_tornado.png','fig_capacity.png','fig_robust.png']: print(' ',f, os.path.getsize(f),'bytes')
