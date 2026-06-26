import numpy as np, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scenarios import make_scenario
from dss import DisasterAwareDSS
plt.rcParams.update({'font.size':9,'figure.dpi':150})

# environment layers (scenario 0)
s=make_scenario(0)
fig,ax=plt.subplots(1,3,figsize=(7.4,2.5))
for a,(lay,ttl,cm) in zip(ax,[(s.fuel0,'Fuel load','YlGn'),(s.value,'Asset value','OrRd'),(s.slope,'Terrain slope','Greys')]):
    im=a.imshow(lay,cmap=cm,origin='lower'); a.set_title(ttl,fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im,ax=a,fraction=0.046,pad=0.04)
iy,ix=np.argwhere(s.B>0.5)[0]
for a in ax: a.plot(ix,iy,'*',color='red',ms=9,mec='k',mew=0.5)
fig.tight_layout(); fig.savefig('fig_sim_environment.png',bbox_inches='tight'); plt.close(fig)

# spatial snapshots baseline vs DSS (same scenario)
sc=2; STEPS=70
sb=make_scenario(sc)
for _ in range(STEPS): sb.step()
sd=make_scenario(sc); dss=DisasterAwareDSS(sd,4,0.05,0.45,90,sc)
for _ in range(STEPS):
    d=dss.decide(); sd.step(U_supp=d['field'])
fig,ax=plt.subplots(1,2,figsize=(6.6,3.1))
for a,(sim,ttl) in zip(ax,[(sb,'Baseline (no DSS)'),(sd,'DisasterAware')]):
    bg=0.6*sim.value/max(sim.value.max(),1e-9)
    rgb=np.dstack([0.92-0.2*bg,0.95-0.3*bg,0.85-0.5*bg])     # light value backdrop
    burned=sim.ever_burned
    rgb[burned]=[0.25,0.18,0.15]                              # burned scar
    rgb[sim.B>0.5]=[0.9,0.35,0.1]                             # active fire
    a.imshow(rgb,origin='lower'); a.set_title('%s\nburned %.0f%%, asset loss %.0f'%(ttl,100*sim.burned_fraction(),sim.asset_loss()),fontsize=8.5)
    a.set_xticks([]); a.set_yticks([])
    # mark settlements
    yy,xx=np.mgrid[0:sim.H,0:sim.W]
    a.contour(sim.value,levels=[0.4],colors='white',linewidths=0.8)
fig.tight_layout(); fig.savefig('fig_sim_snapshots.png',bbox_inches='tight'); plt.close(fig)
print('maps done', sb.burned_fraction(), sd.burned_fraction(), sb.asset_loss(), sd.asset_loss())
