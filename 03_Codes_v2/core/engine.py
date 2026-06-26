"""
DisasterAware session engine with timeline history, checkpoints, scenario
presets and export. Runs two twins (baseline / DSS-managed) from the same
land-use world and stores render-ready snapshots for timeline scrubbing.
"""
import copy, io, csv, json
import numpy as np
from firesim import FireSim, CELL_M, DT_MIN
from dss import DisasterAwareDSS, TYPE_LABEL, TYPE_COLOR
from landuse import legend as lu_legend

class Session:
    def __init__(self):
        self.params=dict(H=80,W=80,seed=7,fuel_density=1.0,wind_speed=0.55,wind_dir=45.0,
                         humidity=0.30,spotting=0.5,n_responders=6,forest_frac=1.2,
                         eta=0.45,eps=0.05,regions=4,dss_enabled=True,event_trigger=0)
        self.params['capacity']=self.params['n_responders']*25.0
        self.ignitions=[]
        self.reset(self.params)

    def _mk(self):
        q=self.params
        return FireSim(H=q['H'],W=q['W'],seed=q['seed'],fuel_density=q['fuel_density'],
                       wind_speed=q['wind_speed'],wind_dir_deg=q['wind_dir'],
                       n_responders=q['n_responders'],forest_frac=q['forest_frac'],
                       humidity=q['humidity'],spotting=q['spotting'])

    def reset(self,p=None):
        if p: self.params.update(p)
        self.params['capacity']=self.params['n_responders']*25.0
        self.baseline=self._mk(); self.managed=self._mk()
        q=self.params
        self.dss=DisasterAwareDSS(self.managed,n_regions=q['regions'],eps_noise=q['eps'],
                                  eta=q['eta'],capacity=q['capacity'],seed=q['seed'])
        self.history=[]; self.last_dss=None; self.last_idle=False; self.ignitions=[]; self.checkpoint_idx=None
        self._snap()

    def clear_fire(self):
        self.baseline.reset(); self.managed.reset()
        self.dss.prev_c=None; self.dss.avail=1.0; self.dss.last=None
        self.history=[]; self.last_dss=None; self.last_idle=False; self.ignitions=[]; self.checkpoint_idx=None
        self._snap()

    def set_params(self,p):
        self.params.update(p); q=self.params
        if 'n_responders' in p:
            # responders change resource layer -> rebuild world keeps fire? simplest: just capacity
            q['capacity']=q['n_responders']*25.0; self.dss.capacity=q['capacity']
        for sim in (self.baseline,self.managed):
            sim.set_wind(speed=q['wind_speed'],dir_deg=q['wind_dir'],humidity=q['humidity'],spotting=q['spotting'])
        self.dss.eps=q['eps']; self.dss.eta=q['eta']

    def ignite(self,cells,radius=1):
        self.baseline.ignite(cells,radius); self.managed.ignite(cells,radius)
        for (y,x) in cells: self.ignitions.append([int(y),int(x),int(radius)])
        self.history[-1]=self._make_snap()

    def step(self,n=1):
        for _ in range(n):
            self.baseline.step()
            apply=self.params['dss_enabled']
            if apply and self.params['event_trigger']>0 and self.managed.active_fire()<self.params['event_trigger']:
                apply=False; self.last_idle=True
            else:
                self.last_idle=False
            if apply:
                d=self.dss.decide(); self.last_dss=d; self.managed.step(U_supp=d['field'])
            else:
                self.last_dss=None; self.managed.step()
            self._snap()

    # ---------- snapshots ----------
    def _codes(self,sim): return ''.join(map(str,sim.codes().flatten().tolist()))
    def _make_snap(self):
        b,m=self.baseline,self.managed; d=self.last_dss
        if d is not None:
            supp=(np.clip(m.last_supp,0,1)*9).astype(np.uint8)
            dom=np.where(d['active'],d['dom']+1,0).astype(np.uint8)
            dssinfo=dict(enabled=True,idle=False,Q=round(d['Q'],3),accepted=bool(d['accepted']),
                         used=round(d['used'],1),capacity=self.params['capacity'],q=d['q'],
                         summary=d['summary'],u=d['u'])
            suppS=''.join(map(str,supp.flatten().tolist())); domS=''.join(map(str,dom.flatten().tolist()))
        else:
            suppS='0'*(m.H*m.W); domS='0'*(m.H*m.W)
            dssinfo=dict(enabled=self.params['dss_enabled'],idle=self.last_idle,Q=None,accepted=None,
                         used=None,capacity=self.params['capacity'],q=None,summary=None,u=None)
        def side(sim):
            return dict(codes=self._codes(sim),burned=round(100*sim.burned_fraction(),1),
                        loss=round(sim.asset_loss(),1),active=sim.active_fire(),
                        ha=round(sim.burned_ha(),1),by_cat=sim.burned_by_cat())
        return dict(step=b.k,minute=round(b.elapsed_min(),0),
                    baseline=side(b), managed=dict(side(m),supp=suppS,dom=domS), dss=dssinfo)
    def _snap(self):
        self.history.append(self._make_snap())
        if len(self.history)>800: self.history=self.history[-800:]

    def series(self):
        h=self.history
        return dict(t=[s['step'] for s in h],
                    base_burned=[s['baseline']['burned'] for s in h],
                    dss_burned=[s['managed']['burned'] for s in h],
                    base_loss=[s['baseline']['loss'] for s in h],
                    dss_loss=[s['managed']['loss'] for s in h])

    # ---------- payloads ----------
    def layers(self):
        b=self.baseline; q=self.params
        u8=lambda a:(np.clip(a,0,1)*255).astype(np.uint8).flatten().tolist()
        gn=int(round(q['regions']**0.5))
        return dict(H=b.H,W=b.W,fuel=u8(b.fuel0),value=u8(b.value),slope=u8(b.slope),access=u8(b.access),
                    cat=''.join(map(str,b.cat.flatten().tolist())),
                    stations=[dict(y=int(y),x=int(x)) for (y,x) in b.stations],
                    grid_n=gn,n_responders=q['n_responders'],capacity=q['capacity'],
                    cell_m=CELL_M,dt_min=DT_MIN,
                    legend=lu_legend(),asset_total=round(b.asset_total(),1),
                    type_legend=[dict(key=k,label=TYPE_LABEL[k],color=TYPE_COLOR[k]) for k in ('m1','m2','m3')])

    def view(self,i=None):
        n=len(self.history)
        if i is None or i<0 or i>=n: i=n-1
        snap=dict(self.history[i]); snap['index']=i; snap['count']=n
        snap['series']=self.series(); snap['asset_total']=round(self.baseline.asset_total(),1)
        snap['live']=(i==n-1); snap['checkpoint']=self.checkpoint_idx
        # savings panel
        bl=snap['baseline']; mg=snap['managed']
        snap['savings']=dict(value_protected=round(bl['loss']-mg['loss'],1),
                             area_protected_ha=round(bl['ha']-mg['ha'],1),
                             pct_loss_cut=round(100*(bl['loss']-mg['loss'])/bl['loss'],0) if bl['loss']>0 else 0)
        return snap

    def concept_layer(self,name):
        f=self.dss.concept_field(name)
        if f is None: return dict(name=name,data=None)
        u8=(np.clip(f,0,1)*255).astype(np.uint8).flatten().tolist()
        return dict(name=name,data=u8,H=self.managed.H,W=self.managed.W)

    def inspect(self,y,x):
        info=self.dss.inspect(y,x) if self.params['dss_enabled'] else None
        if info is None:
            s=self.managed; from landuse import CATS
            return dict(x=int(x),y=int(y),landuse=CATS[int(s.cat[y,x])][1],
                        burning=bool(s.B[y,x]>0.5),fuel=round(float(s.F[y,x]),3),
                        value=round(float(s.value[y,x]),3),note='DSS off — no concept trace')
        return info

    # ---------- checkpoint / restore ----------
    def _capture(self):
        def snap_sim(sim): return dict(B=sim.B.copy(),F=sim.F.copy(),I=sim.I.copy(),tau=sim.tau.copy(),
                                       k=sim.k,ever=sim.ever_burned.copy(),supp=sim.last_supp.copy())
        return dict(base=snap_sim(self.baseline),man=snap_sim(self.managed),
                    prev_c=None if self.dss.prev_c is None else self.dss.prev_c.copy(),
                    avail=self.dss.avail,hist=len(self.history),ign=copy.deepcopy(self.ignitions),
                    last_dss=self.last_dss)
    def set_checkpoint(self):
        self._ckpt=self._capture(); self.checkpoint_idx=len(self.history)-1; return self.checkpoint_idx
    def restore_checkpoint(self):
        c=getattr(self,'_ckpt',None)
        if not c: return self.view()
        def load(sim,d):
            sim.B=d['B'].copy();sim.F=d['F'].copy();sim.I=d['I'].copy();sim.tau=d['tau'].copy()
            sim.k=d['k'];sim.ever_burned=d['ever'].copy();sim.last_supp=d['supp'].copy()
        load(self.baseline,c['base']); load(self.managed,c['man'])
        self.dss.prev_c=None if c['prev_c'] is None else c['prev_c'].copy(); self.dss.avail=c['avail']
        self.history=self.history[:c['hist']+1]; self.ignitions=copy.deepcopy(c['ign']); self.last_dss=c['last_dss']
        return self.view()

    # ---------- presets / export ----------
    def preset(self,name):
        self.clear_fire(); H,W=self.params['H'],self.params['W']; rng=np.random.default_rng()
        if name=='lightning':
            self.set_params(dict(wind_speed=0.7,humidity=0.2))
            pts=[(int(rng.uniform(0.15,0.85)*H),int(rng.uniform(0.15,0.85)*W)) for _ in range(5)]
            self.ignite(pts,radius=0)
        elif name=='arson_town':
            ty,tx=int(H*0.28),int(W*0.80)
            self.ignite([(ty+int(H*0.10),tx-int(W*0.08))],radius=1)
            self.set_params(dict(wind_dir=225,wind_speed=0.65))
        elif name=='dry_windy':
            self.set_params(dict(humidity=0.1,wind_speed=0.9,spotting=1.0))
            self.ignite([(int(H*0.6),int(W*0.25))],radius=1)
        return self.view()

    def export_csv(self):
        out=io.StringIO(); w=csv.writer(out)
        w.writerow(['step','minute','baseline_burned_%','dss_burned_%','baseline_loss','dss_loss','baseline_ha','dss_ha'])
        for s in self.history:
            w.writerow([s['step'],s['minute'],s['baseline']['burned'],s['managed']['burned'],
                        s['baseline']['loss'],s['managed']['loss'],s['baseline']['ha'],s['managed']['ha']])
        return out.getvalue()
    def save_scenario(self):
        return dict(params={k:self.params[k] for k in self.params}, ignitions=self.ignitions)
    def load_scenario(self,sc):
        self.reset(sc.get('params',{}))
        igs=sc.get('ignitions',[])
        for (y,x,r) in igs: self.ignite([(y,x)],radius=r)
        return self.view()
