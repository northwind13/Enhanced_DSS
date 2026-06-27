"""
DisasterAware DSS Core (thesis Chapter 5), M=3 intervention types.
Pipeline: six bounded features -> five-term fuzzification -> four situational
concepts (fire threat, suppression feasibility, asset exposure, urgency) ->
concept rule bases for three intervention types ->
priority-weighted action aggregation -> resource-constrained coordination ->
confidence gating -> satisficing evaluation with graduated fail-safe.

Intervention types (thesis 5.4.2):
  m1 Direct Suppression       (alpha=1.0)  reduce combustion at burning cells
  m2 Preventive Fuel Reduction(alpha=0.7)  fuel break ahead of the front
  m3 Asset Protection Priority(alpha=0.9)  defend high-value cells
Action aggregation: U = (a1 u1 + a2 u2 + a3 u3)/(a1+a2+a3)   (eq. 70)
Evaluation weights: w = (0.35, 0.30, 0.20, 0.15)             (Table 5.18)
"""
import numpy as np

LEVELS = np.array([0.0,0.25,0.5,0.75,1.0])
CENT   = np.array([0.0,0.25,0.5,0.75,1.0])
ALPHA  = dict(m1=1.0, m2=0.7, m3=0.9)            # thesis priority weights (eq. 71)
TYPE_LABEL = {'m1':'Direct suppression','m2':'Preventive fuel reduction','m3':'Asset protection'}
TYPE_COLOR = {'m1':'#22b8cf','m2':'#4a8fd0','m3':'#d65cc8'}

# Five trapezoidal terms over [0,1] forming a partition of unity, designed so
# that each linguistic term occupies an equal area (0.20). Shoulders at the
# ends; ramp width 0.10, core width 0.10 (shoulders 0.15).
_TRAP=[(-1.0,-1.0,0.15,0.25),(0.15,0.25,0.35,0.45),(0.35,0.45,0.55,0.65),
       (0.55,0.65,0.75,0.85),(0.75,0.85,2.0,2.0)]
def fuzzify5(x):
    x=np.clip(x,0,1)
    out=[]
    for (a,b,c,d) in _TRAP:
        up=np.clip((x-a)/(b-a),0,1) if b>a else np.ones_like(x)
        dn=np.clip((d-x)/(d-c),0,1) if d>c else np.ones_like(x)
        out.append(np.minimum(up,dn))
    return np.stack(out,-1)

def _neigh_burn_frac(B):
    H,W=B.shape; acc=np.zeros((H,W))
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            if dy==0 and dx==0: continue
            acc+=np.roll(np.roll(B,dy,0),dx,1)
    return acc/8.0

def _dilate(B,r=1):
    out=B.copy()
    for _ in range(r):
        g=out.copy()
        g[:-1,:]|=out[1:,:]; g[1:,:]|=out[:-1,:]; g[:,:-1]|=out[:,1:]; g[:,1:]|=out[:,:-1]
        out=g
    return out

class DisasterAwareDSS:
    def __init__(self, sim, n_regions=4, eps_noise=0.05, eta=0.45,
                 capacity=90.0, seed=0, gate=True):
        self.sim=sim; self.nreg=n_regions; self.eps=eps_noise; self.eta=eta
        self.capacity=capacity; self.rng=np.random.default_rng(seed); self.gate=gate
        self.prev_c=None; self.avail=1.0
        self._build_rules()

    def _build_rules(self):
        VL,L,M,H,VH=0,1,2,3,4
        # antecedents over concepts (c1 threat, c2 feasibility, c3 exposure, c4 urgency); -1 don't care
        self.rules={
          'm1':[((VH,H,-1,VH),VH),((H,H,-1,H),VH),((VH,M,-1,H),H),((M,M,-1,M),M),
                ((H,VL,-1,H),L),((L,-1,-1,L),VL)],                       # direct suppression
          'm2':[((H,VL,H,M),VH),((M,L,H,M),H),((H,L,M,M),H),((M,M,M,M),M),
                ((H,H,H,H),M),((VL,-1,L,L),VL)],                         # preventive (substitution)
          'm3':[((-1,-1,VH,VH),VH),((-1,-1,H,H),VH),((-1,-1,H,M),H),
                ((-1,-1,M,M),M),((-1,-1,L,-1),VL)],                      # asset protection
        }

    def _infer(self,cm,rules):
        shp=cm.shape[:-2]; num=np.zeros(shp); den=np.zeros(shp)
        for ante,clev in rules:
            w=np.ones(shp)
            for ci,ti in enumerate(ante):
                if ti<0: continue
                w=np.minimum(w,cm[...,ci,ti])
            num+=w*LEVELS[clev]; den+=w
        with np.errstate(invalid='ignore',divide='ignore'):
            return np.where(den>1e-9,num/den,0.0)

    def _features(self):
        s=self.sim
        R0=0.10+0.45*s.F; sp=np.clip(R0*(1+1.8*s.wind_speed+3*s.slope**2),0,1)
        f1=s.I; f2=s.F; f3=sp; f4=s.value
        f5=np.clip(getattr(s,'reach',1.0)*self.avail,0,1)  # resource accessibility (thesis F5)
        nb=_neigh_burn_frac(s.B>0.5)
        f6=np.clip(nb*(0.4+0.6*s.value)+0.3*(s.B>0.5),0,1)
        F=np.stack([f1,f2,f3,f4,f5,f6],-1)
        F=np.clip(F+self.rng.uniform(-self.eps,self.eps,F.shape),0,1)
        return F

    def _concepts(self,F):
        f1,f2,f3,f4,f5,f6=[F[...,i] for i in range(6)]
        c1=np.clip(0.6*f1+0.4*f3,0,1)
        c2=np.clip(0.6*f5+0.4*(1-f2),0,1)
        c3=np.clip(0.6*f4+0.4*f3,0,1)
        c4=np.clip(0.35*c1+0.25*(1-c2)+0.20*c3+0.20*f6,0,1)
        C=np.stack([c1,c2,c3,c4],-1)
        if self.gate and self.prev_c is not None:
            conf=np.clip(1.0-self.eps*2.0,0,1)
            C=conf*C+(1-conf)*self.prev_c
        self.prev_c=C
        return np.clip(C,0,1), (c1,c2,c3,c4)

    def decide(self):
        s=self.sim
        F=self._features(); C,(c1,c2,c3,c4)=self._concepts(F)
        cm=fuzzify5(C)
        u={t:self._infer(cm,self.rules[t]) for t in ('m1','m2','m3')}
        # observation of the front (noisy) + denoise
        trueB=(s.B>0.5)
        flip=self.rng.random(trueB.shape)<(0.45*self.eps)
        Bobs=np.logical_xor(trueB,flip)
        nbf=_neigh_burn_frac(Bobs); Bobs=Bobs&(nbf>=2.0/8.0)
        d1=_dilate(Bobs,1)&(~Bobs)                        # immediate front ring
        d2=_dilate(Bobs,2)&(~_dilate(Bobs,1))             # second ring
        ring1=d1&(s.F>s.eps_fuel); ring2=d2&(s.F>s.eps_fuel)
        band=(_dilate(Bobs,3)&(~Bobs))&(s.F>s.eps_fuel)
        valhi=s.value>0.30
        nearfire=_dilate(Bobs,5)
        # localized candidate effort per intervention type (thesis m1/m2/m3)
        e1=u['m1']*Bobs                                   # direct suppression on burning cells
        contain=np.maximum(u['m1'],u['m2'])               # commit the stronger of direct/preventive
        e2=np.maximum.reduce([u['m2']*band, contain*ring1, 0.75*contain*ring2])  # decisive fuel break ahead
        e3=u['m3']*np.clip(s.value,0,1)*(nearfire&(~Bobs))*valhi  # protect assets near fire
        a1,a2,a3=ALPHA['m1'],ALPHA['m2'],ALPHA['m3']
        eff=np.clip((a1*e1+a2*e2+a3*e3)/a1,0,1)           # priority-weighted aggregation (eq.70)
        # dominant intervention type per actionable cell (for overlay + log)
        stacked=np.stack([a1*e1,a2*e2,a3*e3],0)
        dom=np.argmax(stacked,0); active=eff>0.05
        reach=getattr(s,'reach',np.ones_like(eff))
        eff=eff*(0.4+0.6*reach)                           # resources limited by travel distance
        # ---- coordination: resource normalisation + projection ----
        demand=eff.sum(); rho=min(1.0,self.capacity/max(demand,1e-9))
        field=np.clip(eff*rho,0,1); used=field.sum()
        # ---- satisficing evaluation Q (thesis weights, L1 alignment) ----
        Gk=(s.B>0.5)|band
        def align(Ufield,concept):
            msk=Gk
            if msk.sum()<1: return 1.0
            return float(1.0-np.abs(Ufield[msk]-concept[msk]).mean())
        q1=align(field,c1)            # fire-spread mitigation vs threat
        q2=align(field,c3)            # asset-risk reduction vs exposure
        q3=1.0-min(used/max(self.capacity,1e-9),1.0)   # resource efficiency
        q4=align(field,c4)            # timeliness vs urgency
        Q=0.35*q1+0.30*q2+0.20*q3+0.15*q4
        accepted=Q>=self.eta
        if not accepted:
            field=field*0.5; used=field.sum()
        self.avail=float(np.clip(self.avail-0.0008*used,0.2,1.0))
        # ---- decision summary by type and land-use category ----
        catname={cid:meta[0] for cid,meta in __import__('landuse').CATS.items()} if hasattr(s,'cat') else {}
        summary={}
        for ti,t in enumerate(('m1','m2','m3')):
            cells=active&(dom==ti)
            byc={}
            if hasattr(s,'cat'):
                for cid,nm in catname.items():
                    n=int((cells&(s.cat==cid)).sum())
                    if n>0: byc[nm]=n
            summary[t]=dict(label=TYPE_LABEL[t], color=TYPE_COLOR[t],
                            cells=int(cells.sum()), by_cat=byc,
                            mean=round(float(u[t][Gk].mean()) if Gk.sum() else 0.0,3))
        self.last=dict(F=F, c=(c1,c2,c3,c4), cm=cm, u=u, dom=dom, active=active, field=field)
        return dict(field=field, dom=dom, active=active,
                    Q=float(Q), q=dict(spread=round(q1,3),asset=round(q2,3),
                                       resource=round(q3,3),timeliness=round(q4,3)),
                    accepted=bool(accepted), used=float(used), demand=float(demand),
                    rho=float(rho), summary=summary,
                    u={t:round(float(v.mean()),3) for t,v in u.items()})

    def concept_field(self, name):
        L=getattr(self,'last',None)
        if not L: return None
        idx={'threat':0,'feasibility':1,'exposure':2,'urgency':3}.get(name)
        if idx is None: return None
        return L['c'][idx]

    def inspect(self, y, x):
        L=getattr(self,'last',None)
        if not L: return None
        s=self.sim
        feats=['fire intensity','fuel load','spread potential','asset exposure','resource accessibility','temporal urgency']
        fvals=[round(float(L['F'][y,x,i]),3) for i in range(6)]
        cnames=['fire threat level','suppression feasibility','asset exposure risk','intervention urgency']
        cvals=[round(float(L['c'][i][y,x]),3) for i in range(4)]
        cm=L['cm'][y,x]
        types={}
        for t in ('m1','m2','m3'):
            fired=[]
            for ridx,(ante,clev) in enumerate(self.rules[t]):
                w=1.0
                for ci,ti in enumerate(ante):
                    if ti<0: continue
                    w=min(w,float(cm[ci,ti]))
                if w>0.02: fired.append(dict(rule=ridx+1,weight=round(w,3),level=round(float(LEVELS[clev]),2)))
            fired.sort(key=lambda r:-r['weight'])
            types[t]=dict(label=TYPE_LABEL[t],color=TYPE_COLOR[t],
                          degree=round(float(L['u'][t][y,x]),3),fired=fired[:3])
        from landuse import CATS
        dom=int(L['dom'][y,x]); act=bool(L['active'][y,x])
        domlabel=TYPE_LABEL[['m1','m2','m3'][dom]] if act else 'none'
        return dict(x=int(x),y=int(y),landuse=CATS[int(s.cat[y,x])][1],
                    value=round(float(s.value[y,x]),3),
                    reach=round(float(s.reach[y,x]) if hasattr(s,'reach') else 1.0,3),
                    burning=bool(s.B[y,x]>0.5),fuel=round(float(s.F[y,x]),3),
                    features=list(zip(feats,fvals)),concepts=list(zip(cnames,cvals)),
                    types=types,action=domlabel,applied=round(float(L['field'][y,x]),3))
