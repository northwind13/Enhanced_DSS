"""
DisasterAware concept-based fuzzy decision layer.
Implements the Section III pipeline: six bounded features -> five-term
fuzzification -> four situational concepts (first/second order weighted
aggregation) -> concept rule bases for four intervention types
(Mamdani min-norm firing, firing-weighted singleton defuzzification) ->
regional partition -> resource-constrained coordination (normalisation +
projection) -> confidence gating -> satisficing evaluation with fail-safe.
"""
import numpy as np

LEVELS = np.array([0.0,0.25,0.5,0.75,1.0])   # five singleton consequent levels
CENT   = np.array([0.0,0.25,0.5,0.75,1.0])   # five-term centres (partition of unity)

def fuzzify5(x):
    """Triangular partition of unity -> membership in each of 5 terms.
    x: array (...,) -> returns array (...,5)."""
    x = np.clip(x,0,1)[...,None]
    d = np.abs(x-CENT)/0.25
    return np.clip(1.0-d,0,1)   # adjacent triangles, sum to 1

def _neigh_burn_frac(B):
    H,W=B.shape; acc=np.zeros((H,W))
    for dy in(-1,0,1):
        for dx in(-1,0,1):
            if dy==0 and dx==0: continue
            acc+=np.roll(np.roll(B,dy,0),dx,1)
    return acc/8.0

def _dilate(B, r=1):
    out=B.copy().astype(float)
    for _ in range(r):
        acc=out.copy()
        for dy in(-1,0,1):
            for dx in(-1,0,1):
                acc=np.maximum(acc, np.roll(np.roll(out,dy,0),dx,1))
        out=acc
    return out>0.5

class DisasterAwareDSS:
    def __init__(self, sim, n_regions=4, eps_noise=0.05, eta=0.45,
                 capacity=120.0, seed=0, gate=True, rule_set='concept'):
        self.sim=sim; self.nreg=n_regions; self.eps=eps_noise; self.eta=eta
        self.capacity=capacity; self.rng=np.random.default_rng(seed)
        self.gate=gate; self.rule_set=rule_set
        self.prev_c=None
        self.avail=1.0                      # resource availability (depletes)
        self._build_rules()
        # region label map (nreg x nreg blocks)
        H,W=sim.H,sim.W; g=int(round(np.sqrt(n_regions)))
        self.grid_n=g
        self.reg=np.zeros((H,W),dtype=int)
        for ri in range(g):
            for rj in range(g):
                self.reg[ri*H//g:(ri+1)*H//g, rj*W//g:(rj+1)*W//g]=ri*g+rj

    # ---------- concept rule bases (antecedent term idx per concept, consequent level idx) ----------
    def _build_rules(self):
        # antecedents over concepts (c1 threat, c2 feasibility, c3 exposure, c4 urgency)
        # term idx: 0..4 ; -1 = don't care ; consequent: level idx 0..4
        VL,L,M,H,VH=0,1,2,3,4
        self.rules={
            'm1':[ # direct suppression: threat & urgency high, feasibility ok
                ((VH,-1,-1,VH),VH), ((H,-1,-1,H),H), ((VH,M,-1,H),VH),
                ((M,-1,-1,M),M), ((L,-1,-1,L),VL), ((H,VL,-1,H),M)],
            'm2':[ # preventive fuel reduction: exposure/threat moderate, urgency rising, ahead of front
                ((M,-1,H,M),H), ((H,-1,H,M),VH), ((L,-1,M,M),M), ((VL,-1,L,L),VL),
                ((M,-1,VH,H),VH)],
            'm3':[ # asset protection: exposure high
                ((-1,-1,VH,-1),VH), ((-1,-1,H,H),H), ((-1,-1,M,M),M),
                ((-1,-1,L,-1),VL), ((H,-1,VH,VH),VH)],
            'm4':[ # indirect attack: high spread/threat but low feasibility
                ((VH,VL,-1,H),VH), ((H,L,-1,H),H), ((VH,L,-1,M),H),
                ((M,M,-1,M),M), ((L,-1,-1,L),VL)],
        }
    def _infer(self, cmemb, rules):
        """cmemb: (...,4,5) memberships per concept. Returns intensity (...,)."""
        shp=cmemb.shape[:-2]
        num=np.zeros(shp); den=np.zeros(shp)
        for ante,clev in rules:
            w=np.ones(shp)
            for ci,ti in enumerate(ante):
                if ti<0: continue
                w=np.minimum(w, cmemb[...,ci,ti])   # min-norm
            num+=w*LEVELS[clev]; den+=w
        return np.where(den>1e-9, num/den, 0.0)

    # ---------- feature extraction (with bounded observation noise) ----------
    def _features(self):
        s=self.sim
        # spread potential proxy from observable fuel/wind/slope
        R0=0.10+0.45*s.F; sp=np.clip(R0*(1+1.8*s.wind_speed+3*s.slope**2),0,1)
        f1=s.I                                   # fire intensity
        f2=s.F                                   # fuel load
        f3=sp                                    # spread potential
        f4=s.value                               # asset exposure
        f5=np.clip((1-0.6*s.slope)*self.avail,0,1)   # resource accessibility
        nb=_neigh_burn_frac(s.B>0.5)
        f6=np.clip(nb*(0.4+0.6*s.value)+0.3*(s.B>0.5),0,1)  # temporal urgency
        F=np.stack([f1,f2,f3,f4,f5,f6],-1)
        # bounded epistemic disturbance ||eps||<=eps (uniform, clipped)
        F=np.clip(F+self.rng.uniform(-self.eps,self.eps,F.shape),0,1)
        return F

    # ---------- concepts (first/second order weighted aggregation) ----------
    def _concepts(self,F):
        f1,f2,f3,f4,f5,f6=[F[...,i] for i in range(6)]
        c1=0.6*f1+0.4*f3                         # fire threat level
        c2=0.55*f5+0.45*(1-f2*0)+0; c2=0.55*f5+0.45*(1-np.clip(f2,0,1)*0.0)  # placeholder
        c2=np.clip(0.6*f5+0.4*(1-f2),0,1)        # suppression feasibility (high resource, low fuel load)
        c3=np.clip(0.6*f4+0.4*f3,0,1)            # asset exposure risk
        c4=np.clip(0.35*c1+0.25*(1-c2)+0.20*c3+0.20*f6,0,1)  # second-order urgency
        C=np.stack([c1,c2,c3,c4],-1)
        # confidence gating: blend with persistence prior by gamma (obs confidence)
        if self.gate and self.prev_c is not None:
            conf=np.clip(1.0-self.eps*2.0,0,1)   # scalar confidence proxy
            gamma=conf
            C=gamma*C+(1-gamma)*self.prev_c
        self.prev_c=C
        return np.clip(C,0,1)

    # ---------- one decision cycle -> coordinated suppression field ----------
    def decide(self):
        s=self.sim
        F=self._features(); C=self._concepts(F)
        cm=fuzzify5(C)                            # (...,4,5)
        u={m:self._infer(cm,self.rules[m]) for m in ('m1','m2','m3','m4')}
        # raw effort: direct attack on the burning front, plus a containment band of
        # preventive fuel reduction / indirect attack drawn 3 cells ahead of the front
        # (intervention type m4 builds a fuel break that blocks propagation).
        # the fire front is located from a NOISY burning observation: each cell's
        # sensed status is flipped with probability proportional to eps, so heavier
        # epistemic disturbance mis-targets the containment band.
        trueB=(s.B>0.5)
        flip=self.rng.random(trueB.shape) < (0.45*self.eps)
        Bobs=np.logical_xor(trueB, flip)
        # operational denoising: a sensed-burning cell is kept only if at least two
        # sensed neighbours agree, removing isolated false positives; the band is
        # then dilated so isolated false negatives do not break its continuity.
        nbf=_neigh_burn_frac(Bobs)
        Bobs=Bobs & (nbf >= 2.0/8.0)
        burning=Bobs.astype(float)
        band=(_dilate(Bobs,3)&(~Bobs)).astype(float)*(s.F>s.eps_fuel)
        # downwind containment is most effective; bias the band toward the wind heading
        contain = np.maximum(u['m4'], u['m2'])
        eff = ( 0.75*u['m1']*burning
                + 1.0*contain*band
                + 0.85*u['m3']*np.clip(s.value,0,1)*np.maximum(band,burning) )
        eff=np.clip(eff,0,1)
        # ---- coordination: resource normalisation + projection onto feasible set ----
        demand=eff.sum()
        rho=min(1.0, self.capacity/max(demand,1e-9))     # rho_k
        field=np.clip(eff*rho,0,1)                        # projection (per-cell cap)
        used=field.sum()
        # ---- satisficing evaluation Q (protection, economy, compliance) ----
        protect=np.clip((field*(0.5+0.5*s.value)*(burning+band)).sum()/max(used,1e-9),0,1)
        economy=1.0-min(used/self.capacity,1.0)
        compliance=1.0 if used<=self.capacity+1e-6 else 0.0
        Q=0.5*protect+0.3*economy+0.2*compliance
        accepted=Q>=self.eta
        if not accepted:
            field=field*0.5                               # graduated fail-safe attenuation
            used=field.sum()
        # resource depletion (availability decays with cumulative use)
        self.avail=float(np.clip(self.avail-0.0008*used,0.2,1.0))
        return dict(field=field, Q=float(Q), accepted=bool(accepted),
                    used=float(used), demand=float(demand), rho=float(rho),
                    u={k:float(v.mean()) for k,v in u.items()})
