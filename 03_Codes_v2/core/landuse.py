"""
Land-use / asset layer + resource (first-responder) stations for DisasterAware.
Each cell has a land-use category that sets its fuel (fire side) and its
value-at-risk (decision side, thesis value weights). Suppression resources are
deployed from a set of first-responder stations placed on the map; resource
accessibility decays with travel distance to the nearest station (thesis
Feature 5: z5 = R_avail * exp(-beta*R_time) * G_access).
"""
import numpy as np

# id:(key,name_en,name_tr,emoji,color,fuel_base,V_bld,V_crit,V_pop,V_evac)
CATS = {
 0:('water','Water','Su','🌊','#244b6b',0.00,0.0,0.0,0.0,0.0),
 1:('grass','Grassland','Cayir','🌱','#6f8b3c',0.30,0.0,0.0,0.05,0.10),
 2:('agri','Agriculture','Tarim','🌾','#c2ad57',0.42,0.05,0.0,0.10,0.20),
 3:('grove','Grove','Koru','🌳','#3f7a3f',0.58,0.0,0.0,0.05,0.10),
 4:('forest','Forest','Orman','🌲','#1f5a2a',0.95,0.0,0.0,0.02,0.05),
 5:('resid','Residential','Konut','🏠','#c08552',0.16,0.70,0.10,0.55,0.55),
 6:('urban','City','Sehir','🏙️','#9aa0a6',0.10,0.90,0.30,0.95,0.70),
 7:('crit','Critical facility','Kritik tesis','🏥','#d23b3b',0.10,0.60,1.00,0.30,0.60),
 8:('animal','Livestock','Hayvancilik','🐄','#caa15a',0.40,0.05,0.0,0.20,0.40),
}
VW=dict(bld=0.20,crit=0.40,pop=0.25,evac=0.15)
STATION_ICON='🚒'

def cat_value(cid):
    _,_,_,_,_,_,vb,vc,vp,ve=CATS[cid]
    return VW['bld']*vb+VW['crit']*vc+VW['pop']*vp+VW['evac']*ve

def _grow(mask,r):
    out=mask.copy()
    for _ in range(r):
        g=out.copy(); g[:-1,:]|=out[1:,:]; g[1:,:]|=out[:-1,:]; g[:,:-1]|=out[:,1:]; g[:,1:]|=out[:,:-1]; out=g
    return out

class World:
    def __init__(self,H=80,W=80,seed=7,n_responders=4,forest_frac=1.0):
        self.H,self.W=H,W
        rng=np.random.default_rng(seed)
        cat=np.ones((H,W),dtype=int)
        yy,xx=np.mgrid[0:H,0:W]
        # ---- forests: several blobs scaled by forest_frac (more, larger) ----
        nblob=int(round((4+min(H,W)//20)*forest_frac))
        for _ in range(max(2,nblob)):
            cy,cx=rng.integers(int(H*0.05),int(H*0.8)),rng.integers(int(W*0.03),int(W*0.7))
            r=rng.integers(int(min(H,W)*0.10),int(min(H,W)*0.22))
            cat[(yy-cy)**2+(xx-cx)**2<=r*r]=4
        grove=_grow(cat==4,2)&(cat!=4); cat[grove]=3
        # ---- agriculture + livestock band ----
        cat[(yy>int(H*0.74))&(cat==1)]=2
        cat[(yy>int(H*0.88))&(xx<int(W*0.4))&(cat==2)]=8
        # ---- river (water barrier), meanders ----
        riv=np.abs(xx-(int(W*0.5)+(int(min(H,W)*0.07)*np.sin(yy/(H/9.0))).astype(int)))<=max(1,int(W*0.012))
        cat[riv]=0
        # ---- town: residential ring + urban core + critical facility ----
        for (fy,fx,rr) in [(0.28,0.80,0.11),(0.66,0.58,0.07)]:
            ty,tx=int(H*fy),int(W*fx)
            cat[(yy-ty)**2+(xx-tx)**2<=(min(H,W)*rr)**2]=5
            cat[(yy-ty)**2+(xx-tx)**2<=(min(H,W)*rr*0.45)**2]=6
        cyc,cxc=int(H*0.28),int(W*0.80)
        cat[(yy-(cyc+2))**2+(xx-(cxc+2))**2<=max(2,int(min(H,W)*0.03))**2]=7
        self.cat=cat
        # ---- derived fuel + value ----
        fuel=np.zeros((H,W)); val=np.zeros((H,W))
        for cid,meta in CATS.items():
            m=(cat==cid); fuel[m]=meta[5]; val[m]=cat_value(cid)
        fuel=np.clip(fuel+0.05*rng.standard_normal((H,W)),0.0,1.0); fuel[cat==0]=0.0
        self.fuel0=fuel; self.value=np.clip(val,0,1)
        # ---- terrain accessibility G_access (0..1): water/steep is hard ----
        self.slope=np.clip(0.15+0.25*np.exp(-((xx-W*0.6)**2)/(2*(W*0.2)**2)),0,1)
        self.access=np.clip(1.0-0.7*self.slope,0,1); self.access[cat==0]=0.0
        # ---- first-responder stations: placed on accessible, safe (low-fuel) cells ----
        self.stations=self._place_stations(rng,n_responders)
        self.dist=self._dist_to_stations()

    def _place_stations(self,rng,n):
        H,W=self.H,self.W; cand=[]
        # prefer cells near settlements / roads: residential or grass with good access
        good=((self.cat==5)|(self.cat==1)|(self.cat==2))&(self.access>0.5)
        ys,xs=np.where(good)
        if len(ys)==0: ys,xs=np.where(self.cat!=0)
        idx=rng.choice(len(ys),size=min(n,len(ys)),replace=False)
        picks=[(int(ys[i]),int(xs[i])) for i in idx]
        # spread them out a bit (greedy farthest-ish): simple shuffle is fine for demo
        return picks[:n]

    def _dist_to_stations(self):
        H,W=self.H,self.W
        yy,xx=np.mgrid[0:H,0:W]
        d=np.full((H,W),1e9)
        for (sy,sx) in self.stations:
            d=np.minimum(d,np.sqrt((yy-sy)**2+(xx-sx)**2))
        return d

def legend():
    return [dict(id=cid,key=m[0],name=m[1],name_tr=m[2],icon=m[3],color=m[4]) for cid,m in CATS.items()]
