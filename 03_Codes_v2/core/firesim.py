"""
DisasterAware Simulation Core (thesis Chapter 4).
Grid cellular automaton with per-cell state s=(B, F_load, I, tau):
hybrid transition operator Phi = burning-status evolution (persistence +
wind-anisotropic Rothermel-type propagation + ignition injection),
fuel-mass evolution (combustion + suppression), descriptive intensity proxy,
and ignition-time memory. External inputs: meteorology (wind), terrain
(slope), fuel, value, resources. The decision layer acts only through the
suppression field U_DSS; it never writes to the state.
"""
import numpy as np
from landuse import World, CATS

CELL_M = 30.0      # physical cell size (metres)
DT_MIN = 2.0       # simulation time step (minutes)
ROS_REF = 90.0     # m/min that a normalised ROS of 1.0 represents (wind-driven crown run)

_NB = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class FireSim:
    def __init__(self, H=60, W=60, dt=1.0, seed=0,
                 theta_ign=0.18, eps_fuel=0.02, burn_frac=0.18,
                 fuel_density=1.0, wind_speed=0.55, wind_dir_deg=45.0,
                 n_responders=4, forest_frac=1.0, humidity=0.30, spotting=0.5):
        self.H, self.W, self.dt = H, W, dt
        self.theta_ign=theta_ign; self.eps_fuel=eps_fuel; self.burn_frac=burn_frac
        self.seed=int(seed); self.rng=np.random.default_rng(seed)
        self.n_responders=int(n_responders); self.forest_frac=float(forest_frac)
        self.build_world(fuel_density)
        self.wind_speed=float(wind_speed)
        self.wind_dir=np.deg2rad(float(wind_dir_deg))
        self.humidity=float(humidity); self.spotting=float(spotting)
        self.reset()

    # ---- external data layers (land-use + resources) ----
    def build_world(self, fuel_density=1.0):
        H,W=self.H,self.W
        self.world=World(H,W,self.seed,n_responders=self.n_responders,forest_frac=self.forest_frac)
        self.cat=self.world.cat
        self.slope=self.world.slope
        self.access=self.world.access
        self.stations=self.world.stations
        self.dist=self.world.dist
        self.fuel0=np.clip(self.world.fuel0*float(fuel_density),0.0,1.0); self.fuel0[self.cat==0]=0.0
        self.value=self.world.value
        # fuel moisture (0 dry .. 1 saturated): driven by humidity, wetter in groves/agri, 1 on water
        base=getattr(self,'humidity',0.30)
        self.moisture=np.clip(base+0.10*(self.cat==3)+0.15*(self.cat==2)-0.05*(self.cat==4),0,1)
        self.moisture[self.cat==0]=1.0
        # resource accessibility field (thesis Feature 5): exp decay with travel distance
        scale=max(8.0, min(H,W)*0.6)
        self.reach=np.clip(np.exp(-self.dist/scale)*self.access+0.12,0,1)

    def reset(self):
        H,W=self.H,self.W
        self.B=np.zeros((H,W)); self.F=self.fuel0.copy()
        self.I=np.zeros((H,W)); self.tau=np.zeros((H,W))
        self.k=0; self.ever_burned=np.zeros((H,W),dtype=bool)
        self.last_supp=np.zeros((H,W))
        self._update_intensity()

    # ---- meteorology setters (UI what-if) ----
    def set_wind(self, speed=None, dir_deg=None, humidity=None, spotting=None):
        if speed is not None: self.wind_speed=float(speed)
        if dir_deg is not None: self.wind_dir=np.deg2rad(float(dir_deg))
        if humidity is not None:
            self.humidity=float(humidity)
            self.moisture=np.clip(self.humidity+0.10*(self.cat==3)+0.15*(self.cat==2)-0.05*(self.cat==4),0,1); self.moisture[self.cat==0]=1.0
        if spotting is not None: self.spotting=float(spotting)

    # ---- ignition injection (UI: one or many points) ----
    def ignite(self, cells, radius=0):
        for (y,x) in cells:
            for dy in range(-radius,radius+1):
                for dx in range(-radius,radius+1):
                    iy,ix=y+dy,x+dx
                    if 0<=iy<self.H and 0<=ix<self.W and self.F[iy,ix]>self.eps_fuel:
                        self.B[iy,ix]=1.0; self.ever_burned[iy,ix]=True; self.tau[iy,ix]=0
        self._update_intensity()

    # ---- Rothermel-type rate of spread ----
    def _ros(self):
        R0=0.10+0.45*self.F
        R=R0*(1.0+1.8*self.wind_speed+3.0*self.slope**2)
        R=R*np.clip(1.0-0.7*self.moisture,0.05,1.0)   # moist fuel spreads slower
        return np.clip(R,0,1.0)
    def ros_mpm(self):
        return self._ros()*ROS_REF

    def _update_intensity(self):
        self.I=np.clip(0.5*self.F+0.3*self.wind_speed+0.2*self.slope,0,1)

    def _propagation(self,R):
        H,W=self.H,self.W; Psi=np.zeros((H,W))
        wd=np.array([np.cos(self.wind_dir),np.sin(self.wind_dir)])
        Bburn=self.B>0.5
        for (dy,dx) in _NB:
            d=np.array([-dy,-dx],dtype=float); d/=np.linalg.norm(d)
            g=max(0.0,float(d@wd))
            if g<=0: continue
            src=np.zeros((H,W))
            ys0,ys1=max(0,dy),H+min(0,dy); xs0,xs1=max(0,dx),W+min(0,dx)
            yt0,yt1=max(0,-dy),H+min(0,-dy); xt0,xt1=max(0,-dx),W+min(0,-dx)
            contrib=(R*Bburn).astype(float)
            src[yt0:yt1,xt0:xt1]=contrib[ys0:ys1,xs0:xs1]
            Psi+=g*src
        return Psi

    def step(self, U_supp=None):
        if U_supp is None: U_supp=np.zeros((self.H,self.W))
        self.last_supp=U_supp
        R=self._ros(); Psi=self._propagation(R)
        burning=self.B>0.5
        persist=burning&(self.F>self.eps_fuel)
        propagate=(Psi>self.theta_ign)&(self.F>self.eps_fuel)
        Bnext=(persist|propagate).astype(float)
        F_red=np.clip(3.5*U_supp,0,1)*self.F   # a committed suppression cell becomes a fuel break
        comb=burning.astype(float)*self.F*self.burn_frac
        Fnext=np.clip(self.F-comb-F_red,0,None)
        Bnext=Bnext*(Fnext>self.eps_fuel)
        newign=(~burning)&(Bnext>0.5); active=burning&(Bnext>0.5)
        taunext=np.where(newign,0.0,np.where(active,self.tau+self.dt,0.0))
        self.B,self.F,self.tau=Bnext,Fnext,taunext
        self._update_intensity(); self.ever_burned|=(self.B>0.5)
        self._spotting(); self.k+=1

    def _spotting(self):
        if self.spotting<=0: return
        by,bx=np.where(self.B>0.5); nb=len(by)
        if nb==0: return
        rate=0.03*self.spotting*self.wind_speed
        n=int(rate*nb) + (1 if self.rng.random()<(rate*nb)%1 else 0)
        if n<1: return
        wr,wc=np.cos(self.wind_dir),np.sin(self.wind_dir)
        idx=self.rng.choice(nb,size=min(n,nb),replace=False)
        for i in idx:
            dist=self.rng.integers(2,6)
            ty=int(round(by[i]+wr*dist+self.rng.normal(0,0.9)))
            tx=int(round(bx[i]+wc*dist+self.rng.normal(0,0.9)))
            if 0<=ty<self.H and 0<=tx<self.W and self.F[ty,tx]>self.eps_fuel and self.rng.random()<0.6:
                self.B[ty,tx]=1.0; self.ever_burned[ty,tx]=True; self.tau[ty,tx]=0

    # ---- metrics (real units) ----
    def cell_ha(self): return (CELL_M*CELL_M)/10000.0
    def burned_ha(self): return float(self.ever_burned.sum())*self.cell_ha()
    def elapsed_min(self): return self.k*DT_MIN
    # ---- metrics ----
    def burned_fraction(self): return float(self.ever_burned.mean())
    def asset_loss(self): return float((self.value*self.ever_burned).sum())
    def asset_total(self): return float(self.value.sum())
    def active_fire(self): return int((self.B>0.5).sum())
    def burned_by_cat(self):
        out={}
        for cid,meta in CATS.items():
            m=(self.cat==cid)
            out[meta[0]]=dict(burned=int((self.ever_burned&m).sum()),
                              active=int(((self.B>0.5)&m).sum()))
        return out

    # ---- render: per-cell code 0 unburned, 1 burned scar, 2 active fire ----
    def codes(self):
        c=np.zeros((self.H,self.W),dtype=np.uint8)
        c[self.ever_burned]=1; c[self.B>0.5]=2
        return c
