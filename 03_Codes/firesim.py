"""
DisasterAware wildfire simulation core.
Grid-based cellular automaton implementing the hybrid fire-spread transition
operator of the thesis (Chapter 4): per-cell state s=(B, F_load, I, tau),
Rothermel-type rate of spread with wind anisotropy over an 8-neighbourhood,
combustion-driven fuel depletion, suppression-driven reduction, and a
descriptive intensity proxy. Physics is deterministic; the decision layer
acts only through the suppression input field U_DSS.
"""
import numpy as np

# 8-neighbourhood offsets and their unit direction vectors
_NB = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class FireSim:
    def __init__(self, H=60, W=60, dt=1.0, seed=0,
                 theta_ign=0.18, eps_fuel=0.02, burn_frac=0.18,
                 rng=None):
        self.H, self.W, self.dt = H, W, dt
        self.theta_ign = theta_ign       # ignition threshold Theta_ign
        self.eps_fuel = eps_fuel         # extinction fuel threshold
        self.burn_frac = burn_frac       # combustion fraction per step
        self.rng = rng or np.random.default_rng(seed)
        # ---- static data layers ----
        # Topography: slope (0..1) with a ridge; aspect direction (radians)
        yy, xx = np.mgrid[0:H, 0:W]
        self.slope = 0.15 + 0.25*np.exp(-((xx-W*0.6)**2)/(2*(W*0.18)**2))
        self.slope = np.clip(self.slope, 0, 1)
        # Fuel load map (0..1): heterogeneous, denser band
        base = 0.55 + 0.35*np.exp(-((yy-H*0.5)**2)/(2*(H*0.30)**2))
        noise = 0.12*self.rng.standard_normal((H,W))
        self.fuel0 = np.clip(base+noise, 0.05, 1.0)
        # Asset value map (0..1): two settlements
        self.value = np.zeros((H,W))
        for (cy,cx,s,a) in [(int(H*0.30),int(W*0.75),H*0.07,1.0),
                            (int(H*0.70),int(W*0.55),H*0.06,0.8)]:
            self.value += a*np.exp(-(((yy-cy)**2+(xx-cx)**2)/(2*s**2)))
        self.value = np.clip(self.value,0,1)
        # ---- meteorology (time-varying wind) ----
        self.wind_speed = 0.55          # normalised 0..1
        self.wind_dir = np.deg2rad(45)  # blowing toward NE
        self.reset()

    def reset(self, ign=None, wind_dir=None, wind_speed=None):
        H,W = self.H,self.W
        self.B = np.zeros((H,W))           # burning status {0,1}
        self.F = self.fuel0.copy()         # fuel load
        self.I = np.zeros((H,W))           # intensity proxy
        self.tau = np.zeros((H,W))         # ignition time
        self.k = 0
        self.ever_burned = np.zeros((H,W), dtype=bool)
        if wind_dir is not None: self.wind_dir=wind_dir
        if wind_speed is not None: self.wind_speed=wind_speed
        if ign is None: ign=(int(H*0.62), int(W*0.22))
        iy,ix=ign
        self.B[iy,ix]=1.0; self.ever_burned[iy,ix]=True; self.tau[iy,ix]=0
        self._update_intensity()

    # ---- Rothermel-type rate of spread f_ros(fuel, wind, slope) ----
    def _ros(self):
        # R = R0 * (1 + phi_w + phi_s), bounded to [0,1]
        R0 = 0.10 + 0.45*self.F                       # base spread grows with fuel
        phi_w = 1.8*self.wind_speed                   # wind factor
        phi_s = 3.0*(self.slope**2)                   # slope factor (Rothermel form)
        R = R0*(1.0+phi_w+phi_s)
        return np.clip(R, 0, 1.0)

    def _update_intensity(self):
        # f_intensity(fuel, wind, slope) in [0,1]; no feedback from I_k
        self.I = np.clip(0.5*self.F + 0.3*self.wind_speed + 0.2*self.slope, 0, 1)

    def _propagation_influence(self, R):
        """Psi_k(x,y): wind-weighted sum of ROS from burning neighbours."""
        H,W = self.H,self.W
        Psi = np.zeros((H,W))
        wd = np.array([np.cos(self.wind_dir), np.sin(self.wind_dir)])  # (dy?,dx?) -> treat as (row,col)
        Bburn = self.B>0.5
        for (dy,dx) in _NB:
            # neighbour (i,j) = (x+dy? ) ; influence flows neighbour -> target
            # direction neighbour->target is (-dy,-dx); align with wind
            d = np.array([-dy,-dx], dtype=float); d/=np.linalg.norm(d)
            gdir = max(0.0, float(d@wd))             # cos clipping -> anisotropy
            if gdir<=0: continue
            src = np.zeros((H,W))
            ys0,ys1 = max(0,dy),H+min(0,dy)
            xs0,xs1 = max(0,dx),W+min(0,dx)
            yt0,yt1 = max(0,-dy),H+min(0,-dy)
            xt0,xt1 = max(0,-dx),W+min(0,-dx)
            # value at neighbour (shifted) contributes to target
            contrib = (R*Bburn).astype(float)
            src[yt0:yt1, xt0:xt1] = contrib[ys0:ys1, xs0:xs1]
            Psi += gdir*src
        return Psi

    def step(self, U_supp=None, ign_inject=None):
        """Advance one step. U_supp: suppression effort field in [0,1]; ign_inject: bool map."""
        if U_supp is None: U_supp = np.zeros((self.H,self.W))
        R = self._ros()
        Psi = self._propagation_influence(R)
        burning = self.B>0.5
        # burning status update: persistence OR propagation OR ignition
        persist = burning & (self.F > self.eps_fuel)
        propagate = (Psi > self.theta_ign) & (self.F > self.eps_fuel)
        Bnext = (persist | propagate).astype(float)
        if ign_inject is not None:
            Bnext = np.maximum(Bnext, ign_inject.astype(float))
        # fuel update: combustion (only burning) + suppression reduction
        F_red = np.clip(0.97*U_supp*self.F, 0, self.F)     # f_supp mapping
        comb = burning.astype(float)*self.F*self.burn_frac
        Fnext = np.clip(self.F - comb - F_red, 0, None)
        # cells whose fuel collapsed below threshold cannot stay/!ignite
        Bnext = np.where(Fnext<=self.eps_fuel, np.minimum(Bnext, persist*0+ (persist& (Fnext>self.eps_fuel))), Bnext)
        Bnext = Bnext*(Fnext>self.eps_fuel)
        # tau update
        newign = (burning==False) & (Bnext>0.5)
        active = burning & (Bnext>0.5)
        taunext = np.where(newign, 0.0, np.where(active, self.tau+self.dt, 0.0))
        # commit
        self.B, self.F, self.tau = Bnext, Fnext, taunext
        self._update_intensity()
        self.ever_burned |= (self.B>0.5)
        self.k += 1
        return dict(R=R, Psi=Psi)

    # ---- metrics helpers ----
    def burned_fraction(self):
        return self.ever_burned.mean()
    def asset_loss(self):
        return float((self.value*self.ever_burned).sum())
    def asset_total(self):
        return float(self.value.sum())
    def active_fire(self):
        return float((self.B>0.5).sum())
