"""Central configuration / tunable constants for DisasterAware.
Physical units and the thesis-calibrated weights live here so they can be
changed in one place. Imported by firesim / dss / engine where relevant.
"""
# ---- physical units ----
CELL_M = 30.0       # cell size (metres)
DT_MIN = 2.0        # time step (minutes)
ROS_REF = 90.0      # m/min represented by a normalised ROS of 1.0

# ---- fire model ----
THETA_IGN = 0.18    # ignition (propagation) threshold
EPS_FUEL  = 0.02    # extinction fuel threshold
BURN_FRAC = 0.18    # combustion fraction per step
SUPP_GAIN = 3.5     # suppression effectiveness (committed effort -> fuel break)

# ---- DSS (thesis Chapter 5) ----
ALPHA = dict(m1=1.0, m2=0.7, m3=0.9)           # intervention priority weights (eq. 71)
EVAL_W = dict(spread=0.35, asset=0.30, resource=0.20, timeliness=0.15)  # Table 5.18
VALUE_W = dict(bld=0.20, crit=0.40, pop=0.25, evac=0.15)                # value layer 4.2.4
CAP_PER_FR = 25.0   # suppression capacity per first responder
