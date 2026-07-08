"""Optional standard fuel model catalogue.

The thesis uses four representative fuel classes. For validation against
operational tools it is useful to map the standard catalogues (Anderson 13,
Scott and Burgan 40) onto those classes. This is a lookup only; the simulator
still runs on the four internal classes with the associated load and moisture.
"""

from __future__ import annotations

from .config import FUEL_NAME_TO_ID

# Anderson 13 (1982) mapped to the internal class, a nominal fuel load [0,1]
# and a nominal dead fuel moisture. Grass models -> grass, brush -> shrub,
# timber litter -> pine_litter/hardwood, slash -> hardwood.
ANDERSON_13 = {
    1:  ("Short grass",            "grass",       0.5, 0.06),
    2:  ("Timber grass/understory","grass",       0.7, 0.07),
    3:  ("Tall grass",             "grass",       1.0, 0.06),
    4:  ("Chaparral",              "shrub",       1.0, 0.09),
    5:  ("Brush",                  "shrub",       0.6, 0.10),
    6:  ("Dormant brush",          "shrub",       0.7, 0.09),
    7:  ("Southern rough",         "shrub",       0.5, 0.11),
    8:  ("Closed timber litter",   "hardwood",    0.5, 0.12),
    9:  ("Hardwood litter",        "hardwood",    0.6, 0.10),
    10: ("Timber litter/understory","pine_litter",1.0, 0.09),
    11: ("Light logging slash",    "hardwood",    0.6, 0.10),
    12: ("Medium logging slash",   "pine_litter", 0.9, 0.09),
    13: ("Heavy logging slash",    "pine_litter", 1.0, 0.08),
}

# A small Scott and Burgan (2005) subset by code.
SCOTT_BURGAN = {
    "GR1": ("Short sparse dry grass", "grass",       0.4, 0.06),
    "GR2": ("Low load dry grass",     "grass",       0.7, 0.06),
    "GS1": ("Grass-shrub low load",   "shrub",       0.6, 0.09),
    "SH2": ("Shrub moderate load",    "shrub",       0.8, 0.10),
    "TU1": ("Timber-understory low",  "pine_litter", 0.7, 0.10),
    "TL2": ("Timber litter low load", "hardwood",    0.6, 0.11),
    "TL5": ("Timber litter high load","pine_litter", 1.0, 0.09),
}


def catalog(name: str) -> dict:
    return {"Anderson 13": ANDERSON_13, "Scott & Burgan": SCOTT_BURGAN}.get(name, {})


def resolve(name: str, code) -> tuple:
    """Return (internal_fuel_id, load, moisture) for a standard model code."""
    entry = catalog(name).get(code)
    if entry is None:
        return FUEL_NAME_TO_ID["grass"], 0.7, 0.08
    _, cls, load, moist = entry
    return FUEL_NAME_TO_ID[cls], load, moist
