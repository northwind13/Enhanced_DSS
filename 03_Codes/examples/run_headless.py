"""Headless example run.

Builds the wildland urban interface scenario, runs it to quiescence and prints a
cost report plus a coarse burn map. Use this to verify the engine without the
dashboard:

    python examples/run_headless.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from disasteraware import Simulator, scenarios, compute_costs


def ascii_burn_map(sim, downsample=4):
    burned = sim.ever_burned
    active = sim.state.burning > 0.5
    ny, nx = burned.shape
    rows = []
    for y in range(0, ny, downsample):
        line = []
        for x in range(0, nx, downsample):
            ys, xs = slice(y, y + downsample), slice(x, x + downsample)
            if active[ys, xs].any():
                line.append("@")
            elif burned[ys, xs].any():
                line.append("#")
            elif sim.world.fuel.ftype[ys, xs].max() == 0:
                line.append("~")
            elif sim.world.fuel.ftype[ys, xs].max() >= 2:
                line.append("T")
            else:
                line.append(".")
        rows.append("".join(line))
    return "\n".join(rows)


def main():
    world = scenarios.wui_interface()
    sim = Simulator(world)
    history = sim.run()

    print(f"Simulation finished after {sim.state.step} steps "
          f"({len(history)} recorded).")
    print()
    print("Legend: '.' unburned grass  'T' forest  '#' burned  "
          "'@' active fire  '~' non fuel")
    print(ascii_burn_map(sim))
    print()

    rep = compute_costs(sim)
    print("Cost report")
    print("-" * 40)
    for key, value in rep.to_dict().items():
        if isinstance(value, float):
            print(f"  {key:<32} {value:>15,.2f}")
        else:
            print(f"  {key:<32} {value:>15}")


if __name__ == "__main__":
    main()
