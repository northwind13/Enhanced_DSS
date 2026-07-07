"""Closed-loop runner: couples the DSS to the simulation core (article Fig. 1).

Step 0 is the site setup: sensor assets and resource units are deployed on
the map. At every simulation step the runner then: lets the sensor network
sample the hidden state, gives every regional agent its region-restricted
composite observation with the per-cell confidence, merges the regional
decisions through the non-inferential coordinator (which tasks the resource
units), and injects the resulting ResourceLayer into the simulator as U_DSS
via Simulator.step(resource_override=...). The loop closes strictly through
the interfaces of the simulation core.

Without an explicit sensor list an ideal full-coverage network is used, and
without a unit fleet the demand is mapped onto U_Res fields directly, so
idealized baseline runs remain available for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from disasteraware.core import Simulator, StepDiagnostics
from disasteraware.costs import compute_costs, CostReport

from .agent import RegionalAgent
from .sensing import Sensor, SensorNetwork
from .units import ResourceUnit
from .coordinator import Coordinator, GlobalDecision
from .mitigation import MitigationTracker, mitigated_costs
from .rules import default_rule_base
from .trace import AuditLog, build_trace


@dataclass
class DSSStepResult:
    diag: StepDiagnostics
    global_decision: GlobalDecision
    costs: CostReport               # physical report (no effect model)
    mitigated: CostReport           # with protective-intervention discounts


class DSSRunner:
    """Hierarchically distributed DSS closed loop over a Simulator."""

    def __init__(self, sim: Simulator,
                 n_regions: Tuple[int, int] = (2, 2),
                 sensors: Optional[List[Sensor]] = None,
                 units: Optional[List[ResourceUnit]] = None,
                 quality_threshold: float = 0.6,
                 suppression_budget: Optional[float] = None,
                 epsilon: float = 0.0,
                 seed: Optional[int] = None):
        """n_regions : (rows, cols) rectangular partition of the grid.
        sensors   : sensor assets; None uses an ideal full-coverage network.
        units     : resource fleet; None maps demand onto fields directly.
        epsilon   : extra disturbance for the ideal-network fallback path.
        """
        self.sim = sim
        self.epsilon = float(epsilon)
        self.seed = seed
        ny, nx = sim.world.shape

        # step 0: site setup (sensing side)
        if sensors is not None:
            self.network = SensorNetwork(sensors, (ny, nx), seed=seed)
        elif epsilon > 0:
            self.network = None      # legacy noisy ideal-observer path
        else:
            self.network = SensorNetwork.ideal((ny, nx))

        self.agents: List[RegionalAgent] = []
        rows, cols = n_regions
        for r in range(rows):
            for c in range(cols):
                y0 = r * ny // rows
                y1 = (r + 1) * ny // rows - 1
                x0 = c * nx // cols
                x1 = (c + 1) * nx // cols - 1
                self.agents.append(RegionalAgent(
                    agent_id=f"A{r}{c}", region=(x0, y0, x1, y1),
                    grid_shape=(ny, nx), rule_base=default_rule_base()))

        self.coordinator = Coordinator(
            sim.world, quality_threshold=quality_threshold, units=units,
            suppression_budget=suppression_budget)
        self.audit = AuditLog()
        self.history: List[DSSStepResult] = []
        self.mitigation = MitigationTracker(shape=sim.world.shape)

    # -------------------------------------------------------------------- step
    def step(self) -> DSSStepResult:
        """One full DSS-in-the-loop simulation step."""
        k = self.sim.state.step

        # 1. sensing: the network samples the hidden state
        if self.network is not None:
            self.network.sample(self.sim, k)

        # 2. regional decisions on the sensed composite
        decisions = []
        for agent in self.agents:
            if self.network is not None:
                obs, kappa = self.network.composite(
                    k, region_mask=agent.region_mask)
                decisions.append(agent.decide(obs, self.sim.world,
                                              kappa=kappa))
            else:
                from disasteraware.observation import observe
                x0, y0, x1, y1 = agent.region
                obs = observe(self.sim, epsilon=self.epsilon, seed=self.seed,
                              region=(x0, y0, x1, y1))
                decisions.append(agent.decide(obs, self.sim.world,
                                              epsilon=self.epsilon))

        # 3. merge through the non-inferential coordinator (tasks the units)
        if self.network is not None:
            obs_full, _ = self.network.composite(k)
        else:
            from disasteraware.observation import observe
            obs_full = observe(self.sim, epsilon=0.0)
        gdec = self.coordinator.merge(decisions, obs_full)

        # 4. audit trail (intrinsic traceability)
        for dec in decisions:
            self.audit.append(build_trace(dec, gdec))

        # 5. inject U_DSS and advance the environment
        diag = self.sim.step(resource_override=gdec.resource_layer)

        # 6. effect model: record protective intensities at ignition time
        self.mitigation.update(gdec.intervention, self.sim.ever_burned)
        costs = compute_costs(self.sim)
        mitigated = mitigated_costs(self.sim, self.mitigation)

        result = DSSStepResult(diag=diag, global_decision=gdec, costs=costs,
                               mitigated=mitigated)
        self.history.append(result)
        return result

    # --------------------------------------------------------------------- run
    def run(self, n_steps: Optional[int] = None,
            stop_when_quiescent: bool = True) -> List[DSSStepResult]:
        limit = n_steps if n_steps is not None else self.sim.cfg.max_steps
        for _ in range(limit):
            result = self.step()
            if (stop_when_quiescent and result.diag.n_burning == 0
                    and self.sim.state.step > 1 and self.sim.ever_burned.any()):
                break
        return self.history

    def reset(self) -> None:
        self.sim.reset()
        for agent in self.agents:
            agent.reset()
        self.audit = AuditLog()
        self.history = []
        self.mitigation.reset()
        if self.network is not None:
            self.network = SensorNetwork(self.network.sensors,
                                         self.network.shape, seed=self.seed)
