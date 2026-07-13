"""DisasterAware Simulation Framework.

A grid based, discrete time wildfire simulator implementing the hybrid fire
spread model of DisasterAware. It
exposes a clean Python core plus an editable World model so that a decision
support system can later be built on top of it.

Quick start:

    from disaster_phyengine import Simulator, scenarios, compute_costs

    world = scenarios.wui_interface()
    sim = Simulator(world)
    sim.run()
    print(compute_costs(sim).to_dict())
"""

from .config import (SimConfig, SpreadParams, SuppressionParams,
                     IntensityParams, ValueWeights, CostParams,
                     FuelModel, FUEL_MODELS, FUEL_NAME_TO_ID)
from .layers import (MeteoLayer, TopoLayer, FuelLayer, ValueLayer, ResourceLayer)
from .world import World, Asset, IgnitionEvent
from .state import SimulationState
from .core import Simulator, StepDiagnostics
from .costs import compute_costs, CostReport
from .interaction import InteractionOperator
from .observation import observe, Observation
from . import behavior
from . import fuel_moisture
from . import fuels_standard
from . import scenarios
from . import io_utils
from . import terrain
from . import viz

__version__ = "0.2.1"   # region overlays, clock/night render, perf pass
# Bumped on EVERY engine change; the app refuses to run against a stale
# in-process engine (Streamlit hot-reloads the app but never the packages).
ENGINE_BUILD = 35

__all__ = [
    "SimConfig", "SpreadParams", "SuppressionParams", "IntensityParams",
    "ValueWeights", "CostParams", "FuelModel", "FUEL_MODELS", "FUEL_NAME_TO_ID",
    "MeteoLayer", "TopoLayer", "FuelLayer", "ValueLayer", "ResourceLayer",
    "World", "Asset", "IgnitionEvent", "SimulationState",
    "Simulator", "StepDiagnostics", "compute_costs", "CostReport",
    "InteractionOperator", "observe", "Observation",
    "behavior", "fuel_moisture", "fuels_standard",
    "scenarios", "io_utils", "terrain", "viz",
]
