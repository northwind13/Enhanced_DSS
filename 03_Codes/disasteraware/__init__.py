"""DisasterAware Simulation Framework.

A grid based, discrete time wildfire simulator implementing the hybrid fire
spread model of the DisasterAware PhD thesis (Chapter 4 and appendices). It
exposes a clean Python core plus an editable World model so that a decision
support system can later be built on top of it.

Quick start:

    from disasteraware import Simulator, scenarios, compute_costs

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
from . import scenarios
from . import io_utils
from . import terrain
from . import viz

__version__ = "0.1.0"

__all__ = [
    "SimConfig", "SpreadParams", "SuppressionParams", "IntensityParams",
    "ValueWeights", "CostParams", "FuelModel", "FUEL_MODELS", "FUEL_NAME_TO_ID",
    "MeteoLayer", "TopoLayer", "FuelLayer", "ValueLayer", "ResourceLayer",
    "World", "Asset", "IgnitionEvent", "SimulationState",
    "Simulator", "StepDiagnostics", "compute_costs", "CostReport",
    "InteractionOperator", "scenarios", "io_utils", "terrain", "viz",
]
