"""DisasterAware Decision Support System core.

Implements the concept-based fuzzy reasoning pipeline of the article
(Sections III.B to III.F) with the input space made concrete: sensor assets
(satellite, aerial, in-situ, field reports) sample the hidden wildfire state
into a composite observation with per-cell confidence, and resource units
(crews, engines, aircraft, dozers) realize the physical interventions
through unit tasking. On top of that sit the ten bounded features, the
five-term trapezoidal fuzzification, the four-level concept hierarchy with
confidence gating, rule-based intervention generation over six intervention
types, a non-inferential coordinator with a graduated fail-safe, the
protective-intervention effect model, the Eq. 9 decision cost, and intrinsic
traceability.

The DSS couples to the simulation core only through the observation
interface and the intervention injection point
(Simulator.step(resource_override=...)). It never reads or writes the
authoritative state directly.
"""

from .fuzzy import TERMS, FivePartition
from .features import FEATURE_NAMES, FeatureSet, extract_features, \
    observation_confidence
from .concepts import (BASE_CONCEPTS, MID_CONCEPTS, HIGHER_CONCEPTS,
                       TOP_CONCEPTS, DECISION_CONCEPTS, ALL_CONCEPTS,
                       compute_concepts, gate_concepts)
from .rules import INTERVENTION_TYPES, Rule, RuleBase, default_rule_base
from .inference import fire_rules
from .quality import decision_quality, DRIVER_MAP
from .sensing import Sensor, SensorNetwork, SENSOR_PRESETS
from .units import ResourceUnit, Assignment, assign_units, default_fleet, \
    UNIT_PRESETS
from .agent import RegionalAgent, LocalDecision
from .coordinator import Coordinator, GlobalDecision
from .mitigation import MitigationTracker, mitigated_costs
from .evaluate import evaluate_intervention, CostBreakdown, \
    DEFAULT_J_WEIGHTS, RESPONSE_COST_COEFF
from .trace import DecisionTrace, AuditLog
from .loop import DSSRunner, DSSStepResult

__all__ = [
    "TERMS", "FivePartition",
    "FEATURE_NAMES", "FeatureSet", "extract_features",
    "observation_confidence",
    "BASE_CONCEPTS", "MID_CONCEPTS", "HIGHER_CONCEPTS", "TOP_CONCEPTS",
    "DECISION_CONCEPTS", "ALL_CONCEPTS", "compute_concepts", "gate_concepts",
    "INTERVENTION_TYPES", "Rule", "RuleBase", "default_rule_base",
    "fire_rules", "decision_quality", "DRIVER_MAP",
    "Sensor", "SensorNetwork", "SENSOR_PRESETS",
    "ResourceUnit", "Assignment", "assign_units", "default_fleet",
    "UNIT_PRESETS",
    "RegionalAgent", "LocalDecision",
    "Coordinator", "GlobalDecision",
    "MitigationTracker", "mitigated_costs",
    "evaluate_intervention", "CostBreakdown", "DEFAULT_J_WEIGHTS",
    "RESPONSE_COST_COEFF",
    "DecisionTrace", "AuditLog",
    "DSSRunner", "DSSStepResult",
]
