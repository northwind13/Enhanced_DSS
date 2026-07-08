"""DisasterAware Decision Support System (rebuilt, phase 1).

Phase 1a implements the observation side of the architecture:
the domain is partitioned into Local DSS regions (one agent per region,
plus one Global DSS coordinator), each agent observes its own cells and
extracts the ten bounded observation features. Sensors are
emulated in this phase: the observation reads the live simulation state
directly; the sensor catalogue arrives in a later phase.
"""

from .regions import Region, partition, partition_n
from .features import (ten_features, FEATURE_ORDER, FEATURE_META,
                       FEATURE_SYM, FEATURE_NAME, FEATURE_MEASURES)
from .sensors import (Sensor, SensorNetwork, SENSOR_CATALOG, CHANNELS,
                      LAMBDA_CONF, CHANNEL_SYMBOL, suggest_network)
from .fuzzy import (TERMS, fuzzify, term_vector, expected_value,
                    default_partition)
from .concepts import (HIERARCHY, DECISION_CONCEPTS, CONCEPT_LABEL,
                       GatedConcepts, infer_concepts, crisp, RHO_PERSIST)
from .rules import (Rule, SEED_RULES, INTERVENTIONS, INTERVENTION_LABEL,
                    evaluate_rules)

__all__ = ["Region", "partition", "partition_n", "ten_features",
           "FEATURE_ORDER", "FEATURE_META", "FEATURE_SYM", "FEATURE_NAME",
           "FEATURE_MEASURES", "Sensor", "SensorNetwork", "SENSOR_CATALOG",
           "CHANNELS", "LAMBDA_CONF", "CHANNEL_SYMBOL", "suggest_network",
           "TERMS", "fuzzify", "term_vector", "expected_value",
           "default_partition", "HIERARCHY", "DECISION_CONCEPTS",
           "CONCEPT_LABEL", "GatedConcepts", "infer_concepts", "crisp",
           "RHO_PERSIST", "Rule", "SEED_RULES", "INTERVENTIONS",
           "INTERVENTION_LABEL", "evaluate_rules"]

# bumped on every dss change; checked by the app freshness gate
DSS_BUILD = 5
