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
                       FEATURE_SYM, FEATURE_NAME, FEATURE_MEASURES,
                       FEATURE_CHANNELS, feature_confidence)
from .sensors import (Sensor, SensorNetwork, SENSOR_CATALOG, CHANNELS,
                      LAMBDA_CONF, CHANNEL_SYMBOL, suggest_network)
from .fuzzy import (TERMS, fuzzify, term_vector, expected_value,
                    default_partition, REGISTRY, PartitionRegistry)
from .concepts import (HIERARCHY, DECISION_CONCEPTS, CONCEPT_LABEL,
                       GatedConcepts, infer_concepts, crisp, RHO_PERSIST,
                       concept_gates)
from .rules import (Rule, SEED_RULES, INTERVENTIONS, INTERVENTION_LABEL,
                    evaluate_rules, ALPHA_MIN)
from .actions import (resource_suggestion, decision_to_resources,
                      suggest_resource_items, build_resource_layer,
                      pool_efficiency, RESOURCE_KINDS, resource_kind_label)
from .evaluate import (clone_sim, forecast_cost, candidate_vs_noaction,
                       quality_Q, graduated_failsafe, CONCEPT_FAMILY)
from .persist import (save_learned, load_learned, load_parts,
                      load_vocab, merge_learned,
                      prune_learned, wipe_learned)
from .state import GeneratedState, config_id
from .resolve import resolve_active_set, ActiveSet
from .adapt import (make_runtime_rules, StageController, AdaptOutcome,
                    stage1_evfis, stage2_resolution, stage3_generative,
                    genai_status, genai_config, genai_probe,
                    genai_timeout)
from .decision_log import DecisionLog, DecisionRecord, RunLogger
from .loop import DecisionEngine, counterfactual

__all__ = ["Region", "partition", "partition_n", "ten_features",
           "FEATURE_ORDER", "FEATURE_META", "FEATURE_SYM", "FEATURE_NAME",
           "FEATURE_MEASURES", "FEATURE_CHANNELS", "feature_confidence",
           "concept_gates", "Sensor", "SensorNetwork", "SENSOR_CATALOG",
           "CHANNELS", "LAMBDA_CONF", "CHANNEL_SYMBOL", "suggest_network",
           "TERMS", "fuzzify", "term_vector", "expected_value",
           "default_partition", "REGISTRY", "PartitionRegistry",
           "HIERARCHY", "DECISION_CONCEPTS",
           "CONCEPT_LABEL", "GatedConcepts", "infer_concepts", "crisp",
           "RHO_PERSIST", "Rule", "SEED_RULES", "INTERVENTIONS",
           "INTERVENTION_LABEL", "evaluate_rules", "ALPHA_MIN",
           "resource_suggestion", "decision_to_resources",
           "suggest_resource_items", "build_resource_layer",
           "pool_efficiency", "RESOURCE_KINDS", "resource_kind_label",
           "clone_sim", "forecast_cost", "candidate_vs_noaction",
           "quality_Q", "graduated_failsafe", "CONCEPT_FAMILY",
           "make_runtime_rules", "StageController", "AdaptOutcome",
           "save_learned", "load_learned", "load_parts", "load_vocab",
           "merge_learned",
           "prune_learned", "wipe_learned",
           "GeneratedState", "config_id", "resolve_active_set",
           "ActiveSet",
           "DecisionLog", "DecisionRecord", "RunLogger", "genai_status",
           "genai_config", "genai_probe", "genai_timeout",
           "DecisionEngine",
           "counterfactual"]

# bumped on every dss change; checked by the app freshness gate
DSS_BUILD = 90
