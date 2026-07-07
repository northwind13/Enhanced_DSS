"""Rule base for the intervention decision layer (article Eq. 7-8, Table A.IV).

A rule maps a concept antecedent to an intervention consequent on the same
five linguistic terms used everywhere in the pipeline:

    IF fire_threat_level is High AND suppression_feasibility is Low
    THEN suppression_effort = High, containment_line = Maximum

Antecedents are conjunctions (min t-norm) over decision-concept terms;
consequents assign a singleton intensity per intervention type. The seed rule
base combines two layers: a monotone base policy that maps each decision
concept onto its primary intervention type across all five terms, so that a
defined output exists everywhere in the concept space, and the specific
exception rules R1-R5 of the article that dominate in their region through
conjunction strength. Rule-base evolution (Phase 2/3) refines beyond this
seed.

The rule base is deliberately a mutable, serializable object: the Phase 2
evolution operators (membership modification, resolution increase) and the
Phase 3 generation loop (LLM proposer + RL admission) operate on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Tuple

from .fuzzy import canonical_term, FivePartition
from .concepts import DECISION_CONCEPTS

INTERVENTION_TYPES: Tuple[str, ...] = (
    "suppression_effort",
    "resource_deployment",
    "containment_line",
    "asset_protection",
    "evacuation",
    "public_warning",
)

# readable consequent labels -> intensity singletons
CONSEQUENT_LEVELS = {
    "NONE": 0.0,
    "MONITOR": 0.25, "LOW": 0.25,
    "ADVISORY": 0.5, "MEDIUM": 0.5,
    "HIGH": 0.75, "ISSUED": 0.75,
    "MAXIMUM": 1.0, "IMMEDIATE": 1.0, "VERY HIGH": 1.0, "VH": 1.0,
}


def consequent_level(label) -> float:
    if isinstance(label, (int, float)):
        v = float(label)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"consequent intensity out of [0,1]: {v}")
        return v
    key = str(label).strip().upper()
    if key not in CONSEQUENT_LEVELS:
        raise ValueError(f"unknown consequent label: {label!r}")
    return CONSEQUENT_LEVELS[key]


@dataclass
class Rule:
    """A single linguistic decision rule."""

    rule_id: str
    antecedent: Dict[str, str]          # signal name -> linguistic term
    consequent: Dict[str, float]        # intervention type -> intensity [0,1]
    weight: float = 1.0
    origin: str = "seed"                # seed | modified | generated

    def __post_init__(self):
        self.antecedent = {k: canonical_term(v) for k, v in self.antecedent.items()}
        self.consequent = {k: consequent_level(v) for k, v in self.consequent.items()}
        for itype in self.consequent:
            if itype not in INTERVENTION_TYPES:
                raise ValueError(f"unknown intervention type: {itype!r}")
        if self.weight <= 0:
            raise ValueError("rule weight must be positive")

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "antecedent": dict(self.antecedent),
                "consequent": dict(self.consequent), "weight": self.weight,
                "origin": self.origin}

    def text(self) -> str:
        ant = " AND ".join(f"{k} is {v}" for k, v in self.antecedent.items())
        con = ", ".join(f"{k} = {v:.2f}" for k, v in self.consequent.items())
        return f"IF {ant} THEN {con}"


@dataclass
class RuleBase:
    """Mutable container of rules plus the shared term partition."""

    rules: List[Rule] = dc_field(default_factory=list)
    partition: FivePartition = dc_field(default_factory=FivePartition)

    def add(self, rule: Rule) -> None:
        if any(r.rule_id == rule.rule_id for r in self.rules):
            raise ValueError(f"duplicate rule id: {rule.rule_id}")
        self.rules.append(rule)

    def remove(self, rule_id: str) -> None:
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

    def __len__(self) -> int:
        return len(self.rules)

    def to_dict(self) -> dict:
        return {"rules": [r.to_dict() for r in self.rules],
                "plateau": self.partition.plateau}


def default_rule_base() -> RuleBase:
    """Seed rule base: monotone base policy plus exception rules R1-R5.

    The base policy gives each decision concept a monotone term ladder onto
    its primary intervention type, guaranteeing a fired rule everywhere; the
    exception rules dominate in their region through conjunction strength.
    """
    rb = RuleBase()

    # --- article Table A.IV representative rules ---------------------------
    rb.add(Rule("R1", {"fire_threat_level": "High",
                       "suppression_feasibility": "Low"},
                {"suppression_effort": "High", "containment_line": "Maximum"}))
    rb.add(Rule("R2", {"asset_risk": "High", "evacuation_pressure": "High"},
                {"asset_protection": "High", "evacuation": "Issued"}))
    rb.add(Rule("R3", {"intervention_urgency": "Very High",
                       "suppression_feasibility": "High"},
                {"resource_deployment": "High", "suppression_effort": "High"}))
    rb.add(Rule("R4", {"evacuation_pressure": "Very High",
                       "access_road_status": "Low"},
                {"evacuation": "Immediate", "public_warning": "Maximum"}))
    rb.add(Rule("R5", {"fire_threat_level": "Low", "asset_risk": "Low"},
                {"suppression_effort": "Low", "public_warning": "Monitor"}))

    # --- monotone base policy: concept term ladder -> primary intervention -
    ladder = ("VL", "L", "M", "H", "VH")
    level = ("None", "Low", "Medium", "High", "Maximum")
    primary = {
        "fire_threat_level": "suppression_effort",
        "suppression_feasibility": "resource_deployment",
        "asset_risk": "asset_protection",
        "evacuation_pressure": "evacuation",
        "intervention_urgency": "public_warning",
    }
    for concept in DECISION_CONCEPTS:
        for term, lvl in zip(ladder, level):
            rb.add(Rule(f"C_{concept}_{term}", {concept: term},
                        {primary[concept]: lvl}, weight=0.25))

    # containment follows spread threat: fire threat ladder onto containment
    for term, lvl in zip(ladder, level):
        rb.add(Rule(f"C_containment_{term}", {"fire_threat_level": term},
                    {"containment_line": lvl}, weight=0.25))

    return rb
