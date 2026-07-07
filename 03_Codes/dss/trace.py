"""Intrinsic traceability and audit log (article Section III.F).

Every applied intervention is traced back to the concepts that drove it, the
rules those concepts fired together with their strengths, the observation
confidence behind each concept, and the resource-unit taskings that realized
it. The trace is recorded per agent and per step with no post-hoc
attribution; it is the level at which an operator inspects or overrides a
recommendation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DecisionTrace:
    """One agent-step audit record."""

    step: int
    agent_id: str
    confidence_mean: float
    quality: float
    fail_safe_applied: bool
    concept_means: Dict[str, float]          # regional mean activation
    top_rules: List[Dict]                    # [{rule_id, mean_strength}]
    intervention_means: Dict[str, float]     # applied regional mean intensity
    suppression_scale: float = 1.0
    assignments: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_trace(decision, global_decision, top_k: int = 5) -> DecisionTrace:
    """Assemble the audit record for one LocalDecision within a GlobalDecision."""
    mask = decision.region_mask
    if mask is None or not mask.any():
        mask = np.ones(next(iter(decision.concepts.values())).shape, dtype=bool)

    concept_means = {k: float(v[mask].mean())
                     for k, v in decision.concepts.items()}

    strengths = [(rid, float(s[mask].mean())) for rid, s in decision.firings]
    strengths.sort(key=lambda t: t[1], reverse=True)
    top_rules = [{"rule_id": rid, "mean_strength": round(s, 4)}
                 for rid, s in strengths[:top_k] if s > 0]

    interv_means = {k: float(v[mask].mean())
                    for k, v in global_decision.intervention.items()}

    assigns = [{"unit_id": a.unit_id, "kind": a.kind,
                "target": list(a.target),
                "travel_time": round(a.travel_time, 2)}
               for a in global_decision.assignments
               if mask[a.target[1], a.target[0]]]

    return DecisionTrace(
        step=decision.step,
        agent_id=decision.agent_id,
        confidence_mean=round(decision.confidence_mean, 4),
        quality=round(decision.quality, 4),
        fail_safe_applied=bool(
            global_decision.fail_safe_applied.get(decision.agent_id, False)),
        concept_means={k: round(v, 4) for k, v in concept_means.items()},
        top_rules=top_rules,
        intervention_means={k: round(v, 4) for k, v in interv_means.items()},
        suppression_scale=round(global_decision.suppression_scale, 4),
        assignments=assigns,
    )


@dataclass
class AuditLog:
    """Append-only log of decision traces for a whole run."""

    records: List[DecisionTrace] = field(default_factory=list)

    def append(self, trace: DecisionTrace) -> None:
        self.records.append(trace)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        payload = json.dumps([r.to_dict() for r in self.records], indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    def by_agent(self, agent_id: str) -> List[DecisionTrace]:
        return [r for r in self.records if r.agent_id == agent_id]
