"""Regional DSS agent: the full input-to-decision pipeline on one region.

Each agent i runs the pipeline of article Fig. 2 on its own region-restricted
sensed observation: feature extraction (Eq. 3), fuzzification (Eq. 4),
concept construction (Eq. 5), confidence gating with a persistence prior
(Eq. 6), and rule-based intervention generation (Eq. 7, 15-16). The agent
carries its own rule base (evolved independently in Phase 2/3) and its own
concept priors. Inference lives entirely here; the coordinator performs none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import extract_features, FeatureSet
from .concepts import compute_concepts, gate_concepts
from .rules import RuleBase, default_rule_base
from .inference import fire_rules
from .quality import decision_quality


@dataclass
class LocalDecision:
    """Candidate regional intervention with its full reasoning trace."""

    agent_id: str
    step: int
    intervention: Dict[str, np.ndarray]      # per-type intensity fields
    concepts: Dict[str, np.ndarray]          # gated concept activations
    features: FeatureSet
    firings: List[Tuple[str, np.ndarray]]    # (rule_id, strength field)
    quality: float
    confidence_mean: float
    region_mask: Optional[np.ndarray] = None


class RegionalAgent:
    """Local DSS agent D^i serving one rectangular region of the grid."""

    def __init__(self, agent_id: str,
                 region: Tuple[int, int, int, int],
                 grid_shape: Tuple[int, int],
                 rule_base: Optional[RuleBase] = None):
        """region : (x0, y0, x1, y1) inclusive window in grid coordinates."""
        self.agent_id = agent_id
        self.region = region
        self.rule_base = rule_base if rule_base is not None else default_rule_base()
        ny, nx = grid_shape
        x0, y0, x1, y1 = region
        xa, xb = sorted((int(x0), int(x1)))
        ya, yb = sorted((int(y0), int(y1)))
        mask = np.zeros((ny, nx), dtype=bool)
        mask[ya:yb + 1, xa:xb + 1] = True
        self.region_mask = mask
        self._prior_concepts: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ step
    def decide(self, obs, world, epsilon: float = 0.0,
               kappa: Optional[np.ndarray] = None) -> LocalDecision:
        """Run the full local pipeline on the sensed observation.

        kappa : per-cell confidence from the sensor network; when omitted
            the epsilon-based fallback confidence is used.
        """
        feats = extract_features(obs, world, epsilon=epsilon,
                                 region_mask=self.region_mask, kappa=kappa)
        raw = compute_concepts(feats)
        gated = gate_concepts(raw, self._prior_concepts, feats.confidence)
        # persistence prior for the next step (Eq. 6)
        self._prior_concepts = {k: v.copy() for k, v in gated.items()}

        # rule antecedents may reference decision concepts and raw features
        signals: Dict[str, np.ndarray] = dict(feats.values)
        signals.update({k: gated[k] for k in gated})

        intervention, firings = fire_rules(self.rule_base, signals)
        intervention = {k: np.where(self.region_mask, v, 0.0)
                        for k, v in intervention.items()}

        q = decision_quality(intervention, gated, region_mask=self.region_mask)
        kappa_mean = float(feats.confidence[self.region_mask].mean()) \
            if self.region_mask.any() else 0.0

        return LocalDecision(
            agent_id=self.agent_id, step=obs.step,
            intervention=intervention, concepts=gated, features=feats,
            firings=firings, quality=q, confidence_mean=kappa_mean,
            region_mask=self.region_mask)

    def reset(self) -> None:
        self._prior_concepts = None
