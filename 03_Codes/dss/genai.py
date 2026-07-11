"""Generative rule proposer for the DisasterAware DSS (adaptation loop).

The staged adaptation loop escalates only when the cheaper stages fail
(parameter tuning, then resolution increase, then a GENERATIVE PROPOSAL). This
module implements the generative stage with the Anthropic Claude API: given a
compact, auditable summary of the current decision situation, Claude drafts
ONE new candidate fuzzy rule expressed in the framework's own vocabulary
(decision concepts and admissible interventions, fuzzy terms VL..VH). The
result is a real ``dss.rules.Rule`` that can be dropped straight into
``evaluate_rules``.

A proposal is only a SUGGESTION. It never acts on its own: it must pass the
verification gate and the satisficing acceptance test, and the operator can
override it (Explainability & Governance layer). If the SDK or an API key is
missing, the functions degrade gracefully (return ``None``) so the DSS keeps
running without generative help.

Setup:
    pip install anthropic
    # set the key in the environment (never hard-code it):
    #   Windows:  setx ANTHROPIC_API_KEY "sk-ant-..."
    #   bash:     export ANTHROPIC_API_KEY=sk-ant-...

Wiring example:
    from dss import genai
    from dss.rules import SEED_RULES, evaluate_rules
    rule = genai.propose_rule(genai.ProposalContext(...))
    if rule is not None:                       # then verify before trusting it
        intens, trace = evaluate_rules(concepts, features,
                                       rules=list(SEED_RULES) + [rule])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .concepts import DECISION_CONCEPTS
from .rules import INTERVENTIONS, Rule
from .fuzzy import TERMS                          # ("VL", "L", "M", "H", "VH")

DEFAULT_MODEL = "claude-sonnet-5"                 # or "claude-opus-4-8"


@dataclass
class ProposalContext:
    """Compact, auditable summary handed to the generative proposer."""
    concept_activations: dict                     # {decision_concept: 0..1}
    observation_confidence: float                 # 0..1
    decision_cost_trend: list                     # recent J_k (newest last)
    rule_count: int
    mean_firing_strength: float
    residual_hotspot: str                         # where cost concentrates
    available_interventions: list = field(
        default_factory=lambda: list(INTERVENTIONS))


def _rule_tool() -> dict:
    return {
        "name": "propose_rule",
        "description": "Propose exactly ONE new fuzzy decision rule.",
        "input_schema": {
            "type": "object",
            "properties": {
                "antecedent": {
                    "type": "object",
                    "description": "decision concept -> fuzzy term that fires "
                                   "the rule (use a subset of the concepts).",
                    "properties": {c: {"type": "string", "enum": list(TERMS)}
                                   for c in DECISION_CONCEPTS},
                },
                "consequent": {
                    "type": "object",
                    "description": "intervention -> fuzzy term (its intensity).",
                    "properties": {i: {"type": "string", "enum": list(TERMS)}
                                   for i in INTERVENTIONS},
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence: why this lowers the cost.",
                },
            },
            "required": ["antecedent", "consequent", "rationale"],
        },
    }


_SYSTEM = (
    "You are the generative proposer inside a wildfire decision-support "
    "system. You do NOT act: you only draft ONE candidate fuzzy rule that a "
    "verification gate checks before it can ever run. Use only the given "
    "decision concepts, interventions and fuzzy terms (VL,L,M,H,VH). Choose "
    "the least invasive action that addresses the residual cost, never use an "
    "intervention outside the available list, and treat life safety as the "
    "dominant priority."
)


def available() -> bool:
    """True if the Claude SDK and an API key are both present."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def propose_rule(ctx: ProposalContext,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = 800) -> Optional[Rule]:
    """Ask Claude for one candidate rule. Returns a ``Rule`` or ``None``."""
    if not available():
        return None
    import anthropic
    client = anthropic.Anthropic()                # reads ANTHROPIC_API_KEY
    user = (
        "Current decision situation (values normalized 0..1 unless noted):\n"
        f"- decision concept activations: {ctx.concept_activations}\n"
        f"- observation confidence: {ctx.observation_confidence:.2f}\n"
        f"- recent decision cost J_k (newest last): {ctx.decision_cost_trend}\n"
        f"- rule base: {ctx.rule_count} rules, mean firing "
        f"{ctx.mean_firing_strength:.2f}\n"
        f"- residual cost concentrates at: {ctx.residual_hotspot}\n"
        f"- decision concepts: {list(DECISION_CONCEPTS)}\n"
        f"- available interventions: {ctx.available_interventions}\n"
        f"- fuzzy terms: {list(TERMS)}\n\n"
        "Propose ONE new rule (via the propose_rule tool) that would lower "
        "the residual cost without over-committing resources."
    )
    try:
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=_SYSTEM,
            tools=[_rule_tool()],
            tool_choice={"type": "tool", "name": "propose_rule"},
            messages=[{"role": "user", "content": user}],
        )
    except Exception:                             # network / auth / quota
        return None
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" \
                and block.name == "propose_rule":
            return _to_rule(dict(block.input), ctx)
    return None


def _to_rule(payload: dict, ctx: ProposalContext) -> Optional[Rule]:
    """Validate the model output and build a real Rule, or return None."""
    try:
        ant: List = []
        for c, t in payload.get("antecedent", {}).items():
            if c in DECISION_CONCEPTS and str(t) in TERMS:
                ant.append((c, str(t)))
        con: List = []
        for i, t in payload.get("consequent", {}).items():
            if i in ctx.available_interventions and str(t) in TERMS:
                con.append((i, str(t)))
        if not ant or not con:                    # nothing usable / unsafe
            return None
        note = "GenAI (Claude): " + str(payload.get("rationale", ""))[:280]
        return Rule(name="genai_proposed", antecedents=ant,
                    consequents=con, note=note)
    except (AttributeError, TypeError, ValueError):
        return None
