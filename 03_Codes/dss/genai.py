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

Setup (Claude Code subscription, no API key):
    #   install Claude Code, then run:  claude
    #   inside it type:  /login   (choose your Pro/Max plan, not the console)

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


_SYSTEM = (
    "You are the generative proposer inside a wildfire decision-support "
    "system. You do NOT act: you only draft ONE candidate fuzzy rule that a "
    "verification gate checks before it can ever run. Use only the given "
    "decision concepts, interventions and fuzzy terms (VL,L,M,H,VH). Choose "
    "the least invasive action that addresses the residual cost, never use an "
    "intervention outside the available list, and treat life safety as the "
    "dominant priority."
)


def _cli_available() -> bool:
    """True if the Claude Code command-line tool is on PATH. This path uses
    the user's Claude subscription (Pro/Max/Team) via `claude -p`, so no
    ANTHROPIC_API_KEY is needed."""
    import shutil
    return shutil.which("claude") is not None


def transport_mode() -> str:
    """Which proposer transport stage 3 will use: 'cli' (Claude Code on the
    user's subscription) or 'none'. Stage 3 runs only through Claude Code;
    there is no API-key path."""
    return "cli" if _cli_available() else "none"


def available() -> bool:
    """True if stage 3 can reach a model, i.e. the Claude Code CLI is present
    (it runs on the user's subscription)."""
    return _cli_available()


def _propose_via_cli(ctx: "ProposalContext", user: str,
                     model: Optional[str] = None,
                     timeout: float = 120.0) -> Optional[Rule]:
    """Ask the Claude Code CLI (`claude -p`) for one rule, using the user's
    Claude subscription instead of an API key. Returns a Rule or None."""
    import json
    import re
    import subprocess
    if not _cli_available():
        return None
    prompt = (
        _SYSTEM + "\n\n" + user + "\n\n"
        "Return ONLY a JSON object (no prose, no code fence) with keys: "
        '"antecedent" (an object mapping a subset of the decision concepts '
        'to a fuzzy term), "consequent" (an object mapping a subset of the '
        'interventions to a fuzzy term), and "rationale" (one sentence). '
        "Every fuzzy term must be one of VL, L, M, H, VH.")
    cmd = ["claude"]
    if model:
        cmd += ["--model", model]
    cmd += ["-p", prompt, "--output-format", "json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8",
                             errors="replace",
                             timeout=timeout)
    except Exception:
        return None
    if res.returncode != 0 or not res.stdout.strip():
        return None
    # `--output-format json` wraps the reply: {"type":"result",
    # "result":"...text...", ...}. Fall back to the raw text otherwise.
    inner = res.stdout
    try:
        wrap = json.loads(res.stdout)
        if isinstance(wrap, dict) and "result" in wrap:
            inner = wrap["result"]
    except Exception:
        pass
    m = re.search(r"\{.*\}", inner, re.S)          # first JSON object
    if not m:
        return None
    try:
        payload = json.loads(m.group(0))
    except Exception:
        return None
    return _to_rule(payload, ctx)


def propose_rule(ctx: ProposalContext,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = 800) -> Optional[Rule]:
    """Ask Claude for one candidate rule via the Claude Code CLI (`claude -p`)
    on the user's subscription. Returns a ``Rule`` or ``None``."""
    if not _cli_available():
        return None
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
        "Propose ONE new rule that would lower the residual cost without "
        "over-committing resources."
    )
    return _propose_via_cli(ctx, user, model=current_model())


def current_model() -> str:
    """The PINNED generative model for this campaign: the environment
    override if set, the module default otherwise. Never None — an
    unpinned call would silently take whatever the CLI defaults to
    that day, and the campaign could not state which engine produced
    its results. The run logger writes this value into meta.json."""
    return os.environ.get("DSS_GENAI_MODEL") or DEFAULT_MODEL


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
