"""Proposers for the generative stage of the open intervention space.

The stage-3 hook of the adaptation loop, ``dss.adapt._genai_propose``, is
a plain function ``(situation, timeout, engine, mission) -> dict | None``.
Everything that decides whether a proposal ACTS (the gate chain and the
reseeded shadow forecasts) lives behind that hook and never sees who
wrote the proposal. This module puts three proposers behind the same
hook so they can be compared on equal terms:

  LLMProposer     the language model through the Claude Code CLI (the
                  path the thesis used)
  EvoProposer     an evolutionary search over the SAME grammar (rules,
                  macros, clause actuators, intermediate concepts), with
                  the cheap gates as a filter and the shadow forecast as
                  fitness
  RandomProposer  uniform sampling of the grammar through the cheap gates

A campaign installs one with ``install(engine, proposer)`` or by setting
``adapt._genai_propose = proposer``; the ledger names it through
``proposer_name``. The search proposers read the trial context that the
stage leaves on the engine (``engine._trial_ctx``) so they can price a
candidate on the simulator exactly as the gates will, and they count the
forecasts they spend so budgets can be compared.

Nothing here can admit a rule. A proposer only returns candidates; the
gates in ``adapt._stage3_generative`` decide, for every proposer alike.
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import adapt as _a
from .actions import EFFECTS, SECTORS
from .concepts import DECISION_CONCEPTS, HIERARCHY
from .features import FEATURE_ORDER
from .fuzzy import TERMS
from .rules import (DOCTRINE_INTERVENTIONS, INTERVENTIONS, Rule,
                    DISCOVERABLE_INTERVENTIONS)


# ---------------------------------------------------------------- context
@dataclass
class TrialContext:
    """What the stage knows when it asks for a proposal: enough to run the
    cheap gates and to price a candidate on the shadow simulator."""
    build_override: Callable
    sim: object
    rules: List[Rule]
    horizon: int
    eff: Dict[str, np.ndarray]
    crisp: Dict[str, float]
    engine: object
    coverage_gap: bool = False
    situation: str = ""
    mission: str = ""
    # filled lazily
    _j_base: Dict[Tuple, float] = field(default_factory=dict)

    def dominant(self) -> Dict[str, str]:
        return _a._dominant_terms(self.eff)


# --------------------------------------------------------- cheap gates
def cheap_gates(prop: dict, ctx: TrialContext) -> Optional[str]:
    """The gates that cost no simulation, in the order the stage applies
    them. Returns the failing gate's name or None. The proposal is
    repaired in place (relevance) and a package is normalized in place,
    exactly as the stage would do before the rollouts."""
    eng = ctx.engine
    _a._repair_relevance(prop, ctx.eff)
    err = (_a._g1_g2(prop, engine=eng) or _a._fires_now(prop, ctx.eff)
           or _a._availability(prop, ctx.sim)
           or _a._pool_saturation(prop, engine=eng)
           or _a._ground_proposal(prop, ctx.sim, engine=eng))
    if err:
        return err
    pkg = bool(prop.get("new_concept") or prop.get("new_intervention"))
    if pkg:
        if eng is None:
            return "G2 package (no engine context)"
        perr = _a._validate_package(prop, eng)
        if perr:
            return perr
        ncn = (prop.get("new_concept") or {}).get("name")
        nin = (prop.get("new_intervention") or {}).get("name")
        uses = ((ncn and any(v == ncn for v, _t in prop["antecedents"]))
                or (nin and any(i == nin for i, _x in
                                prop["consequents"])))
        if not uses:
            return "G2 package (the rule does not use the new object)"
        return None
    ants = [(str(v), str(t)) for v, t in prop["antecedents"]]
    if _a._cell_covered(ctx.rules, ants):
        return "G2 duplicate cell"
    return None


# ------------------------------------------------------------ fitness
def forecast_gain(prop: dict, ctx: TrialContext, reseed: int = 101
                  ) -> Tuple[float, float, float]:
    """Price a candidate on the shadow simulator the way G3 does: the base
    extended by the candidate against the base without it, one seeded
    future. Returns (gain, j_with, j_without); gain > 0 is better.
    A package is installed on the engine for the measurement and removed
    afterwards, so nothing persists."""
    eng = ctx.engine
    key = (reseed,)
    if key not in ctx._j_base:
        ctx._j_base[key] = _a._cva(ctx.build_override, ctx.sim,
                                   list(ctx.rules), ctx.horizon,
                                   reseed=reseed)[0]
    j0 = ctx._j_base[key]
    undo = None
    if prop.get("new_concept") or prop.get("new_intervention"):
        undo = _a._install_package(prop, eng)
    rules = list(ctx.rules)
    rules.append(Rule("_trial",
                      [(v, t) for v, t in prop["antecedents"]],
                      [(i, float(x)) for i, x in prop["consequents"]],
                      note="proposer fitness trial"))
    try:
        j1 = _a._cva(ctx.build_override, ctx.sim, rules, ctx.horizon,
                     reseed=reseed)[0]
    finally:
        if undo:
            _a._uninstall_package(undo, eng)
    return float(j0 - j1), float(j1), float(j0)


# ------------------------------------------------------------ grammar
_LEVELS = {"VL": 0, "L": 1, "M": 2, "H": 3, "VH": 4}


class Grammar:
    """Sampling and mutation over the closed vocabulary. Every object it
    produces is syntactically what the gates expect; whether it is
    relevant, affordable or useful is for the gates to say."""

    def __init__(self, ctx: TrialContext, rng: np.random.Generator,
                 withheld: Sequence[str] = ()):
        self.ctx = ctx
        self.rng = rng
        eng = ctx.engine
        self.concepts = list(DECISION_CONCEPTS) + [
            c for c in (getattr(eng, "hierarchy", {}) or {})
            if c not in HIERARCHY and c not in DECISION_CONCEPTS]
        from .rules import withheld as _whd
        wh = set(withheld) | set(_whd())
        self.interventions = [i for i in INTERVENTIONS if i not in wh] + \
            list((getattr(eng, "macros", {}) or {}).keys())
        self.base_channels = [i for i in INTERVENTIONS if i not in wh]
        self.concept_inputs = list(FEATURE_ORDER) + [
            c for c in HIERARCHY if c not in DECISION_CONCEPTS
            and c != "operational_priority"] + list(DECISION_CONCEPTS)
        self.has_water = bool(np.asarray(
            ctx.sim.world.fuel.ftype == 5).any()) \
            if hasattr(ctx.sim, "world") else True
        self.dom = ctx.dominant()
        self._n = 0

    # -- antecedents: terms that hold now, or '>=' of a lower term, so
    #    the relevance gate is passed by construction most of the time
    def antecedents(self, k: Optional[int] = None) -> List[list]:
        k = k or int(self.rng.integers(1, 4))
        cs = list(self.rng.choice(len(DECISION_CONCEPTS),
                                  size=min(k, 5), replace=False))
        out = []
        for ci in cs:
            c = DECISION_CONCEPTS[int(ci)]
            t = self.dom.get(c, "M")
            if self.rng.random() < 0.4 and _LEVELS[t] > 0:
                lower = TERMS[max(0, _LEVELS[t] - int(self.rng.integers(1, 3)))]
                out.append([c, ">=" + lower])
            else:
                out.append([c, t])
        return out

    def intensity(self) -> float:
        return float(np.round(self.rng.uniform(0.5, 1.0), 2))

    def consequents(self, k: Optional[int] = None,
                    must_include: Optional[str] = None) -> List[list]:
        k = k or int(self.rng.integers(1, 4))
        pool = list(self.interventions)
        picks = []
        if must_include:
            picks.append(must_include)
            pool = [p for p in pool if p != must_include]
        n_more = max(0, k - len(picks))
        if n_more and pool:
            idx = self.rng.choice(len(pool), size=min(n_more, len(pool)),
                                  replace=False)
            picks += [pool[int(i)] for i in idx]
        return [[p, self.intensity()] for p in picks]

    def clause(self) -> dict:
        effects = list(EFFECTS)
        if not self.has_water:
            effects = [e for e in effects if e not in ("draft", "coat")]
        eff = str(self.rng.choice(effects))
        sec = str(self.rng.choice(list(SECTORS)))
        rin = int(self.rng.integers(0, 8))
        rout = int(min(15, rin + self.rng.integers(1, 7)))
        return {"effect": eff, "sector": sec, "range": [rin, rout],
                "amount": float(np.round(self.rng.uniform(0.6, 1.0), 2))}

    def _name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self.rng.integers(1000, 9999)}_{self._n}"

    def clause_actuator(self) -> dict:
        n = int(self.rng.integers(1, 4))
        return {"name": self._name("sited"),
                "clauses": [self.clause() for _ in range(n)]}

    def macro(self) -> dict:
        n = int(self.rng.integers(2, 4))
        idx = self.rng.choice(len(self.base_channels),
                              size=min(n, len(self.base_channels)),
                              replace=False)
        comp = [[self.base_channels[int(i)],
                 float(np.round(self.rng.uniform(0.4, 1.0), 2))]
                for i in idx]
        comp[0][1] = 1.0
        return {"name": self._name("bundle"), "composition": comp}

    def concept(self) -> dict:
        n = int(self.rng.integers(2, 5))
        idx = self.rng.choice(len(self.concept_inputs),
                              size=min(n, len(self.concept_inputs)),
                              replace=False)
        w = self.rng.uniform(0.2, 1.0, size=len(idx))
        w = w / w.sum()
        return {"name": self._name("concept"), "level": "intermediate",
                "inputs": [[self.concept_inputs[int(i)],
                            float(np.round(float(x), 3))]
                           for i, x in zip(idx, w)]}

    def sample(self, kind: Optional[str] = None) -> dict:
        """One syntactically complete proposal. kind in
        {rule, macro, actuator, concept}; None draws one."""
        if kind is None:
            if os.environ.get("DSS_NO_COMPOSED", "").strip() == "1":
                # closed-vocabulary search (revision K13): rules and macros
                # over the six families only, no composed intervention
                kind = str(self.rng.choice(["rule", "macro"], p=[0.7, 0.3]))
            elif _a.concepts_off():
                # same relative odds of the other three kinds
                kind = str(self.rng.choice(
                    ["rule", "macro", "actuator"],
                    p=[0.35 / 0.85, 0.15 / 0.85, 0.35 / 0.85]))
            else:
                kind = str(self.rng.choice(
                    ["rule", "macro", "actuator", "concept"],
                    p=[0.35, 0.15, 0.35, 0.15]))
        if kind == "rule":
            return {"antecedents": self.antecedents(),
                    "consequents": self.consequents(),
                    "rationale": "search: plain rule"}
        if kind == "macro":
            m = self.macro()
            return {"antecedents": self.antecedents(),
                    "consequents": self.consequents(
                        k=int(self.rng.integers(1, 3)),
                        must_include=m["name"]),
                    "new_intervention": m,
                    "rationale": "search: composite intervention"}
        if kind == "actuator":
            act = self.clause_actuator()
            # a new object stands on its own most of the time, which is
            # what G5b asks for and what makes the gain attributable
            k = 1 if self.rng.random() < 0.75 else 2
            return {"antecedents": self.antecedents(),
                    "consequents": self.consequents(
                        k=k, must_include=act["name"]),
                    "new_intervention": act,
                    "rationale": "search: sited intervention"}
        c = self.concept()
        ants = self.antecedents(k=int(self.rng.integers(0, 2)))
        ants.append([c["name"], str(self.rng.choice(["H", "VH", ">=M"]))])
        return {"antecedents": ants,
                "consequents": self.consequents(),
                "new_concept": c,
                "rationale": "search: intermediate concept"}

    # -- mutation and crossover on proposals
    def mutate(self, prop: dict) -> dict:
        p = copy.deepcopy(prop)
        r = self.rng.random()
        ni = p.get("new_intervention")
        if ni and ni.get("clauses") and r < 0.5:
            cl = ni["clauses"][int(self.rng.integers(len(ni["clauses"])))]
            what = self.rng.random()
            if what < 0.25:
                cl["effect"] = self.clause()["effect"]
            elif what < 0.5:
                cl["sector"] = str(self.rng.choice(list(SECTORS)))
            elif what < 0.75:
                cl["range"] = self.clause()["range"]
            else:
                cl["amount"] = float(np.round(
                    np.clip(cl["amount"] + self.rng.normal(0, 0.15),
                            0.1, 1.0), 2))
            ni["name"] = self._name("sited")
        elif r < 0.75 and p.get("consequents"):
            i = int(self.rng.integers(len(p["consequents"])))
            p["consequents"][i][1] = float(np.round(
                np.clip(p["consequents"][i][1] + self.rng.normal(0, 0.15),
                        0.05, 1.0), 2))
        else:
            p["antecedents"] = self.antecedents()
        return p

    def crossover(self, a: dict, b: dict) -> dict:
        p = copy.deepcopy(a)
        q = b
        na, nb = p.get("new_intervention"), q.get("new_intervention")
        if na and nb and na.get("clauses") and nb.get("clauses"):
            # swap one clause
            i = int(self.rng.integers(len(na["clauses"])))
            j = int(self.rng.integers(len(nb["clauses"])))
            na["clauses"][i] = copy.deepcopy(nb["clauses"][j])
            na["name"] = self._name("sited")
        elif self.rng.random() < 0.5:
            p["antecedents"] = copy.deepcopy(q["antecedents"])
        return p


# ---------------------------------------------------------- proposers
class Proposer:
    """Callable with the hook signature. Subclasses implement
    ``candidates(ctx, k)`` and get budget accounting for free."""
    name = "base"

    def __init__(self):
        self.proposer_name = self.name
        self.calls = 0
        self.forecasts = 0
        self.candidates_tried = 0

    def __call__(self, situation, timeout=None, engine=None, mission=""):
        self.calls += 1
        ctx = getattr(engine, "_trial_ctx", None) if engine else None
        if ctx is not None:
            ctx.situation, ctx.mission = situation, mission
        prop = self.propose(ctx, situation, timeout, engine, mission)
        if isinstance(prop, dict):
            prop["_source"] = self.name
        return prop

    def propose(self, ctx, situation, timeout, engine, mission):
        raise NotImplementedError


class LLMProposer(Proposer):
    """The language model through the CLI. Thin wrapper so the ledger and
    the budget counters treat it like the others."""
    name = "llm"

    def __init__(self, transport=None):
        super().__init__()
        self._transport = transport or _a._genai_propose_cli_entry()

    def propose(self, ctx, situation, timeout, engine, mission):
        return self._transport(situation, timeout=timeout, engine=engine,
                               mission=mission)


class SearchProposer(Proposer):
    """Common machinery: a sample is kept only if it clears the cheap
    gates; a candidate is priced on one seeded shadow forecast; the best
    of k is returned, the rest are kept on the proposer for inspection.
    The forecast budget per call bounds what the search may spend."""
    name = "search"

    def __init__(self, seed: int = 0, forecast_budget: int = 36,
                 withheld: Sequence[str] = (), k: int = 3,
                 prior=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.forecast_budget = int(forecast_budget)
        self.withheld = tuple(withheld)
        self.k = int(k)
        self.prior = prior
        self.last_ranked: List[Tuple[float, dict]] = []
        self.last_filtered = 0

    def _passes(self, prop: dict, ctx: TrialContext) -> bool:
        try:
            return cheap_gates(prop, ctx) is None
        except Exception:
            return False

    def _score(self, prop: dict, ctx: TrialContext) -> float:
        before = _a.CVA_CALLS
        try:
            gain, _j1, _j0 = forecast_gain(prop, ctx)
        except Exception:
            gain = -1e9
        self.forecasts += _a.CVA_CALLS - before
        if self.prior is not None:
            try:
                self.prior.observe_trial(prop, ctx, gain)
            except Exception:
                pass
        return gain

    def propose(self, ctx, situation, timeout, engine, mission):
        if ctx is None:
            return None
        spent0 = self.forecasts
        ranked = self.search(ctx)
        if not ranked:
            # NO CANDIDATE TO PRICE is a verdict of this call, not an
            # unreachable model. None would be read by stage 3 as "model
            # unreachable" and retire the open layer for the rest of the
            # run (loop.py, _genai_dead); the skip marker records the call
            # as an ordinary rejection and the next window tries again.
            return {"_skip": "no candidate passed the cheap gates",
                    "_search": dict(priced=0,
                                    forecasts=self.forecasts - spent0,
                                    filtered=self.last_filtered)}
        # THE GATES READ TWO FUTURES. A search that optimizes one seed
        # overfits it and is turned away at G4 or G5 on the other, so the
        # finalists are priced on the second seed as well and ranked by
        # the worse of the two, which is the quantity the gates test.
        finalists = []
        _one = os.environ.get("DSS_ONE_FUTURE") == "1"
        for g1, p in ranked[: self.k]:
            if _one:
                g2 = g1       # K25 gate ablation: selection on one future
            elif self.forecasts - spent0 < self.forecast_budget + self.k:
                before = _a.CVA_CALLS
                try:
                    g2 = forecast_gain(p, ctx, reseed=202)[0]
                except Exception:
                    g2 = -1e9
                self.forecasts += _a.CVA_CALLS - before
            else:
                g2 = g1
            finalists.append((min(g1, g2), g1, g2, p))
        finalists.sort(key=lambda t: -t[0])
        self.last_ranked = [(f[0], f[3]) for f in finalists] + \
            [(g, p) for g, p in ranked[self.k:]]
        if finalists[0][0] <= 1e-9:
            # nothing priced lowered the forecast on both futures: do not
            # hand the gates a candidate that is known to fail
            return {"_skip": f"{len(ranked)} candidates priced, none "
                             "lowered the forecast on both futures",
                    "_search": dict(priced=len(ranked),
                                    forecasts=self.forecasts - spent0,
                                    filtered=self.last_filtered,
                                    best_gain=float(finalists[0][0]))}
        best = copy.deepcopy(finalists[0][3])
        best["_candidates"] = [dict(gain=float(f[0]), gain_a=float(f[1]),
                                    gain_b=float(f[2]), proposal=f[3])
                               for f in finalists]
        best["_search"] = dict(priced=len(ranked),
                               forecasts=self.forecasts - spent0,
                               filtered=self.last_filtered)
        return best

    def search(self, ctx) -> List[Tuple[float, dict]]:
        raise NotImplementedError


class RandomProposer(SearchProposer):
    """Uniform samples of the grammar, cheap gates as the only filter,
    priced until the forecast budget is spent."""
    name = "random"

    def search(self, ctx):
        g = Grammar(ctx, self.rng, self.withheld)
        spent0 = self.forecasts
        ranked = []
        tries = 0
        while (self.forecasts - spent0 < self.forecast_budget
               and tries < 20 * max(1, self.forecast_budget)):
            tries += 1
            p = g.sample()
            if not self._passes(p, ctx):
                self.last_filtered += 1
                continue
            self.candidates_tried += 1
            ranked.append((self._score(p, ctx), p))
        ranked.sort(key=lambda t: -t[0])
        return ranked


class LLMScreenProposer(SearchProposer):
    """The language model as the source of candidates for a screened search.

    One model call asks for several DIFFERENT proposals in the schema of the
    single-shot proposer. Each is passed through the cheap gates and priced
    on the shadow simulator exactly as a search candidate is; whatever is
    left of the forecast budget is spent on local mutations of the best
    model candidates (the same mutation operator the evolutionary search
    uses). The finalists are priced on the second seed and the best goes to
    the gate chain, as for every search proposer. With the same forecast
    budget as RandomProposer, the comparison between the two isolates where
    the candidates come from: the model, or the grammar.

    The origin of the candidate handed to the gates ('llm' or 'mutation') is
    recorded in its search record, so the ledger can say how many admitted
    objects the model wrote itself."""
    name = "llmscreen"

    def __init__(self, seed: int = 0, forecast_budget: int = 36,
                 withheld: Sequence[str] = (), k: int = 3, prior=None,
                 n_model: Optional[int] = None, fill: Optional[bool] = None,
                 transport=None):
        super().__init__(seed, forecast_budget, withheld, k, prior)
        self._transport = transport or _a._genai_propose_cli_entry()
        self.n_model = int(n_model or os.environ.get("DSS_LLM_SCREEN_N", 6))
        self.fill = (os.environ.get("DSS_LLM_SCREEN_FILL", "1") != "0"
                     if fill is None else bool(fill))
        self._timeout = None
        self.model_candidates = 0
        self.model_kept = 0
        self._last = (0, 0)

    def _ask(self, ctx) -> List[dict]:
        ask = (str(ctx.situation or "")
               + f"\nThis time return {self.n_model} DIFFERENT proposals, "
               "each one complete and following the schema exactly, as ONE "
               'minified JSON object of the form {"candidates": [P1, P2, ...]}. '
               "Make them differ in tactic (rule, macro, sited intervention, "
               "sector, band), not only in numbers. Each must hold for the "
               "PRESENT reading.")
        raw = self._transport(ask, timeout=self._timeout, engine=ctx.engine,
                              mission=ctx.mission)
        if not isinstance(raw, dict):
            return []
        for key in ("candidates", "proposals", "items"):
            lst = raw.get(key)
            if isinstance(lst, list):
                return [c for c in lst if isinstance(c, dict)][: 2 * self.n_model]
        if raw.get("antecedents") or raw.get("consequents"):
            return [raw]
        return []

    def propose(self, ctx, situation, timeout, engine, mission):
        self._timeout = timeout
        best = super().propose(ctx, situation, timeout, engine, mission)
        if isinstance(best, dict):
            rec = best.setdefault("_search", {})
            rec["model_candidates"], rec["model_kept"] = self._last
            if not best.get("_skip"):
                rec["origin"] = best.get("_origin", "llm")
        return best

    def search(self, ctx) -> List[Tuple[float, dict]]:
        spent0 = self.forecasts
        ranked: List[Tuple[float, dict]] = []
        cands = self._ask(ctx)
        kept = 0
        pool: List[Tuple[float, dict]] = []
        for c in cands:
            p = copy.deepcopy(c)
            p["_origin"] = "llm"
            if not self._passes(p, ctx):
                self.last_filtered += 1
                continue
            kept += 1
            self.candidates_tried += 1
            g = self._score(p, ctx)
            ranked.append((g, p))
            pool.append((g, p))
        self.model_candidates += len(cands)
        self.model_kept += kept
        self._last = (len(cands), kept)
        if self.fill and pool:
            gram = Grammar(ctx, self.rng, self.withheld)
            tries = 0
            while (self.forecasts - spent0 < self.forecast_budget
                   and tries < 20 * max(1, self.forecast_budget)):
                tries += 1
                pool.sort(key=lambda t: -t[0])
                parent = pool[int(self.rng.integers(min(3, len(pool))))][1]
                try:
                    q = gram.mutate(parent)
                except Exception:
                    continue
                q["_origin"] = "mutation"
                if not self._passes(q, ctx):
                    self.last_filtered += 1
                    continue
                self.candidates_tried += 1
                g = self._score(q, ctx)
                ranked.append((g, q))
                pool.append((g, q))
        ranked.sort(key=lambda t: -t[0])
        return ranked


class RandomVarProposer(LLMScreenProposer):
    """Random seeds with the variation operator of the model search
    (revision K14): six uniform samples of the grammar take the place of
    the six model candidates, and the rest of the forecast budget goes to
    local variations of the best, exactly as in LLMScreenProposer. Pairs
    with the full VODA configuration; the two differ only in where the
    six seeds come from."""
    name = "randomvar"

    def __init__(self, seed: int = 0, forecast_budget: int = 36,
                 withheld: Sequence[str] = (), k: int = 3, prior=None,
                 n_model: Optional[int] = None):
        SearchProposer.__init__(self, seed, forecast_budget, withheld, k,
                                prior)
        self._transport = None
        self.n_model = int(n_model or os.environ.get("DSS_LLM_SCREEN_N", 6))
        self.fill = True
        self._timeout = None
        self.model_candidates = 0
        self.model_kept = 0
        self._last = (0, 0)

    def _ask(self, ctx) -> List[dict]:
        g = Grammar(ctx, self.rng, self.withheld)
        out = []
        for _ in range(self.n_model):
            c = g.sample()
            c["rationale"] = "search: random seed"
            out.append(c)
        return out


EXPERT_TACTICS = [
    # the six hand-written tactics of the article (Table IV), in the
    # proposal schema; the doctrine-seeded control (revision K12) uses them
    # as the six seeds of the variation search in place of model candidates
    {"antecedents": [["fire_threat_level", "H"]],
     "consequents": [["head_retardant_line", 1.0]],
     "new_intervention": {"name": "head_retardant_line", "clauses": [
         {"effect": "coat", "sector": "head", "range": [2, 6], "amount": 1.0}]}},
    {"antecedents": [["asset_exposure_risk", "H"]],
     "consequents": [["structure_protection", 1.0]],
     "new_intervention": {"name": "structure_protection", "clauses": [
         {"effect": "wet", "sector": "populated", "range": [0, 2], "amount": 1.0},
         {"effect": "coat", "sector": "assets", "range": [0, 3], "amount": 0.8}]}},
    {"antecedents": [["suppression_feasibility", "H"], ["fire_threat_level", "M"]],
     "consequents": [["flank_line", 0.9]],
     "new_intervention": {"name": "flank_line", "clauses": [
         {"effect": "clear", "sector": "flank", "range": [0, 2], "amount": 1.0}]}},
    {"antecedents": [["suppression_feasibility", "M"]],
     "consequents": [["rear_anchor", 0.8]],
     "new_intervention": {"name": "rear_anchor", "clauses": [
         {"effect": "wet", "sector": "rear", "range": [0, 2], "amount": 1.0}]}},
    {"antecedents": [["evacuation_pressure", "H"]],
     "consequents": [["evacuation_belt", 1.0]],
     "new_intervention": {"name": "evacuation_belt", "clauses": [
         {"effect": "prime", "sector": "populated", "range": [0, 4], "amount": 1.0},
         {"effect": "evacuate", "sector": "populated", "range": [0, 2], "amount": 1.0}]}},
    {"antecedents": [["suppression_feasibility", "VH"], ["fire_threat_level", "H"]],
     "consequents": [["burnout_behind_line", 0.8]],
     "new_intervention": {"name": "burnout_behind_line", "clauses": [
         {"effect": "clear", "sector": "head", "range": [5, 6], "amount": 1.0},
         {"effect": "ignite", "sector": "head", "range": [3, 4], "amount": 0.6}]}},
]


class ExpertVarProposer(RandomVarProposer):
    """Doctrine-seeded search (revision K12): the six hand-written tactics
    of Table IV take the place of the six model candidates, and the rest of
    the forecast budget goes to the same local variations. Pairs with the
    full GODA configuration and with RandomVarProposer; the three differ
    only in where the six seeds come from."""
    name = "expertvar"

    def _ask(self, ctx) -> List[dict]:
        """The six tactics as written; a tactic whose antecedent does not
        hold on the present reading is replaced by the first of up to 20
        local variations that does, so that every call starts from six
        seeds, as the model-seeded search does."""
        gram = Grammar(ctx, self.rng, self.withheld)
        out = []
        for c in EXPERT_TACTICS:
            c = copy.deepcopy(c)
            c["rationale"] = "search: hand-written seed"
            if self._passes(copy.deepcopy(c), ctx):
                out.append(c)
                continue
            for _ in range(20):
                try:
                    q = gram.mutate(c)
                except Exception:
                    continue
                if self._passes(copy.deepcopy(q), ctx):
                    q["rationale"] = "search: hand-written seed, varied"
                    out.append(q)
                    break
        return out


class ExpertTuneProposer(SearchProposer):
    """Hand-written tactics tuned on the forecast (revision K30). The six
    tactics of Table IV are kept as written in antecedent and effect, and
    the first time a tactic's antecedent holds in a world its band and
    amount are chosen among six variants (amount 0.6, 0.8, 1.0; band as
    written or shifted one ring outward) by the same one-future forecast
    that prices every search candidate, six forecasts per tactic and 36
    per world. The tuned tactic is then installed as doctrine without the
    gates, as the untuned doctrine of K11 is. No candidate is handed to
    the gate chain. Pairs with K11 (untuned) and K12 (seeds of a search)."""
    name = "experttune"
    AMOUNTS = (0.6, 0.8, 1.0)
    SHIFTS = (0, 1)

    def __init__(self, seed: int = 0, forecast_budget: int = 36,
                 withheld: Sequence[str] = (), k: int = 3, prior=None):
        super().__init__(seed, forecast_budget, withheld, k, prior)
        self.tuned: Dict[int, dict] = {}
        self.per_tactic = max(1, self.forecast_budget // len(EXPERT_TACTICS))

    @staticmethod
    def _variants(c: dict) -> List[dict]:
        out = []
        for amt in ExpertTuneProposer.AMOUNTS:
            for sh in ExpertTuneProposer.SHIFTS:
                q = copy.deepcopy(c)
                for cl in q["new_intervention"]["clauses"]:
                    a0 = float(cl.get("amount", 1.0))
                    cl["amount"] = round(min(1.0, a0 * amt), 3)
                    r = list(cl.get("range") or [2, 6])
                    cl["range"] = [min(6, r[0] + sh), min(6, r[1] + sh)]
                q["_tune"] = dict(amount=amt, shift=sh)
                out.append(q)
        return out

    @staticmethod
    def _record(c: dict, t_min: float) -> dict:
        ants = " AND ".join(f"{v.replace('_', ' ')} is {t}"
                            for v, t in c["antecedents"])
        cons = ", ".join(f"{i.replace('_', ' ')} {float(x):.2f}"
                         for i, x in c["consequents"])
        return dict(rule=f"IF {ants} THEN {cons}", kind="actuator",
                    clauses=[dict(cl) for cl in
                             c["new_intervention"]["clauses"]],
                    composition=None, concept=None, t_min=float(t_min),
                    source="expert-tuned", accepted=True)

    def propose(self, ctx, situation, timeout, engine, mission):
        if ctx is None:
            return None
        from experiments.open_space.campaign import _install_scripted
        eng = ctx.engine
        try:
            t_now = float(ctx.sim.state.step) * float(
                getattr(ctx.sim.cfg, "step_minutes", 1.0))
        except Exception:
            t_now = 0.0
        installed = []
        for i, c in enumerate(EXPERT_TACTICS):
            if i in self.tuned:
                continue
            probe = copy.deepcopy(c)
            try:
                if _a._fires_now(probe, ctx.eff) is not None:
                    continue
            except Exception:
                continue
            best, best_g = None, None
            spent0 = self.forecasts
            for q in self._variants(c):
                if self.forecasts - spent0 >= self.per_tactic:
                    break
                if not self._passes(copy.deepcopy(q), ctx):
                    self.last_filtered += 1
                    continue
                self.candidates_tried += 1
                g = self._score(q, ctx)
                if best_g is None or g > best_g + 1e-9:
                    best, best_g = q, g
            if best is None:
                best, best_g = copy.deepcopy(c), float("nan")
                best["_tune"] = dict(amount=1.0, shift=0, note="nominal")
            rec = self._record(best, t_now)
            rec["tune"] = dict(best.get("_tune") or {}, gain=best_g,
                               forecasts=self.forecasts - spent0)
            _install_scripted(eng, rec)
            self.tuned[i] = rec
            installed.append(rec)
        return {"_skip": f"expert tactics tuned and installed: "
                         f"{len(installed)} now, {len(self.tuned)} total",
                "_search": dict(priced=len(installed),
                                forecasts=self.forecasts,
                                filtered=self.last_filtered,
                                tuned=[dict(name=r["clauses"] and
                                            r["rule"].split(" THEN ")[1],
                                            **{k: (None if isinstance(v, float)
                                                   and v != v else v)
                                               for k, v in r["tune"].items()})
                                       for r in installed])}


class RandomBlindProposer(Proposer):
    """The random counterpart of the single-shot model proposer: one uniform
    sample of the grammar that clears the cheap gates, handed to the gate
    chain without any shadow forecast. Pairs with the LLM proposer at a
    proposer forecast budget of zero."""
    name = "randomblind"

    def __init__(self, seed: int = 0, withheld: Sequence[str] = ()):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.withheld = tuple(withheld)

    def propose(self, ctx, situation, timeout, engine, mission):
        if ctx is None:
            return None
        g = Grammar(ctx, self.rng, self.withheld)
        for _ in range(400):
            p = g.sample()
            self.candidates_tried += 1
            try:
                ok = cheap_gates(p, ctx) is None
            except Exception:
                ok = False
            if ok:
                return p
        return None


class EvoProposer(SearchProposer):
    """Evolutionary search over the grammar. Population of `pop`, `gens`
    generations, tournament selection, mutation and clause crossover.
    The cheap gates filter every individual BEFORE a forecast is spent,
    so the budget goes to admissible candidates only. The initial
    population is seeded from the composition prior when one is given."""
    name = "evo"

    def __init__(self, seed: int = 0, forecast_budget: int = 36,
                 withheld: Sequence[str] = (), k: int = 3, prior=None,
                 pop: int = 12, gens: int = 3):
        super().__init__(seed, forecast_budget, withheld, k, prior)
        self.pop, self.gens = int(pop), int(gens)

    def _initial(self, g: Grammar, ctx) -> List[dict]:
        out = []
        tries = 0
        while len(out) < self.pop and tries < 40 * self.pop:
            tries += 1
            p = None
            if self.prior is not None and self.rng.random() < 0.5:
                try:
                    p = self.prior.suggest(g, ctx)
                except Exception:
                    p = None
            if p is None:
                p = g.sample()
            if self._passes(p, ctx):
                out.append(p)
            else:
                self.last_filtered += 1
        return out

    def search(self, ctx):
        g = Grammar(ctx, self.rng, self.withheld)
        spent0 = self.forecasts
        scored: Dict[str, Tuple[float, dict]] = {}

        def key(p):
            return repr((p.get("antecedents"), p.get("consequents"),
                         p.get("new_intervention"), p.get("new_concept")))

        def budget_left():
            return self.forecasts - spent0 < self.forecast_budget

        popn = self._initial(g, ctx)
        for p in popn:
            if not budget_left():
                break
            k_ = key(p)
            if k_ not in scored:
                self.candidates_tried += 1
                scored[k_] = (self._score(p, ctx), p)
        for _gen in range(self.gens):
            if not budget_left() or len(scored) < 2:
                break
            ranked = sorted(scored.values(), key=lambda t: -t[0])
            elite = [p for _s, p in ranked[: max(2, len(ranked) // 3)]]
            children = []
            while len(children) < self.pop and budget_left():
                a = elite[int(self.rng.integers(len(elite)))]
                if self.rng.random() < 0.4 and len(elite) > 1:
                    b = elite[int(self.rng.integers(len(elite)))]
                    c = g.crossover(a, b)
                else:
                    c = g.mutate(a)
                if not self._passes(c, ctx):
                    self.last_filtered += 1
                    if len(children) == 0 and self.rng.random() < 0.3:
                        c = g.sample()
                        if not self._passes(c, ctx):
                            continue
                    else:
                        continue
                k_ = key(c)
                if k_ in scored:
                    continue
                children.append(c)
                self.candidates_tried += 1
                scored[k_] = (self._score(c, ctx), c)
        ranked = sorted(scored.values(), key=lambda t: -t[0])
        return ranked


class TemplateProposer(Proposer):
    """The deterministic stand-in of the thesis ladder campaign, behind
    the common interface (it cannot write a clause actuator or a
    concept; kept as the historical reference arm)."""
    name = "template"

    def __init__(self, seed: int = 0):
        super().__init__()
        from experiments.offline_proposer import make_template_proposer
        self._fn = make_template_proposer(seed)

    def propose(self, ctx, situation, timeout, engine, mission):
        return self._fn(situation, timeout=timeout, engine=engine,
                        mission=mission)


# ------------------------------------------------- composition prior
class CompositionPrior:
    """What the simulator has taught the search so far.

    A table over context buckets x (effect, sector) pairs holding the
    number of trials, the number of positive forecast gains and the mean
    gain. It is updated from every priced candidate (search proposers)
    and from every gate verdict the stage files (all proposers), and it
    is used to seed the evolutionary population. It is the measurable
    form of 'the decision layer learns from the simulator which
    compositions of its physical primitives help'. Serializable to a
    dict so a campaign can persist and inspect it.
    """

    def __init__(self):
        self.table: Dict[Tuple[str, str, str], List[float]] = {}

    @staticmethod
    def bucket(ctx: TrialContext) -> str:
        eng = ctx.engine
        sat = bool(getattr(eng, "_pool_saturated", False))
        try:
            water = bool(np.asarray(
                ctx.sim.world.fuel.ftype == 5).any())
        except Exception:
            water = False
        try:
            fire = np.asarray(ctx.sim.state.burning) > 0.5
            from .actions import _asset_cores
            near = bool(_asset_cores(ctx.sim.world, fire).any())
        except Exception:
            near = False
        return f"{'sat' if sat else 'ok'}|{'w' if water else 'nw'}|" \
               f"{'a' if near else 'na'}"

    @staticmethod
    def pairs(prop: dict) -> List[Tuple[str, str]]:
        ni = prop.get("new_intervention") or {}
        if ni.get("clauses"):
            return [(str(c.get("effect")), str(c.get("sector", "ring")))
                    for c in ni["clauses"][:3]]
        if ni.get("composition"):
            return [("macro", str(a)) for a, _b in ni["composition"]]
        return [("rule", str(i)) for i, _x in
                (prop.get("consequents") or [])]

    def observe_trial(self, prop: dict, ctx: TrialContext, gain: float):
        b = self.bucket(ctx)
        for e, s in self.pairs(prop):
            row = self.table.setdefault((b, e, s), [0.0, 0.0, 0.0])
            row[0] += 1
            row[1] += 1.0 if gain > 0 else 0.0
            row[2] += (gain - row[2]) / row[0]

    def observe_verdict(self, prop: dict, ctx: TrialContext,
                        accepted: bool, dJ: float):
        self.observe_trial(prop, ctx, -float(dJ) if accepted else -1e-6)

    def suggest(self, g: Grammar, ctx: TrialContext) -> Optional[dict]:
        """A clause actuator built from the pairs with the best record in
        this bucket; None when the table is empty for the bucket."""
        b = self.bucket(ctx)
        rows = [(k[1], k[2], v) for k, v in self.table.items()
                if k[0] == b and k[1] in EFFECTS]
        if not rows:
            return None
        rows.sort(key=lambda t: -(t[2][1] / max(t[2][0], 1)
                                  + 0.1 * t[2][2]))
        top = rows[: max(1, min(3, len(rows)))]
        n = int(g.rng.integers(1, len(top) + 1))
        clauses = []
        for e, s, _v in top[:n]:
            c = g.clause()
            c["effect"], c["sector"] = e, s
            clauses.append(c)
        act = {"name": g._name("sited"), "clauses": clauses}
        return {"antecedents": g.antecedents(),
                "consequents": [[act["name"], g.intensity()]],
                "new_intervention": act,
                "rationale": "search: prior-seeded sited intervention"}

    def to_dict(self) -> dict:
        return {"|".join(k): list(v) for k, v in self.table.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "CompositionPrior":
        p = cls()
        for k, v in (d or {}).items():
            parts = tuple(k.split("|"))
            if len(parts) == 5:
                key = ("|".join(parts[:3]), parts[3], parts[4])
                p.table[key] = list(v)
        return p


# ------------------------------------------------------------- install
def install(engine, proposer: Proposer) -> Callable:
    """Put `proposer` behind the stage-3 hook; returns the previous hook
    so a campaign can restore it."""
    prev = _a._genai_propose
    _a._genai_propose = proposer
    try:
        engine._proposer = proposer
    except Exception:
        pass
    return prev


def restore(prev: Callable) -> None:
    _a._genai_propose = prev
