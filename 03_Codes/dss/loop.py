"""The Layer 4 decision engine: rules -> evaluate -> adapt -> apply.

One DecisionEngine instance owns the runtime rule base, the per-region
gating state, the value-based stage controller and the decision log. Once per
DECISION CYCLE it produces the composed resource override that the
simulation then holds until the next cycle:

  1. per region: features (+confidences) -> gated concepts -> rules fire
     -> intervention intensities -> quality gate / graduated fail-safe
  2. global coordination: the operational-priority concept sets each
     region's attention share; monitored regions get their offensive
     tempo attenuated (the coordinator, not the local agent, owns this)
  3. the composed candidate is forecast on a shadow copy; if the
     satisficing test J <= J_TH passes it is applied as-is; otherwise
     the stage controller (associative search) picks ONE adaptation
     stage (evFIS / resolution / generative) for this cycle, keeps it
     only if the forecast improves, and the controller is rewarded with
     the realized cost reduction.

Everything is written to the DecisionLog for the backward trace and the
counterfactual replay of the analysis view.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .features import ten_features, feature_confidence
from .concepts import (infer_concepts, concept_gates, crisp,
                       GatedConcepts)
from .rules import evaluate_rules
from .actions import decision_to_resources
from .evaluate import (candidate_vs_noaction, quality_Q,
                       graduated_failsafe, forecast_cost,
                       physical_cost)
from .adapt import (make_runtime_rules, StageController, stage1_evfis,
                    stage2_resolution, stage3_generative, AdaptOutcome)
from .decision_log import DecisionLog, DecisionRecord


class DecisionEngine:
    """The four-layer decision loop of Chapter 4, one instance per run.

    The engine owns everything that happens between a simulation step
    and the orders that step receives: it reads the observation, gates
    the concepts, fires the rule base, scores the candidate on cost and
    on quality, lets the coordinator ration the shared pool, and, when
    the standing decision demonstrably falls short, runs one adaptation
    stage under the gates of Section 4.5.3.

    It is deliberately a single object rather than a pipeline of
    functions. A cycle needs the previous cycle to interpret it: the
    persistence prior, the stage controller's value table and the
    learned store are all state that outlives one decision, and passing
    them between free functions would mean passing the whole engine.
    """
    def __init__(self, regions, base_pool=None, network=None,
                 j_threshold: float = 0.35, eta: float = 0.60,
                 cycle_steps: int = 3, horizon_steps: int = 12,
                 evfis_step: float = 0.05, adapt_on: bool = True,
                 evfis_on: bool = True,
                 genai_on: bool = False, ctrl_eps: float = 0.10,
                 ctrl_lr: float = 0.05, attention_thr: float = 0.35,
                 # THE ABSOLUTE FLOOR IS OFF BY DEFAULT. It exists so
                 # that "a region worth this much is attended whatever
                 # the leader is doing" can be said if it is wanted;
                 # what it may not do is masquerade as the relative
                 # test, which is what the old OR branch did.
                 attention_thr_abs: float | None = None,
                 # the share a monitored region keeps at zero priority
                 s_min: float = 0.50,
                 # a hard cap on the attended count, for sweeping the
                 # count independently of the threshold that makes it
                 k_max: int | None = None,
                 min_gain: float = 0.05, run_logger=None,
                 spread_tighten: float = 1.0,
                 void_tighten: float = 1.0,
                 rel_physical: bool = False,
                 cycle_min: float | None = None,
                 horizon_min: float | None = None,
                 seed_profile: str = "minimal",
                 learned_store: str | None = None,
                 # ONE revision by default: the mission brief promises
                 # the model exactly one corrected retry per rejection
                 # ("comes back to you once"), and the sensitivity
                 # study found the budget flat, so the code now keeps
                 # the promise the prompt makes
                 revision_budget: int = 1,
                 use_evfis: bool = True, use_genai: bool = True,
                 state_path: str | None = None):
        self.regions = list(regions)
        self.base_pool = base_pool
        self.network = network
        self.j_threshold = float(j_threshold)
        # eta is a QUALITY gate on [0, 1]; anything else is a
        # configuration accident and would crush every order through
        # the graduated fail-safe (Q/eta), so it is clamped here
        self.eta = float(min(1.0, max(0.0, eta)))
        # how many times a rejected Stage-3 proposal may return to the
        # generator with the gate verdict before the stage gives up
        self.revision_budget = max(0, int(revision_budget))
        self.cycle_steps = max(1, int(cycle_steps))
        self.horizon_steps = max(2, int(horizon_steps))
        # minute-based scheduling wins over step counts when given: a
        # 1-minute live tick must not shrink the decision cycle to one
        # minute and the lookahead to a few minutes
        self.cycle_min = None if cycle_min is None else float(cycle_min)
        self.horizon_min = (None if horizon_min is None
                            else float(horizon_min))
        self.evfis_step = float(evfis_step)
        self.adapt_on = bool(adapt_on)
        # LIVE modules: whether the adaptation stages RUN during the sim
        self.genai_on = bool(genai_on)
        self.evfis_on = bool(evfis_on)
        # LOAD flags: whether the PRE-LEARNED rules of each stage are brought
        # in from the store (independent of whether the live module runs)
        self.use_evfis = bool(use_evfis)
        self.use_genai = bool(use_genai)
        # set once GenAI proves unproductive this run (unreachable, or its
        # proposals keep getting rejected): stage 3 is then retired so the
        # controller stops spending the scarce adaptation budget on it and
        # evFIS (stages 1/2) reclaims those cycles
        self._genai_dead = False
        self._genai_fails = 0
        # DECISION MODE: which adaptation stages the controller may pick.
        # evFIS covers stage 1 (tuning) and stage 2 (resolution);
        # GenAI is stage 3. Both off = pure fuzzy seed base.
        self.stages_allowed = tuple(
            ([1, 2] if self.evfis_on else [])
            + ([3] if self.genai_on else []))
        if not self.stages_allowed:
            self.adapt_on = False
        self.attention_thr = float(attention_thr)
        self.attention_thr_abs = (None if attention_thr_abs is None
                                  else float(attention_thr_abs))
        self.s_min = float(s_min)
        self.k_max = None if k_max is None else int(k_max)
        # what the last cycle attended, so the coordinator can say how
        # many of its regions it is actually funding
        self.last_k = 0
        self.last_n_fire = 0
        # ADAPTIVE satisficing (thesis: the bound tightens): a candidate
        # must either clear the absolute threshold OR beat the no-action
        # forecast by at least min_gain relative margin; otherwise the
        # adaptation stages engage even when the absolute cost is small
        self.min_gain = float(min_gain)
        # SYMPTOMS TIGHTEN THE ASPIRATION, THEY DO NOT BYPASS IT.
        # A spreading fire or a silent rule base raises the standard the
        # standing decision has to meet, instead of opening the gate on
        # their own. At 1.0 the bound collapses to zero whenever the
        # symptom is present, which reproduces the earlier OR gate
        # exactly; below 1.0 the satisficing bound stays in the causal
        # path and J_TH can be measured.
        self.spread_tighten = float(spread_tighten)
        self.void_tighten = float(void_tighten)
        # WHICH COST THE RELATIVE MARGIN READS. The total cost prices the
        # decision itself through the response and delay terms, which a run
        # that never intervenes does not pay, so a relative test written on
        # the total penalises acting by construction: the same argument the
        # thesis already makes for cross-run comparison. With this flag the
        # ceiling keeps the total cost (it asks whether the incident is
        # still expensive) while the margin reads the physical cost (it
        # asks whether acting buys a better fire).
        self.rel_physical = bool(rel_physical)
        self.run_logger = run_logger
        # TWO PROFILES: the five-rule seed base (default) and the forty of
        # the written doctrine. The 22-rule "core" block is retired, and
        # any other name - a stale flag in a store, an old script - is read
        # as "minimal" rather than silently starting a run on a base it
        # did not ask for.
        _p = str(seed_profile or "").lower()
        self.seed_profile = "full" if _p.startswith("full") else "minimal"
        self.rules = make_runtime_rules(self.seed_profile)
        # the persistent learned-rule store survives fires, engines
        # and MAPS: knowledge accumulated anywhere is reloaded here
        # ---- ENGINE-LOCAL VOCABULARY (open decision space) ----
        # Admitted concept/intervention packages grow these; the
        # module-level base stays pristine. New intermediate concepts
        # enrich the aggregation hierarchy; new decision concepts add
        # antecedent variables (catalog x5) and declare an answering
        # family; macro interventions expand to base channels.
        import copy as _copy
        from .concepts import HIERARCHY as _BASE_H
        from .concepts import DECISION_CONCEPTS as _BASE_DC
        from .evaluate import CONCEPT_FAMILY as _BASE_CF
        self.hierarchy = _copy.deepcopy(dict(_BASE_H))
        self.decision_concepts = list(_BASE_DC)
        self.concept_family = _copy.deepcopy(dict(_BASE_CF))
        self.macros: dict = {}
        self.learned_store = learned_store
        # a NEW engine starts from the pristine thesis membership registry;
        # reset FIRST, then the store's stage-2 term inserts are re-applied
        # by merge_learned ONLY when evFIS is loaded (order matters: a reset
        # AFTER the merge used to wipe the just-loaded inserts)
        from .adapt import reset_partitions
        reset_partitions()
        if learned_store:
            from .persist import merge_learned
            # the toggles gate WHICH pre-learned rules are USED: evFIS off ->
            # stage 1/2 (tuned seeds, A# rules, term inserts) are NOT loaded;
            # GenAI off -> stage 3 (G# rules, generated concepts, macros) are
            # NOT loaded. Both off = pure seed base.
            merge_learned(self.rules, learned_store,
                          profile=self.seed_profile, engine=self,
                          use_evfis=self.use_evfis,
                          use_genai=self.use_genai)
        # what came OUT OF THE STORE, so a view can tell restored vocabulary
        # apart from vocabulary this engine actually invented. The rules
        # already carry that provenance in their note; concepts and macros
        # had no equivalent, so every restored concept looked freshly made.
        from .concepts import HIERARCHY as _BASE_H
        self.loaded_concepts = {c for c in self.hierarchy
                                if c not in _BASE_H}
        self.loaded_macros = set(self.macros)
        # ---- GENERATED-STATE STORE (production / consumption split) ----
        # The adaptation stages no longer edit the live rule objects and keep
        # them: they append a record here, and the set the inference reasons
        # with is DERIVED from this store on every cycle. That is what makes
        # a consumption flag able to revert anything at all.
        from .state import GeneratedState
        self.state_path = state_path
        self.gstate = GeneratedState.load(
            state_path or "logs/dss_generated_state.json",
            active_rule_set=self.seed_profile)
        self.gstate.set_flags(active_rule_set=self.seed_profile,
                              evfis_active=self.evfis_on,
                              genai_active=self.genai_on,
                              use_stage12_rules=self.use_evfis,
                              use_stage3_rules=self.use_genai)
        self.resolve_warnings: list = []
        self.persist_errors: list = []
        # CARRY THE CONTROLLER'S EXPERIENCE ACROSS RUNS. One fire offers only
        # a few dozen adaptation attempts, nowhere near enough for an
        # epsilon-greedy value table to converge, so a fresh table every run
        # meant the stage choice never got past exploration. The table is
        # keyed by map: it is restored on the same terrain and reset on
        # different terrain, because the worth of a stage is a property of
        # the scene, not of the algorithm.
        self.map_key = None
        self.controller_restored = False
        self._evfis_rules = None
        self.run_stats = self._fresh_run_stats()
        if state_path:
            self._sync_active_set()
        self.mission_brief: str | None = None
        self._avail_checked = False
        self._shelved_macros: dict = {}
        self.controller = StageController(eps=ctrl_eps, lr=ctrl_lr)
        # PERFORMANCE THROTTLES (live-run economics, not physics):
        # adaptation trials and the 45-min no-harm re-forecast are the
        # two expensive items of a cycle (each shadow run simulates
        # 45 min of physics). With a 1-min decision cycle they must
        # not run EVERY cycle: trials respect a cooldown, the no-harm
        # verdict is reused while the composed orders are unchanged.
        # 5 min between trials: with the 1-min cycle the old 10 min
        # allowed only ~5 attempts in a 50-min run, which starved the
        # generative stage (epsilon-greedy rarely reached it at all)
        self.adapt_cooldown_min = 5.0
        # A REJECTION DECIDED FOR FREE DOES NOT BUY THE FULL COOLDOWN.
        # The cooldown exists to ration expensive work: a shadow forecast is
        # 45 minutes of physics, a model call blocks on the network. But some
        # attempts are turned down without doing either, by a set lookup or a
        # form check (stage 2 finding the antecedent cell already covered is
        # the common one, and it repeats identically as long as the situation
        # holds). Charging those the same five minutes meant a third of all
        # adaptation windows were spent on decisions that cost nothing, and in
        # long runs stage 3 never got a turn at all. Such an attempt is
        # refunded down to this much instead, so the next cycle can offer the
        # window to a different stage.
        self.adapt_retry_min = 1.0
        self.noharm_recheck_min = 10.0
        self._adapt_last_min = None
        self._nh_last = None      # (t_min, order_sig, phys_c, phys_0)
        self._prev_ever = None    # spread trigger: ever_burned memory
        self.last_global = None   # the Global DSS's explicit decision
        self.log = DecisionLog()
        self.cycles: list = []        # full per-cycle chronicle
        self.gaters: Dict[str, GatedConcepts] = {
            r.name: GatedConcepts() for r in self.regions}
        self.last_override = None
        self.last_actions = None
        self.last_cycle_step = -10 ** 9

    # ------------------------------------------------------------ cycle
    def _perceive(self, sim):
        """Fixed per-cycle perception: features, gates and priors."""
        ctx = {}
        for r in self.regions:
            f = ten_features(sim, r, network=self.network,
                             pool=self.base_pool)
            fc = feature_confidence(self.network, r)
            gates = concept_gates(fc, hierarchy=self.hierarchy)
            ctx[r.name] = dict(region=r, f=f, fc=fc, gates=gates)
        return ctx

    def _effective(self, ctx_r, commit_step=None):
        """Gated activations; commit the persistence prior only when a
        real cycle step is given (trials pass None and stay pure)."""
        g = self.gaters[ctx_r["region"].name]
        act = infer_concepts(ctx_r["f"], hierarchy=self.hierarchy)
        if commit_step is not None:
            return g.gate(act, ctx_r["gates"], step=commit_step)
        prev = g.prev
        eff = {}
        for name, a in act.items():
            gv = float(np.clip(ctx_r["gates"].get(name, 1.0), 0.0, 1.0))
            p = prev.get(name, a)
            eff[name] = np.clip(gv * a + (1.0 - gv) * g.rho * p, 0.0, 1.0)
        return eff

    def _decide_regions(self, sim, ctx, rules, commit_step=None):
        """Intensities per region under a given rule base (+ Q gate and
        coordination); returns per-region dicts and the composed pairs."""
        rows = {}
        prios = {}
        for name, c in ctx.items():
            eff = self._effective(c, commit_step)
            u, trace = evaluate_rules(eff, c["f"], rules,
                                      macros=self.macros)
            cr = crisp(eff)
            q = quality_Q(cr, u, family=self.concept_family)
            rows[name] = dict(eff=eff, crisp=cr, u=dict(u),
                              u_rules=dict(u), q=q,
                              trace=[(r.name, w) for r, w in trace
                                     if w > 0.01])
            prios[name] = cr.get("operational_priority", 0.0)
        pmax = max(prios.values()) if prios else 1.0
        # THE ATTENDED SET IS A RELATIVE TEST, and only a relative one.
        # It used to read "at or above a fraction of the leader OR at or
        # above the threshold outright", but with priorities in [0,1]
        # and a threshold in [0,1] the second condition can never bind:
        # tau * pmax <= tau always, so the looser arm fires first and
        # the absolute arm is unreachable. It is gone. An absolute floor
        # that genuinely bites is its own parameter, off by default, so
        # the two ideas cannot be confused again.
        cut = self.attention_thr * max(pmax, 1e-9)
        _order = sorted(prios, key=lambda n: -prios[n])
        _att = [n for n in _order if prios[n] >= cut]
        if self.attention_thr_abs is not None:
            _att = [n for n in _att
                    if prios[n] >= float(self.attention_thr_abs)]
        # A CAP ON THE COUNT, so the count can be swept on its own. The
        # threshold and the count move together by construction, and a
        # study that varies only the threshold cannot say whether the
        # effect comes from how many regions are served or from how
        # hard the rest are derated.
        if self.k_max is not None:
            _att = _att[:max(1, int(self.k_max))]
        att_set = set(_att)
        self.last_k = len(att_set)
        self.last_n_fire = sum(1 for n in prios if prios[n] > 1e-6)
        for name, c in ctx.items():
            att = name in att_set
            # A MONITORED REGION IS DERATED, NOT SILENCED. The floor is
            # a parameter rather than a constant because it decides
            # whether the allocation can reach a corner at all: with a
            # floor of one half a region can never be starved below
            # half strength, however little it is worth.
            share = 1.0 if att else self.s_min + (
                1.0 - self.s_min) * (prios[name] / max(pmax, 1e-9))
            # COORDINATED ACCEPTANCE THRESHOLD: the coordinator sends
            # back a per-region quality gate along with the share. An
            # attended region keeps the base gate eta; a monitored
            # region gets a tightened gate (raised toward 1 as its
            # share falls), so an offensive order in a low-priority
            # region must justify itself with a higher decision
            # quality before it draws on the shared capacity. The
            # life-safety orders stay outside the gate as always.
            eta_r = 1.0 - share * (1.0 - self.eta)
            u2, fs = graduated_failsafe(rows[name]["u_rules"],
                                        rows[name]["q"], eta_r)
            rows[name]["u"] = u2
            rows[name]["fs"] = fs
            rows[name]["eta"] = float(eta_r)
            for k in ("suppression_effort", "resource_deployment"):
                rows[name]["u"][k] = rows[name]["u"][k] * share
            rows[name]["share"] = float(share)
            rows[name]["attended"] = bool(att)
        pairs = []
        for name, c in ctx.items():
            _u_p = dict(rows[name]["u"])
            _u_p["_share"] = float(rows[name]["share"])
            pairs.append((c["region"], _u_p))
        # ---- GLOBAL DSS: an explicit, loggable decision ----
        # It reads every region's operational priority, ranks them,
        # assigns the shares (which scale the offensive tempo above
        # AND steer the budget concentration in the allocator), and
        # states it in one line.
        # GLOBAL LOGISTICS DIRECTIVE: the coordinator allocates the
        # shared AERIAL capacity, and the water supply is part of that
        # allocation. When the focus region fights a real fire, the
        # map holds water and the region's own rules under-order the
        # drafting, the coordinator floors it: send the helicopters
        # to the water, wherever the water sits. Allocation, not
        # inference: no feature, no rule, no concept is evaluated.
        _directives = []
        try:
            import numpy as _np_w
            _has_w = bool(_np_w.asarray(
                sim.world.fuel.ftype == 5).any())
        except Exception:
            _has_w = False
        if _has_w and prios:
            _hot0 = max(prios, key=lambda n: prios[n])
            _rd0 = rows.get(_hot0)
            if (_rd0 is not None
                    and float(_rd0["crisp"].get("fire_threat_level",
                                                0.0)) >= 0.35
                    and float(_rd0["u"].get("water_drafting",
                                            0.0)) < 0.6):
                _rd0["u"]["water_drafting"] = 0.6
                _directives.append(
                    f"{_hot0} is DIRECTED to draft water by air "
                    "(helicopter shuttle to the nearest water body)")
        _rank = sorted(prios, key=lambda n: -prios[n])
        self.last_global = dict(
            ranking=[(n, round(float(prios[n]), 3)) for n in _rank],
            shares={n: round(float(rows[n]["share"]), 3)
                    for n in rows},
            attended=[n for n in rows if rows[n]["attended"]],
            thresholds={n: round(float(rows[n]["eta"]), 3)
                        for n in rows},
            directives=list(_directives),
            hotspot=(_rank[0] if _rank else None),
            # THE COUNT IS THE OPERATIONAL QUANTITY. The threshold is
            # only the coordinate it is expressed in; what the
            # coordinator does is fund k of its N regions, and until
            # this was recorded neither the operator nor the write-up
            # could state it.
            k=int(self.last_k),
            n_regions=len(rows),
            n_fire=int(self.last_n_fire),
            statement=("Global DSS: "
                       + f"attending {self.last_k} of {len(rows)} "
                         "regions; "
                       + (f"focus on {_rank[0]} "
                          f"(priority {prios[_rank[0]]:.2f}); "
                          if _rank else "")
                       + ", ".join(
                           f"{n} share {rows[n]['share']:.2f}"
                           + ("" if rows[n]["attended"]
                              else " (monitor)")
                           for n in _rank)
                       + ("; " + "; ".join(_directives)
                          if _directives else "")))
        return rows, pairs

    def _observed_burning(self, sim):
        """The fire the DSS may ACT on: only what the sensor network actually
        observes. With no coverage over the fire the fused observation is
        empty there, so the DSS cannot dispatch suppression to a fire it has
        not detected (matching the partial-observation design). Without a
        network at all, the true state is used (full observability)."""
        net = self.network
        _obs = getattr(net, "obs", None) if net is not None else None
        if _obs is not None and "burning" in _obs:
            return np.asarray(_obs["burning"]) > 0.5
        return sim.state.burning > 0.5

    def _note_water_ferry(self, sim, rows) -> None:
        """Make the water logistics VISIBLE in the Global statement:
        when a region orders drafting and the nearest water body sits
        in ANOTHER region, the coordinator says so, because on the
        map the shuttle crosses the border and an operator reading
        only the shares would never learn where the water came from."""
        if not self.last_global:
            return
        import numpy as _np
        wat = _np.asarray(sim.world.fuel.ftype == 5)
        if not wat.any():
            return
        wy, wx = _np.where(wat)
        fire = sim.state.burning > 0.5
        notes = []
        for reg in self.regions:
            name = reg.name
            rd = rows.get(name)
            if rd is None or float(rd["u"].get("water_drafting",
                                               0.0)) <= 0.3:
                continue
            sy, sx = reg.slices()
            fmask = _np.zeros_like(fire)
            fmask[sy, sx] = fire[sy, sx]
            if not fmask.any():
                continue
            fy, fx = _np.where(fmask)
            d2 = ((wx[None, :] - fx[:, None]) ** 2
                  + (wy[None, :] - fy[:, None]) ** 2)
            k = int(_np.unravel_index(_np.argmin(d2), d2.shape)[1])
            wxx, wyy = int(wx[k]), int(wy[k])
            src = next((r.name for r in self.regions
                        if r.x0 <= wxx < r.x1 and r.y0 <= wyy < r.y1),
                       None)
            dm = float(_np.sqrt(d2.min())) * float(
                getattr(sim.cfg, "cell_size_m", 30.0))
            if src and src != name:
                notes.append(f"water ferried from {src}'s water "
                             f"({dm:.0f} m) to {name}")
        if notes:
            self.last_global["statement"] += " · " + "; ".join(notes)
            self.last_global["water_ferry"] = notes

    def _override(self, sim, pairs, keep_actions=False):
        """Turn per-region intensities into a resource field the
        simulator can apply.

        The engine decides in the language of intensities; the physics
        engine acts on cells. This is the only place the two meet.
        Suppression is aimed at the OBSERVED fire rather than the true
        one, so a fire the sensor network has not reported cannot be
        fought, which is what makes coverage a decision input rather
        than a detail of the display.
        """
        world = sim.world
        _burn = self._observed_burning(sim)
        if keep_actions:
            ov, acts = decision_to_resources(
                world, _burn, pairs, self.base_pool,
                return_actions=True, macros=self.macros)
            self.last_actions = acts
            return ov
        return decision_to_resources(world, _burn, pairs,
                                     self.base_pool,
                                     macros=self.macros)

    # ------------------------------------------- per-run adaptation tally
    @staticmethod
    def _fresh_run_stats() -> dict:
        """What the adaptation did over THIS simulation.

        Kept on the engine rather than recovered from the log, so the view
        works while the run is still going. Answering "did evFIS engage at
        all, and if not why" used to mean reading cycles.jsonl by hand."""
        # seq0 scopes the store to THIS run. Step numbers restart with every
        # fire, so a view that joined records to cycles on the step alone
        # pulled in records from earlier runs that happened to reach the same
        # step. The store sequence is global and monotonic, so anything at or
        # above the value it had when the run started belongs to this run.
        return dict(start_step=None, seq0=None, cycles=0,
                    satisficing_failed=0,
                    tried=0, accepted=0, rejected=0,
                    blocked={}, per_stage={}, gates={},
                    reasons={}, withheld=0, dJ_accepted=0.0,
                    j_series=[], phys_series=[], persist_failed=0,
                    cooldown_refunds=0, stage2_prefiltered=0)

    def _tally_cycle(self, step, j_c, j_0, bound, deficit_on, gap,
                     adapt_due, menu):
        """Record what the cycle decided and why, before any stage runs.

        The tally is written even when no stage engages, because the
        interesting question is usually the negative one: a run in which
        adaptation never fired has to be able to say whether the gate
        never opened, or opened and found nothing to try.
        """
        s = self.run_stats
        if s["start_step"] is None:
            s["start_step"] = int(step)
            try:
                s["seq0"] = int(self.gstate.next_seq())
            except Exception:
                s["seq0"] = None
        s["cycles"] += 1
        s["j_series"].append((int(step), float(j_c), float(j_0),
                              float(bound)))
        if not (deficit_on or gap):
            return
        s["satisficing_failed"] += 1
        # the gate opened; record WHY nothing was tried when nothing was
        if not self.adapt_on:
            k = ("adaptation master switch off (evFIS and GenAI both off)"
                 if not self.stages_allowed
                 else "adaptation master switch off")
            s["blocked"][k] = s["blocked"].get(k, 0) + 1
        elif not adapt_due:
            s["blocked"]["cooldown between adaptation trials"] = \
                s["blocked"].get("cooldown between adaptation trials", 0) + 1
        elif not menu:
            s["blocked"]["no runnable stage (GenAI retired, evFIS off)"] = \
                s["blocked"].get(
                    "no runnable stage (GenAI retired, evFIS off)", 0) + 1

    def _tally_outcome(self, stage, outcome):
        """Record the verdict on one adaptation trial.

        Rejections are counted with the name of the gate that refused
        them, which is what makes the funnel of Table 5.11 possible:
        without the reason, a low acceptance rate cannot be told apart
        from a stage that was never given anything to work with.
        """
        s = self.run_stats
        s["tried"] += 1
        ps = s["per_stage"].setdefault(int(stage),
                                       dict(tried=0, accepted=0, dJ=0.0))
        ps["tried"] += 1
        if outcome.accepted:
            s["accepted"] += 1
            ps["accepted"] += 1
            ps["dJ"] += float(outcome.dJ or 0.0)
            s["dJ_accepted"] += float(outcome.dJ or 0.0)
        else:
            s["rejected"] += 1
            # FULL first segment: the 70-char cut chopped every reason
            # mid-sentence in the run analysis panel
            r = (outcome.detail or "rejected").split(" | ")[0]
            s["reasons"][r] = s["reasons"].get(r, 0) + 1
        # stage 3 keeps its own funnel: which gate turned a proposal away
        info = outcome.info or {}
        if int(stage) == 3:
            # a live model call blocks the decision cycle, and the decision
            # cycle blocks the animation frame, so the wait is recorded: a
            # multi-second cycle should read as "waiting for Claude", not as
            # a frozen application
            try:
                from .adapt import LAST_GENAI_MS as _gms
                s.setdefault("genai_ms", []).append(round(float(_gms), 1))
            except Exception:
                pass
            v = ((info.get("gates") or {}).get("verdict")
                 or info.get("reason") or ("admitted" if outcome.accepted
                                           else "rejected"))
            s["gates"][str(v)[:70]] = s["gates"].get(str(v)[:70], 0) + 1

    # ------------------------------ generated state: derive, then record
    def bind_map(self, map_key: str | None) -> bool:
        """Tell the engine which scene it is on, and restore the controller.

        Called by the dashboard whenever the map identity changes. The value
        table survives fires on the SAME map and is dropped on a different
        one. Returns True when experience was carried over.
        """
        self.map_key = None if map_key is None else str(map_key)
        if not getattr(self, "state_path", None):
            return False
        try:
            self.controller_restored = self.gstate.load_controller(
                self.controller, self.map_key)
        except Exception as exc:
            self.persist_errors.append(
                f"stage controller: {type(exc).__name__}: {exc}")
            self.controller_restored = False
        return self.controller_restored

    # ------------------------------------------ where the learning happens
    ENGAGED_FIRE = 0.05      # a region counts as engaged above this intensity

    def _adapt_region(self, rows, ctx, cov_by_region):
        """Which region's situation the adaptation stage works on.

        NOT the coordinator's hotspot. The coordinator ranks on operational
        priority, which is about where the capacity should go. The adaptation
        stages are about something else entirely: stage 2 instantiates an
        antecedent cell the base does not cover, and stage 3 answers a
        situation the base cannot express. Both are coverage operations.

        Sending them to the highest-priority region sent them, run after run,
        to the region the base already covered BEST: measured over 3812
        cycles, the old selector picked one region 83% of the time and that
        region also had the highest mean fired weight (0.487 against 0.258).
        Stage 2 was then turned away with "cell already covered" in 150 of
        its 162 attempts, which is what a growth stage aimed at a full cell
        does.

        So the adaptation goes to the region the base is QUIETEST about,
        among the regions that actually have fire. The fire filter matters:
        a quiet corner with nothing burning has the lowest coverage of all
        for the trivial reason that nothing is happening there, and learning
        rules for an empty situation is worse than not learning.

        Returns (region_name, why) so the views can state the choice.
        """
        engaged = [n for n in rows
                   if float(ctx[n]["f"].get("fire_intensity", 0.0))
                   > self.ENGAGED_FIRE]
        why = "least covered among the regions with fire"
        if not engaged:
            # nothing is burning anywhere: there is no coverage question to
            # answer, so fall back to the coordinator's own ranking
            engaged = list(rows)
            why = "no region has fire, so the coordinator's hotspot stands"
            return (max(engaged,
                        key=lambda n: rows[n]["crisp"].get(
                            "operational_priority", 0.0)), why)
        pick = min(engaged, key=lambda n: (
            float(cov_by_region.get(n, 0.0)),
            -float(rows[n]["crisp"].get("operational_priority", 0.0))))
        return pick, why

    def _sync_active_set(self) -> None:
        """Rebuild what the inference reasons with, from the baseline plus
        whatever the consumption flags allow. Runs every cycle: the active set
        is derived, never accumulated, so a flag turned off genuinely reverts
        instead of leaving an already-mutated object behind."""
        from .resolve import resolve_active_set
        from .adapt import make_runtime_rules, reset_partitions
        from .fuzzy import REGISTRY
        a = resolve_active_set(self.gstate, make_runtime_rules, REGISTRY,
                               reset_partitions)
        self.rules = a.rules
        self.hierarchy = a.hierarchy
        self.decision_concepts = a.decision_concepts
        self.macros = a.macros
        self.resolve_warnings = a.warnings
        self.active_idle = a.idle
        self.applied_mods = a.applied_mods

    def _apply_availability(self, sim) -> None:
        """MAP-DEPENDENT AVAILABILITY (spec 6.2 spirit): learned
        knowledge survives across maps, but a tactic the CURRENT map
        cannot supply must not be active on it. On a map without
        water, a macro whose composition or clauses need water is
        SHELVED (kept in the store, dropped from the active set) and
        a rule ordering it is deactivated, both with visible
        warnings. Nothing is deleted: the same lineage reactivates
        on the next map that has water."""
        import numpy as _np
        try:
            has_water = bool(_np.asarray(
                sim.world.fuel.ftype == 5).any())
        except Exception:
            return
        if has_water:
            return
        _dep = {"water_drafting", "retardant_drop"}

        def _needs_water(md):
            if any(str(a) in _dep
                   for a, _b in md.get("composition", [])):
                return True
            return any(str(c.get("effect")) in ("draft", "coat")
                       for c in (md.get("clauses") or []))
        for name in list(self.macros):
            if _needs_water(self.macros[name]):
                self._shelved_macros[name] = self.macros.pop(name)
                _w = (f"{name}: needs a water body and this map has "
                      "none; shelved here, kept in the store")
                if _w not in self.resolve_warnings:
                    self.resolve_warnings.append(_w)
        bad = _dep | set(self._shelved_macros)
        for r in self.rules:
            if getattr(r, "active", True) and any(
                    str(iv) in bad for iv, _v in r.consequents):
                r.active = False
                _w = ("rule " + r.name + ": orders "
                      + ", ".join(sorted({str(iv) for iv, _v
                                          in r.consequents
                                          if str(iv) in bad}))
                      + ", unavailable on this waterless map; the "
                      "rule sleeps here and wakes on a map with "
                      "water")
                if _w not in self.resolve_warnings:
                    self.resolve_warnings.append(_w)

    def mission(self, sim) -> str:
        """The standing brief, built ONCE per engine/incident. A
        byte-identical prefix across the run's model calls, so the
        provider-side prompt cache can reuse it; also written to the
        run log as mission_brief.md for the operator."""
        if self.mission_brief is None:
            from .mission import build_mission_brief
            try:
                self.mission_brief = build_mission_brief(
                    sim.world, self.base_pool)
            except Exception:
                self.mission_brief = ""
            lg = getattr(self, "run_logger", None)
            if lg is not None and self.mission_brief:
                try:
                    import os as _os_mb
                    with open(_os_mb.path.join(lg.dir,
                                               "mission_brief.md"),
                              "w", encoding="utf-8") as f:
                        f.write(self.mission_brief)
                except Exception:
                    pass
        return self.mission_brief

    def _evfis_base(self):
        """The base evFIS tunes and measures against: baseline plus the WHOLE
        modification chain, whatever the consumption flag says. Cutting the
        chain would invalidate the stored `before` values and with them both
        the reverse-order revert and the forward replay."""
        from .resolve import evfis_chain_set
        from .adapt import make_runtime_rules, reset_partitions
        from .fuzzy import REGISTRY
        if self.use_evfis:
            # consumption is on, so the active set already IS the chain
            return self.rules
        a = evfis_chain_set(self.gstate, make_runtime_rules, REGISTRY,
                            reset_partitions)
        return a.rules

    @staticmethod
    def _snap_rules(rules):
        """The consequents of a rule base, for before-and-after diffing.

        A tuning step changes numbers inside existing rules, so the only
        way to record what it did is to photograph the base on each side
        of the step and subtract.
        """
        return {r.name: [(str(i), float(v)) for i, v in r.consequents]
                for r in rules}

    @staticmethod
    def _snap_parts():
        """The membership partitions, photographed the same way.

        The resolution stage moves boundaries and inserts terms, which
        changes what every rule on that variable means; the change is
        invisible in the rule text and shows only here.
        """
        from .fuzzy import REGISTRY
        return {v: {t: [float(x) for x in abcd]
                    for t, abcd in REGISTRY.get(v).items()}
                for v in REGISTRY.variables()}

    def _record_changes(self, stage, before_rules, before_parts,
                        after_rules, trigger=None):
        """Turn what a stage actually changed into store records.

        Diffing is deliberate: the stages keep their existing in-place
        implementation and this reads the result, so the acceptance logic and
        the persistence format stay independent of each other."""
        from .concepts import HIERARCHY as _BASE_HIER
        after = self._snap_rules(after_rules)
        trig = dict(trigger or {})
        section = ("genai_rules" if stage == 3
                   else "evfis_rule_modifications")
        # new rules
        for r in after_rules:
            if r.name in before_rules:
                continue
            spec = dict(name=r.name,
                        antecedents=[list(a) for a in r.antecedents],
                        consequents=[[str(i), float(v)]
                                     for i, v in r.consequents],
                        note=r.note, strength=float(getattr(r, "strength",
                                                            0.0)))
            if stage == 3:
                # spec already carries `name`; passing it again raised
                # TypeError, and the caller's blanket except swallowed it, so
                # every admitted stage-3 rule was silently dropped instead of
                # being stored
                self.gstate.append("genai_rules",
                                   dict(spec,
                                        depends_on_concepts=[
                                            c for c, _t in r.antecedents
                                            if c not in _BASE_HIER],
                                        trigger=trig),
                                   source_stage=3, save=False)
            else:
                self.gstate.append(
                    "evfis_rule_modifications",
                    dict(base_rule_id=r.name,
                         base_rule_set=self.seed_profile,
                         modification_type="rule_add",
                         before={"rule": None}, after={"rule": spec},
                         trigger=trig),
                    source_stage=2, save=False)
        # retuned consequents of rules that already existed
        for name, cons in after.items():
            old = before_rules.get(name)
            if old is None or old == cons:
                continue
            self.gstate.append(
                "evfis_rule_modifications",
                dict(base_rule_id=name, base_rule_set=self.seed_profile,
                     modification_type="consequent_update",
                     before={"consequents": old}, after={"consequents": cons},
                     trigger=trig),
                source_stage=1, save=False)
        # membership moves and inserted terms
        now_parts = self._snap_parts()
        for var, part in now_parts.items():
            old = before_parts.get(var)
            if old == part:
                continue
            _ins = old is not None and set(part) - set(old)
            self.gstate.append(
                "evfis_rule_modifications",
                dict(variable=var, base_rule_set=self.seed_profile,
                     modification_type=("term_insert" if _ins
                                        else "membership_shift"),
                     before={"partition": old or {}},
                     after={"partition": part},
                     trigger=trig),
                source_stage=(2 if _ins else 1), save=False)
        self.gstate.save()

    def record_package(self, concepts=None, macros=None, trigger=None):
        """Stage 3 vocabulary growth, recorded as its own records so the
        consumption flag can deactivate the whole package at once."""
        for name, spec in (concepts or {}).items():
            lvl, ins = spec
            self.gstate.append(
                "genai_concepts",
                dict(name=name, layer=int(lvl),
                     inputs=[{"name": a, "weight": float(b)}
                             for a, b in ins],
                     outputs=[{"name": "activation", "range": [0, 1]}],
                     aggregation="weighted-sum-gated",
                     trigger=dict(trigger or {})),
                source_stage=3, save=False)
        for name, spec in (macros or {}).items():
            self.gstate.append(
                "genai_interventions",
                dict(name=name,
                     composition=[{"channel": a, "weight": float(b)}
                                  for a, b in spec.get("composition", [])],
                     clauses=list(spec.get("clauses") or []),
                     depends_on_concepts=[],
                     trigger=dict(trigger or {})),
                source_stage=3, save=False)
        self.gstate.save()

    # ------------------------------------------------------------ public
    def maybe_decide(self, sim):
        """Called every simulation step; recomputes on cycle boundaries."""
        step = int(sim.state.step)
        _dt = float(getattr(sim.cfg, "step_minutes", 1.0))
        _cyc = (max(1, int(round(self.cycle_min / _dt)))
                if self.cycle_min is not None else self.cycle_steps)
        if (self.last_override is not None
                and step - self.last_cycle_step < _cyc):
            return self.last_override
        ov = self.decide(sim)
        self.last_cycle_step = step
        return ov

    def new_fire(self) -> None:
        """Fire reset: the ENGINE SURVIVES. Everything learned (rules,
        memberships, controller value table) is knowledge, not decision state, and
        stays; only the per-run transients are dropped so the next
        fire starts with a clean decision slate."""
        self.last_override = None
        self.last_withheld = False
        self.last_actions = None
        # THE CHRONICLE AND THE LOG ARE DECISION STATE, so they go with the
        # fire that produced them. They used to survive it, and since every
        # view reads the LAST cycle or the whole list, the dashboard, the
        # step tables and the decision log all went on showing the previous
        # fire after a reset: the burned area, the cost bars and the agent
        # rows were the old run's, on a map where nothing was burning.
        self.cycles = []
        self.log.records = []
        self.last_global = None
        self.last_adapt_region = None
        self.last_adapt_why = ""
        self.resolve_warnings = []
        self._prev_ever = None
        self.last_cycle_step = -10 ** 9
        # a new fire is a new run: the tally describes ONE simulation
        self.run_stats = self._fresh_run_stats()
        self._nh_last = None
        self._nh_zero = None
        self._adapt_last_min = None
        # GenAI retirement is PER FIRE: a fresh fire gives stage 3 a new chance
        self._genai_dead = False
        self._genai_fails = 0
        for g in self.gaters.values():
            g.prev = {}
            g.step = None
        if self.learned_store:
            from .persist import save_learned
            try:
                save_learned(self.rules, self.learned_store,
                             profile=self.seed_profile,
                             engine=self,
                             use_evfis=self.use_evfis,
                             use_genai=self.use_genai)
                if getattr(self, "run_logger", None) is not None:
                    self.run_logger.save_rules(
                        self.rules, self.seed_profile, self)
            except Exception:
                pass

    def rewind_to(self, step: int) -> None:
        """A simulator rewind must take the DSS with it: the standing
        orders, the no-harm cache, the gating priors and every log
        entry AFTER the rewind point are decision state and roll
        back; learned rules and the controller value table are knowledge and stay
        (they persist across fires too)."""
        k = int(step)
        self.last_override = None
        self.last_actions = None
        self.last_withheld = False
        self.last_global = None
        self._nh_last = None
        self._nh_zero = None
        self._adapt_last_min = None
        self._prev_ever = None
        self._genai_dead = False
        self._genai_fails = 0
        for g in self.gaters.values():
            g.prev = {}
            g.step = None
        self.log.records = [r for r in self.log.records
                            if r.step <= k]
        self.cycles = [c for c in self.cycles
                       if int(c.get("step", 0)) <= k]

    def decide(self, sim):
        """Run one decision cycle and return the orders it produces.

        The order of work is fixed and each step depends on the one
        before it: observe, gate, fire the rules, price the candidate
        against a no-action forecast, apply the quality gate per region,
        let the coordinator ration the pool, and only then consider
        adaptation. Nothing an adaptation stage proposes reaches the
        field in this cycle without passing the same two tests the
        standing decision just passed.
        """
        sim._dss_hmin = self.horizon_min    # stage forecasts read this
        # DERIVE the active set first. Consumption flags therefore take effect
        # on the very next cycle, with no engine rebuild and no restart.
        if getattr(self, "state_path", None):
            self.gstate.set_flags(evfis_active=self.evfis_on,
                                  genai_active=self.genai_on,
                                  use_stage12_rules=self.use_evfis,
                                  use_stage3_rules=self.use_genai)
            self._sync_active_set()
        # availability AFTER the sync: the derived set is filtered by
        # what THIS map can physically supply, every cycle
        self._apply_availability(sim)
        step = int(sim.state.step)
        ctx = self._perceive(sim)

        def build(rules):
            _, pairs = self._decide_regions(sim, ctx, rules)
            return self._override(sim, pairs)

        rows, pairs = self._decide_regions(sim, ctx, rules=self.rules)
        ov = self._override(sim, pairs, keep_actions=True)
        try:
            self._note_water_ferry(sim, rows)
        except Exception:
            pass
        # regions where a GenAI-generated rule (G#) fired this cycle, so the
        # map can flag the generated orders distinctly from the base ones
        self.last_genai_regions = {
            name for name, row in rows.items()
            if any(str(rn).startswith("G") and w > 0.05
                   for rn, w in row.get("trace", []))}
        rep_c = forecast_cost(sim, ov, self.horizon_steps,
                              horizon_min=self.horizon_min)
        rep_0 = forecast_cost(sim, None, self.horizon_steps,
                              horizon_min=self.horizon_min)
        j_c, j_0 = float(rep_c.j_total), float(rep_0.j_total)
        sim._dss_j0 = j_0    # adaptation trials reuse this
        outcome = AdaptOutcome(0, False, "rules as-is", dJ=j_c - j_0)
        stage = 0
        bucket = ""
        # WHERE THE LEARNING WOULD GO, recorded even when no stage runs, so
        # the view never has to guess which region an attempt belonged to
        self.last_adapt_region = None
        self.last_adapt_why = ""
        _p_c = float(getattr(rep_c, "j_physical", j_c))
        _p_0 = float(getattr(rep_0, "j_physical", j_0))
        if self.rel_physical:
            need_rel = (1.0 - self.min_gain) * _p_0
            need = min(self.j_threshold, need_rel)
        else:
            need = min(self.j_threshold, (1.0 - self.min_gain) * j_0)
        _dtm = float(getattr(sim.cfg, "step_minutes", 1.0))
        _now_min = step * _dtm
        _adapt_due = (self._adapt_last_min is None
                      or _now_min - self._adapt_last_min
                      >= self.adapt_cooldown_min)
        # COVERAGE GAP: with a thin seed base (minimal/core profiles)
        # the cost trigger alone almost never fires the GROWTH stages
        # because the direct-attack doctrine already floors the
        # physical outcome. But a live fire on which no rule speaks
        # louder than a whisper IS a deficit of the rule base itself,
        # so it triggers adaptation on its own and routes it to the
        # rule-creating stages (2/3).
        fired_all = []
        _cov_by_region: Dict[str, float] = {}
        for name in rows:
            eff = rows[name]["eff"]
            _, tr = evaluate_rules(eff, ctx[name]["f"], self.rules,
                                   macros=self.macros)
            fired_all.extend(tr)
            # how loudly the base speaks about THIS region, kept per region
            # because that is what decides where the adaptation is sent
            _cov_by_region[name] = max((w for _r, w in tr), default=0.0)
        _covw = max((w for _r, w in fired_all), default=0.0)
        for _r_s, _w_s in fired_all:
            _r_s.strength = float(getattr(_r_s, "strength", 0.0)
                                  + _w_s)
        _fire_on = bool((sim.state.burning > 0.5).any())
        # a coverage VOID: a live fire on which no rule speaks louder than a
        # whisper (max fired weight < 0.45). There is nothing to TUNE here, so
        # only the rule-CREATING stages (2/3) can answer it.
        _void = _fire_on and _covw < 0.45
        _gap = _void
        # SPREAD TRIGGER: the mission is a fire that does NOT spread.
        # If ever_burned grew since the last decision even though the
        # orders were applied, satisficing on J alone is NOT enough:
        # the growth itself engages the adaptation stages.
        _ever_now = int(sim.ever_burned.sum())
        _growth = (0 if self._prev_ever is None
                   else max(0, _ever_now - self._prev_ever))
        self._prev_ever = _ever_now
        _spread = _fire_on and _growth > 0
        _gap = _gap or _spread
        # the symptoms act by tightening the bound, so the acceptance test
        # is the single gate and the bound is always in the causal path
        _tighten = 1.0
        if _spread:
            _tighten *= (1.0 - self.spread_tighten)
        if _void:
            _tighten *= (1.0 - self.void_tighten)
        need_eff = need * _tighten
        if self.rel_physical:
            # the ceiling is read on the total, the margin on the physical
            _deficit_on = ((j_c > self.j_threshold * _tighten
                            or _p_c > need_rel * _tighten)
                           and j_0 > 1e-6)
        else:
            _deficit_on = j_c > need_eff and j_0 > 1e-6
        if (self.adapt_on and _deficit_on
                and not _adapt_due):
            _left = self.adapt_cooldown_min - (_now_min
                                               - self._adapt_last_min)
            outcome = AdaptOutcome(
                0, False,
                f"adaptation on cooldown ({_left:.0f} min left); "
                "seed-base decision stands, its orders are applied",
                dJ=j_c - j_0)
        # the cycle is tallied BEFORE the stage runs, so a cycle where the
        # gate opened but nothing was tried still records the reason
        self._tally_cycle(step, j_c, j_0, need_eff, _deficit_on, _gap,
                          _adapt_due,
                          self.stages_allowed if self.adapt_on else ())
        if self.adapt_on and _deficit_on and _adapt_due:
            self._adapt_last_min = _now_min
            deficit = max(j_c - self.j_threshold,
                          (j_c - need_eff) / max(j_0, 1e-6) * 0.1)
            if _gap:
                deficit = max(deficit, 0.05,
                              min(0.3, _growth / 50.0))
            bucket = self.controller.bucket(deficit, gap=_gap)
            # THE TARGET REGION IS CHOSEN FIRST. The menu filter below asks
            # what stage 2 would find THERE, so the region has to be known
            # before the menu is built.
            hot, _hot_why = self._adapt_region(rows, ctx, _cov_by_region)
            self.last_adapt_region = hot
            self.last_adapt_why = (
                f"{_hot_why} (coverage "
                + ", ".join(f"{_n} {float(_cov_by_region.get(_n, 0.0)):.2f}"
                            for _n in rows) + ")")
            _menu = self.stages_allowed
            if _void:
                # a coverage void cannot be TUNED (no rule fires there): only
                # the rule-creating stages (2/3) answer it. This stops stage 1
                # from being picked and rejected on an empty cell while stage 2
                # (resolution) starves.
                _menu = (tuple(x for x in self.stages_allowed
                               if x in (2, 3))
                         or self.stages_allowed)
            # GenAI (stage 3) is dropped only once it has PROVEN unproductive
            # this run (unreachable model, or repeated rejections) via
            # _genai_dead. Reachability is judged by actually calling `claude`
            # in stage 3 (same path as the Test button), NOT by shutil.which,
            # which on Windows can miss a runnable `.cmd` shim and wrongly
            # skip stage 3 even when the model answers fine.
            if 3 in _menu and self._genai_dead:
                _menu = tuple(s for s in _menu if s != 3)
            # PREDICTIVE FILTER, NOT RETIREMENT. Stage 2 instantiates the
            # antecedent cell of the CURRENT situation. When that cell is
            # already covered and the situation is crisp, the stage is
            # certain to be refused, so offering it wastes the pick. This
            # asks the stage itself (stage2_would_be_refused) rather than
            # guessing, and it drops stage 2 from THIS CYCLE only: nothing
            # is retired, so the moment the cell space grows (a new concept,
            # a new term, a situation not seen before) the stage is back on
            # the menu by itself.
            #
            # A coverage VOID is exempt: there the cell is open by
            # definition, which is precisely when stage 2 is wanted.
            if 2 in _menu and not _void:
                try:
                    from .adapt import stage2_would_be_refused
                    if stage2_would_be_refused(self.rules,
                                               rows[hot]["eff"],
                                               rows[hot]["crisp"]):
                        _m2 = tuple(x for x in _menu if x != 2)
                        # never filter the menu empty: a stage that cannot
                        # win is still better than no adaptation at all
                        if _m2:
                            _menu = _m2
                            self.run_stats["stage2_prefiltered"] = \
                                self.run_stats.get(
                                    "stage2_prefiltered", 0) + 1
                except Exception:
                    pass
            if not _menu:
                outcome = AdaptOutcome(
                    0, False,
                    "no runnable adaptation stage (GenAI unreachable and "
                    "evFIS off)")
            else:
                stage = self.controller.select(deficit, stages=_menu,
                                                gap=_gap)
                # PRODUCTION runs on the stage's own base, independent of what
                # the inference is allowed to consume. With stage 1-2
                # consumption off this is the evFIS chain, so the learner
                # keeps its continuity while the DSS runs on factory values.
                _work = (self._evfis_base()
                         if (stage in (1, 2)
                             and getattr(self, "state_path", None))
                         else self.rules)
                _b_rules = self._snap_rules(_work)
                _b_parts = self._snap_parts()
                _b_vocab = (set(self.hierarchy), set(self.macros))
                # what this attempt is about to spend, measured rather than
                # assumed: the cooldown refund below reads the difference
                from . import adapt as _ad_mod
                _spend0 = (_ad_mod.CVA_CALLS, _ad_mod.GENAI_CALLS)
                if stage == 1:
                    outcome = stage1_evfis(build, sim, _work,
                                           fired_all, self.horizon_steps,
                                           step_size=self.evfis_step)
                elif stage == 2:
                    outcome = stage2_resolution(
                        build, sim, _work, rows[hot]["eff"],
                        rows[hot]["crisp"], self.horizon_steps,
                        coverage_gap=_gap, cov_w=_covw)
                else:
                    outcome = stage3_generative(
                        build, sim, _work, rows[hot]["eff"],
                        rows[hot]["crisp"], self.horizon_steps,
                        coverage_gap=_gap, cov_w=_covw, engine=self)
                    # retire stage 3 for the rest of this run when GenAI is
                    # UNPRODUCTIVE: unreachable, or two proposals in a row got
                    # rejected. GenAI is expensive (a live model call) and each
                    # failed attempt steals a cycle from the productive evFIS
                    # tuning, so a stubbornly-rejected GenAI drags the whole
                    # run below evFIS-only.
                    if (outcome.info or {}).get("reason") \
                            == "model unreachable":
                        self._genai_dead = True
                    elif outcome.accepted:
                        self._genai_fails = 0
                    elif (outcome.info or {}).get("reason") \
                            == "model timeout":
                        pass    # a slow answer is not a bad proposal
                    else:
                        self._genai_fails += 1
                        if self._genai_fails >= 3:
                            self._genai_dead = True
                outcome.stage = stage
                # REFUND A FREE REJECTION. Neither a shadow forecast nor a
                # model call happened, so the attempt rationed nothing and
                # holding the window shut for the full cooldown only starves
                # the other stages. A model timeout is NOT free: it already
                # spent the wall clock and would just repeat.
                _forecasts = _ad_mod.CVA_CALLS - _spend0[0]
                _calls = _ad_mod.GENAI_CALLS - _spend0[1]
                outcome.info = dict(outcome.info or {})
                outcome.info["forecasts"] = int(_forecasts)
                outcome.info["model_calls"] = int(_calls)
                if (not outcome.accepted) and _forecasts == 0 and _calls == 0:
                    self._adapt_last_min = (
                        _now_min - (self.adapt_cooldown_min
                                    - self.adapt_retry_min))
                    self.run_stats["cooldown_refunds"] = \
                        self.run_stats.get("cooldown_refunds", 0) + 1
                self._tally_outcome(stage, outcome)
                _rw = -outcome.dJ
                # a REJECTED attempt wasted a cycle: give it a small negative
                # reward so the controller does not stay stuck on a stage that
                # keeps failing (which was starving stage 2 / stage 3)
                if not outcome.accepted:
                    _rw = min(_rw, -0.004)
                if outcome.accepted and (outcome.info or {}).get("package"):
                    _rw -= 0.02      # G5: vocabulary growth costs margin
                self.controller.update(deficit, stage, reward=_rw, gap=_gap)
                if getattr(self, "state_path", None):
                    # written every update, not at shutdown: a dashboard is
                    # closed by closing the window, and an exit hook that
                    # never runs is a memory that never persists
                    try:
                        self.gstate.save_controller(self.controller,
                                                    map_key=self.map_key)
                    except Exception as _exc_c:
                        self.persist_errors.append(
                            f"stage controller: {type(_exc_c).__name__}: "
                            f"{_exc_c}")
                if outcome.accepted and getattr(self, "state_path", None):
                    # PERSIST WHAT CHANGED, not the whole engine. The record
                    # carries before/after, so a wipe can revert it and a
                    # restart can replay it in the order it happened.
                    try:
                        _trig = dict(step=int(sim.state.step),
                                     minute=float(_now_min),
                                     deficit=float(deficit),
                                     stage=int(stage))
                        _nc = {c: self.hierarchy[c]
                               for c in set(self.hierarchy) - _b_vocab[0]}
                        _nm = {m: self.macros[m]
                               for m in set(self.macros) - _b_vocab[1]}
                        if _nc or _nm:
                            self.record_package(concepts=_nc, macros=_nm,
                                                trigger=_trig)
                        self._record_changes(stage, _b_rules, _b_parts,
                                             _work, trigger=_trig)
                        # the inference set is re-derived next cycle, so the
                        # change reaches the DSS only if consumption allows it
                        self._sync_active_set()
                    except Exception as _exc_p:
                        # never silent: an accepted adaptation that fails to
                        # persist looks exactly like an adaptation that never
                        # happened, and the panels then contradict each other
                        self.persist_errors.append(
                            f"step {int(sim.state.step)}, stage {stage}: "
                            f"{type(_exc_p).__name__}: {_exc_p}")
                        self.run_stats["persist_failed"] = \
                            self.run_stats.get("persist_failed", 0) + 1
                if outcome.accepted and self.learned_store:
                    from .persist import save_learned
                    try:
                        save_learned(self.rules, self.learned_store,
                                 profile=self.seed_profile,
                                 engine=self,
                                 use_evfis=self.use_evfis,
                                 use_genai=self.use_genai)
                        if getattr(self, "run_logger", None) is not None:
                            self.run_logger.save_rules(
                                self.rules, self.seed_profile, self)
                    except Exception:
                        pass
                if not outcome.accepted:
                    outcome.detail = (outcome.detail + " | "
                                      if outcome.detail else "") + (
                        "seed-base decision stands, its orders are "
                        "applied")
            if outcome.accepted:
                rows, pairs = self._decide_regions(sim, ctx,
                                                   rules=self.rules)
                ov = self._override(sim, pairs, keep_actions=True)
                rep_c = forecast_cost(sim, ov, self.horizon_steps,
                                      horizon_min=self.horizon_min)
                j_c = float(rep_c.j_total)

        # NO-HARM FAIL-SAFE: if even the (possibly adapted) candidate
        # is forecast to end WORSE than doing nothing, the offensive
        # allocation is withheld for this cycle: life-safety orders
        # (evacuation, public warning) stand, no capacity is
        # fielded, no response cost is paid. A decision support
        # system must never knowingly buy a worse outcome.
        _phys_c = physical_cost(rep_c, sim.cfg.cost)
        _phys_0 = physical_cost(rep_0, sim.cfg.cost)
        # a SHORT lookahead cannot see the benefit of wetting and
        # line building (they land 30-60 min out), so the no-harm
        # comparison always re-runs at >= 45 min even when the UI
        # horizon is shorter; and orders are withheld only when the
        # candidate is forecast CLEARLY worse, not merely equal
        # (equal now often means better later)
        if (self.horizon_min or 0) < 45.0:
            _q = lambda x: int(round(4.0 * np.log1p(max(x, 0.0))))
            _sig = (0, 0, 0) if ov is None else (
                _q(float(ov.rcap.sum())),
                _q(float((ov.rcap > 1e-9).sum())),
                _q(float(np.clip(ov.ravail, 0, 1).sum())))
            _nh = self._nh_last
            if (_nh is not None and _nh[1] == _sig
                    and _now_min - _nh[0] < self.noharm_recheck_min):
                _phys_c, _phys_0 = _nh[2], _nh[3]
            else:
                _rc45 = forecast_cost(sim, ov, self.horizon_steps,
                                      horizon_min=45.0)
                # the NO-ACTION 45-min future does not depend on the
                # candidate, so it is keyed on the state alone: changing the
                # override used to re-run it for the identical answer, which
                # doubled the cost of every no-harm recheck
                _nh0 = getattr(self, "_nh_zero", None)
                if _nh0 is not None and _nh0[0] == int(sim.state.step):
                    _phys_0 = _nh0[1]
                else:
                    _r045 = forecast_cost(sim, None, self.horizon_steps,
                                          horizon_min=45.0)
                    _phys_0 = physical_cost(_r045, sim.cfg.cost)
                    self._nh_zero = (int(sim.state.step), _phys_0)
                _phys_c = physical_cost(_rc45, sim.cfg.cost)
                self._nh_last = (_now_min, _sig, _phys_c, _phys_0)
        self.last_withheld = bool(_phys_c > _phys_0 + 1e-3)
        # the PHYSICAL comparison is the one that says whether the orders are
        # buying a better fire. The total J also carries the price of acting,
        # so it sits above no-action whenever anything is fielded, and reading
        # only the total makes a working DSS look like a failing one.
        self.run_stats["phys_series"].append(
            (int(step), float(_phys_c), float(_phys_0)))
        if self.last_withheld:
            self.run_stats["withheld"] += 1
        if self.last_withheld:
            # EVERYTHING except the life-safety orders is withheld: a
            # plan judged worse than no action must not keep flying
            # retardant, lighting counter-fires or running macros
            # through the gap in a fixed channel list (that leak let
            # a vetoed plan keep acting for half an hour)
            for name in rows:
                for k_off in list(rows[name]["u"].keys()):
                    if k_off in ("evacuation", "public_warning",
                                 "_share"):
                        continue
                    rows[name]["u"][k_off] = 0.0
            ov = None
            self.last_actions = dict(
                supp=None, cont=None, prot=None,
                regions=[dict(name=n,
                              box=(ctx[n]["region"].x0,
                                   ctx[n]["region"].y0,
                                   ctx[n]["region"].x1,
                                   ctx[n]["region"].y1),
                              u=dict(rows[n]["u"]))
                         for n in rows])
            outcome.detail = (outcome.detail + " | " if
                              outcome.detail else "") + (
                "NO-HARM: no PHYSICAL improvement forecast "
                f"(phys {_phys_c:.4f} vs {_phys_0:.4f}), offensive "
                "orders withheld; life-safety orders stand")
        # commit priors + log
        for name, c in ctx.items():
            self._effective(c, commit_step=step)
            self.log.add(DecisionRecord(
                step=step, region=name, features=dict(c["f"]),
                feature_conf=dict(c["fc"]),
                concepts=dict(rows[name]["crisp"]),
                gates=dict(c["gates"]), fired=list(rows[name]["trace"]),
                intensities=dict(rows[name]["u"]),
                quality=float(rows[name]["q"]),
                failsafe=bool(rows[name]["fs"]),
                stage=int(outcome.stage if outcome.accepted else 0),
                stage_tried=int(stage),
                stage_detail=outcome.detail,
                ctrl_bucket=bucket,
                j_forecast=float(j_c), j_noaction=float(j_0),
                j_threshold=float(self.j_threshold),
                coord_share=float(rows[name]["share"]),
                attended=bool(rows[name]["attended"])))
        _sm = float(getattr(sim.cfg, "step_minutes", 1.0))
        if self.run_logger is not None:
            for rec in self.log.at(step):
                self.run_logger.log_decision(
                    rec, self.log.stage_story(rec), step_minutes=_sm)
        # ---- full cycle chronicle: EVERYTHING that fed and left
        # this decision, one JSON object per cycle ----
        from dataclasses import asdict as _asdict
        from disaster_phyengine.costs import compute_costs as _cc
        try:
            _rep = _cc(sim)
            # ALL FIVE TERMS PLUS BOTH AGGREGATES. The chronicle carried
            # four of the five and neither aggregate, so the dashboard could
            # not show the delay term at all and could not put the physical
            # outcome beside the decision cost.
            _costs = dict(j_total=_rep.j_total, j_burn=_rep.j_burn,
                          j_asset=_rep.j_asset, j_pop=_rep.j_pop,
                          j_resp=_rep.j_resp, j_delay=_rep.j_delay,
                          j_physical=_rep.j_physical)
        except Exception:
            _costs = {}
        _w = sim.world

        def _afl(key):
            """A number the allocator reported for this cycle, if it ran."""
            v = (getattr(self, "last_actions", None) or {}).get(key)
            return None if v is None else float(v)

        cyc = dict(
            global_dss=self.last_global,
            step=step, t_min=step * _sm,
            sim=dict(
                burning=int((sim.state.burning > 0.5).sum()),
                burned=int(sim.ever_burned.sum()),
                wws_mean=float(_w.meteo.wws.mean()),
                prec_mean=float(_w.meteo.prec.mean()),
                fmoist_mean=float(_w.fuel.fmoist.mean())),
            costs=_costs,
            pool=(dict(cells=int((self.base_pool.rcap > 0).sum()),
                       rcap_total=float(self.base_pool.rcap.sum()),
                       # WHAT THE ORDERS ASKED FOR AND WHERE IT WENT.
                       # The ratio of demand to budget says whether the
                       # cycle was rationing at all, and the focus share
                       # whether the rationing concentrated. Both are
                       # needed to read the attention threshold, which
                       # can do nothing when nothing is scarce.
                       demand=_afl("demand"), budget=_afl("budget"),
                       focus_share=_afl("funded_focus_share"),
                       demand_per_region=(
                           (getattr(self, "last_actions", None) or {})
                           .get("demand_per_region")))
                  if self.base_pool is not None else None),
            sensors=(self.network.status()
                     if self.network is not None else None),
            forecast=dict(j_candidate=float(j_c),
                          j_noaction=float(j_0),
                          j_threshold=float(self.j_threshold),
                          satisficing_bound=float(need)),
            stage_controller=dict(selected_stage=int(stage), bucket=bucket,
                    eps=float(self.controller.eps),
                    value_table={f"{b}/s{st_}": round(v, 4)
                             for (b, st_), v in self.controller.q.items()}),
            no_harm_withheld=bool(getattr(self, "last_withheld",
                                          False)),
            adapt_region=getattr(self, "last_adapt_region", None),
            adapt_region_why=getattr(self, "last_adapt_why", ""),
            adaptation=dict(stage=int(outcome.stage),
                            tried=int(stage),
                            accepted=bool(outcome.accepted),
                            detail=outcome.detail,
                            dJ=float(outcome.dJ),
                            info=outcome.info),
            regions={name: dict(
                features={k: round(float(v), 4)
                          for k, v in ctx[name]["f"].items()},
                feature_conf={k: round(float(v), 3)
                              for k, v in ctx[name]["fc"].items()},
                gates={k: round(float(v), 3)
                       for k, v in ctx[name]["gates"].items()},
                concepts_effective={k: round(float(v), 4)
                                    for k, v in
                                    rows[name]["crisp"].items()},
                fired=rows[name]["trace"][:12],
                orders_from_rules={k: round(float(v), 3)
                                   for k, v in
                                   rows[name]["u_rules"].items()},
                orders_final={k: round(float(v), 3)
                              for k, v in rows[name]["u"].items()},
                quality=round(float(rows[name]["q"]), 3),
                failsafe=bool(rows[name]["fs"]),
                coord_share=round(float(rows[name]["share"]), 3),
                attended=bool(rows[name]["attended"]))
                for name in rows})
        self.cycles.append(cyc)
        if len(self.cycles) > 300:
            self.cycles = self.cycles[-300:]
        if self.run_logger is not None:
            self.run_logger.log_cycle(cyc)
            if self.last_global:
                self.run_logger.log_global(step, step * _sm,
                                           self.last_global)
        self.last_override = ov
        if self.learned_store:
            from .persist import save_learned
            try:
                save_learned(self.rules, self.learned_store,
                             profile=self.seed_profile,
                             engine=self,
                             use_evfis=self.use_evfis,
                             use_genai=self.use_genai)
                if getattr(self, "run_logger", None) is not None:
                    self.run_logger.save_rules(
                        self.rules, self.seed_profile, self)
            except Exception:
                pass
        # the per-run analysis goes to disk every cycle, so the figures it
        # feeds can be rebuilt from the run directory without replaying
        if getattr(self, "run_logger", None) is not None:
            try:
                self.run_logger.save_analysis(self)
            except Exception as _exc_a:
                self.persist_errors.append(
                    f"analysis log: {type(_exc_a).__name__}: {_exc_a}")
        return ov


def counterfactual(sim, from_step: int, to_step: int | None = None,
                   step_hook=None):
    """What would have happened WITHOUT the DSS orders from `from_step`
    on: clone the live simulator (snapshots included), rewind the clone
    and rerun history with no resource override. The clone carries the
    rng state saved in the snapshot, so the divergence is attributable
    to the withdrawn orders alone. Returns (clone, report)."""
    from .evaluate import clone_sim
    from disaster_phyengine.costs import compute_costs
    s2 = clone_sim(sim, keep_snapshots=True)
    k = int(from_step)
    if not s2.rewind(k):
        # the exact step may have been evicted by the snapshot
        # budget; the NEAREST EARLIER snapshot is a superset
        # counterfactual (even more orders withdrawn), never a
        # narrower one
        _avail = [q for q in s2.rewindable_steps if q <= k]
        if not _avail or not s2.rewind(int(max(_avail))):
            return None, None
    end = int(to_step if to_step is not None else sim.state.step)
    while s2.state.step < end:
        if step_hook is not None:
            # replay the EXOGENOUS drivers (weather, EMC fuel drying)
            # exactly as the factual run saw them at this minute;
            # they are order-independent and must not freeze at the
            # rewind point
            step_hook(s2.world, int(s2.state.step))
        s2.step(resource_override=None)
    return s2, compute_costs(s2)
