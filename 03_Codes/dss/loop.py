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
    def __init__(self, regions, base_pool=None, network=None,
                 j_threshold: float = 0.35, eta: float = 0.60,
                 cycle_steps: int = 3, horizon_steps: int = 12,
                 evfis_step: float = 0.05, adapt_on: bool = True,
                 evfis_on: bool = True,
                 genai_on: bool = False, ctrl_eps: float = 0.10,
                 ctrl_lr: float = 0.05, attention_thr: float = 0.35,
                 min_gain: float = 0.05, run_logger=None,
                 cycle_min: float | None = None,
                 horizon_min: float | None = None,
                 seed_profile: str = "full",
                 learned_store: str | None = None,
                 revision_budget: int = 3,
                 use_evfis: bool = True, use_genai: bool = True):
        self.regions = list(regions)
        self.base_pool = base_pool
        self.network = network
        self.j_threshold = float(j_threshold)
        self.eta = float(eta)
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
        # ADAPTIVE satisficing (thesis: the bound tightens): a candidate
        # must either clear the absolute threshold OR beat the no-action
        # forecast by at least min_gain relative margin; otherwise the
        # adaptation stages engage even when the absolute cost is small
        self.min_gain = float(min_gain)
        self.run_logger = run_logger
        self.seed_profile = str(seed_profile or "full")
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
        self.controller = StageController(eps=ctrl_eps, lr=ctrl_lr)
        # PERFORMANCE THROTTLES (live-run economics, not physics):
        # adaptation trials and the 45-min no-harm re-forecast are the
        # two expensive items of a cycle (each shadow run simulates
        # 45 min of physics). With a 1-min decision cycle they must
        # not run EVERY cycle: trials respect a cooldown, the no-harm
        # verdict is reused while the composed orders are unchanged.
        self.adapt_cooldown_min = 10.0
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
        for name, c in ctx.items():
            att = prios[name] >= self.attention_thr * max(pmax, 1e-9) \
                or prios[name] >= self.attention_thr
            share = 1.0 if att else 0.5 + 0.5 * (
                prios[name] / max(pmax, 1e-9))
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
        _rank = sorted(prios, key=lambda n: -prios[n])
        self.last_global = dict(
            ranking=[(n, round(float(prios[n]), 3)) for n in _rank],
            shares={n: round(float(rows[n]["share"]), 3)
                    for n in rows},
            attended=[n for n in rows if rows[n]["attended"]],
            thresholds={n: round(float(rows[n]["eta"]), 3)
                        for n in rows},
            hotspot=(_rank[0] if _rank else None),
            statement=("Global DSS: "
                       + (f"focus on {_rank[0]} "
                          f"(priority {prios[_rank[0]]:.2f}); "
                          if _rank else "")
                       + ", ".join(
                           f"{n} share {rows[n]['share']:.2f}"
                           + ("" if rows[n]["attended"]
                              else " (monitor)")
                           for n in _rank)))
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

    def _override(self, sim, pairs, keep_actions=False):
        world = sim.world
        _burn = self._observed_burning(sim)
        if keep_actions:
            ov, acts = decision_to_resources(
                world, _burn, pairs, self.base_pool,
                return_actions=True)
            self.last_actions = acts
            return ov
        return decision_to_resources(world, _burn, pairs, self.base_pool)

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
        self._nh_last = None
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
        sim._dss_hmin = self.horizon_min    # stage forecasts read this
        step = int(sim.state.step)
        ctx = self._perceive(sim)

        def build(rules):
            _, pairs = self._decide_regions(sim, ctx, rules)
            return self._override(sim, pairs)

        rows, pairs = self._decide_regions(sim, ctx, rules=self.rules)
        ov = self._override(sim, pairs, keep_actions=True)
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
        for name in rows:
            eff = rows[name]["eff"]
            _, tr = evaluate_rules(eff, ctx[name]["f"], self.rules,
                                   macros=self.macros)
            fired_all.extend(tr)
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
        _deficit_on = j_c > need and j_0 > 1e-6
        if (self.adapt_on and (_deficit_on or _gap)
                and not _adapt_due):
            _left = self.adapt_cooldown_min - (_now_min
                                               - self._adapt_last_min)
            outcome = AdaptOutcome(
                0, False,
                f"adaptation on cooldown ({_left:.0f} min left); "
                "seed-base decision stands, its orders are applied",
                dJ=j_c - j_0)
        if self.adapt_on and (_deficit_on or _gap) and _adapt_due:
            self._adapt_last_min = _now_min
            deficit = max(j_c - self.j_threshold,
                          (j_c - need) / max(j_0, 1e-6) * 0.1)
            if _gap:
                deficit = max(deficit, 0.05,
                              min(0.3, _growth / 50.0))
            bucket = self.controller.bucket(deficit, gap=_gap)
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
            hot = max(rows, key=lambda n: rows[n]["crisp"].get(
                "operational_priority", 0.0))
            if not _menu:
                outcome = AdaptOutcome(
                    0, False,
                    "no runnable adaptation stage (GenAI unreachable and "
                    "evFIS off)")
            else:
                stage = self.controller.select(deficit, stages=_menu,
                                                gap=_gap)
                if stage == 1:
                    outcome = stage1_evfis(build, sim, self.rules,
                                           fired_all, self.horizon_steps,
                                           step_size=self.evfis_step)
                elif stage == 2:
                    outcome = stage2_resolution(
                        build, sim, self.rules, rows[hot]["eff"],
                        rows[hot]["crisp"], self.horizon_steps,
                        coverage_gap=_gap, cov_w=_covw)
                else:
                    outcome = stage3_generative(
                        build, sim, self.rules, rows[hot]["eff"],
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
                    else:
                        self._genai_fails += 1
                        if self._genai_fails >= 2:
                            self._genai_dead = True
                outcome.stage = stage
                _rw = -outcome.dJ
                # a REJECTED attempt wasted a cycle: give it a small negative
                # reward so the controller does not stay stuck on a stage that
                # keeps failing (which was starving stage 2 / stage 3)
                if not outcome.accepted:
                    _rw = min(_rw, -0.004)
                if outcome.accepted and (outcome.info or {}).get("package"):
                    _rw -= 0.02      # G5: vocabulary growth costs margin
                self.controller.update(deficit, stage, reward=_rw, gap=_gap)
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
                _r045 = forecast_cost(sim, None, self.horizon_steps,
                                      horizon_min=45.0)
                _phys_c = physical_cost(_rc45, sim.cfg.cost)
                _phys_0 = physical_cost(_r045, sim.cfg.cost)
                self._nh_last = (_now_min, _sig, _phys_c, _phys_0)
        self.last_withheld = bool(_phys_c > _phys_0 + 1e-3)
        if self.last_withheld:
            for name in rows:
                for k_off in ("suppression_effort",
                              "resource_deployment",
                              "containment_line",
                              "asset_protection"):
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
            _costs = dict(j_total=_rep.j_total, j_burn=_rep.j_burn,
                          j_asset=_rep.j_asset, j_pop=_rep.j_pop,
                          j_resp=_rep.j_resp)
        except Exception:
            _costs = {}
        _w = sim.world
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
                       rcap_total=float(self.base_pool.rcap.sum()))
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
