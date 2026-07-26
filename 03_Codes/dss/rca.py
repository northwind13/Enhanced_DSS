"""Root cause analysis: the after-action review as part of the loop.

When a run ends, the operator can ask for an RCA. The run's own logs
are compiled into an evidence pack, a strong model (Opus by default)
writes the review an incident analyst would write, and the same
answer carries a machine-readable recommendation list. Applying the
recommendations feeds the findings BACK into the system: rules,
interventions and concepts enter the persistent store as
post-incident knowledge, tuning settings go back to the panel, and
infrastructure advice (a sensor here, a depot there) is staged so
the next run actually starts better. The rerun then SHOWS whether
the review was right, which is the whole point of a review.

Admission note: RCA products bypass the live G3-G5 rollout gates on
purpose. They are offline lessons from a full incident, not
mid-incident gambles; the RERUN is their A/B test, and every record
is marked "post-incident review" so the provenance stays visible.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

_SENSOR_KINDS = ("satellite", "aerial", "ground_camera", "in_situ",
                 "field_report", "public_report")
_DEPOT_KINDS = ("depot", "helibase")
_SETTING_KEYS = {"eta": "dss_eta", "j_threshold": "dss_jth",
                 "cycle_min": "dss_cycle_min",
                 "horizon_min": "dss_horizon_min",
                 "min_gain": "dss_min_gain"}
# hard ranges: an out-of-range advice is a model slip, never applied
# blindly (an eta of 15 once crushed every order to 4% for whole runs)
_SETTING_RANGE = {"eta": (0.0, 1.0), "j_threshold": (0.0, 1.0),
                  "cycle_min": (1.0, 240.0),
                  "horizon_min": (5.0, 480.0),
                  "min_gain": (0.0, 0.5)}


# ------------------------------------------------------------ evidence
def build_evidence(run_dir: str, engine=None, sim=None) -> str:
    """Compact, factual pack: what happened, cycle by cycle summary."""
    L = ["=== RUN EVIDENCE ==="]
    # trajectory
    try:
        rows = list(csv.DictReader(open(os.path.join(run_dir,
                                                     "steps.csv"))))
        pk = max(int(r["burning"]) for r in rows)
        L.append(f"- trajectory: {len(rows)} steps; burning start "
                 f"{rows[0]['burning']}, peak {pk}, end "
                 f"{rows[-1]['burning']}; total burned "
                 f"{rows[-1]['burned']} cells; alloc rcap start "
                 f"{rows[0]['alloc_rcap_sum']}, end "
                 f"{rows[-1]['alloc_rcap_sum']}; j_resp end "
                 f"{rows[-1]['j_resp']}")
        L.append("- fire out at the end: "
                 + ("YES" if int(rows[-1]["burning"]) == 0 else "NO"))
    except Exception as exc:
        L.append(f"- steps.csv unreadable ({type(exc).__name__})")
    # cycles: stages, vetoes, per-agent order profile early vs late
    try:
        cyc = [json.loads(x) for x in
               open(os.path.join(run_dir, "cycles.jsonl"),
                    encoding="utf-8")]
        tried: Dict[int, int] = {}
        acc: Dict[int, int] = {}
        veto = [c.get("step") for c in cyc if c.get("no_harm_withheld")]
        for c in cyc:
            a = c.get("adaptation") or {}
            if a.get("stage"):
                tried[a["stage"]] = tried.get(a["stage"], 0) + 1
                if a.get("accepted"):
                    acc[a["stage"]] = acc.get(a["stage"], 0) + 1
        L.append(f"- adaptation: tried {tried}, accepted {acc}")
        L.append(f"- no-harm vetoes: {len(veto)} cycles"
                 + (f", steps {veto[0]}..{veto[-1]}" if veto else ""))

        def _profile(cs):
            out = {}
            for c in cs:
                for rn, rd in (c.get("regions") or {}).items():
                    u = rd.get("orders_final") or {}
                    o = out.setdefault(rn, {})
                    for k, v in u.items():
                        if k == "_share":
                            continue
                        o[k] = o.get(k, 0.0) + float(v)
            n = max(1, len(cs))
            return {rn: {k: round(v / n, 2) for k, v in o.items()
                         if v / n > 0.05}
                    for rn, o in out.items()}
        k3 = max(1, len(cyc) // 3)
        L.append(f"- orders, first third (mean): {_profile(cyc[:k3])}")
        L.append(f"- orders, last third (mean): {_profile(cyc[-k3:])}")
        g_last = next((c.get("global_dss") for c in reversed(cyc)
                       if c.get("global_dss")), {})
        L.append(f"- last global statement: "
                 f"{(g_last or {}).get('statement', '-')}")
        # observed concept ranges: the ground truth for antecedents.
        # A recommended rule whose antecedents never held in this run
        # could never have fired; the review must aim inside these.
        obs: Dict[str, Dict[str, list]] = {}
        for c in cyc:
            for rn, rd in (c.get("regions") or {}).items():
                for k, v in (rd.get("concepts_effective")
                             or {}).items():
                    obs.setdefault(rn, {}).setdefault(k, []).append(
                        float(v))
        if obs:
            L.append("=== OBSERVED CONCEPT VALUES (min..max per "
                     "region; rule antecedents MUST be satisfiable "
                     "in these ranges, terms: VL<0.2, L~0.3, M~0.5, "
                     "H~0.7, VH>0.8) ===")
            for rn, d in obs.items():
                L.append(f"- {rn}: " + "; ".join(
                    f"{k} {min(v):.2f}..{max(v):.2f}"
                    for k, v in d.items()))
    except Exception as exc:
        L.append(f"- cycles.jsonl unreadable ({type(exc).__name__})")
    # infrastructure geometry
    try:
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        L.append(f"- sensors: {meta.get('sensors')}")
        _dep = [{k: d.get(k) for k in ("kind", "x", "y")}
                for d in (meta.get("depots") or [])]
        L.append(f"- depots: {_dep}")
        _eng = {k: meta.get("engine", {}).get(k)
                for k in ("regions", "seed_profile", "cycle_min")}
        L.append(f"- map: {meta.get('map')} | engine: {_eng}")
    except Exception:
        pass
    if engine is not None:
        from .concepts import DECISION_CONCEPTS as _DC
        from .concepts import HIERARCHY as _BH
        from .rules import INTERVENTIONS as _IV
        from .fuzzy import TERMS as _TM
        _lc = [c for c in (engine.hierarchy or {}) if c not in _BH]
        L.append("=== VOCABULARY (use EXACTLY these names) ===")
        L.append("- concepts: " + ", ".join(list(_DC) + _lc))
        L.append("- terms: " + ", ".join(_TM)
                 + " (or '>=TERM' for a rising threat; write '>=M', "
                 "NEVER a number like '>=0.35')")
        L.append("- interventions/channels: "
                 + ", ".join(list(_IV) + list(engine.macros or {})))
        L.append("=== CURRENT SETTINGS (propose only values inside "
                 "the stated range; anything outside is discarded "
                 "unread) ===")
        for k, (lo, hi) in _SETTING_RANGE.items():
            cur = {"eta": engine.eta,
                   "j_threshold": engine.j_threshold,
                   "cycle_min": engine.cycle_min,
                   "horizon_min": engine.horizon_min,
                   "min_gain": getattr(engine, "min_gain", 0.05),
                   }.get(k)
            L.append(f"- {k} = {cur} (valid {lo}..{hi})")
        L.append("- active rules: "
                 + "; ".join(r.text() for r in engine.rules
                             if getattr(r, "active", True))[:1500])
        if engine.macros:
            L.append("- learned interventions: "
                     + ", ".join(engine.macros))
        for w in (engine.resolve_warnings or [])[:6]:
            L.append(f"- warning: {w}")
    if sim is not None:
        try:
            import numpy as np
            wat = np.asarray(sim.world.fuel.ftype == 5)
            L.append(f"- water on map: {int(wat.sum())} cells")
            L.append(f"- population evacuated: "
                     f"{sim.population_evacuated:.0f}")
            # FIRE GEOGRAPHY: without it, a reviewer guesses the fire
            # position from sensor coordinates and stages depots on
            # the quiet side of the map.
            ign = [(int(e.x), int(e.y)) for e in
                   (getattr(sim.world, "ignitions", None) or [])]
            fp = (np.asarray(sim.ever_burned)
                  | (np.asarray(sim.state.burning) > 0.5))
            L.append("=== FIRE GEOGRAPHY (all placement advice must "
                     "serve THIS footprint; the applier refuses "
                     "sensors/depots more than ~25 cells from it) ===")
            if ign:
                L.append(f"- ignition points (x,y): {ign[:40]}")
            if fp.any():
                ys, xs = np.where(fp)
                L.append(f"- fire footprint: {int(fp.sum())} cells, "
                         f"x {int(xs.min())}..{int(xs.max())}, "
                         f"y {int(ys.min())}..{int(ys.max())}, "
                         f"centroid ({xs.mean():.0f},{ys.mean():.0f})")
            for rg in (getattr(engine, "regions", None) or []
                       if engine is not None else []):
                sl = rg.slices()
                L.append(f"- region {rg.name}: x {rg.x0}.."
                         f"{rg.x1 - 1}, y {rg.y0}..{rg.y1 - 1}; "
                         f"footprint cells inside: "
                         f"{int(fp[sl].sum())}")
        except Exception as exc:
            L.append(f"- fire geography unavailable "
                     f"({type(exc).__name__})")
    # the standing brief describes the map and doctrine
    try:
        mb = open(os.path.join(run_dir, "mission_brief.md"),
                  encoding="utf-8").read()
        L.append(mb)
    except OSError:
        pass
    return "\n".join(L)


_PROMPT = """You are the after-action review officer of a wildfire
decision support exercise. The evidence pack of one full run follows.
Write the review a serious incident analyst would write, in clear
prose, no lists of platitudes:
1. VERDICT: success, partial, or failure, with the numbers.
2. ROOT CAUSES ranked by weight (capacity? sensing? doctrine? gate
   behaviour? logistics? geometry? rule-mix capture, i.e. defensive
   rules firing strongly and diluting the offensive channels in the
   weighted mix while the fire grows?), each with its evidence line.
3. WHAT WOULD HAVE BEEN BETTER: concrete, this-map-specific.
Then output the machine part after a line reading exactly `### JSON`:
{"verdict": "success|partial|failure",
 "recommendations": [
  {"type": "setting", "key": "eta|j_threshold|cycle_min|horizon_min|min_gain", "value": number, "why": "..."},
  {"type": "sensor", "kind": "satellite|aerial|ground_camera|in_situ|field_report|public_report", "x": int, "y": int, "why": "..."},
  {"type": "depot", "kind": "depot|helibase", "x": int, "y": int, "why": "..."},
  {"type": "rule", "antecedents": [["concept", "TERM"]], "consequents": [["intervention", 0..1]], "why": "..."},
     TERM is one of VL,L,M,H,VH or '>=L'/'>=M'/'>=H' - a TERM, NEVER a number like '>=0.35',
  {"type": "tune_rule", "name": "existing rule name", "consequents": [["intervention", 0..1]], "why": "..."},
  {"type": "intervention", "name": "...", "composition": [["channel", w]], "rule": {"antecedents": [...], "consequents": [...]}, "why": "..."},
  {"type": "concept", "name": "...", "inputs": [["feature-or-concept", w]], "rule": {"antecedents": [...], "consequents": [...]}, "why": "..."}
 ]}
Recommend ONLY what this map and this run's evidence support (no
water tactics on a dry map, sensors where the blindness actually
was). At most 6 recommendations, each with a one-line why.
PLACEMENT DISCIPLINE: sensor and depot x,y must lie inside the map
and within about 25 cells of the FIRE GEOGRAPHY footprint or the
corridor it threatens; do NOT infer the fire position from sensor
coordinates, use the footprint given. A placement on the quiet side
of the map is refused by the applier.
RULE GROUNDING: rule antecedents must be satisfiable inside the
OBSERVED CONCEPT VALUES ranges of this run ('>=' counts from its
threshold up). A rule whose antecedents never held in this run could
never have fired and is refused; if a concept stayed low while the
fire raged, that mismatch belongs in the prose as a root cause, not
in an antecedent.
KEEP THE PROSE PART UNDER 300 WORDS: a review is judged by what it
pins down, not by its length.
=== EVIDENCE ===
"""


def run_rca(evidence: str, model: str | None = None,
            timeout: float = 240.0):
    """(markdown_report, recommendations dict) via the Claude CLI."""
    import subprocess
    model = model or os.environ.get("DSS_RCA_MODEL", "opus")
    cmd = ["claude", "--model", model, "-p", _PROMPT + evidence,
           "--output-format", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8",
                             errors="replace",
                         timeout=timeout)
    if res.returncode != 0 or not res.stdout.strip():
        raise RuntimeError("RCA model call failed: "
                           + (res.stderr or "no output")[:200])
    inner = res.stdout
    try:
        wrap = json.loads(res.stdout)
        if isinstance(wrap, dict) and "result" in wrap:
            inner = str(wrap["result"])
    except Exception:
        pass
    if "### JSON" in inner:
        report, tail = inner.split("### JSON", 1)
    else:
        report, tail = inner, "{}"
    i, j = tail.find("{"), tail.rfind("}")
    recs = {}
    if 0 <= i < j:
        try:
            recs = json.loads(tail[i:j + 1])
        except Exception:
            recs = {}
    return report.strip(), recs


# ------------------------------------------------- background runner
_JOBS: Dict[str, dict] = {}


def load_saved(run_dir: str):
    """(report, recs) if an RCA was already produced for this run."""
    p = os.path.join(run_dir, "root_cause_analysis.md")
    if not os.path.exists(p):
        return None
    txt = open(p, encoding="utf-8").read()
    if "### JSON" in txt:
        rep, tail = txt.split("### JSON", 1)
        try:
            i, j = tail.find("{"), tail.rfind("}")
            return rep.strip(), json.loads(tail[i:j + 1])
        except Exception:
            return rep.strip(), {}
    return txt, {}


def start_async(run_dir: str, evidence: str,
                model: str | None = None) -> None:
    """Launch the review in a thread; poll() reads the outcome.

    The caller must NEVER block on this. The review takes one to three
    minutes on the deep model, and the whole point of running it in a
    thread is that the fire, the panels and the navigation stay usable
    while it reads the logs.
    """
    import threading
    import time as _t
    if run_dir in _JOBS and _JOBS[run_dir].get("state") == "running":
        return
    job = {"state": "running", "report": None, "recs": None,
           "error": None, "model": str(model or "opus"),
           "started_at": _t.time(), "finished_at": None}
    _JOBS[run_dir] = job

    def _work():
        try:
            rep, recs = run_rca(evidence, model=model)
            save_rca(run_dir, rep, recs)
            job.update(state="done", report=rep, recs=recs,
                       finished_at=_t.time())
        except Exception as exc:
            job.update(state="error", error=str(exc)[:300],
                       finished_at=_t.time())
    threading.Thread(target=_work, daemon=True).start()


def poll(run_dir: str) -> dict:
    """Where the review stands, WITHOUT blocking.

    Falls back to the file on disk: the thread saves the report before it
    marks the job done, so a review that finished while the process was
    restarted (or one produced by an earlier session) is still found
    rather than looking as though it never ran.
    """
    job = _JOBS.get(run_dir)
    if job:
        return job
    saved = load_saved(run_dir)
    if saved:
        return {"state": "done", "report": saved[0], "recs": saved[1],
                "error": None, "model": None, "started_at": None,
                "finished_at": None, "from_disk": True}
    return {"state": "idle"}


def elapsed_s(run_dir: str) -> float:
    """How long the running review has been going, for the status line."""
    import time as _t
    job = _JOBS.get(run_dir) or {}
    t0 = job.get("started_at")
    if not t0:
        return 0.0
    return float((job.get("finished_at") or _t.time()) - t0)


def save_rca(run_dir: str, report: str, recs: dict) -> None:
    with open(os.path.join(run_dir, "root_cause_analysis.md"), "w",
              encoding="utf-8") as f:
        f.write(report + "\n\n### JSON\n"
                + json.dumps(recs, indent=1, ensure_ascii=False))


# --------------------------------------------------------------- apply
def _self_test(prop: dict, rule, engine) -> str | None:
    """MECHANICAL CONSISTENCY CHECK of an applied artifact, the same
    physics-facing questions the actuation audit asks: does the rule
    FIRE when its antecedents hold, does a composition reach the base
    channels, does a clause actuator export its own intensity for the
    allocator, does a new concept actually compute in the hierarchy?
    A product that fails is uninstalled, never left half-wired."""
    import numpy as np
    from .rules import evaluate_rules, INTERVENTIONS
    from .fuzzy import TERMS
    eff = {}
    for cn, t in rule.antecedents:
        tb = t[2:] if str(t).startswith(">=") else str(t)
        vec = np.zeros(len(TERMS))
        vec[TERMS.index(tb) if tb in TERMS else -1] = 1.0
        eff[str(cn)] = vec
    u, trace = evaluate_rules(eff, {}, [rule], macros=engine.macros)
    w = max((wt for r0, wt in trace if r0 is rule), default=0.0)
    if w < 0.3:
        return f"self-test: the rule does not fire (w={w:.2f})"
    ni = prop.get("new_intervention") or {}
    name = ni.get("name")
    if name and ni.get("composition"):
        if not any(u.get(ch, 0.0) > 0.05 for ch in INTERVENTIONS):
            return ("self-test: the composition reaches no base "
                    "channel")
    if name and ni.get("clauses"):
        if u.get(name, 0.0) < 0.3:
            return ("self-test: the clause actuator's intensity does "
                    "not reach the allocator")
    nc = prop.get("new_concept") or {}
    if nc.get("name"):
        from .concepts import infer_concepts
        try:
            act = infer_concepts({}, hierarchy=engine.hierarchy)
            if nc["name"] not in act:
                return ("self-test: the concept does not compute in "
                        "the hierarchy")
        except Exception as exc:
            return f"self-test: hierarchy broke ({type(exc).__name__})"
    return None


def _normalize_terms(prop: dict) -> list:
    """Read a reviewer's numeric threshold as the nearest term.

    Models sometimes write '>=0.35' where the grammar wants '>=L'.
    The intent is clear, so the applier translates instead of
    refusing: nearest TERM_CENTER, '>=' kept. Returns the list of
    translations made, for the applied-message."""
    from .fuzzy import TERM_CENTER
    notes = []
    for pair in (prop.get("antecedents") or []):
        if len(pair) != 2:
            continue
        t = str(pair[1])
        ge = t.startswith(">=")
        body = t[2:] if ge else t
        try:
            x = float(body)
        except ValueError:
            continue
        term = min(TERM_CENTER, key=lambda k: abs(
            TERM_CENTER[k] - max(0.0, min(1.0, x))))
        pair[1] = (">=" + term) if ge else term
        notes.append(f"term {t} read as {pair[1]}")
    return notes


def _placement_error(x: int, y: int, sim) -> str | None:
    """A placement must serve the incident: inside the map, near the
    fire footprint. The analyzed run's own burn scar is the yardstick,
    the same footprint the evidence pack showed the reviewer."""
    if sim is None:
        return None
    try:
        import numpy as np
        fp = (np.asarray(sim.ever_burned)
              | (np.asarray(sim.state.burning) > 0.5))
    except Exception:
        return None
    ny, nx = fp.shape
    if not (0 <= x < nx and 0 <= y < ny):
        return f"({x},{y}) is outside the {nx}x{ny} map"
    if not fp.any():
        return None
    import numpy as np
    ys, xs = np.where(fp)
    d = int(np.min(np.maximum(np.abs(xs - x), np.abs(ys - y))))
    if d > 25:
        return (f"({x},{y}) is {d} cells from the fire footprint; "
                "advice must serve the incident area")
    return None


def apply_recommendations(recs: dict, engine=None, sim=None):
    """Feed the review back into the system.

    Returns (applied, skipped, session_updates, sensors, depots):
    vocabulary and rules go through the SAME validators as live
    stage-3 output and land in the persistent store marked as
    post-incident knowledge; settings come back as session updates
    for the panel; sensor/depot advice comes back as staged rows."""
    from .adapt import (_g1_g2, _validate_package, _install_package,
                        _uninstall_package, _next_rule_name,
                        _availability)
    from .rules import Rule
    applied: List[str] = []
    skipped: List[str] = []
    session: Dict[str, Any] = {}
    sensors: List[dict] = []
    depots: List[dict] = []
    for rec in (recs or {}).get("recommendations", []):
        t = str(rec.get("type", ""))
        why = str(rec.get("why", ""))[:160]
        if t == "setting":
            key = _SETTING_KEYS.get(str(rec.get("key")))
            if key is None:
                skipped.append(f"setting {rec.get('key')}: unknown key")
                continue
            lo, hi = _SETTING_RANGE[str(rec.get("key"))]
            val = float(rec.get("value"))
            if not (lo <= val <= hi):
                skipped.append(
                    f"setting {rec.get('key')}={val}: outside the "
                    f"valid range [{lo}, {hi}], not applied")
                continue
            session[key] = val
            applied.append(f"setting {rec.get('key')} -> "
                           f"{rec.get('value')} ({why})")
        elif t == "sensor":
            if str(rec.get("kind")) not in _SENSOR_KINDS:
                skipped.append(f"sensor {rec.get('kind')}: unknown kind")
                continue
            _pe = _placement_error(int(rec.get("x", -1)),
                                   int(rec.get("y", -1)), sim)
            if _pe:
                skipped.append(f"sensor {rec.get('kind')}: {_pe}")
                continue
            sensors.append(dict(kind=str(rec["kind"]),
                                x=int(rec["x"]), y=int(rec["y"])))
            applied.append(f"sensor {rec['kind']} @ "
                           f"({rec['x']},{rec['y']}) ({why})")
        elif t == "depot":
            if str(rec.get("kind")) not in _DEPOT_KINDS:
                skipped.append(f"depot {rec.get('kind')}: unknown kind")
                continue
            _pe = _placement_error(int(rec.get("x", -1)),
                                   int(rec.get("y", -1)), sim)
            if _pe:
                skipped.append(f"depot {rec.get('kind')}: {_pe}")
                continue
            depots.append(dict(kind=str(rec["kind"]), x=int(rec["x"]),
                               y=int(rec["y"]), radius=5, cap=0.8,
                               avail=1.0, t_disp=8.0,
                               label="RCA advice"))
            applied.append(f"depot {rec['kind']} @ "
                           f"({rec['x']},{rec['y']}) ({why})")
        elif t in ("tune_rule", "rule_update", "modify_rule"):
            if engine is None:
                skipped.append(f"{t}: no engine")
                continue
            tgt = next((r for r in engine.rules
                        if r.name == str(rec.get("name"))), None)
            if tgt is None:
                skipped.append(f"tune_rule {rec.get('name')}: no such "
                               "rule in the active base")
                continue
            from .rules import INTERVENTIONS as _IVS
            cons = rec.get("consequents") or []
            if not cons or any(
                    str(i) not in _IVS and i not in engine.macros
                    or not (0.0 <= float(v) <= 1.0)
                    for i, v in cons):
                skipped.append(f"tune_rule {tgt.name}: consequents "
                               "must cite known channels in [0, 1]")
                continue
            before = [[i, float(v)] for i, v in tgt.consequents]
            tgt.consequents = [(str(i), float(v)) for i, v in cons]
            tgt.note = ((tgt.note + " | " if tgt.note else "")
                        + "evFIS: RCA tune"
                        + (f": {why}" if why else ""))
            gst = getattr(engine, "gstate", None)
            if gst is not None and getattr(engine, "state_path", None):
                gst.append("evfis_rule_modifications",
                           dict(base_rule_id=tgt.name,
                                base_rule_set=engine.seed_profile,
                                modification_type="consequent_update",
                                before={"consequents": before},
                                after={"consequents":
                                       [[i, float(v)] for i, v
                                        in tgt.consequents]},
                                trigger=dict(source="rca")),
                           source_stage=1)
            applied.append(f"tune_rule {tgt.name} ({why})")
        elif t in ("rule", "intervention", "concept"):
            if engine is None:
                skipped.append(f"{t}: no engine to install into")
                continue
            prop = _rec_to_prop(rec, t)
            _tn = _normalize_terms(prop) if prop else []
            if _tn:
                why = (why + " | " if why else "") + "; ".join(_tn)
            err = _g1_g2(prop, engine=engine) or (
                _validate_package(prop, engine)
                if (prop.get("new_concept")
                    or prop.get("new_intervention")) else None)
            if err is None and sim is not None:
                # the MAP must be able to supply the tactic (no water
                # advice on a waterless map)
                err = _availability(prop, sim)
            if err:
                skipped.append(f"{t} {rec.get('name', '')}: {err}")
                continue
            undo = _install_package(prop, engine)
            name = _next_rule_name("G", engine.rules, engine)
            r = Rule(name,
                     [(v, tm) for v, tm in prop["antecedents"]],
                     [(i, float(x)) for i, x in prop["consequents"]],
                     note="post-incident review (RCA)"
                          + (f" | WHY: {why}" if why else ""))
            engine.rules.append(r)
            st_err = _self_test(prop, r, engine)
            if st_err:
                engine.rules.remove(r)
                try:
                    _uninstall_package(undo, engine)
                except Exception:
                    pass
                skipped.append(f"{t} {rec.get('name', '')}: {st_err}; "
                               "uninstalled")
                continue
            _persist(engine, prop, r)
            applied.append(f"{t} -> rule {name} ({why})")
        else:
            skipped.append(f"unknown recommendation type {t!r}")
    return applied, skipped, session, sensors, depots


def _rec_to_prop(rec: dict, t: str) -> dict:
    rule = rec.get("rule") or {}
    prop = dict(antecedents=rule.get("antecedents")
                or rec.get("antecedents") or [],
                consequents=rule.get("consequents")
                or rec.get("consequents") or [],
                rationale=rec.get("why", "post-incident review"))
    if t == "intervention":
        ni = dict(name=rec.get("name"))
        if rec.get("clauses"):
            ni["clauses"] = rec["clauses"]
        else:
            ni["composition"] = rec.get("composition") or []
        prop["new_intervention"] = ni
    if t == "concept":
        prop["new_concept"] = dict(name=rec.get("name"),
                                   level=rec.get("level",
                                                 "intermediate"),
                                   inputs=rec.get("inputs") or [])
    return prop


def _persist(engine, prop: dict, rule) -> None:
    """The RCA products go to the SAME stores the live stages use."""
    gst = getattr(engine, "gstate", None)
    if gst is not None and getattr(engine, "state_path", None):
        nc = prop.get("new_concept")
        ni = prop.get("new_intervention")
        if nc:
            engine.record_package(
                concepts={nc["name"]: engine.hierarchy[nc["name"]]},
                trigger=dict(source="rca"))
        if ni:
            engine.record_package(
                macros={ni["name"]: engine.macros[ni["name"]]},
                trigger=dict(source="rca"))
        gst.append("genai_rules",
                   dict(name=rule.name,
                        antecedents=[list(a) for a in rule.antecedents],
                        consequents=[[i, float(v)] for i, v
                                     in rule.consequents],
                        note=rule.note,
                        depends_on_concepts=[
                            c for c, _t in rule.antecedents
                            if c in (prop.get("new_concept") or {}).get(
                                "name", "")],
                        trigger=dict(source="rca")),
                   source_stage=3)
    ls = getattr(engine, "learned_store", None)
    if ls:
        try:
            from .persist import save_learned
            save_learned(engine.rules, ls,
                         profile=engine.seed_profile, engine=engine,
                         use_evfis=engine.use_evfis, use_genai=True)
        except Exception:
            pass
