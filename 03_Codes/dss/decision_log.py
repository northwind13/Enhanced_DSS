"""Decision log: the traceability record of Layer 4.

One record per (decision cycle, region) plus one coordinator record per
cycle. The log carries everything needed to answer "which decisions
were taken, why, and what would have happened without them": the
feature vector and its confidences, the gated concepts, the fired rules
with their strengths, the adaptation outcome, the forecast costs and
the applied intensities. The analysis view replays counterfactuals by
cloning the live simulator, rewinding the CLONE to the decision step
and re-running history without the selected orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DecisionRecord:
    step: int
    region: str
    features: Dict[str, float]
    feature_conf: Dict[str, float]
    concepts: Dict[str, float]           # crisp effective activations
    gates: Dict[str, float]
    fired: List[tuple]                   # (rule name, strength)
    intensities: Dict[str, float]
    quality: float
    failsafe: bool
    stage: int = 0                       # 0 = rules as-is (accepted)
    stage_tried: int = 0                 # which stage the controller selected
    stage_detail: str = ""
    ctrl_bucket: str = ""
    j_forecast: float = 0.0
    j_noaction: float = 0.0
    j_threshold: float = 0.0
    coord_share: float = 1.0
    attended: bool = True


class DecisionLog:
    def __init__(self, maxlen: int = 400):
        self.maxlen = int(maxlen)
        self.records: List[DecisionRecord] = []

    def add(self, rec: DecisionRecord) -> None:
        self.records.append(rec)
        if len(self.records) > self.maxlen:
            self.records = self.records[-self.maxlen:]

    def cycles(self) -> List[int]:
        return sorted({r.step for r in self.records})

    def at(self, step: int) -> List[DecisionRecord]:
        return [r for r in self.records if r.step == step]

    @staticmethod
    def stage_story(rec: DecisionRecord) -> str:
        """One-line provenance in the operator's language."""
        names = {1: "evFIS", 2: "resolution increase", 3: "GenAI"}
        if rec.stage_tried == 0:
            return "orders came from the seed rule base (rules as-is)"
        line = (f"stage controller selected stage {rec.stage_tried} "
                f"[{names[rec.stage_tried]}] "
                f"(deficit bucket: {rec.ctrl_bucket or '-'}) \u2192 ")
        if rec.stage:
            line += f"ACCEPTED: {rec.stage_detail}"
        else:
            line += f"rejected ({rec.stage_detail or 'no improvement'})"
        return line

    def why(self, rec: DecisionRecord) -> List[str]:
        """Backward trace of one decision, human readable."""
        out = [f"step {rec.step} \u00b7 {rec.region}",
               self.stage_story(rec)]
        out.append("concepts (gated): " + ", ".join(
            f"{k.replace('_', ' ')}={v:.2f} (\u03b3 {rec.gates.get(k, 1):.2f})"
            for k, v in rec.concepts.items()))
        if rec.fired:
            out.append("fired rules: " + ", ".join(
                f"{n} [{w:.2f}]" for n, w in rec.fired[:8]))
        out.append("orders: " + ", ".join(
            f"{k.replace('_', ' ')}={v:.2f}"
            for k, v in rec.intensities.items() if v > 0.02) or "none")
        out.append(f"forecast J={rec.j_forecast:.3f} vs no-action "
                   f"{rec.j_noaction:.3f} (J_TH {rec.j_threshold:.2f})"
                   + (" \u00b7 FAIL-SAFE attenuated" if rec.failsafe
                      else ""))
        return out


class RunLogger:
    """Persistent run log: every simulation step and every decision
    cycle goes to disk, so runs can be analyzed after the fact.

    <dir>/steps.csv       step, t_min, burning, burned, J terms
    <dir>/decisions.jsonl one line per DecisionRecord (+ stage story)
    <dir>/world.json.gz   full world snapshot -> replayable
    """

    def __init__(self, root: str, tag: str = "run"):
        import os
        import time
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, f"dss_{tag}_{stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self._steps = open(os.path.join(self.dir, "steps.csv"), "a")
        self._steps.write("step,t_min,burning,burned,j_total,"
                          "j_physical,j_burn,"
                          "j_asset,j_pop,j_resp,wws_mean,prec_mean,"
                          "fmoist_mean,alloc_cells,alloc_rcap_sum\n")
        self._dec = open(os.path.join(self.dir, "decisions.jsonl"), "a")
        self._csv = open(os.path.join(self.dir, "decisions.csv"), "a",
                         encoding="utf-8-sig")
        self._csv.write(
            "step;t_min;region;story;stage_tried;stage_accepted;"
            "fired_rules;suppression;deployment;containment;"
            "protection;evacuation;warning;quality;gate_min;"
            "J_forecast;J_noaction;attended\n")
        self._glb = open(os.path.join(self.dir, "global.csv"), "a",
                         encoding="utf-8-sig")
        self._glb.write("step;t_min;hotspot;attended;shares;"
                        "statement\n")

    def save_world(self, world) -> None:
        """Full world snapshot (gzipped JSON): together with meta.json
        (engine + weather settings) the run can be REPLAYED bit for
        bit — rebuild the World, re-create the engine, run."""
        import gzip
        import json
        import os
        with gzip.open(os.path.join(self.dir, "world.json.gz"),
                       "wt") as f:
            json.dump(world.to_dict(), f)

    def write_meta(self, meta: dict) -> None:
        """Scenario + engine + fleet context, once per run (training
        data needs the experiment description, not only traces)."""
        import json
        import os
        with open(os.path.join(self.dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)

    def log_step(self, sim, rep=None, override=None) -> None:
        import numpy as np
        s = sim.state
        w = sim.world
        t = s.step * float(getattr(sim.cfg, "step_minutes", 1.0))
        vals = ["", "", "", "", "", ""]
        if rep is not None:
            vals = [f"{rep.j_total:.5f}",
                    f"{getattr(rep, 'j_physical', 0.0):.5f}",
                    f"{rep.j_burn:.5f}",
                    f"{rep.j_asset:.5f}", f"{rep.j_pop:.5f}",
                    f"{rep.j_resp:.5f}"]
        if override is not None:
            ac = int((override.rcap > 1e-6).sum())
            asum = float(override.rcap.sum())
        else:
            ac, asum = 0, 0.0
        self._steps.write(
            f"{s.step},{t:.1f},{int((s.burning > 0.5).sum())},"
            f"{int(sim.ever_burned.sum())}," + ",".join(vals) + ","
            f"{float(w.meteo.wws.mean()):.2f},"
            f"{float(w.meteo.prec.mean()):.2f},"
            f"{float(w.fuel.fmoist.mean()):.3f},{ac},{asum:.1f}\n")
        self._steps.flush()

    def log_global(self, step, t_min, g: dict) -> None:
        """One row per decision cycle: WHAT the Global DSS decided."""
        try:
            self._glb.write(
                f"{step};{t_min};{g.get('hotspot')};"
                + "|".join(g.get("attended", [])) + ";"
                + "|".join(f"{k}={v}" for k, v in
                           (g.get("shares") or {}).items()) + ";"
                + str(g.get("statement", "")).replace(";", ",")
                + "\n")
            self._glb.flush()
        except Exception:
            pass

    def log_cycle(self, cyc: dict) -> None:
        """One JSON object per decision cycle: simulation state,
        costs, pool, sensor status, forecast, stage-controller choice with its
        value table, the adaptation attempt with every trial and its
        reject reason, and the full per-region z/concept/order
        detail. This is the training-grade record."""
        import json
        import os
        if not hasattr(self, "_cyc"):
            self._cyc = open(os.path.join(self.dir,
                                          "cycles.jsonl"), "a")
        self._cyc.write(json.dumps(cyc, default=str) + "\n")
        self._cyc.flush()

    def log_decision(self, rec: DecisionRecord, story: str,
                     step_minutes: float = 1.0) -> None:
        import json
        from dataclasses import asdict
        d = asdict(rec)
        d["story"] = story
        self._dec.write(json.dumps(d) + "\n")
        self._dec.flush()
        u = rec.intensities
        fired = " ".join(f"{n}[{w:.2f}]" for n, w in rec.fired[:6])
        self._csv.write(
            f"{rec.step};{rec.step * step_minutes:.0f};{rec.region};"
            f"{story};{rec.stage_tried};{rec.stage};{fired};"
            f"{u.get('suppression_effort', 0):.2f};"
            f"{u.get('resource_deployment', 0):.2f};"
            f"{u.get('containment_line', 0):.2f};"
            f"{u.get('asset_protection', 0):.2f};"
            f"{u.get('evacuation', 0):.2f};"
            f"{u.get('public_warning', 0):.2f};"
            f"{rec.quality:.2f};"
            f"{min(rec.gates.values()) if rec.gates else 1.0:.2f};"
            f"{rec.j_forecast:.4f};{rec.j_noaction:.4f};"
            f"{int(rec.attended)}\n")
        self._csv.flush()
