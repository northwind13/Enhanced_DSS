"""The offline stand-in for the generative stage.

Studies that measure the gate chain have to run without a language
model behind them, and they have to give the same answer twice. This
proposer is what stands in its place: it reads the dominant terms out
of the situation brief, aims a strong rule at them, and answers a named
rejection the way the revision protocol asks. Nothing here is exempt
from G1 to G5, and the funnel records the source as "template", so a
product that survives did so on the same terms a live one would.
"""
from __future__ import annotations


def make_template_proposer(seed: int):
    """Deterministic stand-in for the live model in stage 3."""
    state = {"n": 0}

    def _dom(text):
        import re
        m = re.search(r"Current dominant terms: ([^\n]+)", text)
        out = {}
        if m:
            for kv in m.group(1).split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out[k.strip()] = v.strip()
        return out

    def propose(situation, timeout=None, engine=None, mission=""):
        state["n"] += 1
        d = _dom(situation)
        thr = d.get("fire_threat_level", "M")
        aer = d.get("asset_exposure_risk", "M")
        feas = d.get("suppression_feasibility", "M")
        urg = d.get("intervention_urgency", "M")
        # POOL SATURATED (brief line or a G2d rejection): a competent
        # officer stops ordering more physical work the budget cannot
        # fund. What still helps is sustained capacity from water when
        # the map carries it, otherwise the non-spending channels.
        if ("POOL SATURATED" in situation or "G2d" in situation):
            ants = [["fire_threat_level",
                     (">=" + thr) if thr in ("L", "M", "H") else thr],
                    ["suppression_feasibility", feas]]
            if "a lake/sea lies" in situation:
                return {"antecedents": ants,
                        "consequents": [["water_drafting", 0.9]],
                        "note": "template: pool saturated; raise "
                                "sustained capacity from the water "
                                "body instead of re-dividing the "
                                "budget"}
            return {"antecedents": [["asset_exposure_risk",
                                     (">=" + aer) if aer in
                                     ("L", "M", "H") else aer],
                                    ["intervention_urgency", urg]],
                    "consequents": [["public_warning", 1.0],
                                    ["evacuation", 0.7]],
                    "note": "template: pool saturated and no water "
                            "body; spend nothing, protect people"}
        dup = ("duplicate cell" in situation
               or "G2 duplicate" in situation)
        pkg = dup or (state["n"] % 3 == 0)
        if pkg:
            # a composite package: defense as ONE act (the escape the
            # duplicate-cell gate names)
            return {
                "antecedents": [["asset_exposure_risk", aer],
                                ["fire_threat_level", ">=" + thr
                                 if thr in ("L", "M", "H") else thr]],
                "consequents": [["town_shield", 1.0]],
                "new_intervention": {
                    "name": "town_shield",
                    "composition": [["asset_protection", 1.0],
                                    ["containment_line", 0.8]]},
                "note": "template: defense composite for the exposed "
                        "settlement edge"}
        if feas in ("VL", "L"):
            cons = [["containment_line", 0.9],
                    ["resource_deployment", 0.8],
                    ["asset_protection", 0.7]]
        elif aer in ("H", "VH"):
            cons = [["asset_protection", 0.9],
                    ["evacuation", 0.8],
                    ["containment_line", 0.7]]
        else:
            cons = [["suppression_effort", 0.95],
                    ["resource_deployment", 0.85]]
        ants = [["fire_threat_level",
                 (">=" + thr) if thr in ("L", "M", "H") else thr],
                ["intervention_urgency", urg]]
        return {"antecedents": ants, "consequents": cons,
                "note": "template: strongest sensible answer to the "
                        "present dominant terms"}

    return propose
