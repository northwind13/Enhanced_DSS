"""Active-set resolution: what the DSS actually reasons with this cycle.

The production and the consumption of generated knowledge are two independent
axes. A stage may keep learning and writing to the store while the inference
runs on the untouched factory base (shadow mode), and the accumulated
knowledge may be used while no stage produces anything new (frozen mode).

That only works if the active set is DERIVED, never accumulated in place. The
adaptation stages used to edit the live rule objects and the global partition
registry directly, so turning consumption off could not undo them: the change
had already happened to the objects the inference reads. Here the set is
rebuilt from the baseline every cycle and the stored modifications are
replayed onto it only when their consumption flag allows it.

Two different meanings of OFF, which must not be confused:

  use_stage12_rules = False   REVERT. The baseline rules are present at their
                              factory parameters. The rule count is unchanged.
  use_stage3_rules  = False   DEACTIVATE. The generated rules, concepts and
                              interventions leave the inference entirely, so
                              the counts drop.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from .rules import Rule, INTERVENTIONS
from .concepts import HIERARCHY as BASE_HIERARCHY
from .concepts import DECISION_CONCEPTS as BASE_DECISION_CONCEPTS


class ActiveSet:
    """What one cycle reasons with, plus why anything was left out."""

    def __init__(self, rules, hierarchy, decision_concepts, macros,
                 warnings, idle=False):
        self.rules: List[Rule] = rules
        self.hierarchy: Dict[str, Any] = hierarchy
        self.decision_concepts: List[str] = decision_concepts
        self.macros: Dict[str, Any] = macros
        self.warnings: List[str] = warnings
        self.idle: bool = idle
        # how many stored tunings actually found their target this cycle.
        # The UI reports it, so a tuning that silently applies to nothing
        # cannot be mistaken for a tuning that took effect.
        self.applied_mods: int = 0

    @property
    def counts(self) -> Dict[str, int]:
        return {"rules": len(self.rules),
                "concepts": len(self.hierarchy),
                "interventions": len(INTERVENTIONS) + len(self.macros)}


# --------------------------------------------------------------- evFIS side
def _apply_modification(mod: Dict[str, Any], rules_by_name: Dict[str, Rule],
                        rules: List[Rule], registry, warnings: List[str],
                        use_stage3: bool = True) -> bool:
    """Replay ONE stored evFIS modification onto the freshly built baseline.

    Returns True when the modification actually landed on something.
    """
    kind = str(mod.get("modification_type", ""))
    after = mod.get("after") or {}
    rid = mod.get("base_rule_id")

    if kind == "consequent_update":
        r = rules_by_name.get(rid)
        if r is None:
            # NAME THE CAUSE. "not in the active baseline" reads like a
            # corrupt store; usually it is a plain consequence of the
            # toggles. evFIS tunes whatever fires, including the rules
            # stage 3 generated, so switching stage 3 consumption off also
            # leaves those tunings with nothing to apply to.
            _rid = str(rid or "")
            if _rid[:1] == "G" and not use_stage3:
                warnings.append(
                    f"{mod.get('id')}: evFIS had tuned the GENERATED rule "
                    f"{_rid}, which is not loaded because 'Use stage ③ "
                    f"rules' is off, so the tuning is skipped. Turn stage ③ "
                    f"consumption on to get both back.")
            elif _rid[:1] == "G":
                warnings.append(
                    f"{mod.get('id')}: the tuning targets the generated rule "
                    f"{_rid}, which stage ③ did not bring back this time "
                    f"(its record may have been pruned or its dependencies "
                    f"are unresolved), so the tuning is skipped.")
            elif _rid[:1] == "A":
                warnings.append(
                    f"{mod.get('id')}: the tuning targets the stage ② rule "
                    f"{_rid}, which is not in the active set (its own record "
                    f"may have been pruned), so it is skipped.")
            else:
                warnings.append(
                    f"{mod.get('id')}: rule {_rid} is not in the active seed "
                    f"profile, so this tuning does not apply to it. It was "
                    f"recorded against a different seed base.")
            return False
        r.consequents = [(str(i), float(v))
                         for i, v in after.get("consequents", [])]
        r.note = (r.note + " | " if r.note else "") + "evFIS: tuned"
        r.tuned_from_store = True
        return True

    elif kind in ("membership_shift", "term_insert"):
        var = mod.get("variable")
        part = after.get("partition")
        if not var or not part:
            warnings.append(f"{mod.get('id')}: {kind} without a variable or "
                            f"partition, skipped")
            return False
        for term, abcd in part.items():
            try:
                registry.set_term(var, term, abcd)
            except Exception as exc:
                warnings.append(f"{mod.get('id')}: term {var}.{term} could "
                                f"not be applied ({type(exc).__name__})")
                return False
        return True

    elif kind == "rule_add":
        spec = after.get("rule") or {}
        name = str(spec.get("name") or mod.get("id"))
        if name in rules_by_name:
            return False
        r = Rule(name,
                 [tuple(a) for a in spec.get("antecedents", [])],
                 [(str(i), float(v)) for i, v in spec.get("consequents", [])],
                 note=(spec.get("note") or "") + " | restored (stage 2)",
                 strength=float(spec.get("strength", 0.0)))
        r.from_store = "stage2"
        rules.append(r)
        rules_by_name[name] = r
        return True

    else:
        warnings.append(f"{mod.get('id')}: unknown modification type "
                        f"{kind!r}, skipped")
    return False


# --------------------------------------------------------------- stage 3
def _install_concepts(state, hierarchy, decision_concepts, warnings):
    for c in state.sorted_records("genai_concepts"):
        name = c.get("name")
        if not name:
            continue
        ins = []
        for it in c.get("inputs") or []:
            if isinstance(it, dict):
                ins.append((str(it.get("name")), float(it.get("weight", 0.0))))
            else:
                ins.append((str(it[0]), float(it[1])))
        hierarchy[name] = (int(c.get("layer", c.get("level", 2))), ins)
        if c.get("decision") and name not in decision_concepts:
            decision_concepts.append(name)


def _install_macros(state, macros):
    for m in state.sorted_records("genai_interventions"):
        name = m.get("name")
        if not name:
            continue
        comp = []
        for it in m.get("composition") or []:
            if isinstance(it, dict):
                comp.append((str(it.get("channel")), float(it.get("weight"))))
            else:
                comp.append((str(it[0]), float(it[1])))
        macros[name] = dict(composition=comp)
        if m.get("clauses"):
            macros[name]["clauses"] = list(m["clauses"])


def _install_genai_rules(state, rules, rules_by_name, hierarchy, macros,
                         warnings):
    for r in state.sorted_records("genai_rules"):
        name = str(r.get("name") or r.get("id"))
        if name in rules_by_name:
            continue
        missing = [c for c in (r.get("depends_on_concepts") or [])
                   if c not in hierarchy]
        if missing:
            # never dropped in silence: an unresolved dependency is a real
            # finding about the store, not a detail to hide
            warnings.append(f"{name}: depends on {', '.join(missing)}, which "
                            f"is not available, so the rule is not loaded")
            continue
        ants = [tuple(a) if not isinstance(a, dict)
                else (str(a.get("concept")), str(a.get("term")))
                for a in r.get("antecedents", r.get("antecedent", []))]
        cons = []
        for c in r.get("consequents", r.get("consequent", [])):
            if isinstance(c, dict):
                cons.append((str(c.get("channel")), float(c.get("value", 0))))
            else:
                cons.append((str(c[0]), float(c[1])))
        bad = [i for i, _v in cons
               if i not in INTERVENTIONS and i not in macros]
        if bad:
            warnings.append(f"{name}: orders {', '.join(bad)}, which is "
                            f"neither a base channel nor an available macro, "
                            f"so the rule is not loaded")
            continue
        rr = Rule(name, ants, cons,
                  note=(r.get("note") or "adaptation-born: generative")
                  + " | restored (stage 3)",
                  strength=float(r.get("strength", 0.0)))
        rr.from_store = "stage3"
        rules.append(rr)
        rules_by_name[name] = rr


# --------------------------------------------------------------- entry point
def resolve_active_set(state, make_rules, registry,
                       reset_partitions) -> ActiveSet:
    """Build this cycle's rules, concepts and interventions from the baseline.

    make_rules(profile)  returns a FRESH copy of a seed profile
    registry             the global partition registry
    reset_partitions()   returns every partition to its factory value
    """
    flags = state.flags
    warnings: List[str] = []

    if not flags.get("dss_active", True):
        return ActiveSet([], dict(BASE_HIERARCHY),
                         list(BASE_DECISION_CONCEPTS), {}, warnings,
                         idle=True)

    # BASELINE, always rebuilt: this is what makes revert possible at all
    rules = make_rules(str(flags.get("active_rule_set", "minimal5")))
    rules_by_name = {r.name: r for r in rules}
    hierarchy = copy.deepcopy(dict(BASE_HIERARCHY))
    decision_concepts = list(BASE_DECISION_CONCEPTS)
    macros: Dict[str, Any] = {}
    # the partition registry is global state, so it has to be returned to
    # factory value here as well, or a term inserted in an earlier cycle
    # would survive a consumption flag that is now off
    reset_partitions()

    # STAGE 3 FIRST, THEN THE evFIS REPLAY. evFIS tunes whatever fires,
    # including the rules stage 3 generated, so a stored tuning can name a
    # G# rule. Replaying the tunings before those rules existed meant such a
    # tuning could NEVER find its target: it was skipped every time, even
    # with stage 3 consumption fully on, and the warning blamed the toggle.
    _use3 = flags.get("use_stage3_rules", True)
    if _use3:
        _install_concepts(state, hierarchy, decision_concepts, warnings)
        _install_macros(state, macros)
        _install_genai_rules(state, rules, rules_by_name, hierarchy, macros,
                             warnings)

    applied = 0
    if flags.get("use_stage12_rules", True):
        for mod in state.sorted_records("evfis_rule_modifications"):
            if _apply_modification(mod, rules_by_name, rules, registry,
                                   warnings, use_stage3=_use3):
                applied += 1

    aset = ActiveSet(rules, hierarchy, decision_concepts, macros, warnings)
    aset.applied_mods = applied
    return aset


def evfis_chain_set(state, make_rules, registry, reset_partitions
                    ) -> ActiveSet:
    """The base evFIS itself works on: baseline plus the WHOLE modification
    chain, regardless of the consumption flag.

    Shadow mode depends on this. If the chain were cut whenever consumption
    is off, the stored `before` values would stop describing the state they
    were taken from, and both the reverse-order revert and the forward replay
    would become inconsistent. The consumption flag closes what the inference
    sees, not what the learner remembers.
    """
    rules = make_rules(str(state.flags.get("active_rule_set", "minimal5")))
    rules_by_name = {r.name: r for r in rules}
    warnings: List[str] = []
    reset_partitions()
    for mod in state.sorted_records("evfis_rule_modifications"):
        _apply_modification(mod, rules_by_name, rules, registry, warnings)
    return ActiveSet(rules, copy.deepcopy(dict(BASE_HIERARCHY)),
                     list(BASE_DECISION_CONCEPTS), {}, warnings)
