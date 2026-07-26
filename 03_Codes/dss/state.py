"""Generated-knowledge store: one JSON file, append-only, atomically written.

The store separates two things the old learned-rule file conflated:

  BASELINE      the seed rule sets (minimal5 / core_doctrine / rule42) and the
                six base interventions. These live in the code and are the
                factory value. They are never written here and a wipe never
                deletes them, it only returns them to factory value.

  RUNTIME       everything the adaptation stages produce: evFIS modifications
                of baseline rules, and the rules, concepts and interventions
                stage 3 generates. These live in this file and a wipe clears
                them.

Every record carries `origin`, `source_stage`, a globally monotonic `seq` and
the flags it was produced under. Two things depend on that:

  * WIPE reverts evFIS modifications in REVERSE seq order, so each `before`
    value is restored against the state that produced it.
  * RESTART replays them in FORWARD seq order, so a restarted process holds
    exactly the base it held before shutdown.

Records carry no `active` field on purpose. Whether a record is used is
derived from `origin` and the runtime flags on every cycle, so there is one
source of truth and no second state to keep in sync.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import os
from typing import Any, Dict, List

SCHEMA_VERSION = "1.0"

SECTIONS = ("evfis_rule_modifications", "genai_rules",
            "genai_concepts", "genai_interventions")

# THE PROPOSAL LEDGER IS EVIDENCE, NOT KNOWLEDGE, so it is deliberately NOT
# one of SECTIONS: nothing here is ever resolved into the active rule set, it
# carries no replay order and it does not answer to the consumption flags.
# It exists because the failures used to be thrown away. Every stage 3
# proposal was judged by a gate and then discarded, so across 62 runs 155
# rejections, each with its verdict and its revisions, left no trace, and
# there was nothing for a retrieval step to retrieve. A ledger of what was
# proposed, what the gate said and what the simulation measured is the
# corpus that grounding the proposer needs.
LEDGER = "genai_proposals"
MAX_LEDGER = 2000            # oldest entries drop out first

# which stage owns each section, and therefore which flags gate it
_ORIGIN = {"evfis_rule_modifications": "evfis",
           "genai_rules": "genai",
           "genai_concepts": "genai",
           "genai_interventions": "genai"}

_ID_PREFIX = {"evfis_rule_modifications": "evfis_mod",
              "genai_rules": "gen_rule",
              "genai_concepts": "gen_concept",
              "genai_interventions": "gen_interv"}

DEFAULT_FLAGS = {
    "dss_active": True,
    "active_rule_set": "minimal5",
    "evfis_active": True,
    "genai_active": True,
    "use_stage12_rules": True,
    "use_stage3_rules": True,
}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_state(active_rule_set: str = "minimal5") -> Dict[str, Any]:
    flags = dict(DEFAULT_FLAGS)
    flags["active_rule_set"] = active_rule_set
    d: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "last_updated": _now(),
        "baseline_ref": {
            "rule_sets": ["minimal5", "core_doctrine", "rule42"],
            "intervention_count": 6,
        },
        "runtime_flags": flags,
        # THE STAGE CONTROLLER'S EXPERIENCE, kept across runs. It learns
        # which adaptation stage pays off in which situation class, and one
        # fire only offers a few dozen attempts, far too few to converge. It
        # is scoped by `map_key`: the value of a stage is a property of the
        # terrain and the assets, so a different map starts from scratch.
        "stage_controller": {"map_key": None, "maps": {}},
        LEDGER: [],
    }
    for s in SECTIONS:
        d[s] = []
    return d


def config_id(flags: Dict[str, Any]) -> str:
    """The ablation label of a flag combination, e.g. DSS1-EV0-GA0-U12:1-U3:1.

    Every run of the toggle matrix is one experiment configuration, so it gets
    a stable name that a results table can group by."""
    b = lambda k: 1 if flags.get(k) else 0          # noqa: E731
    return (f"DSS{b('dss_active')}-EV{b('evfis_active')}"
            f"-GA{b('genai_active')}"
            f"-U12:{b('use_stage12_rules')}-U3:{b('use_stage3_rules')}")


class GeneratedState:
    """The runtime-generated knowledge of one installation."""

    def __init__(self, path: str, data: Dict[str, Any] | None = None):
        self.path = str(path)
        self.data = data if data is not None else empty_state()
        self.warnings: List[str] = []

    # ------------------------------------------------------------ load/save
    @classmethod
    def load(cls, path: str, active_rule_set: str = "minimal5"
             ) -> "GeneratedState":
        """Read the store, or start an empty one. A store that cannot be
        parsed is NOT silently replaced: the corrupt file is kept beside the
        new one so nothing is lost without a trace."""
        st = cls(path, empty_state(active_rule_set))
        if not os.path.exists(path):
            return st
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as exc:
            _bad = path + ".corrupt"
            try:
                os.replace(path, _bad)
            except OSError:
                pass
            st.warnings.append(
                f"the store could not be read ({type(exc).__name__}); it was "
                f"moved to {os.path.basename(_bad)} and an empty one started")
            return st
        if str(d.get("schema_version", "")) != SCHEMA_VERSION:
            st.warnings.append(
                f"store schema {d.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION!r}; the file was kept but not loaded")
            return st
        base = empty_state(active_rule_set)
        base.update({k: v for k, v in d.items() if k in base or k in SECTIONS})
        for s in SECTIONS:
            base[s] = list(d.get(s) or [])
        base["runtime_flags"] = {**DEFAULT_FLAGS,
                                 **(d.get("runtime_flags") or {})}
        _sc = d.get("stage_controller")
        if isinstance(_sc, dict):
            # kept verbatim; _sc() migrates the old single-table layout on
            # first use, so an existing store is not thrown away
            base["stage_controller"] = dict(_sc)
        base[LEDGER] = list(d.get(LEDGER) or [])
        st.data = base
        return st

    def save(self) -> None:
        """Atomic write: a half-written store would brick the next start."""
        self.data["last_updated"] = _now()
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # --------------------------------------------------------------- flags
    @property
    def flags(self) -> Dict[str, Any]:
        return self.data["runtime_flags"]

    def set_flags(self, **kw) -> None:
        self.flags.update({k: v for k, v in kw.items() if v is not None})

    @property
    def config_id(self) -> str:
        return config_id(self.flags)

    # -------------------------------------------------------------- records
    def records(self, section: str) -> List[Dict[str, Any]]:
        return self.data.setdefault(section, [])

    def next_seq(self) -> int:
        """Global and monotonic across ALL sections: replay order is one
        sequence, not four independent ones."""
        n = 0
        for s in SECTIONS:
            for r in self.data.get(s, []):
                n = max(n, int(r.get("seq", 0)))
        return n + 1

    def append(self, section: str, record: Dict[str, Any],
               source_stage: int, save: bool = True) -> Dict[str, Any]:
        """Stamp a produced record and append it. The stamp is what makes the
        record replayable and auditable later."""
        if section not in SECTIONS:
            raise ValueError(f"unknown section {section!r}")
        seq = self.next_seq()
        rec = dict(record)
        rec.setdefault("id", f"{_ID_PREFIX[section]}_{seq:04d}")
        rec["seq"] = seq
        rec["origin"] = _ORIGIN[section]
        rec["source_stage"] = int(source_stage)
        rec["produced_under_flags"] = {
            "use_stage12_rules": bool(self.flags.get("use_stage12_rules")),
            "use_stage3_rules": bool(self.flags.get("use_stage3_rules")),
            "evfis_active": bool(self.flags.get("evfis_active")),
            "genai_active": bool(self.flags.get("genai_active")),
        }
        rec.setdefault("timestamp", _now())
        self.records(section).append(rec)
        if save:
            self.save()
        return rec

    def sorted_records(self, section: str) -> List[Dict[str, Any]]:
        return sorted(self.records(section), key=lambda r: int(r.get("seq", 0)))

    # ----------------------------------------------------------------- wipe
    def wipe(self, backup: bool = True) -> Dict[str, int]:
        """Full factory reset of the generated knowledge.

        The evFIS modifications are handed back in REVERSE seq order so the
        caller can revert them against the state that produced them. The
        production flags are turned off: leaving them on would let the stages
        dirty the clean state within seconds, before the user has looked at
        it. The consumption flags and dss_active are kept, because those
        express what the user wants to USE, which a wipe does not change."""
        counts = {s: len(self.records(s)) for s in SECTIONS}
        if backup:
            try:
                bak = os.path.splitext(self.path)[0] + ".bak.json"
                with open(bak, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=1, ensure_ascii=False)
            except OSError:
                pass
        for s in SECTIONS:
            self.data[s] = []
        # the controller's value table is learned knowledge too: leaving it
        # behind would let a wiped store still steer the stage choice with
        # experience gathered from rules that no longer exist
        # the ledger SURVIVES a wipe: it records what was tried and what the
        # gates measured, which stays true after the rules are reset. Use
        # clear_ledger() to drop it deliberately.
        counts["ledger_kept"] = len(self.proposals())
        counts["stage_controller_entries"] = sum(
            len(v.get("q") or {})
            for v in ((self.data.get("stage_controller") or {})
                      .get("maps") or {}).values()) + len(
            (self.data.get("stage_controller") or {}).get("q") or {})
        self.data["stage_controller"] = {"map_key": None, "maps": {}}
        self.flags["evfis_active"] = False
        self.flags["genai_active"] = False
        self.save()
        return counts

    # ------------------------------------------------------ proposal ledger
    def proposals(self) -> List[Dict[str, Any]]:
        return self.data.setdefault(LEDGER, [])

    def append_proposal(self, record: Dict[str, Any],
                        save: bool = True) -> Dict[str, Any]:
        """File one stage 3 attempt, accepted or not.

        Its sequence is its OWN, independent of the knowledge sequence: a
        ledger entry is not replayed and must not shift the order in which
        rules and modifications are restored.
        """
        led = self.proposals()
        rec = dict(record)
        rec["lseq"] = (int(led[-1].get("lseq", 0)) + 1) if led else 1
        rec.setdefault("id", f"prop_{rec['lseq']:05d}")
        rec.setdefault("timestamp", _now())
        rec["config"] = config_id(self.flags)
        led.append(rec)
        if len(led) > MAX_LEDGER:
            del led[:len(led) - MAX_LEDGER]
        if save:
            self.save()
        return rec

    def clear_ledger(self, save: bool = True) -> int:
        """Drop the evidence. Separate from wipe on purpose: a wipe resets
        the generated KNOWLEDGE, and the record of what was tried and what
        the gates said about it stays true either way."""
        n = len(self.proposals())
        self.data[LEDGER] = []
        if save:
            self.save()
        return n

    def ledger_stats(self) -> Dict[str, Any]:
        """What the corpus holds, for the panel and for the retrieval step."""
        led = self.proposals()
        by_gate: Dict[str, int] = {}
        acc = 0
        for r in led:
            if r.get("accepted"):
                acc += 1
            else:
                g = str(r.get("gate") or "unknown")
                by_gate[g] = by_gate.get(g, 0) + 1
        return {"entries": len(led), "accepted": acc,
                "rejected_by_gate": by_gate}

    # --------------------------------------------- stage controller memory
    MAX_CONTROLLER_MAPS = 12     # archived scenes, oldest evicted first

    def _sc(self) -> Dict[str, Any]:
        """The controller memory, migrated from the single-table layout.

        The first version kept ONE table tagged with a map key, so returning
        to a scene the DSS had already worked meant starting from zero, even
        though the experience had been paid for. Now every scene keeps its
        own table and the tag only says which one is current.
        """
        sc = self.data.setdefault(
            "stage_controller", {"map_key": None, "q": {}, "updates": 0})
        if "maps" not in sc:
            sc["maps"] = {}
            _k = sc.get("map_key")
            if _k and (sc.get("q") or {}):
                sc["maps"][str(_k)] = {"q": dict(sc["q"]),
                                       "updates": int(sc.get("updates") or 0),
                                       "seen": _now()}
            sc.pop("q", None)
            sc.pop("updates", None)
        return sc

    def _evict(self, sc: Dict[str, Any]) -> None:
        """Bound the archive: keep the most recently used scenes."""
        maps = sc.get("maps") or {}
        if len(maps) <= self.MAX_CONTROLLER_MAPS:
            return
        for k in sorted(maps, key=lambda k: str(maps[k].get("seen") or ""))[
                :len(maps) - self.MAX_CONTROLLER_MAPS]:
            maps.pop(k, None)

    def load_controller(self, controller, map_key: str | None) -> bool:
        """Restore the value table THIS map earned, if it has one.

        Each scene keeps its own table: what stage 2 is worth on a wooded
        ridge says nothing about what it is worth on a coastal town, so the
        tables never mix. But a scene the DSS has already learned on is not
        forgotten just because another map was opened in between.
        """
        sc = self._sc()
        sc["map_key"] = map_key
        if map_key is None:
            return False
        entry = (sc.get("maps") or {}).get(str(map_key))
        if not entry:
            return False
        restored = False
        for k, v in (entry.get("q") or {}).items():
            b, _, st_ = str(k).rpartition("/")
            if not b or not st_.isdigit():
                continue
            controller.q[(b, int(st_))] = float(v)
            restored = True
        entry["seen"] = _now()
        return restored

    def save_controller(self, controller, map_key: str | None = None,
                        save: bool = True) -> None:
        """Write this scene's value table back, keyed "bucket/stage"."""
        sc = self._sc()
        key = str(map_key if map_key is not None else sc.get("map_key") or "")
        if not key:
            return                      # no scene identity, nothing to file
        sc["map_key"] = key
        sc.setdefault("maps", {})[key] = {
            "q": {f"{b}/{st_}": round(float(v), 6)
                  for (b, st_), v in controller.q.items()},
            "updates": int((sc.get("maps", {}).get(key, {})
                            .get("updates") or 0)) + 1,
            "seen": _now()}
        self._evict(sc)
        if save:
            self.save()

    def controller_maps(self) -> Dict[str, int]:
        """How many learned values each remembered scene holds."""
        return {k: len(v.get("q") or {})
                for k, v in (self._sc().get("maps") or {}).items()}

    def reverted_modifications(self) -> List[Dict[str, Any]]:
        """evFIS modifications newest first, for reverting before a wipe."""
        return list(reversed(self.sorted_records(
            "evfis_rule_modifications")))

    # ------------------------------------------------------------ reporting
    def counts(self) -> Dict[str, int]:
        return {s: len(self.records(s)) for s in SECTIONS}

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)
