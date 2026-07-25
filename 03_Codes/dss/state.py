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
        self.flags["evfis_active"] = False
        self.flags["genai_active"] = False
        self.save()
        return counts

    def reverted_modifications(self) -> List[Dict[str, Any]]:
        """evFIS modifications newest first, for reverting before a wipe."""
        return list(reversed(self.sorted_records(
            "evfis_rule_modifications")))

    # ------------------------------------------------------------ reporting
    def counts(self) -> Dict[str, int]:
        return {s: len(self.records(s)) for s in SECTIONS}

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)
