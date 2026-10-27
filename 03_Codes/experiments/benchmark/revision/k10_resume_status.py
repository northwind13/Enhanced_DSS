"""Read-only K10 checkpoint validation. Never imports or calls a model."""
import argparse
import csv
import os
from pathlib import Path
import sys


FAMILIES = tuple(f"S{i}" for i in range(1, 6))
SEEDS = tuple(range(201, 231))
PHASES = {
    "k8": ("k8_gate_seeds_nc", ("",)),
    "k6": ("k6_coating_nc", ("coat_half", "coat_off", "coat_price_x2",
                                  "coat_stock_20ha", "coat_delay_12min")),
    "k7": ("k7_spread_nc", ("wind_x1.5", "wind_rot_p30", "wind_rot_m30")),
}


def checkpoint_status(root, phase):
    folder, perturbations = PHASES[phase]
    expected = {(p, sc, sd) for p in perturbations
                for sc in FAMILIES for sd in SEEDS}
    path = Path(root) / folder / "runs.csv"
    done = set()
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"scenario", "seed", "arm", "n_concepts"}
            required.add("exp" if phase == "k8" else "perturb")
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: empty file or missing checkpoint columns")
            for line, row in enumerate(reader, 2):
                try:
                    key = (row.get("perturb", ""), row["scenario"],
                           int(row["seed"]))
                    arm = "LLMScreen" if phase == "k8" else "live"
                    if row["arm"] != arm or key not in expected:
                        raise ValueError("unexpected world or arm")
                    if phase == "k8" and row["exp"] != "E9":
                        raise ValueError("unexpected experiment")
                    if key in done:
                        raise ValueError("duplicate completed world")
                    if float(row["n_concepts"]) != 0:
                        raise ValueError("nonzero or invalid new-concept count")
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line}: {exc}") from exc
                done.add(key)
    return len(done), len(expected - done), len(expected)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", choices=("all", *PHASES))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    phases = tuple(PHASES) if args.check in (None, "all") else (args.check,)
    remaining = 0
    try:
        for phase in phases:
            done, todo, total = checkpoint_status(root, phase)
            remaining += todo
            if not args.quiet:
                print(f"{phase.upper()}: {done}/{total} complete; {todo} remaining")
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Invalid K10 checkpoint: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        model = os.environ.get("DSS_GENAI_MODEL", "claude-opus-5")
        print(f"Remaining worlds: {remaining}. Requested model: {model}.")
    return int(remaining > 0) if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
