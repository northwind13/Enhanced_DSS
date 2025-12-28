"""
generate_mf_plots.py

Fully Excel-driven membership function plotting.

Nothing is hardcoded about:
- member count
- member labels
- member shapes
- thresholds/parameters

The script uses MF_Params JSON as the single source of truth.

Required Excel columns:
- Fuzzy Variable
- Units
- Typical Min
- Typical Max
- MF_Params   (JSON list: each element defines one member)
Optional:
- MF_Type     (row-level fallback if a member lacks "type")

Supported MF types (must be declared in MF_Params member 'type' OR MF_Type):
- trapezoid: a,b,c,d
- triangle:  a,b,c
- gaussian:  mu,sigma

If MF_Params is "N/A" or empty, the script SKIPS that row and logs it in audit CSV.

Outputs:
- one PNG per row under outdir
- mf_audit_report.csv: per-row status

Usage:
  python generate_mf_plots.py --excel "DSS_Tables.xlsx" --sheet "Level1_MF_Table" --outdir "mf_plots"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trapmf(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        idx = (x >= a) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)
    idx = (x >= b) & (x <= c)
    y[idx] = 1.0
    if d > c:
        idx = (x > c) & (x <= d)
        y[idx] = (d - x[idx]) / (d - c)
    return np.clip(y, 0.0, 1.0)


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    if b > a:
        idx = (x >= a) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)
    if c > b:
        idx = (x >= b) & (x <= c)
        y[idx] = (c - x[idx]) / (c - b)
    return np.clip(y, 0.0, 1.0)


def gaussmf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in (" ", "_", "-", ".", "+"):
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out[:160] if out else "variable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--outdir", default="mf_plots")
    ap.add_argument("--points", type=int, default=800)
    args = ap.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    required = ["Fuzzy Variable", "Units", "Typical Min", "Typical Max", "MF_Params"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audit: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        fv = str(row["Fuzzy Variable"]).strip()
        units = str(row["Units"]).strip()

        try:
            vmin = float(row["Typical Min"])
            vmax = float(row["Typical Max"])
        except Exception:
            audit.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "Non-numeric Typical Min/Max"})
            continue
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            audit.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "Invalid Min/Max"})
            continue

        raw = row["MF_Params"]
        if not isinstance(raw, str) or raw.strip() in ("", "N/A", "nan"):
            audit.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "MF_Params missing"})
            continue

        try:
            members = json.loads(raw)
            if not isinstance(members, list) or len(members) == 0:
                raise ValueError("MF_Params must be a non-empty list")
        except Exception as e:
            audit.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": f"Invalid MF_Params JSON: {e}"})
            continue

        fallback_type = str(row.get("MF_Type","")).strip().lower()

        x = np.linspace(vmin, vmax, args.points)
        plt.figure()

        ok = True
        for m in members:
            mlabel = str(m.get("label","member"))
            mtype = str(m.get("type", fallback_type)).strip().lower()

            if mtype == "trapezoid":
                y = trapmf(x, float(m["a"]), float(m["b"]), float(m["c"]), float(m["d"]))
            elif mtype == "triangle":
                y = trimf(x, float(m["a"]), float(m["b"]), float(m["c"]))
            elif mtype == "gaussian":
                y = gaussmf(x, float(m["mu"]), float(m["sigma"]))
            else:
                ok = False
                audit.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": f"Unsupported MF type: {mtype}"})
                break

            plt.plot(x, y, label=mlabel)

        if not ok:
            plt.close()
            continue

        plt.title(fv if fv else f"Row_{idx+2}")
        plt.xlabel(f"Value ({units})")
        plt.ylabel("Membership")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")

        fname = sanitize_filename(f"{idx+1}_{fv or 'Row'}_mf.png")
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()

        audit.append({"row": idx+2, "fuzzy_variable": fv, "status": "OK", "reason": ""})

    pd.DataFrame(audit).to_csv(outdir / "mf_audit_report.csv", index=False)
    print(f"Done. Plots: {outdir.resolve()}")
    print(f"Audit: {(outdir / 'mf_audit_report.csv').resolve()}")


if __name__ == "__main__":
    main()
