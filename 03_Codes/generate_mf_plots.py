"""
generate_mf_plots_v3.py

Plots membership functions using ONLY Excel definitions, but fixes a common real-world issue:
MF parameters stored normalized in [0,1] while the variable domain is [Typical Min, Typical Max] in real units.

No new Excel columns are required.

Required Excel columns:
- Fuzzy Variable
- Units
- Typical Min
- Typical Max
- MF_Params   (JSON list of member definitions)

Optional Excel columns:
- MF_Type     (fallback if a member lacks "type")

MF_Params JSON member formats:
- trapezoid: {"label":"Low","type":"trapezoid","a":...,"b":...,"c":...,"d":...}
- triangle:  {"label":"Low","type":"triangle","a":...,"b":...,"c":...}
- gaussian:  {"label":"Med","type":"gaussian","mu":...,"sigma":...}

Auto-rescale rule (inferred, not hardcoded thresholds):
- If ALL members' numeric parameters lie within approximately [-0.01, 1.01]
  AND (Typical Max - Typical Min) > 1
  AND Units is not "0-1"
  THEN rescale parameters from [0,1] -> [Typical Min, Typical Max] for plotting.

This preserves your Excel as the source of truth while preventing "collapsed" plots.

Usage:
  python generate_mf_plots_v3.py --excel "DSS_Tables.xlsx" --sheet "Level1_MF_Table" --outdir "mf_plots"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

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
    return out[:140] if out else "variable"


def member_extent(m: Dict[str, Any], mtype: str) -> Tuple[float, float]:
    if mtype == "trapezoid":
        vals = [float(m[k]) for k in ("a", "b", "c", "d")]
        return min(vals), max(vals)
    if mtype == "triangle":
        vals = [float(m[k]) for k in ("a", "b", "c")]
        return min(vals), max(vals)
    if mtype == "gaussian":
        mu = float(m["mu"])
        sigma = float(m["sigma"])
        return mu - 3*sigma, mu + 3*sigma
    return float("nan"), float("nan")


def rescale_member_01_to_domain(m: Dict[str, Any], mtype: str, vmin: float, vmax: float) -> Dict[str, Any]:
    mm = dict(m)
    def map01(v):
        return vmin + (float(v) - 0.0) * (vmax - vmin)

    if mtype == "trapezoid":
        for k in ("a", "b", "c", "d"):
            mm[k] = map01(mm[k])
    elif mtype == "triangle":
        for k in ("a", "b", "c"):
            mm[k] = map01(mm[k])
    elif mtype == "gaussian":
        mm["mu"] = map01(mm["mu"])
        mm["sigma"] = float(mm["sigma"]) * (vmax - vmin)
    return mm


def plot_member(x: np.ndarray, m: Dict[str, Any], mtype: str) -> np.ndarray:
    if mtype == "trapezoid":
        return trapmf(x, float(m["a"]), float(m["b"]), float(m["c"]), float(m["d"]))
    if mtype == "triangle":
        return trimf(x, float(m["a"]), float(m["b"]), float(m["c"]))
    if mtype == "gaussian":
        return gaussmf(x, float(m["mu"]), float(m["sigma"]))
    raise ValueError(f"Unsupported MF type: {mtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--outdir", default="mf_plots")
    ap.add_argument("--points", type=int, default=800)
    ap.add_argument("--audit_csv", default="mf_audit_report.csv", help="Audit report filename inside outdir.")
    args = ap.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    required = ["Fuzzy Variable", "Units", "Typical Min", "Typical Max", "MF_Params"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audit_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        fv = str(row["Fuzzy Variable"]).strip()
        units = str(row["Units"]).strip()

        try:
            vmin = float(row["Typical Min"])
            vmax = float(row["Typical Max"])
        except Exception:
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "Non-numeric Typical Min/Max"})
            continue
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "Invalid Min/Max ordering"})
            continue

        raw = row["MF_Params"]
        if not isinstance(raw, str) or raw.strip() in ("", "N/A", "nan"):
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "Missing MF_Params"})
            continue

        try:
            members = json.loads(raw)
            if not isinstance(members, list) or len(members) == 0:
                raise ValueError("MF_Params must be a non-empty list")
        except Exception as e:
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": f"Invalid MF_Params JSON: {e}"})
            continue

        fallback_type = str(row.get("MF_Type", "")).strip().lower()

        ext_los, ext_his = [], []
        types_ok = True

        for m in members:
            mtype = str(m.get("type", fallback_type)).strip().lower()
            if mtype not in ("trapezoid", "triangle", "gaussian"):
                types_ok = False
                break
            lo, hi = member_extent(m, mtype)
            ext_los.append(lo); ext_his.append(hi)

        if not types_ok or len(ext_los)==0:
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "SKIP", "reason": "Unsupported or missing MF type"})
            continue

        mf_min = float(np.nanmin(ext_los))
        mf_max = float(np.nanmax(ext_his))

        # Domain mismatch inference
        normalized = (mf_min >= -0.01) and (mf_max <= 1.01)
        wide = (vmax - vmin) > 1.0
        units_norm = units.strip().lower().replace(" ", "") in ("0-1", "[0,1]", "0..1")

        rescaled = False
        plot_members = members

        if normalized and wide and not units_norm:
            plot_members = []
            for m in members:
                mtype = str(m.get("type", fallback_type)).strip().lower()
                mm = rescale_member_01_to_domain(m, mtype, vmin, vmax)
                mm["type"] = mtype
                plot_members.append(mm)
            rescaled = True
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "OK_RESCALED_FOR_PLOT", "reason": "MF params were normalized [0,1]"})
        else:
            audit_rows.append({"row": idx+2, "fuzzy_variable": fv, "status": "OK", "reason": ""})

        x = np.linspace(vmin, vmax, args.points)
        plt.figure()

        for m in plot_members:
            mlabel = str(m.get("label", "member")).strip()
            mtype = str(m.get("type", fallback_type)).strip().lower()
            y = plot_member(x, m, mtype)
            plt.plot(x, y, label=mlabel)

            summary_rows.append({
                "row": idx+2,
                "fuzzy_variable": fv,
                "units": units,
                "member_label": mlabel,
                "member_type": mtype,
                "rescaled_for_plot": rescaled,
                **{k: v for k, v in m.items() if k not in ("label", "type")}
            })

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

    pd.DataFrame(audit_rows).to_csv(outdir / args.audit_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(outdir / "mf_params_summary.csv", index=False)

    print(f"Done. Plots: {outdir.resolve()}")
    print(f"Audit: {(outdir / args.audit_csv).resolve()}")
    print(f"Summary: {(outdir / 'mf_params_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
