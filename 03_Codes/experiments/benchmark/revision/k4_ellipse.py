# -*- coding: utf-8 -*-
r"""K4e: equal-area references for the hindcast of Table V.

For each documented fire, the observed first-12-h footprint (cached FIRMS
detections, as in k4_hindcast --protocol window12) is compared with two
references of the same area placed at the first detection: a disk, and an
ellipse elongated along the ERA5 wind at the ignition hour with the
length-to-breadth ratio of the spread kernel, LB = 1 + 0.06 W_s, the
ignition at the rear focus. Reports probability of detection (POD), false
alarm ratio (FAR) and the Sorensen index for both references and for the
simulated footprints saved by k4_hindcast (mean over seeds). Needs the
validation cache of the hindcast runs (validation/cache/<case>), no key.

  python experiments\benchmark\revision\k4_ellipse.py
"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))
import auto_validate as AV                    # noqa: E402

ORDER = ["canakkale2023", "manavgat2021", "marmaris2021", "milas2021"]
CKPT = os.path.join(HERE, "k4_hindcast", "window12_stop_area")


def _scores(ref, obs):
    ref = ref.astype(bool); obs = obs.astype(bool)
    hit = float((ref & obs).sum())
    pod = hit / max(obs.sum(), 1)
    far = float((ref & ~obs).sum()) / max(ref.sum(), 1)
    dice = 2 * hit / max(ref.sum() + obs.sum(), 1)
    return pod, far, dice


def _disk(nx, ny, cx, cy, area_cells):
    r = math.sqrt(area_cells / math.pi)
    yy, xx = np.mgrid[0:ny, 0:nx]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _ellipse(nx, ny, cx, cy, area_cells, wd_math_deg, lb):
    """Equal-area ellipse, long axis along the wind (math angle, toward
    which the wind blows, y down), the ignition at the rear focus."""
    b = math.sqrt(area_cells / (math.pi * lb))     # semi-minor
    a = lb * b                                     # semi-major
    c = math.sqrt(max(a * a - b * b, 0.0))         # focus distance
    th = math.radians(wd_math_deg)
    ux, uy = math.cos(th), -math.sin(th)           # grid y points down
    ex, ey = cx + c * ux, cy + c * uy              # centre ahead of ignition
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - ex, yy - ey
    u = dx * ux + dy * uy
    v = -dx * uy + dy * ux
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT,
                    help="k4_hindcast checkpoint folder with ckpt_<case>/seed_*.npy")
    ap.add_argument("--out", default=None, help="csv path (default <ckpt>/k4_ellipse.csv)")
    args = ap.parse_args()
    ckpt = args.ckpt
    out = args.out or os.path.join(ckpt, "k4_ellipse.csv")
    rows = []
    cell = 90.0
    print("case, reference, POD, FAR, Sorensen")
    for cid in ORDER:
        case = dict(AV.CASES[cid]); case["hours"] = 12.0
        cache = os.path.join(ROOT, "validation", "cache", cid)
        nx, ny, lons, lats = AV._grid(case, cell)
        pts = AV._download_firms(case, "", cache)
        obs, first, ign_cells = AV._firms_mask_and_ignition(case, pts, nx, ny,
                                                           lons, lats, cell)
        obs = np.asarray(obs, dtype=bool)
        area = float(obs.sum())
        cx, cy = (first[0], first[1])
        iglat = case["north"] - (cy + 0.5) * cell / 110540.0
        iglon = case["west"] + (cx + 0.5) * cell / (
            111320.0 * math.cos(math.radians(iglat)))
        wx = AV._download_weather(case, cache, lat=iglat, lon=iglon)
        h = min(int(first[4].hour), len(wx["wind_speed_10m"]) - 1)
        ws = float(wx["wind_speed_10m"][h])          # m/s (windspeed_unit=ms)
        wd = AV._met_to_math_toward(float(wx["wind_direction_10m"][h]))
        lb = 1.0 + 0.06 * ws
        for name, ref in (("disk", _disk(nx, ny, cx, cy, area)),
                          ("ellipse", _ellipse(nx, ny, cx, cy, area, wd, lb))):
            pod, far, dice = _scores(ref, obs)
            print(f"{case['label']}, {name} (LB {lb:.2f}), {pod:.2f}, {far:.2f}, {dice:.2f}")
            rows.append(dict(case=cid, label=case["label"], reference=name, lb=round(lb, 2),
                             pod=round(pod, 3), far=round(far, 3), sorensen=round(dice, 3), n=1))
        label = case.get("label", cid).replace(" ", "_").replace("/", "_")
        masks = sorted(glob.glob(os.path.join(ckpt, f"ckpt_{label}", "seed_*.npy")))
        if masks:
            sc = np.array([_scores(np.load(m), obs) for m in masks])
            print(f"{case['label']}, simulation (n={len(masks)}), "
                  f"{sc[:,0].mean():.2f}, {sc[:,1].mean():.2f}, {sc[:,2].mean():.2f}")
            rows.append(dict(case=cid, label=case["label"], reference="simulation", lb="",
                             pod=round(float(sc[:,0].mean()), 3), far=round(float(sc[:,1].mean()), 3),
                             sorensen=round(float(sc[:,2].mean()), 3), n=len(masks)))
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
