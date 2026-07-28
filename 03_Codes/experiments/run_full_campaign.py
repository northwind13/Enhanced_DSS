"""ONE command: run the N = 20 campaign, build every table and
figure, and fill the thesis automatically (tracked changes).

    python experiments/run_full_campaign.py

Optional:
    --seeds 20            campaign size (default 20)
    --workers 0           0 = all cores minus one
    --docx PATH           the SKELETON thesis to fill (default:
                          01_Thesis/DISASTERAWARE_PhDThesis_vFinalSim_Test.docx)
    --skip-campaign       only rebuild tables/figures/docx from the
                          existing CSVs

The campaign is RESUMABLE: if the machine sleeps or the run is
interrupted, running the same command again continues where it
stopped and skips everything already done. Output docx gets a
timestamped name next to the skeleton, so nothing is overwritten.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
PY = sys.executable or "python"


def preflight():
    """The campaign needs numpy + matplotlib, the docx fill needs
    python-docx. Whichever Python launched this script must carry
    them; if any is missing it is installed into THAT interpreter, so
    'one command' stays one command on a bare Python too."""
    need = []
    for mod, pkg in (("numpy", "numpy"),
                     ("matplotlib", "matplotlib"),
                     ("docx", "python-docx")):
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if not need:
        return
    print("Missing packages for this Python "
          f"({PY}): {', '.join(need)}")
    print("Installing...", flush=True)
    r = subprocess.run([PY, "-m", "pip", "install", *need])
    if r.returncode != 0:
        print("\npip failed. Either run this script with the "
              "Python environment the app uses (the one that runs "
              "Streamlit), or install manually:\n  "
              f"{PY} -m pip install {' '.join(need)}")
        sys.exit(1)


def run(args, name):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=os.path.dirname(HERE))
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"{name} FAILED (exit {r.returncode}) after {dt:.0f}s")
        sys.exit(r.returncode)
    print(f"{name} done in {dt:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--docx", default=os.path.join(
        ROOT, "01_Thesis",
        "DISASTERAWARE_PhDThesis_vFinalSim_Test.docx"))
    ap.add_argument("--skip-campaign", action="store_true")
    ap.add_argument("--redo", default="",
                    help="comma list of arms to purge and re-run "
                         "(forwarded to campaign5.py)")
    ap.add_argument("--eta", type=float, default=None,
                    help="decision-engine quality gate (forwarded "
                         "to campaign5.py; default 0.60)")
    a = ap.parse_args()

    preflight()
    workers = a.workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)

    if not a.skip_campaign:
        cargs = [os.path.join(HERE, "campaign5.py"),
                 "--seeds", str(a.seeds), "--workers", str(workers)]
        if a.redo:
            cargs += ["--redo", a.redo]
        if a.eta is not None:
            cargs += ["--eta", str(a.eta)]
        run(cargs,
            f"campaign (N={a.seeds}, {workers} workers, resumable)")

    run([os.path.join(HERE, "ladder_report.py"),
         "--seeds", str(a.seeds)],
        "tables + claim chain + figures (balanced at N)")

    if not os.path.exists(a.docx):
        print(f"\nSkeleton docx not found: {a.docx}\n"
              "Pass it with --docx PATH; tables and figures are "
              "already refreshed in experiments/out and "
              "01_Thesis/figures.")
        sys.exit(1)
    stamp = time.strftime("%Y%m%d_%H%M")
    out = os.path.join(
        os.path.dirname(a.docx),
        os.path.splitext(os.path.basename(a.docx))[0]
        + f"_Ch5_filled_{stamp}.docx")
    run([os.path.join(HERE, "fill_thesis.py"), a.docx, out],
        "thesis fill (tracked changes)")
    print(f"\nALL DONE.\n  results: {os.path.join(HERE, 'out')}\n"
          f"  figures: {os.path.join(ROOT, '01_Thesis', 'figures')}\n"
          f"  thesis:  {out}")


if __name__ == "__main__":
    main()
