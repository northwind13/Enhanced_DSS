# -*- coding: utf-8 -*-
r"""Round-8 CPU runs in parallel, then commit and push the results.

  K35  evolutionary searches (composed grammar and six families) under the
       combined model error and with independent gate futures
  K37  closed layer with a family mapped to coating (asset protection,
       containment line), and the evolutionary family search with both
  K38  cross-entropy search over the composed grammar and over the families
  K39  closed layer with confidence gating ablated
  K36  hindcast footprint at 12 h without the area-matched stop (only if
       validation\cache holds the FIRMS and weather inputs)

No language-model call. Every run is resumable; run again to continue.
Workers: W_CPU (default 12) are split across the parallel jobs.

  python experiments\benchmark\revision\run_round8.py
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # 03_Codes
REPO = os.path.dirname(ROOT)
R = os.path.relpath(HERE, ROOT)
PY = sys.executable
W = int(os.environ.get("W_CPU") or 12)
share = lambda f: max(1, int(round(W * f)))

BASE_ENV = dict(DSS_REQUIRE_GENAI="", DSS_WITHHOLD_FAMILY_NAME="",
                DSS_NO_COORD="", DSS_GATE_SEEDS="", DSS_NO_CONCEPT="",
                DSS_FAMILY_COAT="", DSS_NO_CONF_GATE="", DSS_NO_COMPOSED="")
CAMP = os.path.join("experiments", "open_space", "campaign.py")
SEEDS = ["--scenarios", "S1,S2,S3,S4,S5", "--seeds", "201-230"]

JOBS = [
    ("K35", [PY, os.path.join(R, "k35_strong_under_error.py"), "--workers", str(share(0.34))], {}),
    ("K37_ap", [PY, CAMP, "--exp", "E2P", "--arms", "F5Ev", *SEEDS, "--workers", str(share(0.08)),
                "--out", os.path.join(R, "k37_family_coat", "closed_asset_protection")],
     dict(DSS_FAMILY_COAT="asset_protection")),
    ("K37_cl", [PY, CAMP, "--exp", "E2P", "--arms", "F5Ev", *SEEDS, "--workers", str(share(0.08)),
                "--out", os.path.join(R, "k37_family_coat", "closed_containment_line")],
     dict(DSS_FAMILY_COAT="containment_line")),
    ("K37_evo", [PY, CAMP, "--exp", "E15", "--arms", "Evo-fam", *SEEDS, "--workers", str(share(0.17)),
                 "--out", os.path.join(R, "k37_family_coat", "evofam_both")],
     dict(DSS_FAMILY_COAT="asset_protection,containment_line")),
    ("K38", [PY, CAMP, "--exp", "E18", "--arms", "CEM,CEM-fam", *SEEDS, "--workers", str(share(0.25)),
             "--out", os.path.join(R, "k38_cem")], {}),
    ("K39", [PY, CAMP, "--exp", "E2P", "--arms", "F5Ev", *SEEDS, "--workers", str(share(0.08)),
             "--out", os.path.join(R, "k39_no_conf_gate")], dict(DSS_NO_CONF_GATE="1")),
]
cache = os.path.join(ROOT, "validation", "cache")
if os.path.isdir(cache) and os.listdir(cache):
    JOBS.append(("K36", [PY, os.path.join(R, "k4_hindcast.py"), "--stop", "none", "--seeds", "50",
                         "--step", "5", "--out", os.path.join(R, "k4_hindcast_step5")], {}))
else:
    print("K36 skipped: validation\\cache is empty on this machine "
          "(the hindcast inputs were cached on the machine that ran K4)", flush=True)


def main():
    os.chdir(ROOT)
    logdir = os.path.join(HERE, "round8_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    for name, cmd, extra in JOBS:
        env = dict(os.environ, **BASE_ENV, **extra)
        log = open(os.path.join(logdir, f"{name}.log"), "a", encoding="utf-8")
        log.write(f"==== {name} started {time.ctime()}\n"); log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((name, p, log))
        print(f"started {name}: {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    left = {n for n, _p, _l in procs}
    while left:
        time.sleep(60)
        for name, p, log in procs:
            if name in left and p.poll() is not None:
                left.discard(name)
                log.write(f"==== {name} finished {p.returncode} {time.ctime()}\n"); log.close()
                print(f"{name} finished ({p.returncode}) after {(time.time()-t0)/60:.0f} min", flush=True)
        if left:
            print(f"{(time.time()-t0)/60:.0f} min, running: {', '.join(sorted(left))}", flush=True)
    # K36 summary of POD, FAR and Sorensen for the unmatched footprint
    if any(n == "K36" for n, _c, _e in JOBS):
        subprocess.call([PY, os.path.join(R, "k4_ellipse.py"), "--ckpt",
                         os.path.join(R, "k4_hindcast_step5", "window12_stop_none")], cwd=ROOT)
    os.chdir(REPO)
    paths = [os.path.join("03_Codes", R, d) for d in
             ("k35_strong_under_error", "k37_family_coat", "k38_cem", "k39_no_conf_gate",
              "round8_logs", os.path.join("k4_hindcast_step5", "window12_stop_none"))]
    paths = [p for p in paths if os.path.exists(p)]
    subprocess.call(["git", "add", "-A", *paths])
    subprocess.call(["git", "commit", "-m", "Round 8 CPU runs: K35 K37 K38 K39 (K36 if cached)"])
    subprocess.call(["git", "pull", "--rebase", "origin", "main"])
    rc = subprocess.call(["git", "push", "origin", "main"])
    print("==== committed and pushed" if rc == 0 else "==== push failed, run git push by hand", flush=True)


if __name__ == "__main__":
    main()
