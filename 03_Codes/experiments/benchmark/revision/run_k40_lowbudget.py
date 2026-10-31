# -*- coding: utf-8 -*-
r"""K40: the evolutionary and cross-entropy searches at small forecast budgets.

Referee question: the model's margin over the strongest searches is not
significant at the full budget of 36. At small budgets the model candidates
beat random candidates (0 forecasts: single LLM vs single random; 6: the
candidate-only configuration vs the small random control). This run gives the
two structured searches the same small budgets so that the comparison at low
budget is against the best searches, not only against random sampling.
Population and sample sizes are scaled to fit each budget:

  evo_b6    evolutionary, population 3, one generation (3 + 3)
  evo_b12   evolutionary, population 4, two generations (4 + 8)
  cem_b6    cross-entropy, 3 samples per iteration, elite 1 (2 iterations)
  cem_b12   cross-entropy, 4 samples per iteration, elite 1 (3 iterations)

Pairs with final/t6b_voda_model_only (LLM candidates, 6), k21_budget/goda_b12
(GODA, 12), final/t5b_random_screened_small (random, 6), k21_budget/random_b12
and final/t1_reference. CPU only, no model call. Runs in parallel, then
commits and pushes the results. Resumable.

  python experiments\benchmark\revision\run_k40_lowbudget.py
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REPO = os.path.dirname(ROOT)
R = os.path.relpath(HERE, ROOT)
PY = sys.executable
W = int(os.environ.get("W_CPU") or 12)
w = max(1, W // 4)
CAMP = os.path.join("experiments", "open_space", "campaign.py")
SEEDS = ["--scenarios", "S1,S2,S3,S4,S5", "--seeds", "201-230"]
BASE = dict(DSS_REQUIRE_GENAI="", DSS_WITHHOLD_FAMILY_NAME="", DSS_NO_COORD="",
            DSS_GATE_SEEDS="", DSS_NO_CONCEPT="", DSS_FAMILY_COAT="",
            DSS_NO_CONF_GATE="", DSS_NO_COMPOSED="", DSS_EVO_POP="", DSS_EVO_GENS="",
            DSS_CEM_N="", DSS_CEM_ELITE="")
OUT = os.path.join(R, "k40_lowbudget")
JOBS = [
    ("evo_b6", ["--exp", "E15", "--arms", "Evo", "--budget", "6"], dict(DSS_EVO_POP="3", DSS_EVO_GENS="1")),
    ("evo_b12", ["--exp", "E15", "--arms", "Evo", "--budget", "12"], dict(DSS_EVO_POP="4", DSS_EVO_GENS="2")),
    ("cem_b6", ["--exp", "E18", "--arms", "CEM", "--budget", "6"], dict(DSS_CEM_N="3", DSS_CEM_ELITE="1")),
    ("cem_b12", ["--exp", "E18", "--arms", "CEM", "--budget", "12"], dict(DSS_CEM_N="4", DSS_CEM_ELITE="1")),
]


def main():
    os.chdir(ROOT)
    logdir = os.path.join(HERE, "k40_logs")
    os.makedirs(logdir, exist_ok=True)
    procs = []
    for name, args, extra in JOBS:
        env = {**os.environ, **BASE, **extra}
        cmd = [PY, CAMP, *args, *SEEDS, "--workers", str(w), "--out", os.path.join(OUT, name)]
        log = open(os.path.join(logdir, f"{name}.log"), "a", encoding="utf-8")
        log.write(f"==== {name} started {time.ctime()}\n"); log.flush()
        procs.append((name, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                             stderr=subprocess.STDOUT), log))
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
    os.chdir(REPO)
    subprocess.call(["git", "add", "-A", os.path.join("03_Codes", R, "k40_lowbudget"),
                     os.path.join("03_Codes", R, "k40_logs")])
    subprocess.call(["git", "commit", "-m", "K40 low-budget evolutionary and cross-entropy searches"])
    subprocess.call(["git", "pull", "--rebase", "origin", "main"])
    rc = subprocess.call(["git", "push", "origin", "main"])
    print("==== committed and pushed" if rc == 0 else "==== push failed, run git push by hand", flush=True)


if __name__ == "__main__":
    main()
