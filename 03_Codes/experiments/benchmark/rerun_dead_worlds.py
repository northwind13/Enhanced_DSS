"""Remove from a campaign folder the worlds in which the open layer was
retired by a "model unreachable" verdict that was in fact an empty
candidate list (fixed in proposers.py, SearchProposer.propose).

    python rerun_dead_worlds.py <out_dir> <arm>

The rows of those worlds are moved out of runs.csv, curves.csv,
proposals.jsonl and admitted.jsonl into <out_dir>/dead_worlds/ so the
campaign script reruns exactly those worlds. Nothing is deleted."""
import csv, json, os, sys

d, arm = sys.argv[1], sys.argv[2]
P = os.path.join(d, "proposals.jsonl")
dead = set()
with open(P, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("arm") == arm and r.get("source") == "none" \
                and "not reachable" in str(r.get("detail") or ""):
            dead.add((r["scenario"], str(r["seed"])))
bak = os.path.join(d, "dead_worlds")
os.makedirs(bak, exist_ok=True)
if not dead:
    print(f"{d}: no dead worlds for arm {arm}")
    sys.exit(0)

def split_csv(name, key):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return 0
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    head, body = (rows[0], rows[1:]) if name == "runs.csv" else (None, rows)
    keep, out = [], []
    for r in body:
        (out if key(r) in dead else keep).append(r)
    with open(os.path.join(bak, name), "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(out)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if head: w.writerow(head)
        w.writerows(keep)
    return len(out)

def split_jsonl(name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return 0
    keep, out = [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                keep.append(line); continue
            (out if (r.get("arm") == arm and (r["scenario"], str(r["seed"])) in dead) else keep).append(line)
    with open(os.path.join(bak, name), "a", encoding="utf-8") as f:
        f.writelines(out)
    with open(p, "w", encoding="utf-8") as f:
        f.writelines(keep)
    return len(out)

# runs.csv columns: exp,scenario,arm,seed,...  curves.csv: exp,scenario,arm,seed,...
n_runs = split_csv("runs.csv", lambda r: (r[1], r[3]) if r[2] == arm else None)
n_curves = split_csv("curves.csv", lambda r: (r[1], r[3]) if r[2] == arm else None)
n_prop = split_jsonl("proposals.jsonl")
n_adm = split_jsonl("admitted.jsonl")
with open(os.path.join(bak, "worlds.txt"), "a", encoding="utf-8") as f:
    f.write("\n".join(f"{s},{seed}" for s, seed in sorted(dead)) + "\n")
print(f"{d} arm {arm}: {len(dead)} dead worlds moved to {bak} "
      f"(runs {n_runs}, curves {n_curves}, proposals {n_prop}, admitted {n_adm})")
