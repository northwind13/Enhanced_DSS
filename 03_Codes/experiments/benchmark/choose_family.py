"""Pick the withheld family whose loss to the closed layer is largest.

    python choose_family.py <final_dir> fam1 fam2 ...

For each family the closed layer without it (folder t12_closed_<fam>) is
paired with the closed layer with the full doctrine (t1_reference, arm
F5Ev) on the same worlds. The family with the larger mean increase of the
burned area is printed on stdout (one word, for the batch file); the
comparison is written to <final_dir>/t12_choice.txt."""
import csv, math, os, statistics as st, sys

final = sys.argv[1]
fams = sys.argv[2:]
ref = {}
with open(os.path.join(final, "t1_reference", "runs.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["arm"] == "F5Ev":
            ref[(r["scenario"], r["seed"])] = float(r["burned_ha"])
lines, best = [], None
for fam in fams:
    p = os.path.join(final, f"t12_closed_{fam}", "runs.csv")
    if not os.path.exists(p):
        lines.append(f"{fam}: missing {p}")
        continue
    with open(p, encoding="utf-8") as f:
        rows = {(r["scenario"], r["seed"]): float(r["burned_ha"])
                for r in csv.DictReader(f)}
    ks = [k for k in rows if k in ref]
    if len(ks) < 2:
        lines.append(f"{fam}: only {len(ks)} paired worlds")
        continue
    d = [rows[k] - ref[k] for k in ks]
    base = st.mean(ref[k] for k in ks)
    pct = 100 * st.mean(d) / base
    hw = 100 * 1.96 * st.stdev(d) / math.sqrt(len(d)) / base
    worse = sum(x > 0 for x in d)
    lines.append(f"{fam}: closed layer without the family, burned area "
                 f"{pct:+.1f}% +/- {hw:.1f} against the full doctrine, "
                 f"worse in {worse} of {len(ks)} worlds")
    if best is None or pct > best[1]:
        best = (fam, pct)
with open(os.path.join(final, "t12_choice.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
    f.write(f"chosen: {best[0] if best else 'none'}\n")
for l in lines:
    print(l, file=sys.stderr)
if best is None:
    sys.exit(1)
print(best[0])
