"""Remove the records of one arm from a campaign output folder.

Usage: python drop_arm_rows.py <out_dir> <arm>

Used before a rerun when an arm completed some worlds with a broken
proposer: the campaign resumes from runs.csv, so those worlds would
otherwise be kept as if they were valid. runs.csv, curves.csv,
proposals.jsonl and admitted.jsonl are rewritten without the arm;
errors.log is moved aside.
"""
import csv
import json
import os
import sys


def main(out, arm):
    n = 0
    p = os.path.join(out, "runs.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            fields = rd.fieldnames
        keep = [r for r in rows if r.get("arm") != arm]
        n += len(rows) - len(keep)
        if fields:
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(keep)
    p = os.path.join(out, "curves.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        keep = [r for r in rows if len(r) < 3 or r[2] != arm]
        with open(p, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(keep)
    for name in ("proposals.jsonl", "admitted.jsonl"):
        p = os.path.join(out, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            lines = f.readlines()
        keep = []
        for ln in lines:
            try:
                if json.loads(ln).get("arm") == arm:
                    continue
            except ValueError:
                pass
            keep.append(ln)
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(keep)
    p = os.path.join(out, "errors.log")
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        with open(os.path.join(out, "errors_previous.log"), "a",
                  encoding="utf-8") as f:
            f.write(txt)
        os.remove(p)
    print(f"{out}: dropped {n} runs of arm {arm}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
