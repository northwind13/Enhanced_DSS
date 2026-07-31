"""Open-decision-space figures for Chapter 5 (Section 5.4.3).

Reproducible from committed artifacts:
  logs/DSS_*/cycles.jsonl        per-cycle adaptation records of the runs
  logs/dss_generated_state.json  the generative product ledger

Writes to 01_Thesis/figures/:
  fig_openspace_funnel.png    verification funnel: evFIS trials and the
                              stage-3 generative gate cascade
  fig_openspace_products.png  the generated vocabulary: macro composition
                              weights and the intermediate concept
  fig_openspace_timeline.png  anatomy of one recorded run: adaptation
                              events over the incident with the cost curve

Run:  python experiments/plot_openspace.py
"""
from __future__ import annotations
import json, glob, os, re, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "logs")
FIGDIR = os.path.join(ROOT, "..", "01_Thesis", "figures")
DPI = 220
C_TRIED = "#bcd3ea"; C_ACC = "#2166ac"
C_REJ = "#d6604d"; C_TRANS = "#9e9e9e"; C_ADM = "#1a9850"


def _extract_adapt(line):
    """Parse only the adaptation object out of a cycle line, so the large
    sim/sensor payload is never deserialized (keeps the scan fast)."""
    i = line.find('"adaptation":')
    if i < 0:
        return None
    j = line.find('{', i); depth = 0; instr = False; esc = False
    for k in range(j, len(line)):
        c = line[k]
        if esc:
            esc = False; continue
        if c == '\\':
            esc = True; continue
        if c == '"':
            instr = not instr; continue
        if instr:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(line[j:k + 1])
                except Exception:
                    return None
    return None


def aggregate_funnel():
    s1t = s1k = s2t = s2k = 0
    prop = adm = 0
    g = collections.Counter()
    for f in glob.glob(os.path.join(LOGS, "DSS_*", "cycles.jsonl")):
        for line in open(f, encoding="utf-8"):
            if '"adaptation"' not in line:
                continue
            a = _extract_adapt(line)
            if not a:
                continue
            st = a.get("stage", 0); tried = a.get("tried", 0) or 0
            info = a.get("info") or {}
            det = a.get("detail") or ""
            if st == 1:
                s1t += tried
                s1k += sum(1 for t in (info.get("trials") or [])
                           if t.get("kept"))
            elif st == 2:
                s2t += tried
                if a.get("accepted"):
                    s2k += 1
            elif st == 3:
                prop += 1
                if a.get("accepted"):
                    adm += 1
                    continue
                m = re.search(r'rejected at (\w+)', det)
                tok = m.group(1) if m else ""
                if tok == "G3":
                    g["G3 forecast"] += 1
                elif tok.startswith("G2"):
                    g["G1-G2 form/vocab/cell"] += 1
                elif tok == "G5":
                    g["G5 margin"] += 1
                elif tok == "G4":
                    g["G4 robustness"] += 1
                elif ("wait budget" in det or "timed out" in det
                      or "not reach" in det or "requires" in det):
                    g["transport"] += 1
                elif tok.startswith("G1"):
                    g["G1-G2 form/vocab/cell"] += 1
                else:
                    g["G1-G2 form/vocab/cell"] += 1
    return dict(s1t=s1t, s1k=s1k, s2t=s2t, s2k=s2k,
                prop=prop, adm=adm, gates=g)


def fig_funnel(d):
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 4.2),
        gridspec_kw=dict(width_ratios=[1.0, 1.5]))
    x = np.arange(2)
    tried = [d["s1t"], d["s2t"]]; kept = [d["s1k"], d["s2k"]]
    axL.bar(x, tried, 0.55, color=C_TRIED, label="tried")
    axL.bar(x, kept, 0.55, color=C_ACC, label="accepted")
    for xi, tr, ke in zip(x, tried, kept):
        axL.text(xi, tr + max(tried) * 0.02, f"{tr}", ha="center", fontsize=9)
        axL.text(xi, ke + max(tried) * 0.02,
                 f"{ke} ({100*ke/max(tr,1):.0f}%)", ha="center",
                 fontsize=9, color=C_ACC)
    axL.set_xticks(x)
    axL.set_xticklabels(["stage 1\nconsequent step",
                         "stage 2\nresolution increase"], fontsize=9)
    axL.set_ylabel("adaptation trials")
    axL.set_title("evolving-fuzzy stages", fontsize=11)
    axL.legend(frameon=False, fontsize=9); axL.margins(y=0.15)
    order = ["proposals", "G1-G2 form/vocab/cell", "transport",
             "G3 forecast", "G4 robustness", "G5 margin", "admitted"]
    vals = {"proposals": d["prop"], "admitted": d["adm"]}; vals.update(d["gates"])
    ys = [vals.get(k, 0) for k in order]
    cols = [C_TRIED, C_REJ, C_TRANS, C_REJ, C_REJ, C_REJ, C_ADM]
    ypos = np.arange(len(order))[::-1]
    axR.barh(ypos, ys, color=cols)
    for yp, v in zip(ypos, ys):
        axR.text(v + max(ys) * 0.01, yp, f"{v}", va="center", fontsize=9)
    lab = {"proposals": "proposals",
           "G1-G2 form/vocab/cell": "rejected G1-G2\n(form/vocab/cell/relevance)",
           "transport": "transport failure\n(timeout, unreachable)",
           "G3 forecast": "rejected G3\n(forecast)",
           "G4 robustness": "rejected G4\n(robustness)",
           "G5 margin": "rejected G5\n(package margin)",
           "admitted": "admitted"}
    axR.set_yticks(ypos)
    axR.set_yticklabels([lab[k] for k in order], fontsize=8.5)
    axR.set_xlabel("proposals")
    axR.set_title("generative stage 3 gate cascade", fontsize=11)
    axR.margins(x=0.12)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig_openspace_funnel.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def fig_products():
    st = json.load(open(os.path.join(LOGS, "dss_generated_state.json"),
                        encoding="utf-8"))
    ivs = st["genai_interventions"]; concepts = st.get("genai_concepts", [])
    chans = []
    for iv in ivs:
        for c in iv["composition"]:
            if c["channel"] not in chans:
                chans.append(c["channel"])
    names = [iv["name"] for iv in ivs]
    M = np.zeros((len(ivs), len(chans)))
    for i, iv in enumerate(ivs):
        for c in iv["composition"]:
            M[i, chans.index(c["channel"])] = c["weight"]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([c.replace("_", " ") for c in chans],
                       rotation=30, ha="right", fontsize=9)
    DISC = {"water_drafting", "retardant_drop", "tactical_burn"}
    for lbl, ch in zip(ax.get_xticklabels(), chans):
        if ch in DISC:
            lbl.set_color("#b0530a"); lbl.set_fontweight("bold")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(chans)):
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if M[i, j] > 0.6 else "#222")
    ax.set_title("composition weights of the generated macro interventions",
                 fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("weight on base channel", fontsize=9)
    if concepts:
        c = concepts[0]
        inp = " + ".join(f"{i['weight']:.2f} {i['name'].replace('_',' ')}"
                         for i in c["inputs"])
        ax.text(0.0, -0.40,
                "orange columns: discoverable actuators (validated in the "
                "physics, ordered by no seed rule)", transform=ax.transAxes,
                fontsize=8.5, color="#b0530a")
        ax.text(0.0, -0.30,
                f"generated concept (L{c['layer']}): "
                f"{c['name'].replace('_',' ')} = {inp}",
                transform=ax.transAxes, fontsize=9, style="italic")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig_openspace_products.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p


def _pick_run():
    """A representative CONTAINED run for the anatomy view: rich adaptation
    activity across the three stages, but on an incident the layer actually
    brings under control (final burned-area cost not saturated), so the fire
    growth curve rises and then flattens rather than running away. The cost
    of a lost fire would misrepresent the mechanism. Deterministic."""
    best = None; bestscore = -1
    for f in sorted(glob.glob(os.path.join(LOGS, "DSS_*", "cycles.jsonl"))):
        n = ev = acc = s3 = 0; jb_last = None; t0 = None
        for line in open(f, encoding="utf-8"):
            if '"costs"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            n += 1
            if t0 is None:
                t0 = d.get("t_min")
            jb = (d.get("costs") or {}).get("j_burn")
            if jb is not None:
                jb_last = jb
            a = d.get("adaptation") or {}
            if a.get("stage", 0) >= 1 and (a.get("tried") or 0) > 0:
                ev += 1
                if a.get("accepted"):
                    acc += 1
                if a.get("stage") == 3:
                    s3 += 1
        if n < 100 or jb_last is None or jb_last > 0.75:
            continue           # skip lost / saturated fires
        if t0 is None or t0 > 5:
            continue           # start of the incident must be logged
        score = 2 * acc + 3 * s3 + ev
        if score > bestscore:
            bestscore = score; best = f
    if best is None:
        best = sorted(glob.glob(os.path.join(LOGS, "DSS_*",
                                             "cycles.jsonl")))[0]
    return best


def fig_timeline():
    f = _pick_run()
    ts = []; jt = []; bc = []; ev = []
    for line in open(f, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("t_min")
        if t is None:
            continue
        costs = d.get("costs") or {}; sim = d.get("sim") or {}
        ts.append(t); jt.append(costs.get("j_burn", np.nan))
        bc.append(sim.get("burning", np.nan))
        a = d.get("adaptation") or {}
        if a.get("stage", 0) >= 1 and (a.get("tried") or 0) > 0:
            ev.append((t, int(a.get("stage")), bool(a.get("accepted"))))
    order = np.argsort(ts)
    ts = list(np.array(ts)[order]); jt = list(np.array(jt)[order])
    bc = list(np.array(bc)[order])
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(11.0, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.15], hspace=0.08))
    # cumulative burned area (accumulates, then plateaus) - left axis
    l1, = axT.plot(ts, jt, color="#b2452f", lw=1.7,
                   label="cumulative burned area (left)")
    axT.fill_between(ts, 0, jt, color="#b2452f", alpha=0.10)
    axT.set_ylabel("normalized\nburned area", color="#b2452f")
    axT.set_ylim(0, 1.02); axT.tick_params(axis="y", labelcolor="#b2452f")
    axT.set_title("anatomy of one recorded run", fontsize=11)
    axT.grid(axis="y", alpha=0.2)
    # active fire front (rises, then is driven down) - right axis
    axT2 = axT.twinx()
    l2, = axT2.plot(ts, bc, color="#2166ac", lw=1.7,
                    label="active burning cells (right)")
    axT2.set_ylabel("active burning\ncells", color="#2166ac")
    axT2.tick_params(axis="y", labelcolor="#2166ac")
    axT2.set_ylim(0, max(bc) * 1.12 if bc else 1)
    axT.legend(handles=[l1, l2], frameon=False, fontsize=8.5,
               loc="center right")
    lane = {1: 1, 2: 2, 3: 3}
    for (t, stg, acc) in ev:
        axB.plot([t], [lane[stg]], "^" if acc else "x",
                 color=C_ADM if acc else C_REJ, ms=6, mew=1.4)
    axB.legend(handles=[
        Line2D([], [], marker="^", color="w", markerfacecolor=C_ADM,
               markeredgecolor=C_ADM, label="accepted", ms=7),
        Line2D([], [], marker="x", color=C_REJ, label="rejected",
               ms=7, ls="")], frameon=False, fontsize=8, ncol=2,
        loc="upper right")
    axB.set_yticks([1, 2, 3])
    axB.set_yticklabels(["stage 1", "stage 2", "stage 3"], fontsize=8.5)
    axB.set_ylim(0.5, 3.5); axB.set_xlabel("incident time (min)")
    axB.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig_openspace_timeline.png")
    fig.savefig(p, dpi=DPI); plt.close(fig); return p, os.path.basename(os.path.dirname(f))


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = aggregate_funnel()
    print("FUNNEL:", d)
    print("written:", fig_funnel(d))
    print("written:", fig_products())
    p, run = fig_timeline()
    print("written:", p, "| anatomy run:", run)


if __name__ == "__main__":
    main()
