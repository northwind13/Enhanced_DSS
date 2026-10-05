"""Write DisasterAware_IEEESystems_v3.docx (tracked) and _v3_clean.docx
from the v2 article, the v3 manuscript text and the campaign outputs.

  python experiments/open_space/write_article_v3.py \
      --v2 02_Article/ieee_systems/DisasterAware_IEEESystems_v2.docx \
      --out-dir 02_Article/ieee_systems --data 03_Codes/experiments/open_space/out

Every change is a tracked insertion or deletion by author "Reviewer".
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import docx_track as T                                   # noqa: E402
from docx import Document                                # noqa: E402
from docx.oxml.ns import qn                              # noqa: E402
from docx.shared import Inches                           # noqa: E402

T.AUTHOR = "Reviewer"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ------------------------------------------------------------ xml helpers
def _ptext(p):
    return "".join(t.text or "" for t in p.iter()
                   if t.tag in (qn("w:t"), qn("w:delText"))
                   and t.getparent().getparent().tag != qn("w:del"))


def _live_text(p):
    out = []
    for t in p.iter(qn("w:t")):
        r = t.getparent()
        if r.getparent().tag == qn("w:del"):
            continue
        out.append(t.text or "")
    return "".join(out)


def _mark_para_deleted(p):
    """Delete every run and the paragraph mark, so that accepting the
    change removes the paragraph entirely."""
    T.del_runs_of(p)
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = T._el("w:pPr")
        p.insert(0, ppr)
    rpr = ppr.find(qn("w:rPr"))
    if rpr is None:
        rpr = T._el("w:rPr")
        ppr.append(rpr)
    rpr.append(T._el("w:del", **{"w:id": T._nid(), "w:author": T.AUTHOR,
                                 "w:date": T.DATE}))


def _mark_table_deleted(tbl):
    for tr in tbl.iter(qn("w:tr")):
        trpr = tr.find(qn("w:trPr"))
        if trpr is None:
            trpr = T._el("w:trPr")
            tr.insert(0, trpr)
        trpr.append(T._el("w:del", **{"w:id": T._nid(), "w:author": T.AUTHOR,
                                      "w:date": T.DATE}))
        for p in tr.iter(qn("w:p")):
            T.del_runs_of(p)


def _new_para(text, style=None, bold_lead=None, italic=False, like=None):
    """like: an existing paragraph whose pPr and first-run rPr are copied
    (used for reference entries, which carry direct formatting)."""
    p = T._el("w:p")
    src_ppr = like.find(qn("w:pPr")) if like is not None else None
    ppr = copy.deepcopy(src_ppr) if src_ppr is not None else T._el("w:pPr")
    for e in ppr.findall(qn("w:rPr")) + ppr.findall(qn("w:sectPr")):
        ppr.remove(e)
    if style and src_ppr is None:
        ps = T._el("w:pStyle", **{"w:val": style})
        ppr.append(ps)
    rpr = T._el("w:rPr")
    rpr.append(T._el("w:ins", **{"w:id": T._nid(), "w:author": T.AUTHOR,
                                 "w:date": T.DATE}))
    ppr.append(rpr)
    p.append(ppr)
    if bold_lead:
        p.append(T.ins_run(bold_lead, bold=True))
        text = text[len(bold_lead):]
    run = T.ins_run(text, italic=italic)
    if like is not None:
        r0 = like.find(qn("w:r"))
        if r0 is None:
            r0 = next(like.iter(qn("w:r")), None)
        rpr0 = r0.find(qn("w:rPr")) if r0 is not None else None
        if rpr0 is not None:
            run.find(qn("w:r")).insert(0, copy.deepcopy(rpr0))
    p.append(run)
    return p


def _insert_after(anchor, elems):
    cur = anchor
    for e in elems:
        cur.addnext(e)
        cur = e
    return cur


def _new_table(doc, header, rows, widths=None, total=5000, size=14):
    """A column-width table in the template's style: fixed layout, rules
    above and below, 7 pt text, column widths proportional to content."""
    ncol = len(header)
    if widths is None:
        lens = [max([len(str(header[j]))] + [len(str(r[j])) for r in rows])
                for j in range(ncol)]
        lens = [max(l, 3) for l in lens]
        widths = [int(total * l / sum(lens)) for l in lens]
    widths[-1] += total - sum(widths)
    tbl = doc.add_table(rows=1 + len(rows), cols=ncol)
    el = tbl._tbl
    # table properties
    tblpr = el.find(qn("w:tblPr"))
    for ch in list(tblpr):
        tblpr.remove(ch)
    tblpr.append(T._el("w:tblW", **{"w:w": str(total), "w:type": "dxa"}))
    tblpr.append(T._el("w:jc", **{"w:val": "center"}))
    bd = T._el("w:tblBorders")
    for side, sz in (("w:top", "8"), ("w:bottom", "8"), ("w:insideH", "4")):
        bd.append(T._el(side, **{"w:val": "single", "w:sz": sz,
                                 "w:space": "0", "w:color": "auto"}))
    tblpr.append(bd)
    tblpr.append(T._el("w:tblLayout", **{"w:type": "fixed"}))
    cm = T._el("w:tblCellMar")
    for side in ("w:left", "w:right"):
        cm.append(T._el(side, **{"w:w": "30", "w:type": "dxa"}))
    tblpr.append(cm)
    grid = el.find(qn("w:tblGrid"))
    for ch in list(grid):
        grid.remove(ch)
    for w in widths:
        grid.append(T._el("w:gridCol", **{"w:w": str(w)}))
    for j, h in enumerate(header):
        c = tbl.rows[0].cells[j]
        c.paragraphs[0]._p.append(T.ins_run(str(h), bold=True))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            tbl.rows[i + 1].cells[j].paragraphs[0]._p.append(
                T.ins_run(str(v)))
    for r in tbl.rows:
        T._ins_row(r)
        for j, c in enumerate(r.cells):
            tcpr = c._tc.find(qn("w:tcPr"))
            if tcpr is None:
                tcpr = T._el("w:tcPr")
                c._tc.insert(0, tcpr)
            for ch in list(tcpr):
                tcpr.remove(ch)
            tcpr.append(T._el("w:tcW", **{"w:w": str(widths[j]),
                                          "w:type": "dxa"}))
            for p in c.paragraphs:
                ppr = p._p.find(qn("w:pPr"))
                if ppr is None:
                    ppr = T._el("w:pPr")
                    p._p.insert(0, ppr)
                sp = T._el("w:spacing", **{"w:before": "0", "w:after": "0"})
                ppr.append(sp)
                rpr = T._el("w:rPr")
                rpr.append(T._el("w:ins", **{"w:id": T._nid(),
                                             "w:author": T.AUTHOR,
                                             "w:date": T.DATE}))
                rpr.append(T._el("w:sz", **{"w:val": str(size)}))
                ppr.append(rpr)
                for run in p._p.iter(qn("w:r")):
                    rp = run.find(qn("w:rPr"))
                    if rp is None:
                        rp = T._el("w:rPr")
                        run.insert(0, rp)
                    rp.append(T._el("w:sz", **{"w:val": str(size)}))
                    rp.append(T._el("w:szCs", **{"w:val": str(size)}))
    el.getparent().remove(el)
    return el


def _picture_para(doc, path, width_in, caption_style="FigureCaption",
                  caption=None):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_picture(path, width=Inches(width_in))
    el = p._p
    el.getparent().remove(el)
    # wrap the picture run as an insertion, mark the paragraph inserted
    r = el.find(qn("w:r"))
    ins = T._el("w:ins", **{"w:id": T._nid(), "w:author": T.AUTHOR,
                            "w:date": T.DATE})
    el.replace(r, ins)
    ins.append(r)
    ppr = el.find(qn("w:pPr"))
    if ppr is None:
        ppr = T._el("w:pPr")
        el.insert(0, ppr)
    ppr.append(T._el("w:jc", **{"w:val": "center"}))
    rpr = T._el("w:rPr")
    rpr.append(T._el("w:ins", **{"w:id": T._nid(), "w:author": T.AUTHOR,
                                 "w:date": T.DATE}))
    ppr.append(rpr)
    out = [el]
    if caption:
        out.append(_new_para(caption, style=caption_style))
    return out


# ------------------------------------------------------- accept changes
def accept_all(doc):
    body = doc.element.body
    # deleted rows
    for tr in list(body.iter(qn("w:tr"))):
        trpr = tr.find(qn("w:trPr"))
        if trpr is not None and trpr.find(qn("w:del")) is not None:
            tr.getparent().remove(tr)
    for tr in body.iter(qn("w:tr")):
        trpr = tr.find(qn("w:trPr"))
        if trpr is not None:
            for e in trpr.findall(qn("w:ins")):
                trpr.remove(e)
    # deleted paragraphs (mark on pPr/rPr)
    for p in list(body.iter(qn("w:p"))):
        ppr = p.find(qn("w:pPr"))
        if ppr is not None:
            rpr = ppr.find(qn("w:rPr"))
            if rpr is not None and rpr.find(qn("w:del")) is not None:
                p.getparent().remove(p)
                continue
            if rpr is not None:
                for e in rpr.findall(qn("w:ins")):
                    rpr.remove(e)
    for d in list(body.iter(qn("w:del"))):
        par = d.getparent()
        if par is not None:
            par.remove(d)
    for ins in list(body.iter(qn("w:ins"))):
        par = ins.getparent()
        idx = list(par).index(ins)
        for k, child in enumerate(list(ins)):
            par.insert(idx + k, child)
        par.remove(ins)
    return doc


# ------------------------------------------------------------ manuscript
def parse_manuscript(path):
    """Split the markdown into named blocks of (style, text) paragraphs."""
    txt = open(path, encoding="utf-8").read()
    blocks = {}
    cur = None
    for line in txt.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            blocks[cur] = []
            continue
        if cur is None:
            continue
        blocks[cur].append(line)
    out = {}
    for k, lines in blocks.items():
        paras = []
        buf = []

        def flush():
            if buf:
                s = " ".join(x.strip() for x in buf).strip()
                if s:
                    paras.append(s)
                buf.clear()
        for ln in lines:
            if not ln.strip():
                flush()
                continue
            if ln.startswith("### "):
                flush()
                paras.append(("H2", ln[4:].strip()))
                continue
            if ln.startswith("TABLE ") or ln.startswith("|") \
                    or ln.startswith("Notes for the writer") \
                    or ln.startswith("Conventions:") or ln.startswith("---"):
                flush()
                continue
            buf.append(ln)
        flush()
        out[k] = paras
    return out


_SUP = {"10^-4": "10\u207b\u2074", "10^-3": "10\u207b\u00b3", "10^-2": "10\u207b\u00b2"}


def fill_placeholders(text, F):
    for a, b in _SUP.items():
        text = text.replace(a, b)
    def rep(m):
        k = m.group(1)
        return str(F.get(k, m.group(0)))
    return re.sub(r"\{\{([^}]+)\}\}", rep, text)


# ------------------------------------------------------------- numbers
def _load_csv(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_fill(data):
    F = {}
    tdir = os.path.join(data, "tables")
    runs = _load_csv(os.path.join(data, "runs.csv"))
    red = _load_csv(os.path.join(tdir, "E1_rediscovery.csv"))
    pair1 = _load_csv(os.path.join(tdir, "E1_paired.csv"))
    out2 = _load_csv(os.path.join(tdir, "E2_outcomes.csv"))
    pair2 = _load_csv(os.path.join(tdir, "E2_paired.csv"))
    prop3 = _load_csv(os.path.join(tdir, "E3_proposers.csv"))
    out3 = _load_csv(os.path.join(tdir, "E3_outcomes.csv"))
    out4 = _load_csv(os.path.join(tdir, "E4_outcomes.csv"))

    def f(x):
        try:
            return float(x)
        except Exception:
            return float("nan")
    # E1
    byr = {(r["scenario"], r["arm"]): r for r in red}
    for sc in ("S6", "S7", "S8"):
        for arm in ("Evo", "Random"):
            r = byr.get((sc, arm))
            if r:
                n = int(float(r["n"]))
                F[f"E1.any.{sc}.{arm}"] = f"{round(f(r['any_share']) * n)}"
                for tgt in ("draft", "coat", "ignite"):
                    F[f"E1.{tgt}.{sc}.{arm}"] = \
                        f"{round(f(r[tgt + '_share']) * n)}"
                    F[f"E1.{tgt}.{sc}.{arm}.t"] = \
                        f"{f(r[tgt + '_median_min']):.0f}" \
                        if r[tgt + '_median_min'] not in ("", None) else "n/a"
    evo_runs = [r for r in runs if r["exp"] == "E1" and r["arm"] == "Evo"]
    if evo_runs:
        anyk = sum(1 for r in evo_runs if max(f(r["redis_draft"]),
                                              f(r["redis_coat"]),
                                              f(r["redis_ignite"])) == 1)
        F["E1.any_share_evo_pct"] = f"{100 * anyk / len(evo_runs):.0f}"
    rnd_runs = [r for r in runs if r["exp"] == "E1" and r["arm"] == "Random"]
    if rnd_runs:
        anyk = sum(1 for r in rnd_runs if max(f(r["redis_draft"]),
                                              f(r["redis_coat"]),
                                              f(r["redis_ignite"])) == 1)
        F["E1.any_share_rnd_pct"] = f"{100 * anyk / len(rnd_runs):.0f}"
    for tgt in ("draft", "coat", "ignite"):
        ks = [round(f(r[tgt + "_share"]) * int(float(r["n"])))
              for (sc, arm), r in byr.items()]
        if ks:
            F[f"E1.{tgt}.min"] = f"{min(ks)}"
            F[f"E1.{tgt}.max"] = f"{max(ks)}"
    e1a = defaultdict(list)
    for r in runs:
        if r["exp"] == "E1":
            e1a[r["arm"]].append(f(r["n_actuators"]))
    for arm, key in (("Evo", "evo"), ("Random", "rnd")):
        if e1a.get(arm):
            F[f"E1.{key}.act_per_run"] = f"{np.mean(e1a[arm]):.1f}"

    def best(tgt):
        cands = []
        for (sc, arm), r in byr.items():
            n = int(float(r["n"]))
            cands.append((round(f(r[tgt + "_share"]) * n), n, sc, arm))
        if not cands:
            return "n/a"
        k, n, sc, arm = max(cands)
        return f"{k} of {n} on {sc} with the {'evolutionary' if arm == 'Evo' else 'random'} proposer"
    for tgt in ("draft", "coat", "ignite"):
        F[f"E1.{tgt}.best"] = best(tgt)
    for sc in ("S6", "S8"):
        cands = [(round(f(r[f"draft_share"]) * int(float(r["n"]))),
                  int(float(r["n"])), arm) for (s_, arm), r in byr.items()
                 if s_ == sc]
        if cands:
            k, n, arm = max(cands)
            F[f"E1.draft.{sc}.best"] = f"{k} of {n} on {sc}"
    byp = {(r["scenario"], r["arm"]): r for r in pair1}
    for sc in ("S6", "S7", "S8"):
        for arm in ("Evo", "Random"):
            r = byp.get((sc, arm))
            if r:
                F[f"E1.delta.{sc}.{arm}"] = \
                    f"{-f(r['delta_ha']):.0f} ± {f(r['hw']):.0f}"
    for arm in ("Evo", "Random"):
        rr = [r for r in pair1 if r["arm"] == arm]
        if rr:
            F[f"E1.wins.{arm}"] = str(sum(int(float(r["wins"])) for r in rr))
            F[f"E1.n.{arm}"] = str(sum(int(float(r["n"])) for r in rr))
    # E1: shape of the admitted sited interventions
    adm_p = os.path.join(data, "admitted.jsonl")
    if os.path.exists(adm_p):
        ncl, rng = [], defaultdict(list)
        with open(adm_p, encoding="utf-8") as fh:
            for ln in fh:
                a = json.loads(ln)
                if a.get("exp") != "E1" or not a.get("clauses"):
                    continue
                ncl.append(len(a["clauses"]))
                for cl in a["clauses"]:
                    rng[cl["effect"]].append(tuple(cl["range"]))
        if ncl:
            F["E1.sited.n"] = f"{len(ncl)}"
            F["E1.sited.clauses_mean"] = f"{np.mean(ncl):.1f}"
            F["E1.sited.multi_pct"] = f"{100 * np.mean([k > 1 for k in ncl]):.0f}"
            for e in ("draft", "coat", "ignite"):
                if rng.get(e):
                    F[f"E1.sited.{e}.rin"] = f"{np.median([r[0] for r in rng[e]]):.0f}"
                    F[f"E1.sited.{e}.rout"] = f"{np.median([r[1] for r in rng[e]]):.0f}"
    # E2
    by2 = {(r["scenario"], r["arm"]): r for r in out2}
    byp2 = {(r["scenario"], r["arm"]): r for r in pair2}
    fams = [s_ for s_ in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
            if (s_, "F5Ev") in by2]
    for sc in fams:
        for arm in ("Evo", "Random", "F5"):
            r = byp2.get((sc, arm))
            if r:
                F[f"E2.delta.{sc}.{arm}"] = \
                    f"{-f(r['delta_ha']):.0f} ± {f(r['hw']):.0f}"
                F[f"E2.wins.{sc}.{arm}"] = f"{int(f(r['wins']))}"
                F[f"E2.n.{sc}.{arm}"] = f"{int(f(r['n']))}"
        o = by2.get((sc, "F5Ev"))
        if o:
            F[f"E2.closed.{sc}.burn"] = f"{f(o['burned_ha']):.0f}"
        for arm in ("Evo", "Random", "T0"):
            o = by2.get((sc, arm))
            if o:
                F[f"E2.{arm}.{sc}.burn"] = f"{f(o['burned_ha']):.0f}"
                F[f"E2.{arm}.{sc}.succ"] = f"{f(o['success_pct']):.0f}"
    if fams:
        n2 = [int(f(byp2[(sc, "Evo")]["n"])) for sc in fams
              if (sc, "Evo") in byp2]
        F["E2.seeds"] = f"{min(n2)}" if n2 else "n/a"

        def pooled(group, arm):
            c = np.mean([f(by2[(s_, "F5Ev")]["burned_ha"]) for s_ in group])
            o = np.mean([f(by2[(s_, arm)]["burned_ha"]) for s_ in group
                         if (s_, arm) in by2])
            return c, o
        c, o = pooled(fams, "Evo")
        F["E2.closed_mean"] = f"{c:.0f}"
        F["E2.evo_mean"] = f"{o:.0f}"
        F["E2.open_vs_closed_pct"] = f"{100 * (c - o) / c:.0f}"
        c, o = pooled(fams, "Random")
        F["E2.rnd_mean"] = f"{o:.0f}"
        F["E2.rnd_vs_closed_pct"] = f"{100 * (c - o) / c:.0f}"
        hard = [s_ for s_ in fams if s_ in ("S6", "S7", "S8")]
        easy = [s_ for s_ in fams if s_ not in ("S6", "S7", "S8")]
        if hard:
            c, o = pooled(hard, "Evo")
            F["E2.open_vs_closed_S678"] = f"{c - o:.0f}"
            F["E2.open_pct_S678"] = f"{100 * (c - o) / c:.0f}"
        if easy:
            c, o = pooled(easy, "Evo")
            F["E2.open_vs_closed_S15"] = f"{c - o:.0f}"
            F["E2.open_pct_S15"] = f"{100 * (c - o) / c:.0f}"
        # families on which each open arm is better on average
        for arm, key in (("Evo", "evo"), ("Random", "rnd")):
            k = sum(1 for sc in fams if (sc, arm) in byp2
                    and f(byp2[(sc, arm)]["delta_ha"]) < 0)
            F[f"E2.{key}.families_better"] = f"{k}"
        F["E2.n_families"] = f"{len(fams)}"
        # F5 against F5+Ev: largest absolute paired difference
        d5 = [abs(f(byp2[(sc, "F5")]["delta_ha"])) for sc in fams
              if (sc, "F5") in byp2]
        if d5:
            F["E2.f5_vs_f5ev_max"] = f"{max(d5):.0f}"
        bestp = min((r for r in pair2 if r["arm"] in ("Evo", "Random")),
                    key=lambda r: f(r["delta_ha"]), default=None)
        if bestp:
            F["E2.best_family"] = (f"{bestp['scenario']} "
                                   f"({-f(bestp['delta_ha']):.0f} ha with the "
                                   f"{'evolutionary' if bestp['arm'] == 'Evo' else 'random'} proposer)")
        # stabilisation share pooled over families
        for arm, key in (("F5Ev", "closed"), ("Evo", "evo"), ("Random", "rnd")):
            v = [f(by2[(sc, arm)]["success_pct"]) for sc in fams
                 if (sc, arm) in by2]
            if v:
                F[f"E2.{key}.succ_mean"] = f"{np.mean(v):.0f}"
        # relative reduction per family, evolutionary proposer
        for sc in fams:
            if (sc, "Evo") in by2:
                c = f(by2[(sc, "F5Ev")]["burned_ha"])
                o = f(by2[(sc, "Evo")]["burned_ha"])
                F[f"E2.relpct.{sc}.Evo"] = f"{100 * (c - o) / c:.0f}"
    # E3
    by3 = {r["arm"]: r for r in prop3}
    o3 = defaultdict(list)
    a3 = defaultdict(list)
    for r in out3:
        o3[r["arm"]].append(f(r["burned_ha"]))
        a3[r["arm"]].append(f(r["n_actuators"]))
    for arm, key in (("Evo", "evo"), ("Random", "rnd"),
                     ("Evo-noprior", "evo_noprior"),
                     ("Random-noprior", "rnd_noprior")):
        r = by3.get(arm)
        if r:
            att = f(r.get("attempts", 0)) or 1
            adm = f(r.get("admitted", 0))
            F[f"E3.{key}.adm_pct"] = f"{100 * adm / att:.0f}"
            F[f"E3.{key}.fpa"] = f"{f(r.get('forecasts_per_admission')):.0f}"
            ka = f(r.get("kind_actuator", 0)) or 1
            F[f"E3.{key}.act_pct"] = f"{100 * f(r.get('acc_actuator', 0)) / ka:.0f}"
            F[f"E3.{key}.act_tried"] = f"{int(f(r.get('kind_actuator', 0)))}"
            F[f"E3.{key}.act_adm"] = f"{int(f(r.get('acc_actuator', 0)))}"
            F[f"E3.{key}.rule_tried"] = f"{int(f(r.get('kind_rule', 0)))}"
            F[f"E3.{key}.rule_adm"] = f"{int(f(r.get('acc_rule', 0)))}"
            F[f"E3.{key}.rule_pct"] = f"{100 * f(r.get('acc_rule', 0)) / max(f(r.get('kind_rule', 0)), 1):.0f}"
            F[f"E3.{key}.s0_pct"] = f"{100 * f(r.get('rej_S0', 0)) / att:.0f}"
            F[f"E3.{key}.attempts"] = f"{int(att)}"
            F[f"E3.{key}.admitted"] = f"{int(adm)}"
        if o3.get(arm):
            F[f"E3.{key}.burn"] = f"{np.mean(o3[arm]):.0f}"
            F[f"E3.{key}.act_per_run"] = f"{np.mean(a3[arm]):.1f}"
    # paired effect of the prior on the evolutionary proposer, same seeds
    e3 = {(r["scenario"], r["arm"], int(r["seed"])): f(r["burned_ha"])
          for r in runs if r["exp"] == "E3"}
    for sc in ("S6", "S7", "S8"):
        ds = [e3[(sc, "Evo", sd)] - e3[(sc, "Evo-noprior", sd)]
              for (s_, a_, sd) in e3 if s_ == sc and a_ == "Evo"
              and (sc, "Evo-noprior", sd) in e3]
        if ds:
            a = np.asarray(ds)
            hw = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
            F[f"E3.prior.delta.{sc}"] = f"{-a.mean():.0f} ± {hw:.0f}"
            F[f"E3.prior.wins.{sc}"] = f"{int((a < 0).sum())}"
            F[f"E3.prior.n.{sc}"] = f"{len(a)}"
    e3runs = [r for r in runs if r["exp"] == "E3" and r["arm"] == "Evo"]
    if e3runs:
        # wall seconds per adaptation window with a search proposer
        ws = [f(r["wall_s"]) / max(f(r["prop_calls"]), 1) for r in e3runs]
        F["E3.evo.wall"] = f"{np.median(ws):.0f}"
    # E4: one arm per margin, labelled Evo@<margin> by report.py
    prop4 = _load_csv(os.path.join(tdir, "E4_proposers.csv"))
    red4 = _load_csv(os.path.join(tdir, "E4_rediscovery.csv"))
    if out4:
        g = defaultdict(list)
        b = defaultdict(list)
        anyk = defaultdict(list)
        for r in runs:
            if r["exp"] == "E4":
                m = f"{f(r['margin']):g}"
                g[m].append(f(r["n_actuators"]))
                b[m].append(f(r["burned_ha"]))
                anyk[m].append(1.0 if max(f(r["redis_draft"]), f(r["redis_coat"]),
                                          f(r["redis_ignite"])) == 1 else 0.0)
        keys = sorted(g, key=lambda k: f(k))
        for m in keys:
            tag = {"0.0001": "1e-4", "0.001": "1e-3", "0.01": "1e-2"}.get(m, m)
            F[f"E4.act.{tag}"] = f"{np.mean(g[m]):.1f}"
            F[f"E4.burn.{tag}"] = f"{np.mean(b[m]):.0f}"
            F[f"E4.redis.{tag}"] = f"{100 * np.mean(anyk[m]):.0f}"
            pr = next((r for r in prop4 if r["arm"].endswith("@" + m)), None)
            if pr:
                att = f(pr.get("attempts", 0)) or 1
                F[f"E4.adm.{tag}"] = f"{100 * f(pr.get('admitted', 0)) / att:.0f}"
                F[f"E4.fpa.{tag}"] = f"{f(pr.get('forecasts_per_admission')):.0f}"
        if len(keys) >= 2:
            F["E4.burn_delta"] = f"{abs(np.mean(b[keys[-1]]) - np.mean(b[keys[0]])):.0f}"
            F["E4.seeds"] = f"{len(b[keys[0]]) // 3}"
    return F


# ---------------------------------------------------------------- tables
def table_rows_E1(data):
    red = _load_csv(os.path.join(data, "tables", "E1_rediscovery.csv"))
    pair = {(r["scenario"], r["arm"]): r for r in
            _load_csv(os.path.join(data, "tables", "E1_paired.csv"))}
    header = ["Family", "Arm", "draft", "coat", "ignite", "any",
              "vs closed (ha)"]
    rows = []
    for r in sorted(red, key=lambda r: (r["scenario"], r["arm"])):
        n = int(float(r["n"]))

        def cell(t):
            k = round(float(r[t + "_share"]) * n)
            med = r[t + "_median_min"]
            return f"{k}" + (f" @{float(med):.0f}" if med else "")
        p = pair.get((r["scenario"], r["arm"]))
        rows.append([r["scenario"], r["arm"], cell("draft"), cell("coat"),
                     cell("ignite"),
                     f"{round(float(r['any_share']) * n)}",
                     (f"{float(p['delta_ha']):+.0f} ± {float(p['hw']):.0f}"
                      if p else "ref.")])
    return header, rows


def table_rows_E2(data):
    out = _load_csv(os.path.join(data, "tables", "E2_outcomes.csv"))
    pair = {(r["scenario"], r["arm"]): r for r in
            _load_csv(os.path.join(data, "tables", "E2_paired.csv"))}
    header = ["Family", "Arm", "burned (ha)", "J_phys", "ext. %",
              "vs closed (ha)"]
    order = {"T0": 0, "F5": 1, "F5Ev": 2, "Evo": 3, "Random": 4, "LLM": 5}
    rows = []
    for r in sorted(out, key=lambda r: (r["scenario"], order.get(r["arm"], 9))):
        if r["arm"] == "F5":
            continue
        p = pair.get((r["scenario"], r["arm"]))
        rows.append([r["scenario"], r["arm"],
                     f"{float(r['burned_ha']):.0f} ± {float(r['burned_hw']):.0f}",
                     f"{float(r['j_phys']):.3f}",
                     f"{float(r['success_pct']):.0f}",
                     (f"{float(p['delta_ha']):+.0f} ± {float(p['hw']):.0f}"
                      if p else ("ref." if r["arm"] == "F5Ev" else ""))])
    return header, rows


def table_rows_E3(data):
    pr = _load_csv(os.path.join(data, "tables", "E3_proposers.csv"))
    out = defaultdict(list)
    for r in _load_csv(os.path.join(data, "tables", "E3_outcomes.csv")):
        out[r["arm"]].append(float(r["burned_ha"]))
    header = ["Proposer", "prior", "attempts", "admitted", "sited adm.",
              "forec./adm.", "misses", "burned (ha)"]
    rows = []
    for r in sorted(pr, key=lambda r: r["arm"]):
        arm = r["arm"]
        rows.append([arm.replace("-noprior", ""),
                     "off" if arm.endswith("-noprior") else "on",
                     r.get("attempts", ""), r.get("admitted", ""),
                     f"{r.get('acc_actuator', 0)}/{r.get('kind_actuator', 0)}",
                     f"{float(r.get('forecasts_per_admission', 0)):.0f}",
                     r.get("rej_S0", 0),
                     f"{np.mean(out[arm]):.0f}" if out.get(arm) else ""])
    return header, rows


def table_rows_E4(data):
    pr = _load_csv(os.path.join(data, "tables", "E4_proposers.csv"))
    runs = [r for r in _load_csv(os.path.join(data, "runs.csv"))
            if r["exp"] == "E4"]
    header = ["G5 margin", "admitted / attempts", "forec./adm.",
              "sited / run", "redisc. %", "burned (ha)"]
    rows = []
    for r in sorted(pr, key=lambda r: float(r["arm"].split("@")[1])):
        m = r["arm"].split("@")[1]
        rr = [x for x in runs if f"{float(x['margin']):g}" == m]
        act = np.mean([float(x["n_actuators"]) for x in rr]) if rr else 0
        red = np.mean([1.0 if max(float(x["redis_draft"]), float(x["redis_coat"]),
                                  float(x["redis_ignite"])) == 1 else 0.0
                       for x in rr]) if rr else 0
        burn = np.mean([float(x["burned_ha"]) for x in rr]) if rr else 0
        att = float(r.get("attempts", 0)) or 1
        adm = float(r.get("admitted", 0))
        rows.append([{"0.0001": "10\u207b\u2074", "0.001": "10\u207b\u00b3", "0.01": "10\u207b\u00b2"}.get(m, m),
                     f"{int(adm)} / {int(att)} ({100 * adm / att:.0f}%)",
                     f"{float(r.get('forecasts_per_admission', 0)):.0f}",
                     f"{act:.1f}", f"{100 * red:.0f}", f"{burn:.0f}"])
    return header, rows


SCENARIO_TABLE = (
    ["ID", "Landscape", "Pool", "Ignitions", "Weather", "What it tests"],
    [["S1", "Remote forest, 2 settlements", "full", "1 remote",
      "8 m/s, fuel moisture 0.08", "baseline competence"],
     ["S2", "Remote forest", "0.6", "2 remote", "same",
      "allocation under scarcity"],
     ["S3", "WUI, 12 settlements", "full", "1 near the interface", "same",
      "life and asset priority"],
     ["S4", "WUI", "0.6", "2 near the interface", "same",
      "full operational stress"],
     ["S5", "as S4", "0.6", "as S4", "same, outage 0.25, noise 0.10",
      "degraded observation"],
     ["S6", "River valley, lake 8 to 11 cells downwind", "0.4", "2 remote",
      "9 m/s, 0.08", "drafting as the only extra capacity"],
     ["S7", "Mediterranean coast, 12 settlements", "full",
      "2 near the interface", "12 m/s, 0.06", "pre-emptive coating"],
     ["S8", "Rolling hills, 18 h cap", "0.4", "3 remote", "8 m/s, 0.08",
      "spent pool, substitution tactics"]])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--manuscript", default=os.path.join(
        ROOT, "..", "02_Article", "ieee_systems", "v3_manuscript.md"))
    ap.add_argument("--data", default=os.path.join(HERE, "out"))
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    figdir = args.figures or os.path.join(args.data, "figures")
    F = compute_fill(args.data)
    M = parse_manuscript(args.manuscript)
    doc = Document(args.v2)
    body = doc.element.body
    paras = list(body.iter(qn("w:p")))
    text = [_live_text(p) for p in paras]

    def find(prefix, start=0):
        for i in range(start, len(paras)):
            if text[i].strip().startswith(prefix):
                return i
        raise KeyError(prefix)

    def para_list(block, fill=True):
        out = []
        for item in M[block]:
            if isinstance(item, tuple):
                out.append(_new_para(item[1], style="Heading2"))
            else:
                t = fill_placeholders(item, F) if fill else item
                # bold lead sentence for the contribution paragraphs
                m = re.match(r"^((?:Open intervention space over|Rediscovery as|End-to-end measurement)[^.]*\.)", t)
                out.append(_new_para(t, style="Text",
                                     bold_lead=m.group(1) if m else None))
        return out

    # ---- abstract and index terms
    i = find("Abstract")
    _mark_para_deleted(paras[i])
    ab = [p for p in M["ABSTRACT (replaces paragraph #7)"]
          if not isinstance(p, tuple)]
    _insert_after(paras[i], [_new_para(fill_placeholders(ab[0], F)),
                             _new_para(ab[1])])
    j = find("Index Terms")
    _mark_para_deleted(paras[j])

    # ---- introduction contributions (#22-#26)
    i0 = find("DisasterAware answers these gaps")
    i1 = find("This work is a systems contribution")
    for k in range(i0, i1 + 1):
        _mark_para_deleted(paras[k])
    _insert_after(paras[i1], para_list(
        "I. INTRODUCTION (paragraphs #10-#12 kept; #22-#26 replaced by the following)"))

    # ---- II heading and related work
    h2 = find("II. Background")
    _mark_para_deleted(paras[h2])
    _insert_after(paras[h2], [_new_para("II. Related Work and Background",
                                        style="Heading1")])
    hb = find("B. Decision Support Foundations")
    hc = find("C. The Input Space")
    for k in range(hb, hc):
        _mark_para_deleted(paras[k])
    _insert_after(paras[hc - 1], para_list(
        "II. RELATED WORK AND BACKGROUND (heading replaces \"II. Background\"; II.A kept; II.B replaced by B, C, D below; old II.C, II.D, II.E become E, F, G with their text unchanged)"))
    for old, new in (("C. The Input Space", "E. The Input Space: Heterogeneous Sensing to Observation Features"),
                     ("D. Concept-Driven Fuzzy Reasoning", "F. Concept-Driven Fuzzy Reasoning"),
                     ("E. The Decision Space", "G. The Decision Space: Interventions and Their Evaluation")):
        k = find(old)
        _mark_para_deleted(paras[k])
        _insert_after(paras[k], [_new_para(new, style="Heading2")])

    # ---- IV.F pointer
    hf = find("F. Open Intervention Space: Staged Adaptation")
    pf = find("Adaptation has three stages")
    _mark_para_deleted(paras[hf])
    _mark_para_deleted(paras[pf])
    _insert_after(paras[pf], para_list(
        "IV.F (replaced by a pointer paragraph; the staged adaptation text moves to Section V)"))

    # ---- drop Fig. 7 (consequent step, described in the text) and
    #      renumber the figures that follow; new figures are 9 and 10
    def _fig_renumber(t):
        return re.sub(r"Fig\. (\d+)", lambda m: "Fig. " + str(
            {8: 7, 9: 8}.get(int(m.group(1)), int(m.group(1)))), t)
    for p in list(body.iter(qn("w:p"))):
        if p.getparent().tag == qn("w:tc"):
            continue
        t = _live_text(p)
        if not t.strip():
            continue
        if t.strip().startswith("Fig. 7.") and "Consequent step" in t:
            _mark_para_deleted(p)
            prev = p.getprevious()
            # the drawing paragraph that carries the picture
            if prev is not None and prev.find(".//" + qn("w:drawing")) is not None:
                _mark_para_deleted(prev)
            continue
        if re.search(r"Fig\. (8|9)\b", t) and p.find(".//" + qn("w:drawing")) is None:
            is_cap = t.strip().startswith("Fig.")
            _mark_para_deleted(p)
            style = "FigureCaption" if is_cap else "Text"
            _insert_after(p, [_new_para(_fig_renumber(t), style=style)])
    # ---- new Section V before old "V. Results"
    hv = find("V. Results")
    new_v = [_new_para("V. The Open Intervention Space", style="Heading1")]
    new_v += para_list("V. THE OPEN INTERVENTION SPACE (new Heading1, inserted before the old \"V. Results\"; old V becomes VII after the new VI)")
    prev = paras[hv].getprevious()
    _insert_after(prev, new_v)
    # old V heading -> VI. Evaluation Design
    _mark_para_deleted(paras[hv])
    _insert_after(paras[hv], [_new_para("VI. Evaluation Design",
                                        style="Heading1")])
    # ---- old V.C .. V.E deleted; VI.C, VI.D and VII inserted
    hc5 = find("C. Experimental Setup and Sensitivity Analysis")
    hvi = find("VI. Discussion")
    # delete paragraphs between (inclusive hc5, exclusive hvi), and tables
    # whose first cell lies in that range
    rng = set(id(p) for p in paras[hc5:hvi])
    for p in paras[hc5:hvi]:
        if p.getparent().tag == qn("w:tc"):
            continue
        _mark_para_deleted(p)
    for tbl in list(body.iter(qn("w:tbl"))):
        first = next(tbl.iter(qn("w:p")), None)
        if first is not None and id(first) in rng:
            _mark_table_deleted(tbl)
    # insert after the last element before old C heading
    anchor = paras[hc5].getprevious()
    blocks = []
    blocks += para_list("VI. EVALUATION DESIGN (new Heading1; takes old V.A criteria table as A and old V.B hindcast as B unchanged; C, D new)")
    # scenario table after the "C. Scenario Families" paragraph block: put
    # it right after the first paragraph of C
    # find index of H2 "D. Arms" in blocks to insert table before it
    vi_blocks = blocks
    idx_d = next(i for i, e in enumerate(vi_blocks)
                 if _live_text(e).startswith("D. Arms"))
    hdr, rows = SCENARIO_TABLE
    vi_blocks[idx_d:idx_d] = [_new_para("TABLE V", style="TableTitle"),
                              _new_para("Scenario families", style="TableTitle"),
                              _new_table(doc, hdr, rows,
                                         widths=[300, 1450, 400, 950, 950, 950])]
    vii = [_new_para("VII. Results", style="Heading1")]
    res = para_list("VII. RESULTS (old \"V. Results\" renumbered; A criteria and B hindcast kept; old C, D, E replaced by the following)")
    # renumber result subsections C..G -> A..E and tables/figures in text
    remap = {"C. Rediscovery": "A. Rediscovery", "D. Closed and Open": "B. Closed and Open",
             "E. Proposers": "C. Proposers", "F. Gate Margin": "D. Gate Margin",
             "G. Compliance": "E. Compliance"}
    fixed = []
    for e in res:
        t = _live_text(e)
        for a, b in remap.items():
            if t.startswith(a):
                e = _new_para(b + t[len(a):], style="Heading2")
        t = _live_text(e)
        t2 = (t.replace("Table XI", "Table IX").replace("Table VIII", "Table VI")
              .replace("Table IX reports E2", "Table VII reports E2")
              .replace("Table X reports", "Table VIII reports")
              .replace("Fig. 15", "Fig. 9").replace("Fig. 16", "Fig. 10"))
        if t2 != t:
            e = _new_para(t2, style=("Heading2" if "Heading2" in e.xml else "Text"))
        fixed.append(e)
    res = fixed
    # tables and figures at their places
    def after_h2(prefix, elems):
        k = next(i for i, e in enumerate(res) if _live_text(e).startswith(prefix))
        # insert after the first text paragraph following the heading
        res[k + 2:k + 2] = elems
    e1h, e1r = table_rows_E1(args.data)
    e2h, e2r = table_rows_E2(args.data)
    e3h, e3r = table_rows_E3(args.data)
    e4h, e4r = table_rows_E4(args.data)
    f10 = os.path.join(figdir, "fig09_rediscovery.png")
    f11 = os.path.join(figdir, "fig10_paired_burned.png")
    el_a = [_new_para("TABLE VI", style="TableTitle"),
            _new_para("Rediscovery of the withheld tactics (E1): runs out of 30 that admitted an equivalent sited intervention, with the median minute of first admission, and paired burned area against the closed reference", style="TableTitle"),
            _new_table(doc, e1h, e1r)]
    if os.path.exists(f10):
        el_a += _picture_para(doc, f10, 3.4, caption="Fig. 9. Cumulative share of runs in which each withheld tactic had been rediscovered, against incident time, for the two search proposers on S6 to S8 (E1).")
    after_h2("A. Rediscovery", el_a)
    el_b = [_new_para("TABLE VII", style="TableTitle"),
            _new_para("Outcome per scenario family and arm with the full library available (E2): mean and 95 percent half-width over 12 seeds, and paired difference in burned area against the closed reference; the static F5 arm is within 12 ha of F5+Ev on every family and is omitted", style="TableTitle"),
            _new_table(doc, e2h, e2r)]
    if os.path.exists(f11):
        el_b += _picture_para(doc, f11, 3.4, caption="Fig. 10. Paired difference in burned area between the open arms and the closed adaptive reference per scenario family (E2); negative is better for the open arm, bars are means with 95 percent confidence intervals; S6 to S8 are the families built so that the closed doctrine is structurally insufficient.")
    after_h2("B. Closed and Open", el_b)
    el_c = [_new_para("TABLE VIII", style="TableTitle"),
            _new_para("Proposer accounting at an equal forecast budget of 36 shadow forecasts per cycle, with and without the composition prior (E3)", style="TableTitle"),
            _new_table(doc, e3h, e3r,
                       widths=[700, 420, 650, 650, 780, 700, 500, 600])]
    after_h2("C. Proposers", el_c)
    if e4r:
        el_d = [_new_para("TABLE IX", style="TableTitle"),
                _new_para("Vocabulary margin of G5 swept over three decades with the evolutionary proposer on S6 to S8, three library tactics withheld (E4; 10 seeds per margin and family)", style="TableTitle"),
                _new_table(doc, e4h, e4r)]
        after_h2("D. Gate Margin", el_d)
    _insert_after(anchor, vi_blocks + vii + res)

    # ---- VI, VII -> VIII, IX; discussion and conclusion replaced
    hdisc = find("VI. Discussion")
    hconc = find("VII. Conclusion")
    hack = find("Acknowledgment")
    for k in range(hdisc, hack):
        _mark_para_deleted(paras[k])
    _insert_after(paras[hack - 1],
                  [_new_para("VIII. Discussion: Systems Lessons and Limitations", style="Heading1")]
                  + para_list("VIII. DISCUSSION: SYSTEMS LESSONS AND LIMITATIONS (replaces old VI)")
                  + [_new_para("IX. Conclusion and Future Work", style="Heading1")]
                  + para_list("IX. CONCLUSION AND FUTURE WORK (replaces old VII)"))
    # ---- references appended
    last_ref = find("[34]")
    refs = [p for p in M["NEW REFERENCES (appended after [34])"]
            if not isinstance(p, tuple) and p.startswith("[")]
    new_refs = [_new_para(r, like=paras[last_ref]) for r in refs]
    _insert_after(paras[last_ref], new_refs)
    # a section break carried by the last reference paragraph moves to
    # the new last reference, so the references stay in one section
    _sp = paras[last_ref].find(qn("w:pPr"))
    _sect = _sp.find(qn("w:sectPr")) if _sp is not None else None
    if _sect is not None:
        _sp.remove(_sect)
        new_refs[-1].find(qn("w:pPr")).append(_sect)
    # ---- cross references in kept text
    for p in paras:
        t = _live_text(p)
        if "Section IV-F" in t and p.getparent().tag != qn("w:tc"):
            _mark_para_deleted(p)
            _insert_after(p, [_new_para(t.replace("Section IV-F", "Section V"), style="Text")])

    # the template's manuscript-id header line
    for sec in doc.sections:
        for hp in sec.header.paragraphs:
            if "REPLACE THIS LINE" in hp.text:
                for r in hp.runs:
                    r.text = ""
    os.makedirs(args.out_dir, exist_ok=True)
    tracked = os.path.join(args.out_dir, "DisasterAware_IEEESystems_v3.docx")
    doc.save(tracked)
    clean = Document(tracked)
    accept_all(clean)
    clean_p = os.path.join(args.out_dir, "DisasterAware_IEEESystems_v3_clean.docx")
    clean.save(clean_p)
    missing = sorted({m for m in re.findall(r"\{\{([^}]+)\}\}",
                                            "\n".join(_live_text(p) for p in clean.element.body.iter(qn("w:p"))))})
    print("written", tracked, clean_p)
    print("unfilled placeholders:", missing)


if __name__ == "__main__":
    main()
