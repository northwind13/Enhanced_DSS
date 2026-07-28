"""Figures for Table 5.8 - physical outcome per scenario and configuration.

The table reports, for each of the five scenarios and each configuration,
the burned area, the burned forest area, the affected population, the
evacuated population and the time to containment. This script draws it:

  * 15 per-scenario figures - five scenarios x three metrics (burned area,
    burned forest, affected population), each a chart of its own, the
    configurations along the x axis;
  * 3 average figures - the same three metrics averaged over the five
    scenarios.

THE NUMBERS ARE READ OUT OF THE THESIS, not copied into this file. A second
copy of a table is a second table, and the two drift: a value corrected in
the document would leave the figures quietly wrong. The parser finds the
table by its own header row (Burned (ha) / Burned Forest / Affected), so it
does not depend on the table's position either.

    python validation/plot_table58.py
    python validation/plot_table58.py --docx "../01_Thesis/OTHER.docx"
    python validation/plot_table58.py --print      # just show what it read
"""

from __future__ import annotations

import argparse
import html
import os
import re
import zipfile

import matplotlib

# THE BACKEND IS CHOSEN BEFORE PYPLOT IS IMPORTED, which is why this reads
# sys.argv here rather than in main(): with --show the figures have to open
# in a window you can edit and save yourself, and pyplot fixes the backend
# at import time. Without it, the file-writing backend, so the script runs
# the same on a machine with no display.
import sys                                                        # noqa: E402
if not any(a in ("--show", "-s") for a in sys.argv[1:]):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOCX = os.path.normpath(os.path.join(
    HERE, "..", "..", "01_Thesis", "DISASTERAWARE_PhDThesis_Fin1.docx"))
OUTDIR = os.path.join(HERE, "figures", "table58")

#: the three metrics asked for, with the column header they carry in the
#: table and the axis label they need on a figure
METRICS = [
    ("burned_ha", "Burned (ha)", "burned area (ha)"),
    ("burned_forest", "Burned Forest (ha)", "burned forest (ha)"),
    ("affected_pop", "AffectedPop.", "affected population (people)"),
]

#: the configurations the FIGURES show, in the order the thesis lists them.
#: TF40 - the 40-rule doctrine run statically - is not among them: the
#: comparison the chapter makes is what the five-rule seed base does with
#: and without the adaptation stages, and the doctrine arm sat in the
#: middle of every chart answering a question that is no longer asked. It
#: is still PARSED (see CONFIG_PARSE), so the numbers can be printed with
#: --with-t40 without touching the table.
CONFIG_ORDER = ["T0", "TF5", "TF5+Ev", "TF5+Ev+AI"]
CONFIG_PARSE = ["T0", "TF5", "TF40", "TF5+Ev", "TF5+Ev+AI"]
#: WHAT A ROW MAY BE CALLED IN THE DOCUMENT. The thesis renamed the full
#: configuration to T_DisasterAware, and the parser - matching the old
#: string exactly - skipped those rows without a word, so the figures came
#: out missing the very configuration the chapter is about. Labels are
#: normalised (case, spaces, underscores, subscript braces dropped) and
#: looked up here, and anything that looks like a data row but matches
#: nothing is REPORTED rather than silently ignored.
CONFIG_ALIAS = {
    "t0": "T0",
    "tf5": "TF5",
    "tf40": "TF40",
    "tf5+ev": "TF5+Ev",
    "tf5+ev+ai": "TF5+Ev+AI",
    "tdisasteraware": "TF5+Ev+AI",
    "disasteraware": "TF5+Ev+AI",
}


def _config_key(label: str):
    """The configuration a row label names, or None."""
    k = re.sub(r"[\s_{}$]", "", str(label or "")).lower()
    return CONFIG_ALIAS.get(k)
#: WHAT THE FIGURES CALL EACH CONFIGURATION. The keys are the strings the
#: TABLE uses, because that is what the parser matches on; the labels are
#: what the thesis calls them. The full configuration is the system this
#: work is about, so it carries its name rather than a recipe: TF5+Ev+AI
#: reads as one more ablation, T_DisasterAware reads as the thing.
CONFIG_LABEL = {
    "T0": "$T_0$\nno DSS",
    "TF5": "$T_{F5}$\n5 rules",
    "TF40": "$T_{F40}$\n40 rules",
    "TF5+Ev": "$T_{F5+Ev}$\n5 + evFIS",
    "TF5+Ev+AI": "$T_{DisasterAware}$\n5 + evFIS + GenAI",
}
#: the same names without the second line, for the crowded combined view
CONFIG_SHORT = {
    "T0": "$T_0$",
    "TF5": "$T_{F5}$",
    "TF40": "$T_{F40}$",
    "TF5+Ev": "$T_{F5+Ev}$",
    "TF5+Ev+AI": "$T_{DisasterAware}$",
}
#: TYPE SIZES. A figure drawn at 6 inches and printed at 3 loses half of
#: every letter, so the defaults here are deliberately large; --fontscale
#: multiplies all of them for a page that is smaller or larger still.
LABEL_FS = 9.0        # the value written on a bar
TICK_FS = 12.0        # configuration names and the number axis
AXIS_FS = 13.0        # the axis title
TITLE_FS = 14.0       # the panel title (S1 ...)
LEGEND_FS = 12.0


def set_font_scale(k: float) -> None:
    """Scale every type size on the figures by `k`."""
    global LABEL_FS, TICK_FS, AXIS_FS, TITLE_FS, LEGEND_FS
    LABEL_FS *= k
    TICK_FS *= k
    AXIS_FS *= k
    TITLE_FS *= k
    LEGEND_FS *= k


#: one colour per configuration, so a reader who has seen one figure can
#: read the next fourteen without going back to the legend
CONFIG_COLOR = {
    "T0": "#9e9e9e",
    "TF5": "#7fb3d5",
    "TF40": "#2e86c1",
    "TF5+Ev": "#f0b27a",
    "TF5+Ev+AI": "#cb4335",
}


# ------------------------------------------------------------------ reading
def _docx_tables(path: str):
    """Every table in the document, as a list of rows of cell strings."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    out = []
    for tbl in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S):
        rows = []
        for tr in re.findall(r"<w:tr[ >].*?</w:tr>", tbl, re.S):
            cells = []
            for tc in re.findall(r"<w:tc>.*?</w:tc>", tr, re.S):
                txt = html.unescape(re.sub(r"<[^>]+>", "", tc))
                cells.append(" ".join(txt.split()))
            rows.append(cells)
        out.append(rows)
    return out


def _num(cell: str):
    """The value in a cell, and its +/- spread when the table gives one.

    Cells read like "97.0 ± 28.6", "44.5", "0.0" or an em dash for "did not
    happen". A dash is not zero and must not be plotted as zero, so it comes
    back as None and the bar is left out.
    """
    s = (cell or "").replace("−", "-").strip()
    if not s or s in {"—", "-", "–"}:
        return None, None
    parts = re.split(r"±|\+/-", s)
    try:
        val = float(re.sub(r"[^0-9.\-]", "", parts[0]))
    except ValueError:
        return None, None
    sd = None
    if len(parts) > 1:
        try:
            sd = float(re.sub(r"[^0-9.\-]", "", parts[1]))
        except ValueError:
            sd = None
    return val, sd


def read_table58(path: str):
    """Table 5.8 as {scenario: {config: {metric: (value, sd)}}}.

    Found by its header row rather than by position: a table that grows an
    appendix ahead of it must not send this looking at the wrong numbers.
    """
    want = {m[1].lower().replace(" ", "") for m in METRICS}
    for rows in _docx_tables(path):
        if not rows:
            continue
        head = [c.lower().replace(" ", "") for c in rows[0]]
        if not want <= set(head):
            continue
        col = {}
        for key, header, _lab in METRICS:
            col[key] = head.index(header.lower().replace(" ", ""))
        data, scen, unknown = {}, None, []
        for r in rows[1:]:
            if not r or not r[0]:
                continue
            name = r[0].strip()
            # a scenario row carries the name and nothing else
            if re.fullmatch(r"S\d+", name) and not any(c.strip()
                                                       for c in r[1:]):
                scen = name
                data[scen] = {}
                continue
            key = _config_key(name)
            if key is None:
                # a row with numbers in it that we could not name is a
                # configuration the figures would drop in silence
                if scen and any(_num(c)[0] is not None for c in r[1:]):
                    unknown.append(f"{scen}/{name}")
                continue
            if scen is None or key not in CONFIG_PARSE:
                continue
            data[scen][key] = {k: _num(r[col[k]]) if col[k] < len(r)
                               else (None, None) for k, _h, _l in METRICS}
        if unknown:
            print("WARNING: rows whose configuration name is not known "
                  "(add it to CONFIG_ALIAS): " + ", ".join(unknown))
        if data:
            return data
    raise SystemExit(f"Table 5.8 not found in {path}: no table carries the "
                     f"header {sorted(want)}")


#: WHAT THE CAMPAIGN CALLS EACH CONFIGURATION. The campaign's arm
#: names predate the thesis's, and this is the only place the two
#: vocabularies meet.
ARM_TO_CONFIG = {
    "Test0": "T0", "F5": "TF5", "F40": "TF40",
    "F5Ev": "TF5+Ev", "F5EvAI": "TF5+Ev+AI",
}


def read_campaign_csv(path: str):
    """The same structure, read from experiments/out/table58_phys.csv.

    The document is still the source for the FIGURES THAT GO IN IT -
    a value corrected by hand in Word must show up in the plot - but
    a campaign that has just finished has not reached the document
    yet, and waiting for the fill step to plot the run that produced
    it is a detour. Same numbers either way; this is the short road.
    """
    import csv
    data = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cfg = ARM_TO_CONFIG.get(r["arm"])
            if cfg is None or cfg not in CONFIG_PARSE:
                continue
            sc = r["scenario"]
            def _f(k):
                try:
                    return float(r[k])
                except (TypeError, ValueError):
                    return None
            data.setdefault(sc, {})[cfg] = {
                "burned_ha": (_f("burned_ha"), _f("burned_ci")),
                "burned_forest": (_f("forest_ha"), None),
                "affected_pop": (_f("pop_affected"), None),
            }
    if not data:
        raise SystemExit(f"no usable rows in {path}")
    return data


# ------------------------------------------------------------------ drawing
def fit_labels(fig, axes, pad=1.03, rounds=6):
    """Raise the top of the axis until every value label is inside it.

    The number on a bar is drawn at the bar's height, so how far above
    the bar it reaches depends on the length of the string and the
    type size, not on the data. A fixed headroom - 25 % over the
    tallest bar - is therefore a guess, and on the tall thin bars it
    guessed wrong: "3,117" set sideways crossed the frame line.

    So the labels are MEASURED. Matplotlib can only report where a
    text ended up after a draw, and raising the ceiling stretches the
    data-to-pixel scale, which makes the same text cover more data
    units than before - hence the loop rather than one correction.
    """
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
    except Exception:
        return                       # a backend with nothing to draw on
    for _ in range(rounds):
        need = None
        for ax in axes:
            inv = ax.transData.inverted()
            lo, hi = ax.get_ylim()
            for t in ax.texts:
                y = inv.transform((0, t.get_window_extent(rend).y1))[1]
                if need is None or y > need:
                    need = y
        if need is None:
            return
        tops = [ax.get_ylim()[1] for ax in axes]
        if need <= max(tops) / pad:
            return                   # already clear, with room to spare
        for ax in axes:
            ax.set_ylim(top=need * pad)
        fig.canvas.draw()


def _bar(ax, cfgs, vals, errs, ylabel, title):
    xs = np.arange(len(cfgs))
    cols = [CONFIG_COLOR.get(c, "#888888") for c in cfgs]
    # a missing value is missing, not zero: it is left out of the bars and
    # named under the axis instead
    ok = [i for i, v in enumerate(vals) if v is not None]
    ax.bar(xs[ok], [vals[i] for i in ok],
           yerr=([errs[i] if errs[i] is not None else 0.0 for i in ok]
                 if any(errs[i] is not None for i in ok) else None),
           color=[cols[i] for i in ok], width=0.62,
           capsize=4, error_kw=dict(lw=1.1, ecolor="#333333"))
    for i in ok:
        _v = vals[i]
        ax.text(xs[i], _v + (0.02 * max(v for v in vals if v is not None)),
                f"{_v:,.1f}" if _v < 1000 else f"{_v:,.0f}",
                ha="center", va="bottom", fontsize=LABEL_FS + 2)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONFIG_LABEL.get(c, c) for c in cfgs],
                       fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    _top = max([v for v in vals if v is not None] + [1.0])
    ax.set_ylim(0, _top * 1.18)


def per_scenario_figures(data, outdir: str, save: bool = True):
    paths = []
    for scen in sorted(data):
        for key, header, ylabel in METRICS:
            cfgs = [c for c in CONFIG_ORDER if c in data[scen]]
            vals = [data[scen][c][key][0] for c in cfgs]
            errs = [data[scen][c][key][1] for c in cfgs]
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            _bar(ax, cfgs, vals, errs, ylabel,
                 f"{scen} — {header}")
            fig.tight_layout()
            fit_labels(fig, [ax])
            fig.canvas.manager.set_window_title(f"{scen} · {header}")
            p = os.path.join(outdir, f"table58_{scen}_{key}.png")
            if save:
                fig.savefig(p, dpi=300)
                plt.close(fig)
            paths.append(p)
    return paths


def average_figures(data, outdir: str, save: bool = True):
    """The three metrics averaged over the scenarios.

    Plain bars: NO error bar. The spread across five scenarios is not an
    uncertainty about the average - it is the fact that the scenarios
    differ, which is what the fifteen per-scenario figures are for - and
    drawing it here invited the reader to take it for one.
    """
    paths = []
    for key, header, ylabel in METRICS:
        cfgs = [c for c in CONFIG_ORDER if any(c in data[s] for s in data)]
        means = []
        for c in cfgs:
            vals = [data[s][c][key][0] for s in sorted(data)
                    if c in data[s] and data[s][c][key][0] is not None]
            means.append(float(np.mean(vals)) if vals else None)
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        _bar(ax, cfgs, means, [None] * len(cfgs),
             f"mean {ylabel}", f"Average over S1\u2013S5 \u2014 {header}")
        fig.tight_layout()
        fit_labels(fig, [ax])
        fig.canvas.manager.set_window_title(f"AVG · {header}")
        p = os.path.join(outdir, f"table58_AVG_{key}.png")
        if save:
            fig.savefig(p, dpi=300)
            plt.close(fig)
        paths.append(p)
    return paths


#: one colour per METRIC for the combined view, where the configurations
#: are the x axis and the three metrics stand side by side
METRIC_COLOR = {
    "burned_ha": "#c0392b",
    "burned_forest": "#1e8449",
    "affected_pop": "#2471a3",
}


def combined_figure(data, outdir: str, save: bool = True,
                    relative: bool = False, fname: str = None):
    """ONE figure, one number axis, everything comparable.

    Fifteen separate charts answer "how did the configurations do on this
    metric in this scenario" and nothing else: to compare a metric across
    scenarios, or two metrics against each other, the reader has to hold
    three pictures in their head. Here the scenarios are panels sharing a
    single y axis on the left, each configuration is a group, and the three
    metrics stand side by side inside it.

    The metrics do not share a unit - hectares against people - and their
    magnitudes differ by twenty times, so a linear axis would flatten the
    burned area into the baseline. Two honest ways out, and the switch says
    which one you are looking at:

      absolute  a LOG axis, where a bar's height is its order of magnitude
                and the printed value is the number itself;
      relative  every value as a percentage of that scenario's no-DSS run
                (T0 = 100%), which puts the three metrics on one linear
                scale because they are all "what fraction of the damage
                remained".
    """
    scens = sorted(data)
    # A SINGLE PANEL NEEDS ITS OWN WIDTH. The five-panel figure is sized by
    # the number of panels, and asking for one scenario left a 4-inch sheet
    # with the title and the legend running off both edges.
    _w = (3.1 * len(scens) + 1.2) if len(scens) > 1 else 7.4
    fig, axes = plt.subplots(1, len(scens), figsize=(_w, 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.26
    for ax, scen in zip(axes, scens):
        cfgs = [c for c in CONFIG_ORDER if c in data[scen]]
        xs = np.arange(len(cfgs))
        for m, (key, header, _lab) in enumerate(METRICS):
            vals, labs = [], []
            for c in cfgs:
                v = data[scen][c][key][0]
                if relative and v is not None:
                    base = data[scen].get("T0", {}).get(key, (None,))[0]
                    labs.append(v)
                    v = (100.0 * v / base) if base else None
                else:
                    labs.append(v)
                vals.append(v)
            off = (m - 1) * width
            ok = [i for i, v in enumerate(vals) if v is not None]
            # ZERO IS A RESULT, and a log axis cannot draw it: "no one was
            # affected" is the strongest thing this table says about
            # S3/TF5+Ev+AI, and it was the one bar the chart left out. It
            # is drawn as a stub at the floor and labelled 0.
            _floor = 0.6 if not relative else 0.0
            _h = [(vals[i] if vals[i] > 0 else _floor) for i in ok]
            ax.bar(xs[ok] + off, _h, width=width,
                   color=METRIC_COLOR[key],
                   label=header if ax is axes[0] else None)
            for n, i in enumerate(ok):
                _t = ("0" if labs[i] == 0 else
                      (f"{labs[i]:,.0f}" if labs[i] >= 1000
                       else f"{labs[i]:,.1f}"))
                # A FIGURE IS READ AT PRINT SIZE. These figures go into a
                # page at a fraction of the size they are drawn at, so the
                # value on a bar and the name under it have to be set large
                # here to still be legible there.
                ax.text(xs[i] + off, _h[n], _t, ha="center", va="bottom",
                        fontsize=LABEL_FS, rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels([CONFIG_SHORT.get(c, c) for c in cfgs],
                           fontsize=TICK_FS, rotation=30, ha="right")
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_title(scen, fontsize=TITLE_FS)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    if relative:
        axes[0].set_ylabel("share of the no-DSS run $T_0$ (%)",
                           fontsize=AXIS_FS)
        axes[0].axhline(100.0, color="#555555", lw=0.8, ls="--")
        _top = 0.0
        for scen in scens:
            for c in data[scen]:
                for key, _h, _l in METRICS:
                    v = data[scen][c][key][0]
                    b = data[scen].get("T0", {}).get(key, (None,))[0]
                    if v is not None and b:
                        _top = max(_top, 100.0 * v / b)
        axes[0].set_ylim(0, max(_top * 1.25, 120))
    else:
        axes[0].set_yscale("log")
        axes[0].set_ylim(bottom=0.5)
        axes[0].set_ylabel("hectares / people (log scale)",
                           fontsize=AXIS_FS)
    # NO TABLE NAME ON THE FIGURE. A figure printed in the thesis carries
    # its caption from the document; repeating "Table 5.8 - physical
    # outcome per scenario and configuration" inside the image duplicated
    # the caption and squeezed the legend against it. What the figure shows
    # is already said by the panel titles, the axis and the legend - and
    # "relative to T0" is in the axis label, not lost with the title.
    fig.legend(loc="upper center", ncol=len(METRICS), frameon=False,
               fontsize=LEGEND_FS, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    # the panels share one axis, so one measurement serves all five
    fit_labels(fig, list(axes))
    fig.canvas.manager.set_window_title(
        "Table 5.8 combined" + (" (relative)" if relative else ""))
    # NAMED AFTER WHAT IS IN IT. Every combined run wrote the same file, so
    # asking for S1 and then for S3 left one figure on disk and no way to
    # tell which scenario it held.
    _tag = "" if len(scens) > 1 else "_" + scens[0]
    p = os.path.join(outdir, fname or
                     f"table58_combined{_tag}"
                     f"{'_relative' if relative else ''}.png")
    if save:
        fig.savefig(p, dpi=300)
        plt.close(fig)
    return [p]


def combined_average_figure(data, outdir: str, save: bool = True,
                            relative: bool = False):
    """The same comparison, averaged over the scenarios: one panel."""
    avg = {"S1\u2013S5": {}}
    for c in CONFIG_ORDER:
        if not any(c in data[s] for s in data):
            continue
        avg["S1\u2013S5"][c] = {}
        for key, _h, _l in METRICS:
            vals = [data[s][c][key][0] for s in sorted(data)
                    if c in data[s] and data[s][c][key][0] is not None]
            avg["S1\u2013S5"][c][key] = (float(np.mean(vals)) if vals
                                          else None, None)
    return combined_figure(
        avg, outdir, save=save, relative=relative,
        fname=("table58_combined_avg_relative.png" if relative
               else "table58_combined_avg.png"))


def main():
    ap = argparse.ArgumentParser(
        description="Draw Table 5.8 of the thesis: 15 per-scenario figures "
                    "and 3 averages.")
    ap.add_argument("--docx", default=DEFAULT_DOCX)
    ap.add_argument("--csv", default="",
                    help="read the numbers from a campaign result file "
                         "(experiments/out/table58_phys.csv) instead of "
                         "the thesis, for plotting a run that has not "
                         "been written into the document yet")
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--print", dest="show_only", action="store_true",
                    help="print the parsed table and stop")
    ap.add_argument("-s", "--show", action="store_true",
                    help="open the figures in windows instead of writing "
                         "PNGs: edit them (toolbar: 'Configure subplots' "
                         "and 'Edit axis, curve and image parameters') and "
                         "save with the toolbar's save button")
    ap.add_argument("--scenario", default="",
                    help="only these scenarios, e.g. S1 or S1,S3 "
                         "(with --show, so you do not open 18 windows)")
    ap.add_argument("--metric", default="",
                    help="only these metrics: burned_ha, burned_forest, "
                         "affected_pop (comma separated)")
    ap.add_argument("--no-average", action="store_true",
                    help="skip the average figures")
    ap.add_argument("--only-average", action="store_true",
                    help="ONLY the average over the scenarios, no "
                         "per-scenario figures at all")
    ap.add_argument("--combined", action="store_true",
                    help="ONE figure: the scenarios as panels sharing a "
                         "single left axis, the three metrics side by side "
                         "in every configuration")
    ap.add_argument("--each", action="store_true",
                    help="with --combined: one figure PER scenario "
                         "(table58_combined_S1.png ...) instead of the "
                         "five-panel one")
    ap.add_argument("--with-t40", action="store_true",
                    help="also draw the 40-rule static arm (TF40), which "
                         "the figures leave out by default")
    ap.add_argument("--fontscale", type=float, default=1.0,
                    help="multiply every type size on the figures "
                         "(default 1.0; try 1.3 for a small page)")
    ap.add_argument("--relative", action="store_true",
                    help="with --combined: plot each value as a percentage "
                         "of that scenario's no-DSS run (T0 = 100%%), which "
                         "puts hectares and people on one linear scale")
    args = ap.parse_args()

    global CONFIG_ORDER
    if args.with_t40:
        CONFIG_ORDER = list(CONFIG_PARSE)
    if args.fontscale != 1.0:
        set_font_scale(float(args.fontscale))
    data = (read_campaign_csv(args.csv) if args.csv
            else read_table58(args.docx))
    # a subset keeps --show usable: eighteen windows at once is not editing,
    # it is hunting
    if args.scenario:
        want = {w.strip().upper() for w in args.scenario.split(",")}
        data = {k: v for k, v in data.items() if k.upper() in want}
        if not data:
            raise SystemExit(f"no scenario matches {args.scenario}")
    global METRICS
    if args.metric:
        want_m = {w.strip() for w in args.metric.split(",")}
        METRICS = [m for m in METRICS if m[0] in want_m]
        if not METRICS:
            raise SystemExit(f"no metric matches {args.metric}")

    print(f"read Table 5.8 from {args.docx}")
    print(f"  {len(data)} scenario(s) x "
          f"{len(next(iter(data.values())))} configurations")
    for scen in sorted(data):
        for c in CONFIG_ORDER:
            if c not in data[scen]:
                continue
            row = data[scen][c]
            print(f"  {scen} {c:10s} " + "  ".join(
                f"{k}={row[k][0]}" + (f"±{row[k][1]}" if row[k][1] else "")
                for k, _h, _l in METRICS))
    if args.show_only:
        return

    if args.show:
        if args.only_average:
            combined_average_figure(data, args.out, save=False,
                                    relative=args.relative)
        elif args.combined:
            if args.each:
                for _s in sorted(data):
                    combined_figure({_s: data[_s]}, args.out, save=False,
                                    relative=args.relative)
            else:
                combined_figure(data, args.out, save=False,
                                relative=args.relative)
            if not args.no_average:
                combined_average_figure(data, args.out, save=False,
                                        relative=args.relative)
        else:
            per_scenario_figures(data, args.out, save=False)
            if not args.no_average:
                average_figures(data, args.out, save=False)
        n = len(plt.get_fignums())
        print(f"\n{n} figure window(s) open. Edit them from the toolbar "
              "('Configure subplots' for the layout, 'Edit axis, curve and "
              "image parameters' for titles, labels, limits and colours), "
              "then save each one with the toolbar's save button. Nothing "
              "is written to disk by this script in --show mode.")
        plt.show()
        return

    os.makedirs(args.out, exist_ok=True)
    if args.only_average:
        # the chapter asks for the average alone; drawing fifteen
        # per-scenario figures to throw them away is only slow
        p1 = []
        p2 = combined_average_figure(data, args.out,
                                     relative=args.relative)
    elif args.combined and args.each:
        p1 = []
        for _s in sorted(data):
            p1 += combined_figure({_s: data[_s]}, args.out,
                                  relative=args.relative)
        p2 = ([] if args.no_average
              else combined_average_figure(data, args.out,
                                           relative=args.relative))
    elif args.combined:
        p1 = combined_figure(data, args.out, relative=args.relative)
        p2 = ([] if args.no_average
              else combined_average_figure(data, args.out,
                                           relative=args.relative))
    else:
        p1 = per_scenario_figures(data, args.out)
        p2 = [] if args.no_average else average_figures(data, args.out)
    print(f"\n{len(p1)} per-scenario figure(s) + {len(p2)} average(s) "
          f"-> {args.out}")
    for p in p1 + p2:
        print("   ", os.path.basename(p))


if __name__ == "__main__":
    main()
