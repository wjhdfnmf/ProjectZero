"""
Plot cart trainer results as SVG (Python standard library only).
Reuses the Chart class from ../sine_trainer/plot.py.

usage: python plot_cart.py      (run from cart_trainer/)
reads:  outputs/fitness.csv, outputs/rollout.csv, outputs/eval.csv
        outputs/experiments/<config>_s<seed>/ (if ./experiments.sh has been run)
writes: outputs/fitness.svg, outputs/rollout_angle.svg, outputs/rollout_cart.svg,
        outputs/eval_starts.svg, outputs/experiments.svg, outputs/compare_angle.svg
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sine_trainer"))
import plot  # noqa: E402
from plot import Chart  # noqa: E402

OUT = HERE / "outputs"
plot.OUT = OUT  # Chart.save writes into plot.OUT
CONFIGS = [  # (name, color, label)
    ("original", "#969696", "original"),
    ("better_search", "#fd8d3c", "sine search"),
    ("multi_start", "#41ab5d", "multi-start"),
    ("track_fix", "#9e9ac8", "track fix"),
    ("full_penalty", "#4292c6", "full penalty"),
    ("full_walls", "#d7301f", "full walls"),
]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rs, k):
    return [float(r[k]) for r in rs]


def save(chart, name):
    chart.save(name)


def plot_run(run=OUT, suffix=""):
    fit = rows(run / "fitness.csv")
    g, best, avg = col(fit, "generation"), col(fit, "best_cost"), col(fit, "avg_cost")
    lo = 10 ** math.floor(math.log10(max(min(best), 1e-3)))
    hi = 10 ** math.ceil(math.log10(max(avg)))
    c = Chart("Training cost (lower is better)", "generation", "cost per step (log)", (0, g[-1]), (lo, hi), ylog=True)
    c.axes(range(0, int(g[-1]) + 1, max(1, int(g[-1]) // 5)),
           [10 ** e for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1)], "{:g}")
    c.line(g, avg, "#9ecae1", "population avg")
    c.line(g, best, "#d7301f", "best network")
    save(c, f"fitness{suffix}.svg")

    ro = rows(run / "rollout.csv")
    t = [0.0] + col(ro, "time")
    th = [95.0] + col(ro, "theta_deg")
    c = Chart("Pendulum angle, start 95°", "time (s)", "angle (degrees, 90 = upright)", (0, t[-1]), (0, 180))
    c.axes(range(0, int(t[-1]) + 1, 5), [0, 45, 90, 135, 180])
    c.line([0, t[-1]], [90, 90], "#222", "upright", 1, "6,4")
    c.line(t, th, "#d7301f", "pendulum")
    save(c, f"rollout_angle{suffix}.svg")

    x = [0.0] + col(ro, "x")
    f = [0.0] + col(ro, "force")
    span = max(6.0, math.ceil(max(abs(v) for v in x) + 1))
    c = Chart("Cart position and push force", "time (s)", "position (m) / force (N ÷ 10)", (0, t[-1]), (-span, span))
    step = max(1, int(span // 3))
    c.axes(range(0, int(t[-1]) + 1, 5), list(range(-int(span), int(span) + 1, step)))
    c.line([0, t[-1]], [5, 5], "#222", "track edge", 1, "6,4")
    c.line([0, t[-1]], [-5, -5], "#222", "", 1, "6,4")
    c.legend.pop()
    c.line(t, x, "#4292c6", "cart x (m)")
    c.line(t, [v / 10 for v in f], "#fd8d3c", "force ÷ 10 (N)", 1.5)
    save(c, f"rollout_cart{suffix}.svg")

    ev = rows(run / "eval.csv")
    s, dev = col(ev, "start_deg"), col(ev, "final_dev_deg")
    c = Chart("Final tilt from each start angle", "start angle (degrees)", "|angle − 90| at 20 s (degrees)",
              (45, 135), (0, max(30, math.ceil(max(dev) / 10) * 10)))
    c.axes(range(45, 136, 15), range(0, int(c.y1) + 1, max(5, int(c.y1) // 6)))
    c.line([45, 135], [10, 10], "#222", "10° limit", 1, "6,4")
    c.line(s, dev, "#d7301f", "final tilt")
    save(c, f"eval_starts{suffix}.svg")


def plot_experiments():
    base = OUT / "experiments"
    if not base.exists():
        return
    curves = {}
    for name, _, _ in CONFIGS:
        runs = sorted(base.glob(f"{name}_s*/fitness.csv"))
        if runs:
            cols = [col(rows(f), "best_cost") for f in runs]
            n = min(len(c) for c in cols)
            curves[name] = [sorted(c[i] for c in cols)[len(cols) // 2] for i in range(n)]
    if curves:
        gens = max(len(c) for c in curves.values()) - 1
        lo = 10 ** math.floor(math.log10(min(min(c) for c in curves.values())))
        hi = 10 ** math.ceil(math.log10(max(max(c) for c in curves.values())))
        c = Chart("Best training cost, median of 5 seeds", "generation", "cost per step (log)", (0, gens), (lo, hi), ylog=True)
        c.axes(range(0, gens + 1, max(1, gens // 5)), [10 ** e for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1)], "{:g}")
        for name, color, label in CONFIGS:
            if name in curves:
                c.line(range(len(curves[name])), curves[name], color, label, 3 if name == CONFIGS[-1][0] else 2)
        save(c, "experiments.svg")

    # angle from 95 degrees: seed 1 of each config
    c = Chart("Pendulum angle from 95°, seed 1", "time (s)", "angle (degrees, 90 = upright)", (0, 20), (60, 120))
    c.axes(range(0, 21, 5), [60, 75, 90, 105, 120])
    c.line([0, 20], [90, 90], "#222", "upright", 1, "6,4")
    for name, color, label in CONFIGS:
        f = base / f"{name}_s1" / "rollout.csv"
        if f.exists():
            ro = rows(f)
            c.line([0.0] + col(ro, "time"), [95.0] + [min(120, max(60, v)) for v in col(ro, "theta_deg")], color, label)
    save(c, "compare_angle.svg")


if __name__ == "__main__":
    if (OUT / "rollout.csv").exists():
        plot_run()
    plot_experiments()
