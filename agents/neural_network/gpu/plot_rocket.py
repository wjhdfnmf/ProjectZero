"""
Plots for the GPU network and Booster Landing (Python standard library only).
Line charts reuse Chart from ../sine_trainer/plot.py.

usage: python plot_rocket.py [run_dir]      (default: outputs)
reads:  <run_dir>/fitness.csv, trajectories.csv, test.csv
        outputs/bench.csv, outputs/sizes_summary.csv (if present)
writes: <run_dir>/fitness.svg, trajectories.svg, descent.svg
        outputs/bench.svg, outputs/sizes.svg
"""

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "sine_trainer"))
import plot  # noqa: E402
from plot import Chart  # noqa: E402

OUTCOME = {1: ("landed", "#1a9850"), 2: ("crashed", "#d73027"), 3: ("tipped", "#984ea3"),
           4: ("out of bounds", "#ff7f00"), 5: ("timeout", "#999999")}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_run(run):
    plot.OUT = run
    fit = rows(run / "fitness.csv")
    g = [float(r["generation"]) for r in fit]
    best = [float(r["best_cost"]) for r in fit]
    med = [float(r["median_cost"]) for r in fit]
    lo, hi = math.floor(min(best) / 10) * 10, math.ceil(max(med) / 10) * 10
    c = Chart("Training cost (lower is better)", "generation", "mean cost over 16 scenarios", (0, g[-1]), (lo, hi))
    c.axes(range(0, int(g[-1]) + 1, max(1, int(g[-1]) // 5)), range(int(lo), int(hi) + 1, max(10, (int(hi) - int(lo)) // 6 // 10 * 10)))
    c.line(g, med, "#9ecae1", "population median")
    c.line(g, best, "#d7301f", "best network")
    c.save("fitness.svg")

    traj = rows(run / "trajectories.csv")
    test = {int(r["scenario"]): r for r in rows(run / "test.csv")}
    by = {}
    for r in traj:
        by.setdefault(int(r["scenario"]), []).append(r)
    c = Chart("Test landings: flight paths", "x relative to pad (m)", "altitude (m)", (-800, 800), (0, 2600))
    c.axes(range(-800, 801, 400), range(0, 2601, 500))
    c.line([-15, 15], [8, 8], "#000", "landing pad", 6)
    seen = set()
    for s, pts in sorted(by.items()):
        oc = int(test[s]["outcome"])
        name, color = OUTCOME[oc]
        xs = [float(p["x"]) - float(p.get("pad_x") or 0) for p in pts]
        c.line([min(800, max(-800, v)) for v in xs], [float(p["y"]) for p in pts], color, name, 1.5)
        if name in seen:
            c.legend.pop()
        seen.add(name)
    c.save("trajectories.svg")

    # altitude and throttle against time for the first 4 scenarios
    dt = 0.05
    tmax = max(len(p) for p in by.values()) * dt
    c = Chart("Descent: altitude and throttle", "time (s)", "altitude (km) / throttle (0-1)", (0, math.ceil(tmax / 10) * 10), (0, 2.6))
    c.axes(range(0, int(math.ceil(tmax / 10) * 10) + 1, 10), [0, 0.5, 1, 1.5, 2, 2.5])
    colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83"]
    for (s, pts), col in zip(sorted(by.items())[:4], colors):
        t = [int(p["step"]) * dt for p in pts]
        c.line(t, [float(p["y"]) / 1000 for p in pts], col, f"#{s} altitude", 2)
        c.line(t, [float(p["throttle"]) for p in pts], col, f"#{s} throttle", 1, "3,3")
    c.save("descent.svg")


def bars(name, title, groups, series, values, ylabel, log=False, fmt="{:.3g}"):
    """values[group][series] -> grouped bar chart SVG in outputs/"""
    W, H, L, R, T, B = 760, 420, 80, 20, 50, 80
    vmax = max(v for g in values for v in g if v > 0)
    vmin = min(v for g in values for v in g if v > 0)
    if log:
        lo, hi = math.floor(math.log10(vmin)), math.ceil(math.log10(vmax))
        ypos = lambda v: H - B - (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (H - T - B)
        ticks = [10 ** e for e in range(lo, hi + 1)]
    else:
        hi = vmax * 1.1
        ypos = lambda v: H - B - v / hi * (H - T - B)
        ticks = [hi * i / 5 for i in range(6)]
    colors = ["#bdbdbd", "#74a9cf", "#0570b0", "#d7301f"]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="28" font-size="16" font-weight="bold" text-anchor="middle">{title}</text>',
         f'<text x="18" y="{(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 18 {(T + H - B) / 2})">{ylabel}</text>']
    for tv in ticks:
        y = ypos(tv)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="#eee"/>')
        p.append(f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{(f"{tv:.0e}" if log else f"{tv:.0f}")}</text>')
    gw = (W - L - R) / len(groups)
    bw = gw * 0.8 / len(series)
    for gi, g in enumerate(groups):
        x0 = L + gi * gw + gw * 0.1
        for si, v in enumerate(values[gi]):
            if v <= 0:
                continue
            y = ypos(v)
            p.append(f'<rect x="{x0 + si * bw:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" height="{H - B - y:.1f}" fill="{colors[si % 4]}"/>')
            p.append(f'<text x="{x0 + si * bw + bw / 2:.1f}" y="{y - 3:.1f}" text-anchor="middle" font-size="9">{fmt.format(v)}</text>')
        p.append(f'<text x="{L + gi * gw + gw / 2:.1f}" y="{H - B + 16}" text-anchor="middle">{g}</text>')
    for si, s in enumerate(series):
        x = L + si * 170
        p.append(f'<rect x="{x}" y="{H - 30}" width="14" height="14" fill="{colors[si % 4]}"/>')
        p.append(f'<text x="{x + 20}" y="{H - 19}">{s}</text>')
    p.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#333"/>')
    p.append("</svg>")
    (HERE / "outputs" / name).write_text("\n".join(p), encoding="utf-8")
    print("wrote", HERE / "outputs" / name)


def plot_bench():
    f = HERE / "outputs" / "bench.csv"
    if not f.exists():
        return
    rs = rows(f)
    keys = ["v3_cpu", "v4_cpu", "gpu_dataset", "gpu_episodes"]
    bars("bench.svg", "Forward passes per second (log scale)", [r["network"] for r in rs],
         ["v3 CPU (1 thread)", "v4 CPU (1 thread)", "GPU dataset mode", "GPU episode mode"],
         [[float(r[k]) for k in keys] for r in rs], "passes / second", log=True)


GROUP_TITLES = {"easy": "Easy mode, GA",
                "ga_hard20": "Hard mode, ship ±20 m, GA",
                "ga_hard5": "Hard mode, GA",
                "es_hard5": "Hard mode, Evolution Strategies",
                "ga_hard5_mem": "Hard mode, GA + memory"}


def plot_compare(names):
    """median % landed per network size, one bar series per training setup"""
    data = {}
    for n in names:
        f = HERE / "outputs" / f"sizes_{n}_summary.csv"
        if f.exists():
            for r in rows(f):
                data.setdefault(r["hidden"], {}).setdefault(n, []).append(100 * int(r["landed"]) / int(r["test"]))
    present = [n for n in names if any(n in d for d in data.values())]
    if len(present) < 2:
        return
    hs = list(data.keys())
    med = lambda v: sorted(v)[len(v) // 2] if v else 0
    bars("compare_hard.svg", "Hard mode: median landings (%) over 3 seeds",
         [f"hidden {h}" for h in hs], [GROUP_TITLES[n].replace("Hard mode, ", "") for n in present],
         [[med(data[h].get(n, [])) for n in present] for h in hs], "% landed", fmt="{:.1f}")


def plot_sizes(mode):
    f = HERE / "outputs" / f"sizes_{mode}_summary.csv"
    if not f.exists():
        return
    rs = rows(f)
    nets = []
    for r in rs:
        if r["net"] not in nets:
            nets.append(r["net"])
    vals = []
    for n in nets:
        v = [100 * int(r["landed"]) / int(r["test"]) for r in rs if r["net"] == n]
        vals.append(sorted(v))
    n_seeds = max(len(v) for v in vals)
    bars(f"sizes_{mode}.svg", f"{GROUP_TITLES.get(mode, mode)}: % landed on 1,000 new scenarios",
         nets, [f"seed run {i + 1} (sorted)" for i in range(n_seeds)],
         [v + [0] * (n_seeds - len(v)) for v in vals], "% landed", fmt="{:.1f}")


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs"
    if (run / "trajectories.csv").exists():
        plot_run(run)
    if len(sys.argv) <= 1:  # summary plots only for the default run
        plot_bench()
        for g in GROUP_TITLES:
            plot_sizes(g)
        plot_compare(["ga_hard5", "es_hard5", "ga_hard5_mem"])
