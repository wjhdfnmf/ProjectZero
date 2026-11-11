"""
Plot the sine trainer results as SVG (Python standard library only).

usage: python plot.py            (run from sine_trainer/ after ./run)
reads:  outputs/fitness.csv, outputs/snapshots.csv
writes: outputs/fitness.svg, outputs/sine_fit.svg
"""

import csv
import math
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
W, H = 720, 420
M = dict(left=70, right=170, top=50, bottom=55)
COLORS = ["#c7c7c7", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#d7301f"]


def read_csv(name):
    with open(OUT / name, newline="") as f:
        return list(csv.DictReader(f))


class Chart:
    def __init__(self, title, xlabel, ylabel, xr, yr, ylog=False):
        self.ylog = ylog
        self.x0, self.x1 = xr
        self.y0, self.y1 = (math.log10(yr[0]), math.log10(yr[1])) if ylog else yr
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
            f'<rect width="{W}" height="{H}" fill="white"/>',
            f'<text x="{(M["left"] + W - M["right"]) / 2}" y="28" font-size="16" '
            f'text-anchor="middle" font-weight="bold">{title}</text>',
            f'<text x="{(M["left"] + W - M["right"]) / 2}" y="{H - 12}" text-anchor="middle">{xlabel}</text>',
            f'<text x="18" y="{(M["top"] + H - M["bottom"]) / 2}" text-anchor="middle" '
            f'transform="rotate(-90 18 {(M["top"] + H - M["bottom"]) / 2})">{ylabel}</text>',
        ]
        self.legend = []

    def px(self, x):
        return M["left"] + (x - self.x0) / (self.x1 - self.x0) * (W - M["left"] - M["right"])

    def py(self, y):
        if self.ylog:
            y = math.log10(max(y, 10 ** self.y0))
        return H - M["bottom"] - (y - self.y0) / (self.y1 - self.y0) * (H - M["top"] - M["bottom"])

    def axes(self, xticks, yticks, yfmt="{:g}"):
        l, r, t, b = M["left"], W - M["right"], M["top"], H - M["bottom"]
        for x in xticks:
            p = self.px(x)
            self.parts.append(f'<line x1="{p:.1f}" y1="{t}" x2="{p:.1f}" y2="{b}" stroke="#eee"/>')
            self.parts.append(f'<text x="{p:.1f}" y="{b + 18}" text-anchor="middle">{x:g}</text>')
        for y in yticks:
            p = self.py(y)
            self.parts.append(f'<line x1="{l}" y1="{p:.1f}" x2="{r}" y2="{p:.1f}" stroke="#eee"/>')
            self.parts.append(f'<text x="{l - 8}" y="{p + 4:.1f}" text-anchor="end">{yfmt.format(y)}</text>')
        self.parts.append(f'<rect x="{l}" y="{t}" width="{r - l}" height="{b - t}" fill="none" stroke="#333"/>')

    def line(self, xs, ys, color, label, width=2, dash=None):
        pts = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>')
        self.legend.append((color, label, dash))

    def save(self, name):
        x = W - M["right"] + 15
        for i, (color, label, dash) in enumerate(self.legend):
            y = M["top"] + 12 + i * 20
            d = f' stroke-dasharray="{dash}"' if dash else ""
            self.parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" stroke="{color}" stroke-width="3"{d}/>')
            self.parts.append(f'<text x="{x + 30}" y="{y + 4}">{label}</text>')
        self.parts.append("</svg>")
        (OUT / name).write_text("\n".join(self.parts), encoding="utf-8")
        print("wrote", OUT / name)


def plot_fitness():
    rows = read_csv("fitness.csv")
    gen = [int(r["generation"]) for r in rows]
    best = [float(r["best_mse"]) for r in rows]
    avg = [float(r["avg_mse"]) for r in rows]
    lo = 10 ** math.floor(math.log10(max(min(best), 1e-6)))
    hi = 10 ** math.ceil(math.log10(max(avg)))
    c = Chart("Training error (lower is better)", "generation", "mean squared error (log)",
              (0, gen[-1]), (lo, hi), ylog=True)
    step = max(1, gen[-1] // 6)
    yt = [10 ** e for e in range(int(math.log10(lo)), int(math.log10(hi)) + 1)]
    c.axes(range(0, gen[-1] + 1, step), yt, "{:.0e}")
    c.line(gen, avg, "#9ecae1", "population avg")
    c.line(gen, best, "#d7301f", "best network")
    c.save("fitness.svg")


def plot_fit():
    rows = read_csv("snapshots.csv")
    gens = sorted({int(r["generation"]) for r in rows})
    c = Chart("Best network vs sin(x)", "x (radians)", "output", (-math.pi, math.pi), (-1.25, 1.25))
    c.axes([-3, -2, -1, 0, 1, 2, 3], [-1, -0.5, 0, 0.5, 1])
    first = [r for r in rows if int(r["generation"]) == gens[0]]
    c.line([float(r["x"]) for r in first], [float(r["target"]) for r in first], "#222", "sin(x) target", 3, "6,4")
    colors = COLORS[-len(gens):]
    for g, color in zip(gens, colors):
        pts = [r for r in rows if int(r["generation"]) == g]
        c.line([float(r["x"]) for r in pts], [float(r["prediction"]) for r in pts], color, f"gen {g}",
               3 if g == gens[-1] else 2)
    c.save("sine_fit.svg")


if __name__ == "__main__":
    plot_fitness()
    plot_fit()
