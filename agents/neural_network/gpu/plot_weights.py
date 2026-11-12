"""
Paint a trained network's weights as SVG (Python standard library only).

usage: python plot_weights.py <weights.txt> [out.svg]
       default out: weights.svg next to the weights file

Draws, for one network (weights.txt from rocket.exe):
  - the network: every connection colored by sign (blue +, red -), width by size;
    neurons filled by their bias
  - a heatmap of each weight matrix, with values
  - input influence: how strongly each input reaches each output through the
    hidden layers (sum over paths of |w| products), as a bar chart
Layout follows gpu_nn.cuh: per layer W[in][out] then b[out]; tanh everywhere.
For big networks (more than 64 neurons in a layer) only the strongest
connections are drawn in the diagram.
"""

import math
import sys
from pathlib import Path

OBS = {
    "easy": ["x to pad", "altitude", "sideways speed", "vertical speed", "sin tilt", "cos tilt",
             "rotation rate", "propellant"],
}
OBS["hard"] = OBS["easy"] + ["deck sideways speed", "throttle (actual)"]
OBS["real"] = ["x to pad", "height above deck", "sideways speed (rel.)", "vertical speed (rel.)",
               "sin tilt", "cos tilt", "rotation rate", "propellant", "deck sideways speed",
               "throttle (actual)", "engine starts left", "deck vertical speed"]
OUTPUTS = ["throttle", "gimbal"]


def load(path):
    lines = Path(path).read_text().split("\n")
    head = lines[0].split()
    sizes = [int(s) for s in head[0].split("-")]
    words = head[1:]
    hist = "hist" in words
    mode = next((w for w in words if w in OBS), None)
    vals = [float(v) for v in lines[1:] if v.strip()]
    layers, off = [], 0
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        W = [[vals[off + k * n_out + j] for j in range(n_out)] for k in range(n_in)]
        off += n_in * n_out
        b = vals[off:off + n_out]
        off += n_out
        layers.append((W, b))
    if off != len(vals):
        raise SystemExit(f"{path}: expected {off} weights for {head[0]}, found {len(vals)}")
    # input names
    if mode is None:
        n = sizes[0] if not hist else (sizes[0] - 2) // 2
        mode = {8: "easy", 10: "hard", 12: "real"}.get(n)
    names = list(OBS.get(mode, [f"input {i + 1}" for i in range(sizes[0])]))
    if hist:
        names = names + [f"prev. {s}" for s in names] + ["prev. throttle cmd", "prev. gimbal cmd"]
    names = names[:sizes[0]] + [f"input {i + 1}" for i in range(len(names), sizes[0])]
    return sizes, layers, names, mode or "?", head[0], len(vals)


def color(v, vmax):
    """diverging: red (negative) - white - blue (positive)"""
    t = max(-1.0, min(1.0, v / vmax)) if vmax > 0 else 0.0
    if t >= 0:
        r, g, b = 255 - int(205 * t), 255 - int(145 * t), 255 - int(35 * t)
    else:
        t = -t
        r, g, b = 255 - int(35 * t), 255 - int(185 * t), 255 - int(185 * t)
    return f"rgb({r},{g},{b})"


def text_color(v, vmax):
    return "#fff" if vmax > 0 and abs(v) / vmax > 0.55 else "#222"


def influence(layers):
    """|W1| |W2| ... : how strongly each input reaches each output (sum over all paths)"""
    M = [[abs(w) for w in row] for row in layers[0][0]]
    for W, _ in layers[1:]:
        A = [[abs(w) for w in row] for row in W]
        M = [[sum(M[i][k] * A[k][j] for k in range(len(A))) for j in range(len(A[0]))] for i in range(len(M))]
    return M


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(path, out):
    sizes, layers, names, mode, shape, n_params = load(path)
    allw = [w for W, b in layers for row in W for w in row] + [x for _, b in layers for x in b]
    vmax = max(abs(w) for w in allw)
    p = []
    W_TOTAL = 1500

    # ---------------- network diagram ----------------
    L = len(sizes)
    tallest = max(sizes)
    node_gap = 34 if tallest <= 16 else max(3.0, 520 / tallest)
    diag_h = max(520, tallest * node_gap + 80)
    x_cols = [230 + i * (520 / max(1, L - 1)) for i in range(L)]
    top = 90

    def ny(i, n):
        return top + (diag_h - 80 - (n - 1) * node_gap) / 2 + i * node_gap

    p.append(f'<text x="30" y="60" font-size="15" font-weight="bold">Network: every connection '
             f'(blue = positive, red = negative, thicker = larger)</text>')
    # edges (big networks: strongest 600 only)
    edges = []
    for li, (W, _) in enumerate(layers):
        for k, row in enumerate(W):
            for j, w in enumerate(row):
                edges.append((abs(w), li, k, j, w))
    if len(edges) > 600:
        edges = sorted(edges, reverse=True)[:600]
    for a, li, k, j, w in sorted(edges):
        x1, y1 = x_cols[li] + 9, ny(k, sizes[li])
        x2, y2 = x_cols[li + 1] - 9, ny(j, sizes[li + 1])
        width = 0.3 + 4.5 * a / vmax
        op = 0.25 + 0.75 * a / vmax
        p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color(w, vmax)}" '
                 f'stroke-width="{width:.2f}" stroke-opacity="{op:.2f}"/>')
    # nodes (filled by bias; inputs have no bias)
    r = 9 if tallest <= 16 else max(1.5, node_gap * 0.35)
    for li, n in enumerate(sizes):
        for i in range(n):
            x, y = x_cols[li], ny(i, n)
            fill = "#f2f2f2" if li == 0 else color(layers[li - 1][1][i], vmax)
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#444" stroke-width="1"/>')
            if li == 0 and n <= 40:
                p.append(f'<text x="{x - 15:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{esc(names[i])}</text>')
            if li == L - 1:
                p.append(f'<text x="{x + 15:.1f}" y="{y + 4:.1f}" font-size="13" font-weight="bold">'
                         f'{esc(OUTPUTS[i] if i < len(OUTPUTS) else f"output {i + 1}")}</text>')
            if 0 < li < L - 1 and n <= 16:
                p.append(f'<text x="{x:.1f}" y="{y - 13:.1f}" text-anchor="middle" font-size="10" fill="#555">h{i + 1}</text>')
    for li, n in enumerate(sizes):
        label = "inputs" if li == 0 else ("outputs (tanh)" if li == L - 1 else f"hidden {li} (tanh)")
        p.append(f'<text x="{x_cols[li]:.1f}" y="{top - 14}" text-anchor="middle" font-size="12" fill="#555">{label} ({n})</text>')
    p.append(f'<text x="{x_cols[1]:.1f}" y="{diag_h + 12:.0f}" text-anchor="middle" font-size="11" fill="#555">'
             f'neuron fill = its bias</text>')

    # ---------------- heatmaps ----------------
    hx, hy = 860, 90
    p.append(f'<text x="{hx}" y="60" font-size="15" font-weight="bold">Weight matrices (rows = from, columns = to)</text>')
    for li, (W, b) in enumerate(layers):
        n_in, n_out = len(W), len(W[0])
        avail_w = W_TOTAL - hx - 160  # room to the right of the row labels
        cell = max(2, min(30, int(avail_w / max(n_out, 1)), int(420 / max(n_in + 1, 1))))
        show_vals = cell >= 26
        to_names = ([f"h{j + 1}" for j in range(n_out)] if li < len(layers) - 1 else
                    [OUTPUTS[j] if j < len(OUTPUTS) else f"out {j + 1}" for j in range(n_out)])
        from_names = names if li == 0 else [f"h{k + 1}" for k in range(n_in)]
        p.append(f'<text x="{hx}" y="{hy - 6}" font-size="12" fill="#555">layer {li + 1}: '
                 f'{n_in} -> {n_out}, plus bias row</text>')
        for k in range(n_in + 1):
            row = W[k] if k < n_in else b
            for j in range(n_out):
                v = row[j]
                x, y = hx + 140 + j * cell, hy + k * cell
                p.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{color(v, vmax)}"/>')
                if show_vals:
                    p.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 3:.1f}" text-anchor="middle" '
                             f'font-size="9" fill="{text_color(v, vmax)}">{v:.2f}</text>')
            if n_in <= 40:
                lab = from_names[k] if k < n_in else "bias"
                style = ' font-style="italic"' if k == n_in else ""
                p.append(f'<text x="{hx + 134}" y="{hy + k * cell + cell / 2 + 4:.1f}" text-anchor="end" '
                         f'font-size="11"{style}>{esc(lab)}</text>')
        if n_out <= 40:
            for j in range(n_out):
                p.append(f'<text x="{hx + 140 + j * cell + cell / 2:.1f}" y="{hy + (n_in + 1) * cell + 13:.1f}" '
                         f'text-anchor="middle" font-size="11">{esc(to_names[j])}</text>')
        hy += (n_in + 1) * cell + 50

    # colour scale
    sx, sy = hx + 140, hy - 20
    for i in range(41):
        v = -vmax + 2 * vmax * i / 40
        p.append(f'<rect x="{sx + i * 6}" y="{sy}" width="6" height="12" fill="{color(v, vmax)}"/>')
    p.append(f'<text x="{sx}" y="{sy + 26}" font-size="10" text-anchor="middle">{-vmax:.2f}</text>')
    p.append(f'<text x="{sx + 123}" y="{sy + 26}" font-size="10" text-anchor="middle">0</text>')
    p.append(f'<text x="{sx + 246}" y="{sy + 26}" font-size="10" text-anchor="middle">+{vmax:.2f}</text>')

    # ---------------- input influence ----------------
    inf = influence(layers)
    iy = max(diag_h + 60, hy + 30)
    p.append(f'<text x="30" y="{iy}" font-size="15" font-weight="bold">Input influence on each output '
             f'(sum over hidden paths of |weight| products, scaled to the largest)</text>')
    n_out = len(inf[0])
    top_inf = max(max(r_) for r_ in inf) or 1
    order = sorted(range(len(inf)), key=lambda i: -sum(inf[i]))[:24]
    bar_w = 420
    cols = ["#2b6cb0", "#dd6b20"]
    for rank, i in enumerate(order):
        y = iy + 25 + rank * 22
        p.append(f'<text x="250" y="{y + 12}" text-anchor="end" font-size="12">{esc(names[i])}</text>')
        for j in range(n_out):
            w_ = bar_w * inf[i][j] / top_inf
            x0 = 260 + j * (bar_w + 60)
            p.append(f'<rect x="{x0}" y="{y}" width="{w_:.1f}" height="16" fill="{cols[j % 2]}"/>')
            p.append(f'<text x="{x0 + w_ + 5:.1f}" y="{y + 12}" font-size="10" fill="#555">{inf[i][j] / top_inf:.2f}</text>')
    for j in range(n_out):
        p.append(f'<text x="{260 + j * (bar_w + 60)}" y="{iy + 18}" font-size="12" font-weight="bold" fill="{cols[j % 2]}">'
                 f'-> {esc(OUTPUTS[j] if j < len(OUTPUTS) else f"output {j + 1}")}</text>')
    total_h = iy + 25 + len(order) * 22 + 30

    title = (f"Weights of {esc(Path(path).parent.name)}/weights.txt: {shape} network, {mode} mode, "
             f"{n_params} weights (range {min(allw):+.2f} to {max(allw):+.2f})")
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_TOTAL}" height="{total_h:.0f}" '
            f'viewBox="0 0 {W_TOTAL} {total_h:.0f}" font-family="sans-serif">',
            f'<rect width="{W_TOTAL}" height="{total_h:.0f}" fill="white"/>',
            f'<text x="30" y="30" font-size="18" font-weight="bold">{title}</text>']
    Path(out).write_text("\n".join(head + p + ["</svg>"]), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    src = sys.argv[1]
    svg(src, sys.argv[2] if len(sys.argv) > 2 else str(Path(src).with_name("weights.svg")))
