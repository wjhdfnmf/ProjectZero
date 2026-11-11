"""
Reference point: train the SAME 1-8-1 tanh network on the SAME 64 sin(x) samples
with gradient descent (backprop + Adam) instead of evolution.
Standard library only.

usage: python backprop_baseline.py [epochs] [seeds]
"""

import math
import random
import sys
import time

H = 8
SAMPLES = [(-math.pi + 2 * math.pi * i / 63) for i in range(64)]
DATA = [(x / math.pi, math.sin(x)) for x in SAMPLES]


def train(seed, epochs, lr=0.01):
    rnd = random.Random(seed)
    # same init range as Network v3.0: weights in [-1, 1]
    p = {
        "w1": [rnd.uniform(-1, 1) for _ in range(H)], "b1": [rnd.uniform(-1, 1) for _ in range(H)],
        "w2": [rnd.uniform(-1, 1) for _ in range(H)], "b2": [rnd.uniform(-1, 1)],
    }
    m = {k: [0.0] * len(v) for k, v in p.items()}
    v = {k: [0.0] * len(v) for k, v in p.items()}
    b1m, b2m = 0.9, 0.999
    history = []
    for t in range(1, epochs + 1):
        g = {k: [0.0] * len(val) for k, val in p.items()}
        loss = 0.0
        for x, y in DATA:
            h = [math.tanh(p["w1"][j] * x + p["b1"][j]) for j in range(H)]
            out = math.tanh(sum(p["w2"][j] * h[j] for j in range(H)) + p["b2"][0])
            e = out - y
            loss += e * e
            d_out = 2 * e * (1 - out * out) / len(DATA)
            g["b2"][0] += d_out
            for j in range(H):
                g["w2"][j] += d_out * h[j]
                d_h = d_out * p["w2"][j] * (1 - h[j] * h[j])
                g["w1"][j] += d_h * x
                g["b1"][j] += d_h
        history.append(loss / len(DATA))
        for k in p:  # Adam update
            for i in range(len(p[k])):
                m[k][i] = b1m * m[k][i] + (1 - b1m) * g[k][i]
                v[k][i] = b2m * v[k][i] + (1 - b2m) * g[k][i] ** 2
                mh, vh = m[k][i] / (1 - b1m ** t), v[k][i] / (1 - b2m ** t)
                p[k][i] -= lr * mh / (math.sqrt(vh) + 1e-8)
    return history


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print("seed,final_mse,epoch_below_0.01,seconds")
    for s in range(1, seeds + 1):
        t0 = time.time()
        hist = train(s, epochs)
        hit = next((i for i, l in enumerate(hist) if l < 0.01), "never")
        print(f"{s},{hist[-1]:.6g},{hit},{time.time() - t0:.1f}")
