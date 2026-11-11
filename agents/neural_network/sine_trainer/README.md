# Sine Wave Trainer

Evolves the v3.0 neural network (`../network/neural_network.h`) to approximate `sin(x)` on `[-π, π]`. It uses the same evolutionary loop as `trainer_3_0`: evaluate, tournament prune, refill with mutated copies.

- **Network:** 1 → 8 → 8 → 1 with `tanh`. The input is `x / π`, scaled to `[-1, 1]`.
- **Fitness:** mean squared error over 64 evenly spaced points (lower is better).
- **Evolution:** population of 100 for 3,000 generations. Each weight has a 0.3 chance to mutate, with a step that starts at ±0.1 and shrinks to ±0.005 over the run. The best network always survives to the next generation (`KEEP_BEST`).

These defaults are the best settings found in [IMPROVEMENTS.md](IMPROVEMENTS.md). They're at the top of `main.cpp`, and you can override any of them on the command line: `seed`, `hidden` (e.g. `8` for 1-8-1), `pop`, `gens`, `chance`, `mod`, `mut` (`net` or `gauss`), `anneal`, `elite` and `out`.

```bash
make
./run 1                                            # seed 1, default (best) settings
./run 1 hidden=8 chance=0.1 mod=0.2 anneal=0       # original settings, for comparison
python plot.py     # writes outputs/fitness.svg and outputs/sine_fit.svg (standard library only)
```

**[IMPROVEMENTS.md](IMPROVEMENTS.md)** compares 13 configurations over 5 seeds each, plus a gradient descent baseline. To rerun it, use `./experiments.sh` and `python plot.py`; for the baseline, `python backprop_baseline.py`.

Outputs, all in `outputs/`:

| File | Contents |
|---|---|
| `fitness.csv` | Best and average MSE for each generation |
| `snapshots.csv` | The best network's curve at generations 0, 10, 50, 200, 1000 and 3000 |
| `weights.txt` | Weights of the final best network |
| `fitness.svg`, `sine_fit.svg` | The two plots |

With seed 1, the best MSE goes from 0.14 at generation 0 to 0.00057 at generation 3000, in about 80 seconds. The original 1-8-1 settings reach 0.0029 in about 40 seconds.
