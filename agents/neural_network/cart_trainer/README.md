# Cart Balance Trainer

Evolves the v3.0 neural network (`../network/neural_network.h`) to keep an inverted pendulum upright by pushing its cart. The domain is **Cart Balance by Sierra Gonzales** ([sierra121314](https://github.com/sierra121314)), in `domains/_not_mine/cart_balance/`. The training loop is the same as `trainer_3_0`: evaluate, tournament prune, refill with mutated copies.

- **Inputs:** `cos θ`, `sin θ` and `θ̇`, the same as trainer_3_0. With `cartin=1`, the network also sees cart position and velocity.
- **Outputs:** torque (disabled, `max_torque = 0`) and push force, up to ±40 N.
- **Domain:** 1 kg / 1 m pendulum on a 2 kg cart, 10 m track, `dt = 0.1 s`. Each episode is 200 steps (20 s).
- **Cost:** the domain's own per-step cost (lower is better). With `track=1`, the trainer adds +10 for every step the cart spends off the track.

```bash
make
./run 1                                   # seed 1, default settings
./run 1 hidden=4 chance=0.1 mod=0.2 anneal=0 elite=0 starts=95   # trainer_3_0's settings
python plot_cart.py                       # SVG plots into outputs/ (uses ../sine_trainer/plot.py)
./experiments.sh                          # every config x 5 seeds -> outputs/experiments_summary.csv
```

All settings are listed at the top of `main.cpp`.

**[REPORT.md](REPORT.md)** has the results.

| Output | Contents |
|---|---|
| `fitness.csv` | Best and average training cost for each generation |
| `rollout.csv` | The best network balancing from 95°, step by step |
| `eval.csv` | The best network from every start angle from 45° to 135° |
| `weights.txt` | Final best network weights |
