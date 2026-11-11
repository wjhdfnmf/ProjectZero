# GPU Neural Network (v4) + Booster Landing

A GPU rewrite of Neural Network v3.0. A whole population of networks lives on the GPU and is evaluated and evolved in parallel. It's trained on a new problem: **landing a rocket booster**.

**[REPORT.md](REPORT.md)** has the results.

## Files

| File | What it is |
|---|---|
| `gpu_nn.cuh` | The network: topology, flat parameter layout, and three forward passes. `forward_serial` runs on the CPU or one GPU thread. `forward_block` lets one thread block run one network, for episodes. `forward_population` is a batched, tiled matrix multiply over every network × every sample. |
| `gpu_evolve.cuh` | The population on the GPU: Xavier initialization, elitism + truncation selection, and Gaussian mutation, all as GPU kernels. Only one cost per network crosses back to the CPU each generation. |
| `verify_bench.cu` | Builds networks with v3 `neural_network.h`, converts them, checks the GPU gives the same outputs, then benchmarks v3 CPU, v4 CPU and both GPU modes. |
| `rocket.cuh` | Booster Landing physics, observations, commands and cost. Shared by the GPU and CPU (`__host__ __device__`). |
| `rocket.cu` | Booster Landing trainer: GPU episode kernel, CPU reference, training loop, validation, and the held-out test. |
| `sizes.sh` | Trains several network sizes × seeds and writes `outputs/sizes_summary.csv`. |
| `plot_rocket.py` | SVG plots (standard library only, reuses `../sine_trainer/plot.py`). |
| `build.bat` | Windows build with the CUDA toolkit + Visual Studio C++ tools. |

## Build and run

Requirements: an NVIDIA GPU, the CUDA toolkit (tested with 13.5) and Visual Studio C++ build tools (tested with 2022 Build Tools). The default target is `sm_86` (RTX 30 series); `set ARCH=sm_89` for RTX 40, `sm_120` for RTX 50.

```bat
build.bat                      :: builds verify_bench.exe and rocket.exe
verify_bench.exe               :: v3 equivalence check + benchmark -> outputs\bench.csv
rocket.exe                     :: train 8-64-64-2, 400 generations (about 4 minutes on an RTX 3070)
rocket.exe hidden=128,128 seed=2 gens=600
python plot_rocket.py          :: SVG plots into outputs\
```

From Git Bash: `./sizes.sh 400` runs the network size comparison (about 45–60 minutes).

On Linux, the same sources should build with `nvcc -O3 -std=c++17 -arch=sm_86 rocket.cu -o rocket` (not tested; only the Windows build was run).

## Booster Landing in one paragraph

A 20 t booster with 4–7 t of propellant starts 1.5–2.5 km up, up to 600 m to the side of a 30 m pad, drifting up to 40 m/s sideways and falling at 50–90 m/s, tilted up to 14°. It feels a steady wind plus gusts (up to 23 m/s) that it **cannot sense**. It controls throttle (0–100% of 600 kN) and engine gimbal (±8.6°), which tilts the whole rocket. A landing counts if it touches down on the pad at under 3 m/s down and 1.5 m/s sideways, within 5° of vertical and barely rotating. Running out of propellant means free fall.
