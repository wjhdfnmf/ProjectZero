#!/bin/bash
# Run every configuration below for seeds 1-5 in parallel, then summarize.
# usage: ./experiments.sh [parallel_jobs]
# raw runs: outputs/experiments/<name>_s<seed>/   summary: outputs/experiments_summary.csv

set -e
cd "$(dirname "$0")"
make -s
JOBS=${1:-$(nproc)}
SEEDS="1 2 3 4 5"
BASE=outputs/experiments

# name | settings      (each run uses the same total evaluations unless noted)
CONFIGS=(
  "baseline|"
  "no_elite|elite=0"
  "hidden4|hidden=4"
  "hidden16|hidden=16"
  "deep8x8|hidden=8,8"
  "small_steps_often|chance=0.3 mod=0.1"
  "big_steps|mod=0.5"
  "gauss|mut=gauss mod=0.1"
  "anneal|anneal=1"
  "gauss_anneal|mut=gauss mod=0.3 anneal=1"
  "pop200_gens1500|pop=200 gens=1500"
  "combo|hidden=8,8 chance=0.3 mod=0.1"
  "combo_anneal|hidden=8,8 chance=0.3 mod=0.1 anneal=1"
)

jobs=()
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}; args=${c#*|}
  for s in $SEEDS; do
    mkdir -p "$BASE/${name}_s$s"
    jobs+=("./run $s $args out=$BASE/${name}_s$s > $BASE/${name}_s$s/log.txt")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'

# summary: config, seed, final best mse, generation where best mse first < 0.01, runtime
echo "config,settings,seed,best_mse,gen_below_0.01,seconds" > outputs/experiments_summary.csv
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}; args=${c#*|}
  for s in $SEEDS; do
    d="$BASE/${name}_s$s"
    best=$(grep '^best mse' "$d/log.txt" | awk '{print $3}')
    secs=$(grep '^best mse' "$d/log.txt" | awk '{print $5}' | tr -d s)
    hit=$(awk -F, 'NR>1 && $2<0.01 {print $1; exit}' "$d/fitness.csv")
    echo "$name,\"$args\",$s,$best,${hit:-never},$secs" >> outputs/experiments_summary.csv
  done
done
echo "wrote outputs/experiments_summary.csv"
