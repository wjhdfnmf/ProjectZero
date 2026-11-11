#!/bin/bash
# Run every configuration below for seeds 1-5 in parallel, then summarize.
# usage: ./experiments.sh [parallel_jobs] [config_regex]
#   config_regex: only run configs whose name matches (e.g. '^full_'); the summary
#   still covers every config that has results
# raw runs: outputs/experiments/<name>_s<seed>/   summary: outputs/experiments_summary.csv

set -e
cd "$(dirname "$0")"
make -s
JOBS=${1:-$(nproc)}
ONLY=${2:-.}
SEEDS="1 2 3 4 5"
BASE=outputs/experiments

# Each config builds on the one before it.
CONFIGS=(
  "original|hidden=4 chance=0.1 mod=0.2 anneal=0 elite=0 starts=95"
  "better_search|hidden=4 chance=0.3 mod=0.1 anneal=1 elite=1 starts=95"
  "multi_start|hidden=4 chance=0.3 mod=0.1 anneal=1 elite=1 starts=80,85,95,100"
  "track_fix|hidden=4 chance=0.3 mod=0.1 anneal=1 elite=1 starts=80,85,95,100 track=1 cartin=1"
  "track_fix_deep|hidden=8,8 chance=0.3 mod=0.1 anneal=1 elite=1 starts=80,85,95,100 track=1 cartin=1"
  # full problem (balanced AND on the track) with the original, larger fixed mutation + elitism
  "full_penalty|hidden=4 chance=0.1 mod=0.2 anneal=0 elite=1 starts=95 track=1 cartin=1 gens=2000"
  "full_walls|hidden=4 chance=0.1 mod=0.2 anneal=0 elite=1 starts=95 track=1 cartin=1 walls=1 gens=2000"
  "full_penalty_h8|hidden=8 chance=0.1 mod=0.2 anneal=0 elite=1 starts=95 track=1 cartin=1 gens=2000"
)

jobs=()
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}; args=${c#*|}
  [[ "$name" =~ $ONLY ]] || continue
  for s in $SEEDS; do
    mkdir -p "$BASE/${name}_s$s"
    jobs+=("./run $s $args out=$BASE/${name}_s$s > $BASE/${name}_s$s/log.txt")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'

# summary from each run's SUMMARY line
echo "config,seed,train_cost,cost95,maxx95,dev95,balanced95,upright,balanced,total,eval_cost,seconds" > outputs/experiments_summary.csv
for c in "${CONFIGS[@]}"; do
  name=${c%%|*}
  for s in $SEEDS; do
    d="$BASE/${name}_s$s/log.txt"
    grep -q '^SUMMARY' "$d" 2>/dev/null || continue
    tc=$(grep '^best cost' "$d" | awk '{print $3}')
    secs=$(grep '^best cost' "$d" | awk '{print $5}' | tr -d s)
    vals=$(grep '^SUMMARY' "$d" | sed 's/^SUMMARY //; s/[a-z_0-9]*=//g; s/ /,/g')
    echo "$name,$s,$tc,$vals,$secs" >> outputs/experiments_summary.csv
  done
done
echo "wrote outputs/experiments_summary.csv"
