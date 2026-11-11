#!/bin/bash
# Train Booster Landing with several network sizes and seeds (one run at a time:
# each run already fills the GPU), then summarize.
# usage: ./sizes.sh [gens] [easy|hard] [hidden sizes...]   (run from gpu/ after build.bat; Git Bash on Windows)
#   e.g. ./sizes.sh 400 hard 8 32,32 64,64 128,128
# raw runs: outputs/sizes_<mode>/<hidden>_s<seed>/   summary: outputs/sizes_<mode>_summary.csv

set -e
cd "$(dirname "$0")"
GENS=${1:-400}
MODE=${2:-easy}
shift 2 2>/dev/null || true
HIDDEN=${*:-"8 32,32 64,64 128,128"}
SEEDS="1 2 3"
SUM=outputs/sizes_${MODE}_summary.csv

echo "net,hidden,seed,params,landed,test,crashed,tipped,oob,timeout,cost,fuel,train_s,cpu_match" > "$SUM"
for h in $HIDDEN; do
  for s in $SEEDS; do
    d="outputs/sizes_${MODE}/${h//,/x}_s$s"
    mkdir -p "$d"
    ./rocket.exe mode=$MODE hidden=$h seed=$s gens=$GENS out=$d > "$d/log.txt"
    line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
    get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
    echo "$(get net),${h//,/x},$s,$(get params),$(get landed),$(get test),$(get crashed),$(get tipped),$(get oob),$(get timeout),$(get cost),$(get fuel),$(get train_s),$(get cpu_match)" >> "$SUM"
    echo "$MODE hidden=$h seed=$s: landed $(get landed)/$(get test) in $(get train_s)s"
  done
done
echo "wrote $SUM"
