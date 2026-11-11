#!/bin/bash
# Train Booster Landing for several network sizes x seeds (one run at a time:
# each run already fills the GPU), then summarize.
#
# usage: NAME=<group> EXTRA="<rocket.exe settings>" ./sizes.sh <gens> "<hidden sizes>"
#   NAME   label for the output (default easy)
#   EXTRA  settings passed to every run (default mode=easy)
#   SEEDS  default "1 2 3"
# examples (the runs in REPORT.md):
#   NAME=easy    EXTRA="mode=easy" ./sizes.sh 400 "8 32,32 64,64 128,128"
#   NAME=ga_hard EXTRA="mode=hard algo=ga scen=32 sigma=0.03 sigma_end=0.005" ./sizes.sh 800 "8 32,32 64,64 128,128"
#   NAME=es_hard EXTRA="mode=hard algo=es scen=32" ./sizes.sh 800 "8 32,32 64,64 128,128"
# raw runs: outputs/sizes_<NAME>/<hidden>_s<seed>/   summary: outputs/sizes_<NAME>_summary.csv

set -e
cd "$(dirname "$0")"
GENS=${1:-400}
HIDDEN=${2:-"8 32,32 64,64 128,128"}
NAME=${NAME:-easy}
EXTRA=${EXTRA:-mode=easy}
SEEDS=${SEEDS:-"1 2 3"}
SUM=outputs/sizes_${NAME}_summary.csv

[ -f "$SUM" ] || echo "net,hidden,seed,params,landed,test,crashed,tipped,oob,timeout,cost,fuel,train_s,cpu_match" > "$SUM"
for h in $HIDDEN; do
  for s in $SEEDS; do
    d="outputs/sizes_${NAME}/${h//,/x}_s$s"
    if grep -q '^SUMMARY' "$d/log.txt" 2>/dev/null; then echo "skip $d (done)"; continue; fi
    mkdir -p "$d"
    ./rocket.exe $EXTRA hidden=$h seed=$s gens=$GENS out=$d > "$d/log.txt"
    line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
    get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
    echo "$(get net),${h//,/x},$s,$(get params),$(get landed),$(get test),$(get crashed),$(get tipped),$(get oob),$(get timeout),$(get cost),$(get fuel),$(get train_s),$(get cpu_match)" >> "$SUM"
    echo "$NAME hidden=$h seed=$s: landed $(get landed)/$(get test) in $(get train_s)s"
  done
done
echo "wrote $SUM"
