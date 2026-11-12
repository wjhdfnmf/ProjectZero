#!/bin/bash
# Landing-gate experiment (align=): reward being lined up 10 m above the deck, so the
# booster comes straight down. Real mode, 8-node network. Resumable.
# Compares with day 2's from-scratch runs (outputs/day2_summary.csv, group sizes, h8)
# and with a flagship fine-tune without the gate.
# summary: outputs/align_summary.csv

set -e
cd "$(dirname "$0")"
SUM=outputs/align_summary.csv
[ -f "$SUM" ] || echo "group,name,seed,net,landed,test,crashed,tipped,cost,fuel,ignitions,train_s,cpu_match" > "$SUM"
GA="mode=real algo=ga scen=32 sigma=0.03 sigma_end=0.005 hidden=8"
FT="mode=real algo=ga scen=32 sigma=0.01 sigma_end=0.002 hidden=8 init=outputs/flagship_real/weights.txt"

run() {  # run <group> <name> <seed> <settings...>
	local group=$1 name=$2 seed=$3; shift 3
	local d=outputs/align/${group}/${name}_s${seed}
	if grep -q '^SUMMARY' "$d/log.txt" 2>/dev/null; then echo "skip $d"; return; fi
	mkdir -p "$d"
	./rocket.exe seed=$seed "$@" out=$d > "$d/log.txt"
	local line; line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
	get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
	echo "$group,$name,$seed,$(get net),$(get landed),$(get test),$(get crashed),$(get tipped),$(get cost),$(get fuel),$(get ignitions),$(get train_s),$(get cpu_match)" >> "$SUM"
	echo "$(date +%H:%M) $group $name seed $seed: landed $(get landed)/$(get test) in $(get train_s)s"
}

for s in 1 2 3; do
	# from scratch, 800 generations, two gate weights (baseline: day 2 sizes/h8, same settings)
	run scratch align5 $s $GA gens=800 align=5
	run scratch align10 $s $GA gens=800 align=10
	# fine-tune the 85.5% flagship for 400 generations, with and without the gate
	run finetune no_gate $s $FT gens=400
	run finetune align10 $s $FT gens=400 align=10
done
echo "wrote $SUM"
