#!/bin/bash
# Time-bonus experiment: fine-tune the real-mode flagship for 800 generations, seeds 1-3.
#   round "small": small steps (sigma 0.01 -> 0.002), time_bonus = 0 (control), 10, 30
#   round "large": larger steps (sigma 0.03 -> 0.005), time_bonus = 0 (control), 150
#   round "scratch": trained from scratch (not from the flagship), time_bonus = 150
# Resumable.  usage: ./time_bonus.sh     summary: outputs/time_bonus_summary.csv

set -e
cd "$(dirname "$0")"
SUM=outputs/time_bonus_summary.csv
[ -f "$SUM" ] || echo "round,time_bonus,seed,landed,test,crashed,tipped,cost,fuel,ignitions,land_time,train_s,cpu_match" > "$SUM"

run() {  # run <round> <time_bonus> <sigma> <sigma_end> [scratch]
	local round=$1 tb=$2 sg=$3 se=$4 init="init=outputs/flagship_real/weights.txt"
	[ "$5" = scratch ] && init=""
	for s in 1 2 3; do
		local d=outputs/time_bonus/${round}_tb${tb}_s$s
		if grep -q '^SUMMARY' "$d/log.txt" 2>/dev/null; then echo "skip $d"; continue; fi
		mkdir -p "$d"
		./rocket.exe mode=real algo=ga hidden=8 scen=32 sigma=$sg sigma_end=$se gens=800 seed=$s \
			time_bonus=$tb $init out=$d > "$d/log.txt"
		local line; line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
		get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
		echo "$round,$tb,$s,$(get landed),$(get test),$(get crashed),$(get tipped),$(get cost),$(get fuel),$(get ignitions),$(get land_time),$(get train_s),$(get cpu_match)" >> "$SUM"
		echo "$(date +%H:%M) $round time_bonus=$tb seed $s: landed $(get landed)/$(get test), landing time $(get land_time) s, fuel $(get fuel) kg"
	done
}

for tb in 0 10 30; do run small $tb 0.01 0.002; done
for tb in 0 150; do run large $tb 0.03 0.005; done
# from scratch (no flagship start); the control is day2.sh's sizes/h8 runs (same settings, no bonus)
run scratch 150 0.03 0.005 scratch
echo "wrote $SUM"
