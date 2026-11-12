#!/bin/bash
# G-FOLD experiments (GFOLD.md): G-FOLD alone, imitation (behavior cloning) from
# G-FOLD, then GA fine-tuning, compared with networks trained without it.
# Resumable: finished steps are skipped.
# usage: ./gfold.sh    (Git Bash on Windows, from gpu/ after build.bat gfold and build.bat rocket)
# summary: outputs/gfold_summary.csv

set -e
cd "$(dirname "$0")"
OUT=outputs/gfold
SUM=outputs/gfold_summary.csv
mkdir -p $OUT
[ -f "$SUM" ] || echo "group,name,seed,net,landed,test,crashed,tipped,cost,fuel,ignitions,train_s,cpu_match" > "$SUM"
GA="mode=real algo=ga scen=32 sigma=0.03 sigma_end=0.005"

# 1. G-FOLD alone on the 1,000 real-mode test flights
[ -f $OUT/summary.txt ] || ./gfold.exe demo mode=real out=$OUT | tee $OUT/demo_log.txt

# 2. imitation data: G-FOLD flies 2,000 training scenarios (not the test set);
#    every step of every LANDED flight becomes an (input, command) example
[ -f $OUT/bc_data.bin ] || ./gfold.exe data mode=real flights=2000 seed=7 out=$OUT | tee $OUT/data_log.txt

# 3. behavior cloning: 8-node and 64-64 networks learn G-FOLD's commands
for h in 8 64,64; do
	n=${h//,/x}
	[ -f $OUT/bc_$n/weights.txt ] || { mkdir -p $OUT/bc_$n; ./gfold.exe bc data=$OUT/bc_data.bin hidden=$h epochs=40 out=$OUT/bc_$n | tee $OUT/bc_$n/log.txt; }
done

run() {  # run <group> <name> <seed> <settings...>   (rocket.exe; also scores BC-only networks with gens=0)
	local group=$1 name=$2 seed=$3; shift 3
	local d=$OUT/${group}/${name}_s${seed}
	if grep -q '^SUMMARY' "$d/log.txt" 2>/dev/null; then echo "skip $d"; return; fi
	mkdir -p "$d"
	./rocket.exe seed=$seed "$@" out=$d > "$d/log.txt"
	local line; line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
	get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
	echo "$group,$name,$seed,$(get net),$(get landed),$(get test),$(get crashed),$(get tipped),$(get cost),$(get fuel),$(get ignitions),$(get train_s),$(get cpu_match)" >> "$SUM"
	echo "$(date +%H:%M) $group $name seed $seed: landed $(get landed)/$(get test) in $(get train_s)s"
}

# 4. the cloned networks as they are (no training: gens=0 just scores them)
run bc_only h8 1 $GA hidden=8 gens=0 init=$OUT/bc_8/weights.txt
run bc_only h64x64 1 $GA hidden=64,64 gens=0 init=$OUT/bc_64x64/weights.txt

# 5. GA from the cloned networks vs from scratch (same settings, 800 generations, 3 seeds;
#    the from-scratch runs are day 2's "sizes" runs, reused from outputs/day2_summary.csv)
for s in 1 2 3; do
	run bc_ga h8 $s $GA hidden=8 gens=800 init=$OUT/bc_8/weights.txt
	run bc_ga h64x64 $s $GA hidden=64,64 gens=800 init=$OUT/bc_64x64/weights.txt
done
echo "wrote $SUM"
