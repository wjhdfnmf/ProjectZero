#!/bin/bash
# Day 2 experiments on real mode (REPORT.md section 8). Resumable: finished runs are skipped.
# Needs the hard-mode runs from sizes.sh (ga_hard5) as transfer starting points.
# usage: ./day2.sh            (Git Bash on Windows, from gpu/ after build.bat)
# summary: outputs/day2_summary.csv (group, name, seed, SUMMARY fields)

set -e
cd "$(dirname "$0")"
SUM=outputs/day2_summary.csv
[ -f "$SUM" ] || echo "group,name,seed,net,landed,test,crashed,tipped,cost,fuel,ignitions,train_s,cpu_match" > "$SUM"
GA="mode=real algo=ga scen=32 sigma=0.03 sigma_end=0.005"

run() {  # run <group> <name> <seed> <settings...>
	local group=$1 name=$2 seed=$3; shift 3
	local d=outputs/day2/${group}/${name}_s${seed}
	if grep -q '^SUMMARY' "$d/log.txt" 2>/dev/null; then echo "skip $d"; return; fi
	mkdir -p "$d"
	./rocket.exe seed=$seed "$@" out=$d > "$d/log.txt"
	local line; line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
	get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
	echo "$group,$name,$seed,$(get net),$(get landed),$(get test),$(get crashed),$(get tipped),$(get cost),$(get fuel),$(get ignitions),$(get train_s),$(get cpu_match)" >> "$SUM"
	echo "$(date +%H:%M) $group $name seed $seed: landed $(get landed)/$(get test) starts $(get ignitions) in $(get train_s)s"
}

for s in 1 2 3; do
	# 1. network size, from scratch
	for h in 8 16 32,32 64,64; do run sizes "h${h//,/x}" $s $GA hidden=$h gens=800; done
	# 2. transfer from the realistic hard-mode models (same seed's run)
	run transfer h8 $s $GA hidden=8 gens=800 init=outputs/sizes_ga_hard5/8_s$s/weights.txt
	run transfer h64x64 $s $GA hidden=64,64 gens=800 init=outputs/sizes_ga_hard5/64x64_s$s/weights.txt
	# 3. ablation: switch off one real-mode feature at a time (8-node network, from scratch)
	run ablation no_ignition_limit $s $GA hidden=8 gens=800 ignitions=0
	run ablation no_noise $s $GA hidden=8 gens=800 noise=0
	run ablation no_delay $s $GA hidden=8 gens=800 delay=0
	run ablation no_heave $s $GA hidden=8 gens=800 heave=0
	run ablation original_score $s $GA hidden=8 gens=800 score=0
done

# 4. hybrid: from each transferred 8-node model, 400 more generations of
#    (a) GA fine-tuning (small steps) vs (b) ES fine-tuning (equal episodes)
for s in 1 2 3; do
	w=outputs/day2/transfer/h8_s$s/weights.txt
	run hybrid ga_finetune $s mode=real algo=ga scen=32 sigma=0.01 sigma_end=0.002 hidden=8 gens=400 init=$w
	run hybrid es_finetune $s mode=real algo=es scen=32 sigma=0.01 lr=0.003 hidden=8 gens=400 init=$w
done
echo "wrote $SUM"
