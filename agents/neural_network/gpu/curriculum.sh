#!/bin/bash
# Engine-start curriculum on real mode: 400 generations each with 10, then 5, then 3
# engine starts allowed, each stage starting from the previous stage's network.
# Resumable. Needs day2.sh's ga_hard5 runs for the transferred variant.
# usage: ./curriculum.sh     summary: outputs/curriculum_summary.csv (final stage = 3 starts)

set -e
cd "$(dirname "$0")"
SUM=outputs/curriculum_summary.csv
[ -f "$SUM" ] || echo "variant,seed,stage,ignitions,net,landed,test,cost,starts_used,train_s" > "$SUM"
GA="mode=real algo=ga scen=32 sigma=0.03 sigma_end=0.005 gens=400"

stage() {  # stage <variant> <seed> <n> <ignitions> <settings...>
	local v=$1 s=$2 n=$3 ign=$4; shift 4
	local d=outputs/curriculum/${v}_s${s}/stage${n}
	if grep -q '^SUMMARY' "$d/log.txt" 2>/dev/null; then echo "skip $d"; return; fi
	mkdir -p "$d"
	./rocket.exe seed=$s ignitions=$ign "$@" out=$d > "$d/log.txt"
	local line; line=$(grep '^SUMMARY' "$d/log.txt" | sed 's/^SUMMARY //')
	get() { echo "$line" | tr ' ' '\n' | grep "^$1=" | cut -d= -f2; }
	echo "$v,$s,$n,$ign,$(get net),$(get landed),$(get test),$(get cost),$(get ignitions),$(get train_s)" >> "$SUM"
	echo "$(date +%H:%M) $v seed $s stage $n (starts<=$ign): landed $(get landed)/$(get test), starts used $(get ignitions)"
}

for s in 1 2 3; do
	for v in h64_scratch h64_transfer h8_scratch; do
		case $v in
			h64_scratch)  H="hidden=64,64"; INIT="" ;;
			h64_transfer) H="hidden=64,64"; INIT="init=outputs/sizes_ga_hard5/64x64_s$s/weights.txt" ;;
			h8_scratch)   H="hidden=8";     INIT="" ;;
		esac
		p=outputs/curriculum/${v}_s${s}
		stage $v $s 1 10 $GA $H $INIT
		stage $v $s 2 5  $GA $H init=$p/stage1/weights.txt
		stage $v $s 3 3  $GA $H init=$p/stage2/weights.txt
	done
done
echo "wrote $SUM"
