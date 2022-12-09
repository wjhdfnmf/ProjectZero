#!/usr/bin/gnuplot

#
# Neural Network Training Data Plotter
# William Gregory
# October 27, 2016
#

# data location
loc = "~/Documents/programming/GitHub/ProjectZero/domains/gridspace/neural_network/trainer/output_results/"

## output

set title font ",20"
set grid
set key left box

set xlabel "round"
set ylabel "fitness"

set terminal pngcairo size 600,400

## error
set title "Network Fitness"
set output loc."error.png"
plot loc."avg_error.txt"

print "graphs generated"
