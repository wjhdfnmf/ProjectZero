#!/usr/bin/gnuplot

#
# Neural Network Training Data Plotter
# William Gregory
# October 27, 2016
#

# data location
loc = "~/data/working/personal/ProjectZero/agents/neural_network/trainer_3_0/output_results/"
set datafile separator ","

## output

set title font ",20"
set grid
set key left box

set xlabel "round"
set ylabel "fitness"

set xrange [1:101]

set terminal pngcairo size 600,400

## error
set title "Network Fitness"
set output loc."error.png"

plot for [col=1:10] "fitness_history.csv" using 0:col with lines lc col title sprintf("Series %d", col)
#plot loc."../fitness_history.csv"

print "graphs generated"
