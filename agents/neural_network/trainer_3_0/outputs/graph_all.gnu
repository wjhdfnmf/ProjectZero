#!/usr/bin/gnuplot

#
# Neural Network Training Data Plotter
# William Gregory
# October 27, 2016
#

# data location
loc = "~/data/working/personal/ProjectZero/agents/neural_network/trainer_3_0/outputs/"
set datafile separator ","

## output

set title font ",20"
set grid
set key left box

set xlabel "round"
set ylabel "fitness"

set key off

set xrange [1:*]
unset yrange

set terminal pngcairo size 1200,800

## error
set title "Network Fitness"
set output loc."error.png"

#plot for [col=1:10] loc."fitness_history.csv" using 0:col with lines lc col title sprintf("Series %d", col)
#plot loc."fitness_history.csv"

#plot for [i=1:*] loc."fitness_history.csv" using 0:i with lines lw 2 title columnhead(i)
plot for [i=1:*] loc."fitness_history.csv" using 0:i with lines

print "graphs generated"
