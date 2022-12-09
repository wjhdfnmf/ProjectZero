#!/usr/bin/gnuplot

#
# Neural Network Training Data Plotter
# William Gregory
# September 13, 2016
#
# output	(actual v expected)
# error		(average v time)
# 

# gnuplot -e "filename="filename" output_results/graph_all.gnu
#if (!exists(filename)) filename="default.png"

# data location
loc = "~/Documents/programming/GitHub/ProjectZero/neural_network/trainer/output_results/"

## output

# line color, line type, line width, point type, point size
set style line 1 lc rgb "#0060ad" lt 1 lw 2 pt 9 ps 1.0  # blue
set style line 2 lc rgb "#dd181f" lt 1 lw 2 pt 7 ps 1.0  # red

set title font ",20"
set grid
set key left box

set xlabel "inputs"
set ylabel "outputs"

set terminal pngcairo size 600,400

# best network
set title "Network Output - Best"
set output loc."image_best.png"
plot loc."network_output_best.txt" using 1:2 index 0 with linespoints ls 1 title "actual", \
	 ""									 index 1 with linespoints ls 2 title "expected"

# first network
set title "Network Output - First"
set output loc."image_first.png"
plot loc."network_output_first.txt" using 1:2 index 0 with linespoints ls 1 title "actual", \
	 ""									  index 1 with linespoints ls 2 title "expected"

## error
set title "Network Error"
set output loc."error.png"
plot loc."avg_error.txt"


print "graphs generated"
