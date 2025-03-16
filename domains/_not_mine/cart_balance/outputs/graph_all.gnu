#!/usr/bin/gnuplot

loc = "~/data/working/personal/ProjectZero/domains/_not_mine/cart_balance/outputs/"
set datafile separator ","

set title font ",20"
set grid
set key left box

set xlabel "round"
set ylabel "theta"
set xrange [1:*]
set terminal pngcairo size 600,400

## theta
set title "Theta"
set output loc."theta.png"
plot loc."pend_state_log.csv" using 0:4 with lines title "Theta" lc rgb "blue"

## delta theta
set title "Delta Theta"
set output loc."delta_theta.png"
plot loc."pend_state_log.csv" using 0:8 with lines title "Delta Theta" lc rgb "red"

## fitness
set title "Fitness"
set output loc."fitness.png"
plot loc."pend_state_log.csv" using 0:7 with lines title "Fitness" lc rgb "red"

## torque
set title "Torque"
set output loc."torque.png"
plot loc."pend_state_log.csv" using 0:1 with lines title "Torque" lc rgb "#800080"

print "graphs generated"
