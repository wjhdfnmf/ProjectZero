#!/usr/bin/gnuplot

loc = "~/data/working/personal/ProjectZero/domains/_not_mine/cart_balance/outputs/"
set datafile separator ","

#print "theta_dot"
stats loc."pend_state_log.csv" using 5 name "theta_dot" nooutput
#print "delta_theta"
stats loc."pend_state_log.csv" using 8 name "delta_theta" nooutput
#print "fitness"
stats loc."pend_state_log.csv" using 7 name "fitness" nooutput
#print "torque"
stats loc."pend_state_log.csv" using 1 name "torque" nooutput
#print "force"
stats loc."pend_state_log.csv" using 9 name "force" nooutput
#print "position"
stats loc."pend_state_log.csv" using 10 name "position" nooutput

set title font ",20"
set grid
set key left box

set xlabel "round"
set xrange [1:*]
set terminal pngcairo size 600,400

## theta
set title "Theta"
set output loc."theta.png"
set ylabel "theta"
set yrange  [0:360]
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead lt rgb "red" lw 2
set arrow 2 from graph 0, first 180 to graph 1, first 180 nohead lt rgb "red" lw 1
set arrow 3 from graph 0, first 90 to graph 1, first 90 nohead lt rgb "green" lw 1
plot loc."pend_state_log.csv" using 0:4 with points title "Theta" lc rgb "blue"
unset ylabel
unset yrange
unset arrow 1
unset arrow 2
unset arrow 3

## theta dot
set title "Theta Dot"
set output loc."theta_dot.png"
set ylabel "theta dot [deg/s]"
ymin = (theta_dot_min < -1) ? theta_dot_min : -1
ymax = (theta_dot_max > 1) ? theta_dot_max : 1
set yrange [ymin:ymax]
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead lt rgb "green" lw 1
plot loc."pend_state_log.csv" using 0:5 with lines title "Theta Dot" lc rgb "blue"
unset ylabel
unset yrange
unset arrow 1

## delta theta
set title "Delta Theta"
set ylabel "theta [deg]"
set output loc."delta_theta.png"
plot loc."pend_state_log.csv" using 0:8 with lines title "Delta Theta" lc rgb "red"

## fitness
set title "Fitness"
set ylabel "fitness"
set output loc."fitness.png"
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead lt rgb "green" lw 2
plot loc."pend_state_log.csv" using 0:7 with lines title "Fitness" lc rgb "red"
unset ylabel
unset arrow 1

## torque
set title "Torque"
set ylabel "torque [Nm]"
set output loc."torque.png"
ymin = (torque_min < -1) ? torque_min : -1
ymax = (torque_max > 1) ? torque_max : 1
set yrange [ymin:ymax]
plot loc."pend_state_log.csv" using 0:1 with lines title "Torque" lc rgb "#800080"
unset ylabel
unset yrange

## force
set title "Force"
set ylabel "force [N]"
set output loc."force.png"
ymin = (force_min < -1) ? force_min : -1
ymax = (force_max > 1) ? force_max : 1
set yrange [ymin:ymax]
plot loc."pend_state_log.csv" using 0:9 with lines title "Force" lc rgb "#800080"
unset ylabel
unset yrange

## position
set title "Position"
set ylabel "position [m]"
set output loc."position.png"
ymin = (position_min < -1) ? position_min : -1
ymax = (position_max > 1) ? position_max : 1
set yrange [ymin:ymax]
set arrow 1 from graph 0, first 0 to graph 1, first 0 nohead lt rgb "orange" lw 1
plot loc."pend_state_log.csv" using 0:10 with lines title "Position" lc rgb "#800080"
unset ylabel
unset yrange
unset arrow 1

#print ""
#print "CART DOMAIN GRAPHS GENERATED"
