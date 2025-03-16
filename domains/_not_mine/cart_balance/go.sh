#!/bin/bash
make
if [ $? -eq 0 ]; then
	./run
	gnuplot outputs/graph_all.gnu
else
	echo "Make failed, not running ./run"
fi
