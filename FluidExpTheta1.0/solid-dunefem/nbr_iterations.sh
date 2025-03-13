#!/bin/sh

if [ "${1:-}" = "" ]; then
    echo "No target directory specified. Please specify the directory of the solid participant containing the watchpoint, e.g. ./plot-displacement.sh solid-fenics."
    exit 1
fi

FILE="$1/precice-Solid-iterations.log"

if [ ! -f "$FILE" ]; then
    echo "Unable to locate the watchpoint file (precice-Solid-watchpoint-Flap-Tip.log) in the specified solid directory '${1}'. Make sure the specified directory matches the solid participant you used for the calculations."
    exit 1
fi
 
gnuplot -p << EOF                                                               
	set grid                                                                        
	set title 'number of iterations'                                        
	set xlabel 'Number of time windows'                                                           
	set ylabel 'Iterations'                                                 
	plot "$1/precice-Solid-iterations.log" using 1:3 with lines title "$1"
EOF
