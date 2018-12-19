#!/bin/bash
#arguments: $1 is the file, $2 is the feedline to extract from that file
#output file name is currently hardcoded
awk -v FL="$2" '{if(int('\$1'/10000)==FL) print '\$0'}' $1 >> "beammap_20181219_all_clicked.bmap"

