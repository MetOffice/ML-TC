#!/usr/bin/bash

# Batch MASS retrievals
for f in /project/ciid/projects/ML-TC/mass/*moofilter
do
    echo "-----------------------------------"
    echo "Starting ${f}..."
    moo select -I -j 6 ${f} moose:/devfc/u-bw324/field.pp/ $SCRATCH/.
    echo "...done!"
done 
