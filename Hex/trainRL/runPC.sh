#!/bin/bash
for ((it=1; it<=10; it++))
do
    for ((i=1; i<=10; i++))
        do
            echo "epoch $i"
	          python selfplay.py
	          python conquerPC.py
	          python train.py
        done
done

