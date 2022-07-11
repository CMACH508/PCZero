#!/bin/bash
for ((it=1; it<=5; it++))
do
    for ((i=1; i<=10; i++))
        do
            echo "epoch $i"
	    python selfplay.py
            python conquerPC.py
	    python trainPC.py
        done
done
python evaluatePKModel.py
for ((it=1; it<=5; it++))
do
    for ((i=1; i<=10; i++))
        do
            echo "epoch $i"
	    python selfplay.py
            python conquerPC.py
	    python trainPC.py
        done
done
python evaluatePKModel.py
for ((it=1; it<=5; it++))
do
    for ((i=1; i<=10; i++))
        do
            echo "epoch $i"
	    python selfplay.py
            python conquerPC.py
	    python trainPC.py
        done
done
python evaluatePKModel.py
for ((it=1; it<=5; it++))
do
    for ((i=1; i<=10; i++))
        do
            echo "epoch $i"
	    python selfplay.py
            python conquerPC.py
	    python trainPC.py
        done
done
python evaluatePKModel.py







