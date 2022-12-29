#!/bin/bash

#for i in 10 15 20 30 50 100 500 1358
for i in 70
do
    echo "process $i companies data"

    python get_SWOT_dataset.py -n $i --dataset_name "SWOT_quad"
    python get_SWOT_dataset.py -n $i --dataset_name "SWOT_NLI"
done
echo "DONE"
