#!/bin/bash

#for i in 10 15 20 30 50 100 500 1358
for i in 50
do
    echo "process $i companies data"

    CUDA_VISIBLE_DEVICES=0 python ./training_SWOT_NLI.py -n $i
    CUDA_VISIBLE_DEVICES=0 python ./test_SWOT_NLI.py -n $i
done
echo "DONE"
