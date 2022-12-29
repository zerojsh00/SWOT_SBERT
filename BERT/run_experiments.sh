#!/bin/bash

#for i in 10 15 20 30 50 100 500 1358
for i in 70
do
    echo "process $i companies data"

    CUDA_VISIBLE_DEVICES=0,1 python ./training_SWOT_quad.py -n $i
    CUDA_VISIBLE_DEVICES=0,1 python ./test_SWOT_quad.py -n $i

done
echo "DONE"
