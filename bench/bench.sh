#!/bin/bash

echo "Testing: $1"

total_len=6144

for target in 2;
do
    for model in 0 1 2;
    do
        for bsz in 16 32 64;
        do
            echo $target $model $bsz $total_len
            python ./torch_bench_flow.py $target $model $bsz $total_len $1 > bench_log_$target_$model_$bsz.log
        done
    done
done
