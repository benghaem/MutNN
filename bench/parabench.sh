#!/bin/bash

total_len=6144
runtime=0 #pytorch
model=0 #mobilenet
batch_size=1

for pid in {0..0};
do
    echo $target $model $bsz $total_len
    python ./torch_bench_flow.py $runtime $model $batch_size $total_len \
                                 para_out_${pid}.log &
done
