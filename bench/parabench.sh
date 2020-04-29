#!/bin/bash

runtime=0 #pytorch
model_idx=(1 1 1 1 1 1 1 1) #resnet50, vgg
batch_len=4
total_len=6144
of_prefix="pblog"
of_suffix=".log"
run_id=$(date | sha1sum | head -c6)
para_factor=$1

pid=0
while [ $pid -lt ${para_factor} ]
do
    #echo ${model_idx[$pid]}
    python torch_bench_flow.py ${runtime} \
                           ${model_idx[$pid]} \
                           ${batch_len} \
                          ${total_len} \
                           "${of_prefix}_p${pid}_of_${para_factor}_${of_suffix}" > p${pid}.log &
    ((pid++))
done

wait
wait
exit
