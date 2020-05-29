#!/bin/bash
arch=$1
lang=$2
datprefix=$3
outdir=$4
root=$5
mname=$6

if [[ -d $outdir ]]; then
    echo "$outdir exists"
else
    python $root/src/train.py \
        --dataset sigmorphon19task1 \
        --train $datprefix"_train.txt"  \
        --dev $datprefix"_dev.txt" \
        --model $outdir/$lang/$arch"_"$mname --seed 0 \
        --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
        --src_layer 2 --trg_layer 1 --max_norm 5 \
        --arch $arch --estop 1e-8 --epochs 50 --bs 20 --mono --shuffle --patience 10
fi
