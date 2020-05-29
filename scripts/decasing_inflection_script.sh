#!/bin/bash
arch=$1

root=/homes/pjc211/project_m2

python src2/train.py \
    --dataset sigmorphon19task1 \
    --train $root/no_lem_nom_inflector_train_adjs_anim_full.txt  \
    --dev $root/no_lem_nom_inflector_test_adjs_anim_full.txt \
    --model $root/model/hmmfull-half-no-lem-adjs_anim_full --seed 0 \
    --embed_dim 100 --src_hs 200 --trg_hs 200 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 \
    --arch $arch --estop 1e-8 --epochs 50 --bs 20 --mono --shuffle --patience 10


