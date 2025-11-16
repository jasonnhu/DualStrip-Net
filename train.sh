#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
# modify these augments if you want to try other datasets, splits or methods
# dataset: ['chn6', 'mass', 'deepglobe']
# method: ['dualstrip']
# exp: just for specifying the 'save_path'
# split: ['5%', '10%', '20%']
dataset='mass'
method='dualstrip'
exp='dualstrip-net' #deeplabv3plus_r50
split='20%'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path