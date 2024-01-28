#! /bin/bash

set -e

# dataset='AM_FROM_DGL'

# python3 node_classifier/train_cv_model.py --dataset $dataset




# dataset='MDGENRE'

# python3 node_classifier/define_reproducibility_parameters.py

# reproducibility_parameters_file='node_classifier/tmp/reproducibility_parameters.txt'
# nr_lines=$(wc -l < $reproducibility_parameters_file)
# i=0
# while read line; do
#     i=$(( i + 1 ))
#     test $i = $((2)) && ranHashSeed=$line
#     test $i = $((4)) && ranSeed=$line
#     test $i = $((6)) && ranState=$line
#     test $i = $((8)) && workers=$line

# done <"$reproducibility_parameters_file"
# echo -e 'ranHashSeed: \t' $ranHashSeed
# echo -e 'ranSeed: \t' $ranSeed
# echo -e 'ranState: \t' $ranState
# echo -e 'workers: \t' $workers
# echo
# export PYTHONHASHSEED=$ranHashSeed

# python3 node_classifier/train_model.py --dataset $dataset --aproximate_model
# dataset='AIFB'
# kge_model='TransE'
# python3 node_classifier/train_cv_model.py --dataset $dataset --kge_model $kge_model
datasets=(
    'AIFB'
    'MUTAG'
    'AM_FROM_DGL'
    'MDGENRE'
)
kge_models=(
    'ComplEx'
    'distMult'
    'TransE'
    'TransH'
)

for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# for DATASET in "${datasets}"
do
    for KGE_MODEL in 'ComplEx' 'distMult' 'TransE' 'TransH'
    # for KGE_MODEL in "${kge_models}"
    do
        python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL
    done
done
# ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']
# ['RDF2Vec', 'ComplEx', 'distMult', 'TransE', 'TransH']
# ['ComplEx', 'distMult', 'TransE', 'TransH']
