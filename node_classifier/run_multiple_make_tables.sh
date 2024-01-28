#! /bin/bash

set -e

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
    for KGE_MODEL in 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
    # for KGE_MODEL in "${kge_models}"
    do
        python3 node_classifier/make_tables.py --dataset $DATASET --kge_model $KGE_MODEL
    done
done
# ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']
# ['RDF2Vec', 'ComplEx', 'distMult', 'TransE', 'TransH']
# ['ComplEx', 'distMult', 'TransE', 'TransH']
