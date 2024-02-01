#! /bin/bash

set -e

## FOR REPRODUCIBLE RESULTS JUST PUT THE reproducibility_parameters.txt FILE IN THE tmp FOLDER AND COMMENT THE NEXT
## LINE
# python3 node_classifier/define_reproducibility_parameters.py

reproducibility_parameters_file='node_classifier/tmp/reproducibility_parameters.txt'
nr_lines=$(wc -l < $reproducibility_parameters_file)
i=0
while read line; do
    i=$(( i + 1 ))
    test $i = $((2)) && ranHashSeed=$line
    test $i = $((4)) && ranSeed=$line
    test $i = $((6)) && ranState=$line
    test $i = $((8)) && workers=$line

done <"$reproducibility_parameters_file"
echo -e 'ranHashSeed: \t' $ranHashSeed
echo -e 'ranSeed: \t' $ranSeed
echo -e 'ranState: \t' $ranState
echo -e 'workers: \t' $workers
echo
export PYTHONHASHSEED=$ranHashSeed

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




## for testing aifb effectiveness without facts that used all the nwighbours
for DATASET in 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# for DATASET in "${datasets}"
do
    for KGE_MODEL in 'ComplEx' ## 'ComplEx' 'distMult' 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
    # for KGE_MODEL in "${kge_models}"
    do
        python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
    done
done


# ## for updating randomforest with time stats
# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done
# ## for running other ml models
# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model_xgboost.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done




## BEFORE
# for DATASET in 'AIFB' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done

# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'ComplEx' 'distMult' 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done

# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done




## TO KEEP THE SAME PARAMETERS JUST USE --keep_seeds_for_running_multiple_cv_models IN THE SCRIPT AND COMMENT THE NEXT
## LINE
# rm node_classifier/tmp/reproducibility_parameters.txt