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




# ## for testing aifb effectiveness without facts that used all the neighbours
# for DATASET in 'AIFB' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done


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
#         python3 node_classifier/train_cv_model_mlp.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done


# ## for running rf global
# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model_rf_global.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done
# ## for running mlp global
# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model_mlp_global.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done
# ## for running xgb global
# for DATASET in 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# # for DATASET in "${datasets}"
# do
#     for KGE_MODEL in 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
#     # for KGE_MODEL in "${kge_models}"
#     do
#         python3 node_classifier/train_cv_model_xgb_global.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
#     done
# done

## for running rf local with time stats
for DATASET in 'MDGENRE' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# for DATASET in "${datasets}"
do
    for KGE_MODEL in 'TransE' 'TransH' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
    # for KGE_MODEL in "${kge_models}"
    do
        python3 node_classifier/train_cv_model_with_stats.py --dataset $DATASET --kge_model $KGE_MODEL --keep_seeds_for_running_multiple_cv_models
    done
done


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