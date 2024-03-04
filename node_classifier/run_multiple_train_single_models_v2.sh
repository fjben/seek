#! /bin/bash

set -e

# FOR REPRODUCIBLE RESULTS JUST PUT THE reproducibility_parameters.txt FILE IN THE tmp FOLDER AND COMMENT THE NEXT
# LINE
python3 node_classifier/define_reproducibility_parameters.py

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


for DATASET in 'AM_FROM_DGL' ## 'AIFB' 'MUTAG' 'AM_FROM_DGL' 'MDGENRE'
# for DATASET in "${datasets}"
do
    for KGE_MODEL in 'RDF2Vec' ## 'RDF2Vec' 'ComplEx' 'distMult' 'TransE' 'TransH'
    # for KGE_MODEL in "${kge_models}"
    do
        python3 node_classifier/train_model_multi_random.py --dataset $DATASET --aproximate_model
        python3 node_classifier/train_single_model.py --dataset $DATASET --kge_model $KGE_MODEL
    done
done


# TO KEEP THE SAME PARAMETERS JUST USE --keep_seeds_for_running_multiple_cv_models IN THE SCRIPT AND COMMENT THE NEXT
# LINE
## No need, it is being removed in train_single_model
# rm node_classifier/tmp/reproducibility_parameters.txt