#! /bin/bash

set -e


############################################################################### arguments

# # set_workers_for_reproducibility=''
# set_workers_for_reproducibility='--set_workers_for_reproducibility'


############################################################################### script

# python3 node_classifier/define_reproducibility_parameters.py $set_workers_for_reproducibility
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

python3 node_classifier/train_model.py
python3 node_classifier/train_model.py --aproximate_model
