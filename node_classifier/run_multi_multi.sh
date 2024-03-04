#! /bin/bash

set -e

for i in {0..9}; do
    bash node_classifier/run_multiple_train_single_models_v2.sh
done

for i in {0..9}; do
    bash node_classifier/run_multiple_train_single_models_v2_mdgenre.sh
done