##### to improve to a script with arguments in the command line

import sys
import os
import json

import pandas as pd

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

from dataset_nc import concat_rdf_data_path

dataset_name = 'MUTAG'
rdf_data_path = concat_rdf_data_path(dataset_name)

with open(os.path.join(rdf_data_path, 'metadata.json'), 'r') as f:
        ds_metadata = json.load(f)

train_data_path = os.path.join(rdf_data_path, "trainingSet.tsv")
test_data_path = os.path.join(rdf_data_path, "testSet.tsv")

train_data = pd.read_csv(train_data_path, sep="\t")
test_data = pd.read_csv(test_data_path, sep="\t")

train_entities = [entity for entity in train_data[ds_metadata['entities_name']]]
train_labels = list(train_data[ds_metadata['labels_name']])

test_entities = [entity for entity in test_data[ds_metadata['entities_name']]]
test_labels = list(test_data[ds_metadata['labels_name']])

def float_to_binary_class(value):
        if value == 1.0:
                return 'yes'
        elif value == 0.0:
                return 'no'
        else:
                raise Exception("Value must be either 1.0 or 0.0")

train_data[ds_metadata['labels_name']] = list(map(float_to_binary_class, train_labels))
test_data[ds_metadata['labels_name']] = list(map(float_to_binary_class, test_labels))

train_data.to_csv(train_data_path, sep="\t")
test_data.to_csv(test_data_path, sep="\t")