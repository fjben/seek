import csv
import json
import os

from itertools import product

import pandas as pd

# dataset = 'AIFB'
dataset = 'AM_FROM_DGL'
kge_model = 'RDF2Vec'

# results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}_{kge_model}'
# results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model_backup_latest/{dataset}_{kge_model}'
results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model_rf_local_final/{dataset}_{kge_model}'



data_path = f'node_classifier/data/{dataset}'
# model_path = f'node_classifier/cv_model/{dataset}_{kge_model}'
model_data_partitions_path = f'node_classifier/cv_model_data_partitions/{dataset}/data_partitions'
if kge_model == 'RDF2Vec':
    transformer_model_path = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/models/RDF2Vec_{dataset}'
entity_to_neighbours_path = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/trained/entity_to_neighbours.json'
if kge_model == 'RDF2Vec':
    path_embedding_classes = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/trained/neighbours_embeddings.json'
elif kge_model in ['ComplEx', 'distMult', 'TransE', 'TransH']:
    path_embedding_classes = f'Embeddings/node_classification/{kge_model}/{dataset}_{kge_model}_100.json'
# transformer_model_path = f'node_classifier/model/{dataset}/{dataset}_model_-1_RAN/models/RDF2Vec_{dataset}'
# entity_to_neighbours_path = f'node_classifier/model/{dataset}/{dataset}_model_-1_RAN/trained/entity_to_neighbours.json'
# path_embedding_classes = f'node_classifier/model/{dataset}/{dataset}_model_-1_RAN/trained/neighbours_embeddings.json'

with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
    ds_metadata = json.load(f)

test_data = pd.read_csv(os.path.join(data_path, "testSet.tsv"), sep="\t")
train_data = pd.read_csv(os.path.join(data_path, "trainingSet.tsv"), sep="\t")

train_entities = [entity for entity in train_data[ds_metadata['entities_name']]]
train_labels = list(train_data[ds_metadata['labels_name']])

test_entities = [entity for entity in test_data[ds_metadata['entities_name']]]
test_labels = list(test_data[ds_metadata['labels_name']])

entities = train_entities + test_entities
labels = train_labels + test_labels




if kge_model == 'RDF2Vec':
    entity_to_neighbours_path = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/trained/entity_to_neighbours.json'

dict_entities_num_neighbours = {}
with open(entity_to_neighbours_path, 'r') as f:
    entity_to_neighbours = json.load(f)
# for entity in entities:
#     [all_neighbours, all_neighbour_relation] = entity_to_neighbours[entity]
#     dict_entities_num_neighbours[entity.split('/')[-1]] = len(all_neighbours)


individual_explanations_paths = []
for i in range(10):
    sub_path = f'{dataset}_model_{i}_RAN/explanations/individual_explanations'
    individual_explanations_folders = os.listdir(os.path.join(results_path, sub_path))
    for individual_folder in individual_explanations_folders:
        individual_explanations_paths.append(os.path.join(results_path, sub_path, individual_folder))
# print(individual_explanations_paths)

explanation_type = ['necessary', 'sufficient']
explanation_len = ['len1', 'len5']
acceptance_condition = ['threshold', 'class_change']

combinations = list(product(*[explanation_type, explanation_len, acceptance_condition]))
# print(combinations)

# for p in combinations[:1]:
# for p in combinations[-1:]:
for p in combinations:
    print(p)
    file_name = '_'.join(p) + '.csv'
    explanations_dict = dict()
    for individual_expl_path in individual_explanations_paths:
        load_path = os.path.join(individual_expl_path, file_name)
        # print(load_path)
        explanation_rows = []
        with open(load_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            original = next(reader)
            for row in reader:
                # print(row)
                # row = row.remove('') ## this is not working do not understand why
                row = row[:-1]
                # print(row)
                explanation_rows.append(row)
        # print('\n', header)
        # print('\n', original)
        # print('\n', explanation_rows)
        # print(explanation_rows[-1][1])
        # print(explanation_rows[-1])
        if p[0] == 'necessary':
            condition1 = eval(explanation_rows[-1][1]) == True ## explanation condition satisfied
            # print(condition1)
            condition2 = explanation_rows[-1][2] == explanation_rows[-1][3] ## predicted same as true
            # print(condition2)
            if dataset == 'AIFB':
                condition3 = explanation_rows[-1][5] != 'http://swrc.ontoware.org/ontology#Person' ## no Person as explanation
            else:
                condition3 = True
            condition4 = True

            # ## condition5 is to find explanations where the explanation doesn't use all the neighbours and also not the Person neighbour
            current_entity_in_path = individual_expl_path.split('/')[-1]
            all_entity_neighb = entity_to_neighbours['http://purl.org/collections/nl/am/' + current_entity_in_path][0]
            # while 'http://swrc.ontoware.org/ontology#Person' in all_entity_neighb: all_entity_neighb.remove('http://swrc.ontoware.org/ontology#Person')
            # while 'http://swrc.ontoware.org/ontology#PhDStudent' in all_entity_neighb: all_entity_neighb.remove('http://swrc.ontoware.org/ontology#PhDStudent')
            num_neighbours_for_entity_in_path = len(all_entity_neighb)
            num_neigh_in_explan = int(len(explanation_rows[0][5:]) / 2)
            condition5 = True if num_neighbours_for_entity_in_path > num_neigh_in_explan else False
            if current_entity_in_path == 'proxy-28901':
                print(condition5)
                print(num_neighbours_for_entity_in_path)
                print(num_neigh_in_explan)
                print(all_entity_neighb)

            if int(p[1][-1]) == 5:
                condition6 = num_neigh_in_explan >= 2
            else:
                condition6 = True

            # condition5 = True ## for AM condition5 not necessary
            # condition6 = True
        elif p[0] == 'sufficient':
            condition1 = eval(explanation_rows[0][1]) == True ## explanation condition satisfied
            # print(condition1)
            condition2 = explanation_rows[0][2] == explanation_rows[0][3] ## predicted same as true
            # print(condition2)
            if dataset == 'AIFB':
                condition3 = explanation_rows[0][5] != 'http://swrc.ontoware.org/ontology#Person' ## no Person as explanation
            else:
                condition3 = True            
            ## JUST FOR SUFFICIENT TO TRY AND FIND ONE WITH MORE THAN ONE EXPLANATION
            # print(explanation_rows[-1][5:])
            if int(p[1][-1]) == 5:
                condition4 = len(explanation_rows[0][5:]) > 2
            else:
                condition4 = True

            # ## condition5 is to find explanations where the explanation doesn't use all the neighbours and also not the Person neighbour
            # current_entity_in_path = individual_expl_path.split('/')[-1]
            # all_entity_neighb = entity_to_neighbours['http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/' + current_entity_in_path][0]
            # while 'http://swrc.ontoware.org/ontology#Person' in all_entity_neighb: all_entity_neighb.remove('http://swrc.ontoware.org/ontology#Person')
            # while 'http://swrc.ontoware.org/ontology#PhDStudent' in all_entity_neighb: all_entity_neighb.remove('http://swrc.ontoware.org/ontology#PhDStudent')
            # num_neighbours_for_entity_in_path = len(all_entity_neighb)
            # num_neigh_in_explan = int(len(explanation_rows[0][5:]) / 2)
            # condition5 = True if num_neighbours_for_entity_in_path > num_neigh_in_explan else False

            # condition4 = True
            condition5 = True ## for AM condition5 not necessary
            condition6 = True
        # if condition1 and condition2 and condition3:
        ## JUST FOR SUFFICIENT TO TRY AND FIND ONE WITH MORE THAN ONE EXPLANATION
        # if current_entity_in_path == 'id2008instance':
        #     print(condition5)
        # if current_entity_in_path == 'proxy-67987':
        # if current_entity_in_path == 'proxy-66137':
        #         print(condition5)
        #         # print(num_neighbours_for_entity_in_path)
        #         # print(num_neigh_in_explan)
        #         print(all_entity_neighb)
        if condition1 and condition2 and condition3 and condition4 and condition5 and condition6:
            dif = float(original[0]) - float(explanation_rows[-1][0])
            explanations_dict[individual_expl_path.split('/')[-1]] = [float(dif), original, explanation_rows, load_path.split('/')[-5]]

    try:
        values, keys = zip(*sorted(zip(list(explanations_dict.values()), list(explanations_dict.keys()))))
        # print(values)

        ## the lists are sorted ascending so for necessary we want the largest values (the last x) and for sufficient we
        ## want the smallers (the first x)
        number_of_top_explanations = 1
        if p[0] == 'necessary':
            sl = slice(len(keys)-number_of_top_explanations, len(keys))
            for key, value in reversed(list(zip(keys, values))[sl]):
                print(f"'{key}', {{'location': '{value[3]}'}}")
        elif p[0] == 'sufficient':
            sl = slice(0, number_of_top_explanations)
            for key, value in list(zip(keys, values))[sl]:
                print(f"'{key}', {{'location': '{value[3]}'}}")
    except:
        print('no explanations found that respect the search conditions')