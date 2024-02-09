

import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
import warnings

from collections import Counter
from multiprocessing import cpu_count
from collections import defaultdict
from collections import OrderedDict

from joblib import dump, load
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from pyrdf2vec import RDF2VecTransformer
from xgboost import XGBClassifier

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
from utils.logger import Logger
from run_seek_explanations import compute_effectiveness_kelpie, compute_random, \
    compute_effectiveness_global_explainer, compute_performance_metrics_v2


tic_total_script_time = time.perf_counter()


############################################################################### arguments

parser = argparse.ArgumentParser(description="description")
parser.add_argument("--dataset",
                    type=str,
                    choices=['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE'],
                    help="help")
parser.add_argument("--kge_model",
                    type=str,
                    choices=['RDF2Vec', 'ComplEx', 'distMult', 'TransE', 'TransH'],
                    help="help")
parser.add_argument("--keep_seeds_for_running_multiple_cv_models",
                    action="store_true",
                    help="help.")
args = parser.parse_args()
dataset = args.dataset
kge_model = args.kge_model
keep_seeds_for_running_multiple_cv_models = args.keep_seeds_for_running_multiple_cv_models


## for global explanations, adjust these two paths, and the mlp model params (not really needed)
# load_model_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_rf_local_final'
load_model_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_xgb_local_final'

# model_path = f'node_classifier/cv_model_rf_global_final/{dataset}_{kge_model}'
model_path = f'node_classifier/cv_model_xgb_global_final/{dataset}_{kge_model}'


# print(keep_seeds_for_running_multiple_cv_models)
# raise

# max_len_explanations=1
max_len_explanations=5

# explanation_limit='threshold'
# explanation_limit='class_change'

# dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
# dataset = 'MDGENRE'

# aproximate_model=True
# aproximate_model=False

# best_embeddings_params = True
# best_embeddings_params = False

n_splits = 10
# n_splits = 2

verbose = 1


############################################################################### logging

sys.stdout = Logger()


############################################################################### functions

def create_save_dirs(model_path, dataset, model_type, current_model_num):
    current_model_path = os.path.join(model_path, f'{dataset}_model_{current_model_num}_{model_type}')
    current_model_models_path = os.path.join(current_model_path, 'models')
    current_model_models_results_path = os.path.join(current_model_path, 'models_results')
    # os.makedirs(current_model_models_path)
    # os.makedirs(current_model_models_results_path)
    ensure_dir(current_model_models_path, option='overwrite')
    ensure_dir(current_model_models_results_path, option='overwrite')
    if model_type == 'RAN':
        current_model_trained_path = os.path.join(current_model_path, 'trained')
        # os.makedirs(current_model_trained_path)
        ensure_dir(current_model_trained_path, option='overwrite')
    elif model_type == 'RO':
        current_model_trained_path = None

    return current_model_path, current_model_models_path, current_model_models_results_path, current_model_trained_path


def ensure_dir(path, option='make_if_not_exists'):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path.
    """
    # d = os.path.dirname(path)
    d = path
    if option =='overwrite':
        if os.path.exists(d): ## temporary for tests
            shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)


def process_indexes_partition(file_partition):
    file = open(file_partition, 'r')
    indexes_partition = []
    for line in file:
        indexes_partition.append(int(line[:-1]))
    file.close()
    return indexes_partition


def run_partition(entities, labels, filename_output, n_splits, random_state):
    index_partition = 0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # data_partitions_path = os.path.join(filename_output, 'data_partitions')
    data_partitions_path = os.path.join(filename_output)
    ensure_dir(data_partitions_path, option='overwrite')
    train_index_files_designation = os.path.join(data_partitions_path, 'Indexes_crossvalidationTrain_Run')
    test_index_files_designation = os.path.join(data_partitions_path, 'Indexes_crossvalidationTest_Run')
    for indexes_partition_train, indexes_partition_test in skf.split(entities, labels):
        file_crossValidation_train = open(train_index_files_designation + str(index_partition) + '.txt', 'w')
        file_crossValidation_test = open(test_index_files_designation + str(index_partition) + '.txt', 'w')
        for index in indexes_partition_train:
            file_crossValidation_train.write(str(index) + '\n')
        for index in indexes_partition_test:
            file_crossValidation_test.write(str(index) + '\n')
        file_crossValidation_train.close()
        file_crossValidation_test.close()
        index_partition = index_partition + 1

    return train_index_files_designation, test_index_files_designation


def setup_random_seeds(model_path, cpu_num):
    try:
        with open(os.path.join(model_path, 'reproducibility_parameters.txt')) as f:
            lines = f.readlines()
        RANDOM_STATE = int(lines[5])
        workers = int(lines[7])
        if workers != 1:
            warnings.warn('workers parameter is not equal to 1, so the results are not reproducible')
    except:
        warnings.warn('no reproducibility parameters available, so the results are not reproducible')
        RANDOM_STATE = random.randrange(0, 4294967295)
        workers = cpu_num
        ## for debugging purposes
        # RANDOM_STATE = 22
        # workers = 1

    return RANDOM_STATE, workers


def setup_save_paths(model_path, dataset, aproximate_model, index_partition):
    if aproximate_model:
        model_type = 'RAN' ## representation with aggregate neighbours
    else:
        model_type = 'RO' ## representation with original

    # saved_models = os.listdir(model_path)
    # if not saved_models:
    #     last_saved_model_num = -1
    # else:
    #     ## sorts the models first to last using the name ending digits
    #     saved_models.sort(key=lambda x: int(x.split('_')[-2]))
    #     last_saved_model_num = int(saved_models[-1].split('_')[-2])
    # if os.path.exists('node_classifier/tmp/reproducibility_parameters.txt'):
    #     if aproximate_model:
    #         current_model_num = last_saved_model_num
    #         current_model_models_path, \
    #         current_model_models_results_path, \
    #         current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)
    #         shutil.move('node_classifier/tmp/reproducibility_parameters.txt', os.path.join(current_model_models_path, 'reproducibility_parameters.txt'))
    #     else:
    #         current_model_num = last_saved_model_num + 1
    #         current_model_models_path, \
    #         current_model_models_results_path, \
    #         current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)
    #         shutil.copy('node_classifier/tmp/reproducibility_parameters.txt', os.path.join(current_model_models_path, 'reproducibility_parameters.txt'))
    # else:
    #     current_model_num = last_saved_model_num + 1
    #     current_model_models_path, \
    #     current_model_models_results_path, \
    #     current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)
    
    current_model_num = index_partition
    current_model_path, \
    current_model_models_path, \
    current_model_models_results_path, \
    current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)

    return current_model_path, current_model_models_path, current_model_models_results_path, current_model_trained_path


def stats_for_preds(predictions_proba):
        maxs_list = [np.ndarray.max(single_pred) for single_pred in predictions_proba]
        # for single_pred in predictions_proba:
        #     print(numpy.ndarray.max(single_pred))
        mean_preds = np.mean(maxs_list)
        std_preds = np.std(maxs_list)

        return mean_preds, std_preds


############################################################################### script
        
cpu_num = cpu_count()

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

location = os.path.join(data_path, ds_metadata['rdf_file'])
skip_predicates = set(ds_metadata['skip_predicates'])

# ## original specs found in online-learning.py used when best_embeddings_params is False
# vector_size = 100
# sg=0
# max_depth=2
# max_walks=None
# ## specs from LoFI used when best_embeddings_params is True
# if best_embeddings_params:
#     match dataset:
#         case 'AIFB':
#             vector_size=500
#             sg=1
#             max_depth=4
#             max_walks=500
#         case 'MUTAG':
#             vector_size=50
#             sg=1
#             max_depth=4
#             max_walks=500
#         case 'AM_FROM_DGL':
#             vector_size=500
#             sg=1
#             max_depth=2
#             max_walks=500
#         case 'MDGENRE':
#             vector_size=500
#             sg=1
#             max_depth=2
#             max_walks=500

ensure_dir(model_path)

if keep_seeds_for_running_multiple_cv_models:
    shutil.copy('node_classifier/tmp/reproducibility_parameters.txt', os.path.join(model_path, 'reproducibility_parameters.txt'))
else:
    shutil.move('node_classifier/tmp/reproducibility_parameters.txt', os.path.join(model_path, 'reproducibility_parameters.txt'))

RANDOM_STATE, workers = setup_random_seeds(model_path, cpu_num)

# n_jobs = 2 ## original specs found in online-learning.py
n_jobs = cpu_num

print('RANDOM_STATE:\t\t', RANDOM_STATE)
print('workers:\t\t', workers)
print("Number of used cpu:\t", n_jobs, '\n')

## before was partitioning every time, now fixed partitions
# train_index_files_designation, test_index_files_designation = run_partition(entities, labels, model_path, n_splits, RANDOM_STATE)
if not os.listdir(model_data_partitions_path):
    train_index_files_designation, test_index_files_designation = run_partition(entities, labels, model_data_partitions_path, n_splits, RANDOM_STATE)
else:
    ensure_dir(model_data_partitions_path, 'make_if_not_exists')
    train_index_files_designation = os.path.join(model_data_partitions_path, 'Indexes_crossvalidationTrain_Run')
    test_index_files_designation = os.path.join(model_data_partitions_path, 'Indexes_crossvalidationTest_Run')

## I don't think I need this, I can load this information using the /trained data directly from the dict
if kge_model == 'RDF2Vec':
    transformer = RDF2VecTransformer().load(transformer_model_path)
    all_embeddings = transformer._embeddings
    all_entities = transformer._entities
elif kge_model in ['ComplEx', 'distMult', 'TransE', 'TransH']:
    with open(entity_to_neighbours_path, 'r') as f:
        entity_to_neighbours = json.load(f)
    with open(path_embedding_classes, 'r') as f:
        dic_emb_classes = json.load(f)
    all_embeddings = list(dic_emb_classes.values())
    all_entities = list(dic_emb_classes.keys())

with open(entity_to_neighbours_path, 'r') as f:
    entity_to_neighbours = json.load(f)

with open(path_embedding_classes, 'r') as f:
    dic_emb_classes = json.load(f)


def run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                         train_index_files_designation, test_index_files_designation,
                         aproximate_model, RANDOM_STATE, max_len_explanations, explanation_limit,
                         n_jobs, n_partitions,
                         overwrite_invidivual_explanations=False):
    
    ## preprocessing for the global explainer
    all_relations = []
    all_relations_count = []
    all_relation_exists_in_entity_count = []
    for entity in entities:
        [all_neighbours, all_neighbour_relation] = entity_to_neighbours[entity]
        [all_relations.append(relation) for relation in all_neighbour_relation if relation not in all_relations]
    #     [all_relations_count.append(relation) for relation in all_neighbour_relation]

    #     entity_relations = []
    #     for relation in all_neighbour_relation:
    #         if relation not in entity_relations:
    #             entity_relations.append(relation)
    #             all_relation_exists_in_entity_count.append(relation)            

    # all_relations_count = Counter(all_relations_count)
    # all_relation_exists_in_entity_count = Counter(all_relation_exists_in_entity_count)
    # print(all_relations)
    # print(all_relation_exists_in_entity_count)
    # raise

    
    all_results_summary = []
    all_effectiveness_results_lenx = []
    all_effectiveness_results_len1 = []
    all_random_effectiveness_results_lenx = []
    all_random_effectiveness_results_len1 = []
    all_explain_stats = defaultdict(list)
    all_explanations_dict_len1_for_global = OrderedDict()
    all_explain_stats_for_global = defaultdict(list)
    for index_partition in range(0, n_partitions):
        train_index = process_indexes_partition(train_index_files_designation + str(index_partition) + '.txt')
        test_index = process_indexes_partition(test_index_files_designation + str(index_partition) + '.txt')
        print('\n\n\n\n###########################')
        print("######   RUN" + str(index_partition) + "       #######")
        print('###########################\n')

        print(f'Aproximate model: {aproximate_model}')

        current_model_path, \
        current_model_models_path, \
        current_model_models_results_path, \
        current_model_trained_path = setup_save_paths(model_path, dataset, aproximate_model, index_partition)

        entities, labels = np.array(entities), np.array(labels)
        train_entities, train_labels = list(entities[train_index]), list(labels[train_index])
        test_entities, test_labels = list(entities[test_index]), list(labels[test_index])

        entities = train_entities + test_entities
        labels = train_labels + test_labels

        embeddings = get_embeddings(aproximate_model, all_embeddings, all_entities, entity_to_neighbours, entities)

        train_embeddings = embeddings[: len(train_entities)]
        test_embeddings = embeddings[len(train_entities) :]

        # print(len(train_embeddings))
        # print(len(test_embeddings))

        clf_extra, results_summary, effectiveness_results, random_effectiveness_results, explain_stats, \
        explanations_dict_len1_for_global, explain_stats_for_global = train_classifier(train_embeddings, train_labels, test_embeddings,
                                                                       test_labels, test_entities, aproximate_model,
                                                                       entity_to_neighbours, dic_emb_classes,
                                                                       max_len_explanations, explanation_limit,
                                                                       all_relations,
                                                                       current_model_path, n_jobs, RANDOM_STATE, 
                                                                       overwrite_invidivual_explanations=overwrite_invidivual_explanations)

        save_model_results(dataset, clf_extra, current_model_models_path, current_model_models_results_path, current_model_trained_path,
             results_summary)
        
        all_results_summary.append(results_summary)
        all_effectiveness_results_lenx.append(effectiveness_results[0])
        all_effectiveness_results_len1.append(effectiveness_results[1])
        all_random_effectiveness_results_lenx.append(random_effectiveness_results[0])
        all_random_effectiveness_results_len1.append(random_effectiveness_results[1])
        if explain_stats:
            for key, _ in explain_stats.items():
                all_explain_stats[key].extend(explain_stats[key])

        # print(len(explanations_dict_len1_for_global))
        # print(explain_stats_for_global)
        # raise
        # all_explanations_dict_len1_for_global.update(explanations_dict_len1_for_global)
        for key in explanations_dict_len1_for_global.keys():
            all_explanations_dict_len1_for_global[(key, f'fold{index_partition}')] = explanations_dict_len1_for_global[key]
        for key in explain_stats_for_global.keys():
            all_explain_stats_for_global[key].extend(explain_stats_for_global[key])

    # print(len(all_explanations_dict_len1_for_global))
    # [print(len(all_explain_stats_for_global[key])) for key in all_explain_stats_for_global.keys()]
    # raise

    if not any(all_effectiveness_results_lenx) and not any(all_effectiveness_results_len1):
        all_effectiveness_results_lenx = False
        all_effectiveness_results_len1 = False
    if not any(all_random_effectiveness_results_lenx) and not any(all_random_effectiveness_results_len1):
        all_random_effectiveness_results_lenx = False
        all_random_effectiveness_results_len1 = False




    ## global explanations compute effectiveness
    print('DONT FORGET THIS:', explanation_limit)
    print('SAVE ORIGINAL DICTS FROM GLOBAL EXPLANATIONS')
    nec_global_explanations, suf_global_explanations = defaultdict(list), defaultdict(list)
    all_neighbours_results = defaultdict(list)
    # print(all_explanations_dict_len1_for_global)
    # raise
    for key, [nec_global_dict, suf_global_dict] in all_explanations_dict_len1_for_global.items():
        all_neighbours_results[('all_object_properties', 'necessary')].append(nec_global_dict['all_object_properties'])
        all_neighbours_results[('all_object_properties', 'sufficient')].append(suf_global_dict['all_object_properties'])
        # print(key, len(nec_global_dic), len(nec_global_dic))
        # print(nec_global_dic)
        # raise
        for relation in all_relations:
            # print('\nrelation', relation)
            # print(nec_global_dic[relation])
            nec_global_explanations[relation].append(nec_global_dict[relation])
            suf_global_explanations[relation].append(suf_global_dict[relation])
        # print(nec_global_explanations)
        # raise
    # print(all_neighbours_results)
    # raise
    nec_property_score, suf_property_score = {}, {}
    for relation in all_relations:
        # print(key)
        count_satisfied_condition = 0
        for result in nec_global_explanations[relation]:
            if result[1] == True:
                count_satisfied_condition += 1
        nec_property_score[relation] = count_satisfied_condition / len(nec_global_explanations[relation])

        count_satisfied_condition = 0
        for result in suf_global_explanations[relation]:
            if result[1] == True:
                count_satisfied_condition += 1
        suf_property_score[relation] = count_satisfied_condition / len(suf_global_explanations[relation])

    # print(nec_property_score)
    best_nec_object_property = max(nec_property_score, key=nec_property_score.get)
    best_suf_object_property = max(suf_property_score, key=suf_property_score.get)
    # print(nec_property_score)
    # print(suf_property_score)

    # print(summary_results)
    # raise
    # print(nec_global_explanations[best_nec_object_property])

    labels_from_results = [result[2] for result in nec_global_explanations[best_nec_object_property]]
    original_pred_eva = [result[3] for result in nec_global_explanations[best_nec_object_property]]
    pred_eva_necessary_len1 = [result[4] for result in nec_global_explanations[best_nec_object_property]]
    pred_eva_sufficient_len1 = [result[4] for result in suf_global_explanations[best_suf_object_property]]
        
    effectiveness_results_len1_global_explanations = compute_performance_metrics_v2(labels_from_results, original_pred_eva,
                                                        pred_eva_necessary_len1, pred_eva_sufficient_len1)
    # print(effectiveness_results_len1)
    # print(all_relations_count)
    # print(all_relation_exists_in_entity_count)

    # print(all_explanations_dict_len1_for_global.keys())
    # summary_global_explanations
    # print(effectiveness_results_len1_global_explanations)
    # raise

    ## global explanations make global dict
    summary_results = defaultdict(list)
    for expl_t, expl_t_global_expl, prop_score in zip(['necessary', 'sufficient'],
                                          [nec_global_explanations, suf_global_explanations],
                                          [nec_property_score, suf_property_score]):
        predict_proba_all_neighb = [result[0] for result in all_neighbours_results[('all_object_properties', expl_t)]]
        # print(predict_proba_all_neighb)
        # raise
        mean, std = np.mean(predict_proba_all_neighb), np.std(predict_proba_all_neighb)
        summary_results[f'{expl_t}_len1'].append({'all_object_properties': [f'{mean} ({std})', 0]})
        temp_relations = {}
        for relation in all_relations:
            predict_proba_relation_expl_t = [result[0] for result in expl_t_global_expl[relation]]
            mean, std = np.mean(predict_proba_relation_expl_t), np.std(predict_proba_relation_expl_t)
            # summary_results[f'{expl_t}_len1'].append({relation: [f'{mean} ({std})', prop_score[relation]]})
            temp_relations[relation] = [f'{mean} ({std})', prop_score[relation]]
        # print(temp_relations)
        temp_relations = dict(sorted(temp_relations.items(), key=lambda x: float(x[1][0].split(' ')[0]), reverse=True))
        # print(temp_relations)
        # raise
        summary_results[f'{expl_t}_len1'].append(temp_relations)
        # print(summary_results)
        # raise

    with open(os.path.join(model_path, f'explain_stats_for_global_explainer_{explanation_limit}.json'), 'w', encoding ='utf8') as f: 
        json.dump(explain_stats_for_global, f, ensure_ascii = False)

    nec_to_save, suf_to_save = OrderedDict(), OrderedDict()
    for key, [nec_dict, suf_dict] in all_explanations_dict_len1_for_global.items():
        new_key = f'{key[0]}_{key[1]}' ## because to save json doesn't support tuple in keys
        nec_to_save[new_key] = nec_dict
        suf_to_save[new_key] = suf_dict
    # print(nec_to_save)
    # print(suf_to_save)
    # raise
    with open(os.path.join(model_path, f'raw_global_explanations_global_dict_necessary.json'), 'w',
                encoding ='utf8') as f: 
        json.dump(nec_to_save, f, ensure_ascii = False)
    with open(os.path.join(model_path, f'raw_global_explanations_global_dict_sufficient.json'), 'w',
                encoding ='utf8') as f: 
        json.dump(suf_to_save, f, ensure_ascii = False)


    path_explanations = os.path.join(model_path, 'explanations')
    path_global_explanations = os.path.join(model_path, 'global_explanations')
    if overwrite_invidivual_explanations:
        ensure_dir(path_global_explanations, option='overwrite')
    else:
        ensure_dir(path_explanations, option='make_if_not_exists')
        ensure_dir(path_global_explanations, option='make_if_not_exists')

    print(summary_results)
    # raise

    for key, ds in summary_results.items():
        with open(os.path.join(path_global_explanations, f'{key}_{explanation_limit}.csv'), 'w') as f:
            header = ['predict_proba', f'satisfied_{explanation_limit}_ratio', 'all_properties/property', '\n']
            f.write('\t'.join(header))
            for d in ds:
                print(type(d))
                print(d)
                for key, value in d.items():
                    value_list_of_strs = [str(val) for val in value]
                    key_list_of_strs = []
                    single_line_part0 = '\t'.join(value_list_of_strs)
                    # print(single_line_part0)
                    if isinstance(key, str):
                        single_line_part1 = key
                        # print('\n', single_line_part1)
                    else:
                        single_line_part1 = flatten_tupple(key)
                        single_line_part1 = '\t'.join(single_line_part1)
                        # print(single_line_part1)
                    f.write('\t'.join([single_line_part0, single_line_part1, '\n']))




    return all_results_summary, \
           [all_effectiveness_results_lenx, all_effectiveness_results_len1], \
           [all_random_effectiveness_results_lenx, all_random_effectiveness_results_len1], \
           all_explain_stats, \
           effectiveness_results_len1_global_explanations

def get_embeddings(aproximate_model, all_embeddings, all_entities, entity_to_neighbours, entities):
    if aproximate_model:
        embeddings = []
        # for key, (entity, neighbours) in enumerate(entity_to_neighbours.items()):
        for entity in entities:
            [neighbours, _] = entity_to_neighbours[entity]
            entity_neighbours_embeddings = []
            for neighbour in neighbours:
                idx = all_entities.index(neighbour)
                # entity_neighbours_embeddings.append(neighbours_embeddings[idx])
                entity_neighbours_embeddings.append(all_embeddings[idx])
            entity_neighbours_embeddings = np.array(entity_neighbours_embeddings)
            entity_neighbours_embeddings_avg = np.average(entity_neighbours_embeddings, 0)
            embeddings.append(entity_neighbours_embeddings_avg.tolist())
    else:
        embeddings = []
        for entity in entities:
            idx = all_entities.index(entity)
            embeddings.append(all_embeddings[idx])

    return embeddings


def flatten_tupple(tuple_item):
    result = []
    for item in tuple_item:
        if isinstance(item, tuple):
            result.extend(flatten_tupple(item))
        else:
            result.append(item)
    return result


def save_explanation(path_entity_explanations, len_explanations, explanation_limit, single_expl_dict, expl_path_info,
                     expl_type):
    save_path = os.path.join(path_entity_explanations, f'{expl_type}_len{len_explanations}_{explanation_limit}')
    save_path_expl_path = save_path + '_feature_selection_order'

    # print('\n')
    # print(key)
    # print('nec_len')
    # print(single_expl_dict)
    # print('necessary_path_to_best_explanation')
    # print(expl_path)
    # print('\n')
    
    ## cannot save tuple keys as json dict
    # with open(save_path + '.json', 'w', encoding ='utf8') as f: 
    #     json.dump(single_expl_dict, f, ensure_ascii = False)
    with open(save_path + '.csv', 'w') as f:
        header = ['predict_proba', f'satisfied_{explanation_limit}', 'true_label', 'predicted_label', 'explanation_label', 'all_neighbours/explanation_facts', '\n']
        f.write('\t'.join(header))
        for key, value in single_expl_dict.items():
            value_list_of_strs = [str(val) for val in value]
            key_list_of_strs = []
            single_line_part0 = '\t'.join(value_list_of_strs)
            # print(single_line_part0)
            if isinstance(key, str):
                single_line_part1 = key
                # print('\n', single_line_part1)
            else:
                single_line_part1 = flatten_tupple(key)
                single_line_part1 = '\t'.join(single_line_part1)
                # print(single_line_part1)
            f.write('\t'.join([single_line_part0, single_line_part1, '\n']))

    if expl_path_info:
        with open(save_path_expl_path + '.csv', 'w') as f:
            header = ['predict_proba', f'satisfied_{explanation_limit}', 'explanation_label', 'explanation_facts', '\n']
            f.write('\t'.join(header))
            # for key, value in expl_path_info.items():
            for explanation in expl_path_info:
                # print(explanation)
                value_list_of_strs = [str(val) for val in explanation[1]]
                key_list_of_strs = []
                single_line_part0 = '\t'.join(value_list_of_strs)
                # print(single_line_part0)
                if isinstance(explanation[0], str):
                    single_line_part1 = key
                    # print('\n', single_line_part1)
                else:
                    single_line_part1 = flatten_tupple(explanation[0])
                    single_line_part1 = '\t'.join(single_line_part1)
                    # print(single_line_part1)
                f.write('\t'.join([single_line_part0, single_line_part1, '\n']))


def save_effectiveness_results(path_explanations, max_len_explanations, effectiveness_results,
                                       paths_to_explanations, explanations_dicts, path_individual_explanations):
            with open(os.path.join(path_explanations, f'effectiveness_results_len{max_len_explanations}.json'), 'w',
                      encoding ='utf8') as f: 
                json.dump(effectiveness_results[0], f, ensure_ascii = False)
            df = pd.DataFrame([effectiveness_results[0]])
            df.to_csv(os.path.join(path_explanations, f'effectiveness_results_len{max_len_explanations}.csv'),
                      sep='\t')
            with open(os.path.join(path_explanations, f'effectiveness_results_len1.json'), 'w', encoding ='utf8') as f: 
                json.dump(effectiveness_results[1], f, ensure_ascii = False)
            df = pd.DataFrame([effectiveness_results[1]])
            df.to_csv(os.path.join(path_explanations, f'effectiveness_results_len1.csv'), sep='\t')

            necessary_paths_best_expl_dict, sufficient_paths_best_expl_dict = paths_to_explanations
            # print('\n\n\n\n\nhere')
            explanations_dict_lenx, explanations_dict_len1 = explanations_dicts
            for key, [nec_len, suf_len] in explanations_dict_lenx.items():
                necessary_path = necessary_paths_best_expl_dict[key]
                sufficient_path = sufficient_paths_best_expl_dict[key]
                path_entity_explanations = os.path.join(path_individual_explanations, f'{key.split("/")[-1]}')
                # print(path_entity_explanations)
                ensure_dir(path_entity_explanations, option='make_if_not_exists')
                save_explanation(path_entity_explanations, max_len_explanations, explanation_limit, nec_len,
                                 necessary_path, 'necessary')
                save_explanation(path_entity_explanations, max_len_explanations, explanation_limit, suf_len,
                                 sufficient_path, 'sufficient')


            for key, [nec_len, suf_len] in explanations_dict_len1.items():
                path_entity_explanations = os.path.join(path_individual_explanations, f'{key.split("/")[-1]}')
                ensure_dir(path_entity_explanations, option='make_if_not_exists')
                len_explanations = 1
                save_explanation(path_entity_explanations, len_explanations, explanation_limit, nec_len, None,
                                 'necessary')
                save_explanation(path_entity_explanations, len_explanations, explanation_limit, suf_len, None,
                                 'sufficient')


def train_classifier(train_embeddings, train_labels, test_embeddings, test_labels, test_entities, aproximate_model,
                     entity_to_neighbours, dic_emb_classes, max_len_explanations, explanation_limit, all_relations,
                     current_model_path, n_jobs, RANDOM_STATE, overwrite_invidivual_explanations=False):
    
    try:
        current_model_partial_path = '/'.join(current_model_path.split('/')[-2:])
        existing_model_models_path = os.path.join(load_model_path, current_model_partial_path, 'models')
        existing_model_models_results_path = os.path.join(load_model_path, current_model_partial_path, 'models_results')
        # print(existing_model_path)
        # print(os.listdir(existing_model_path))
        # raise
        files_in_model_models_path = os.listdir(existing_model_models_path)
    except:
        files_in_model_models_path = None
        print('No model path given to use to load model.')
    if files_in_model_models_path:
        clf = load(os.path.join(existing_model_models_path, f'classifier_{dataset}'))
        try:
            with open(os.path.join(existing_model_models_path, f'lenc_{dataset}.pickle'), 'rb') as label_classes:
                lenc = pickle.load(label_classes)
        except:
            lenc = None
        clf_extra = [clf, lenc]

        with open(os.path.join(existing_model_models_results_path, 'results_summary.json'), 'r') as f:
            results_summary = json.load(f)
        ## convert str to float
        results_summary = {key: float(value) for key, value in results_summary.items()}
        
    else:
        print('No existing model, fitting new model.')
        # model = RandomForestClassifier(random_state=RANDOM_STATE)
        # param_grid = {"max_depth": [2, 4, 6, 8, 10]} ## RandomForestClassifier()

        # model = MLPClassifier(random_state=RANDOM_STATE)
        # param_grid = {
        #               'hidden_layer_sizes': [(100,), (50,50), (50,100,50)], ## MLPClassifier()
        #               'activation': ['tanh', 'relu'],
        #               'solver': ['sgd', 'adam'],
        #               # 'alpha': [0.0001, 0.05],
        #               # 'learning_rate': ['constant','adaptive'],
        #               }

        model = XGBClassifier(random_state=RANDOM_STATE)
        param_grid = {"max_depth": [2, 4, 6, 8, 10]} ## xgboost
        
        scoring=['accuracy',
                'f1_weighted',
                'f1_macro',
                'precision_weighted',
                'recall_weighted',
                'precision_macro',
                'recall_macro'
                ]
        clf = GridSearchCV(
            model,
            param_grid,
            scoring=scoring,
            refit='f1_weighted'
        )

        if type(clf.estimator).__name__ == 'XGBClassifier':
            lenc = LabelEncoder()
            lenc.fit(train_labels)
            train_labels = lenc.transform(train_labels)
            test_labels = lenc.transform(test_labels)
        else:
            lenc = None

        tic = time.perf_counter()
        clf.fit(train_embeddings, train_labels)
        toc = time.perf_counter()
        classifier_fit_time = toc - tic
        print(f"Fitted classifier model in ({classifier_fit_time:0.4f}s)\n")

        # Evaluate the Support Vector Machine on test embeddings.
        predictions = clf.predict(test_embeddings)
        predictions_proba = clf.predict_proba(test_embeddings)

        acc_scr = accuracy_score(test_labels, predictions)
        f1_scr_wei = f1_score(test_labels, predictions, average='weighted')
        prec_scr_wei = precision_score(test_labels, predictions, average='weighted')
        reca_scr_wei = recall_score(test_labels, predictions, average='weighted')
        f1_scr_macro = f1_score(test_labels, predictions, average='macro')
        prec_scr_macro = precision_score(test_labels, predictions, average='macro')
        reca_scr_macro = recall_score(test_labels, predictions, average='macro')

        print(
            f"Predicted {len(test_entities)} entities with\n"
            + f"\t{acc_scr * 100 :.4f}% ACCURACY\n"
            + f"\t{f1_scr_wei * 100 :.4f}% F1-WEIGHTED\n"
            + f"\t{prec_scr_wei * 100 :.4f}% PRECISION-WEIGHTED\n"
            + f"\t{reca_scr_wei * 100 :.4f}% RECALL-WEIGHTED\n"
            + f"\t{f1_scr_macro * 100 :.4f}% F1-MACRO\n"
            + f"\t{prec_scr_macro * 100 :.4f}% PRECISION-MACRO\n"
            + f"\t{reca_scr_macro * 100 :.4f}% RECALL-MACRO"
        )
        print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
        print(confusion_matrix(test_labels, predictions))

        mean_preds, std_preds = stats_for_preds(predictions_proba)
        print("Mean in probability of predicted class:\t\t\t", mean_preds)
        print("Standard deviation in probability of predicted class:\t", std_preds, '\n')

        results_summary = {
            'classifier_fit_time': classifier_fit_time,
            'acc_scr': acc_scr,
            'f1_scr_wei': f1_scr_wei,
            'prec_scr_wei': prec_scr_wei,
            'reca_scr_wei': reca_scr_wei,
            'f1_scr_macro': f1_scr_macro,
            'prec_scr_macro': prec_scr_macro,
            'reca_scr_macro': reca_scr_macro,
            'mean_preds': mean_preds,
            'std_preds': std_preds
        }

        if type(clf.estimator).__name__ == 'XGBClassifier':
            test_labels = lenc.inverse_transform(test_labels)
        clf_extra = [clf, lenc]

    if aproximate_model:
        dataset_labels = list(zip(test_entities, test_labels))
        path_explanations = os.path.join(current_model_path, 'explanations')
        path_individual_explanations = os.path.join(path_explanations, 'individual_explanations')
        if overwrite_invidivual_explanations:
            ensure_dir(path_explanations, option='overwrite')
            ensure_dir(path_individual_explanations, option='overwrite')
        else:
            ensure_dir(path_explanations, option='make_if_not_exists')
            ensure_dir(path_individual_explanations, option='make_if_not_exists')

        # # compute_effectiveness_kelpie(dataset_labels, path_embedding_classes, entity_to_neighbours_path,
        # #                              path_file_model, model_stats_path, path_explanations, max_len_explanations)
        # effectiveness_results, \
        # explanations_dicts, \
        # paths_to_explanations, \
        # explain_stats = compute_effectiveness_kelpie(dataset_labels, dic_emb_classes, entity_to_neighbours, clf_extra,
        #                                              results_summary, path_explanations, max_len_explanations,
        #                                              explanation_limit, n_jobs)
        # save_effectiveness_results(path_explanations, max_len_explanations, effectiveness_results,
        #                            paths_to_explanations, explanations_dicts, path_individual_explanations)
        # random_effectiveness_results = compute_random(dataset_labels, clf_extra, dic_emb_classes, entity_to_neighbours, explanations_dicts)
        # # print(random_effectiveness_results)
        # # raise
        effectiveness_results = [False, False]
        random_effectiveness_results = [False, False]
        explain_stats = False

        # fact_qtt_keys = [f'necessary_len{max_len_explanations}', 'necessary_len1_facts_qtt',
        #                  f'sufficient_len{max_len_explanations}', 'sufficient_len1_facts_qtt']
        # facts_qtt_info_for_random = {key: explain_stats[key] for key in fact_qtt_keys}
        # effectiveness_results, \
        # explanations_dicts, \
        # paths_to_explanations, \
        # explain_stats = compute_effectiveness_kelpie(dataset_labels, dic_emb_classes, entity_to_neighbours, clf,
        #                                              results_summary, path_explanations, max_len_explanations,
        #                                              explanation_limit, n_jobs, facts_qtt_info_for_random)
        # save_effectiveness_results(path_explanations, max_len_explanations, effectiveness_results,
        #                            paths_to_explanations, explanations_dicts, path_individual_explanations)
            
        ## global explanations
        explanations_dict_len1_for_global, \
        explain_stats_for_global = compute_effectiveness_global_explainer(dataset_labels, dic_emb_classes, entity_to_neighbours, clf_extra,
                                                     results_summary, path_explanations, max_len_explanations,
                                                     explanation_limit, all_relations, n_jobs)
        # print('\n\n', explanations_dict_len1_for_global)
        # print('\n\n', explain_stats_for_global)
        # raise

    else:
        effectiveness_results = [False, False]
        random_effectiveness_results = [False, False]
        explain_stats = False
        explanations_dict_len1_for_global = False
        explain_stats_for_global = False

    return clf_extra, results_summary, effectiveness_results, random_effectiveness_results, explain_stats, \
        explanations_dict_len1_for_global, explain_stats_for_global


def save_model_results(dataset, clf_extra, current_model_models_path, current_model_models_results_path, current_model_trained_path,
             results_summary):
    # transformer.save(os.path.join(current_model_models_path, f'RDF2Vec_{dataset}')) ## save transformer model

    clf, lenc = clf_extra
    dump(clf, os.path.join(current_model_models_path, f'classifier_{dataset}')) ## save node classification model

    if lenc:
        with open(os.path.join(current_model_models_path, f'lenc_{dataset}.pickle'), 'wb') as file:
            pickle.dump(lenc, file, pickle.HIGHEST_PROTOCOL)

    ## save grid search cv results although they are also saved with the joblib.dump
    df = pd.DataFrame(clf.cv_results_)
    df.to_csv(os.path.join(current_model_models_results_path, 'classifier_cv_results_.csv'), sep='\t')

    ## save grid search cv best estimator although it is also saved with the joblib.dump
    with open(os.path.join(current_model_models_results_path, 'classifier_best_estimator_.json'), 'w', encoding ='utf8') as f: 
            json.dump(str(clf.best_estimator_), f, ensure_ascii = False)

    ## save results summary for test set
    # print(results_summary)
    results_summary = {key: str(value) for key, value in results_summary.items()}
    with open(os.path.join(current_model_models_results_path, 'results_summary.json'), 'w', encoding ='utf8') as f: 
            json.dump(results_summary, f, ensure_ascii = False)
    df = pd.DataFrame([results_summary])
    df.to_csv(os.path.join(current_model_models_results_path, 'results_summary.csv'), sep='\t')

    ## save dictionary with embeddings for each neighbour, save dicionary with neighbours for each entity
    # if aproximate_model:
    #     with open(os.path.join(current_model_trained_path, 'neighbours_embeddings.json'), 'w', encoding ='utf8') as f: 
    #         json.dump(dic_emb_classes, f, ensure_ascii = False)
    #     with open(os.path.join(current_model_trained_path, 'entity_to_neighbours.json'), 'w', encoding ='utf8') as f: 
    #         json.dump(entity_to_neighbours, f, ensure_ascii = False)
            

def global_results_dict(all_results):
    df = pd.DataFrame(all_results)
    columns = list(df.columns)
    df = df.set_axis(['mean_' + column for column in columns], axis=1)
    mean_dict = OrderedDict(df.mean())
    df = df.set_axis(['std_' + column for column in columns], axis=1)
    std_dict = OrderedDict(df.std())
    global_results = dict()
    for key_mean, key_std in zip(mean_dict.keys(), std_dict.keys()):
        global_results[key_mean] = mean_dict[key_mean]
        global_results[key_std] = std_dict[key_std]

    return global_results
    

def save_global_results(aproximate_model, all_results_summary, all_effectiveness_results,
                        all_random_effectiveness_results, all_explain_stats,
                        max_len_explanations, explanation_limit, effectiveness_results_len1_global_explanations):
    global_results = global_results_dict(all_results_summary)
    # df = pd.DataFrame(all_results_summary)
    # columns = list(df.columns)
    # df = df.set_axis(['mean_' + column for column in columns], axis=1)
    # mean_dict = OrderedDict(df.mean())
    # df = df.set_axis(['std_' + column for column in columns], axis=1)
    # std_dict = OrderedDict(df.std())
    # global_results = dict()
    # for key_mean, key_std in zip(mean_dict.keys(), std_dict.keys()):
    #     global_results[key_mean] = mean_dict[key_mean]
    #     global_results[key_std] = std_dict[key_std]

    if all_effectiveness_results[0] and all_effectiveness_results[1]:
        global_effectiveness_results_lenx = global_results_dict(all_effectiveness_results[0])
        global_effectiveness_results_len1 = global_results_dict(all_effectiveness_results[1])

    if all_random_effectiveness_results[0] and all_random_effectiveness_results[1]:
        global_random_effectiveness_results_lenx = global_results_dict(all_random_effectiveness_results[0])
        global_random_effectiveness_results_len1 = global_results_dict(all_random_effectiveness_results[1])

    # for d_to_print, description in zip([global_results, global_effectiveness_results],
    #                                    ['Global Classifier Results', 'Global Explainer Results']):
    #     print(f'\n##########     {description}     ##########')
    #     for key, value in d_to_print.items():
    #         print(f'{key}: {value}')
    #     print('\n')
            
    print(f'\n##########     Global Classifier Results     ##########')
    for key, value in global_results.items():
        print(f'{key}: {value}')
    print('\n')

    if all_effectiveness_results[0]:
        print(f'\n##########     Global Local Explainer Results for Maximum Length of Explanation of {max_len_explanations}     ##########')
        for key, value in global_effectiveness_results_lenx.items():
            print(f'{key}: {value}')
        print('\n')

    if all_effectiveness_results[1]:
        print(f'\n##########     Global Local Explainer Results for Maximum Length of Explanation of 1     ##########')
        for key, value in global_effectiveness_results_len1.items():
            print(f'{key}: {value}')
        print('\n')

    if effectiveness_results_len1_global_explanations:
        print(f'\n##########     Global Global Explainer Results for Maximum Length of Explanation of 1     ##########')
        for key, value in effectiveness_results_len1_global_explanations.items():
            print(f'{key}: {value}')
        print('\n')

    if all_effectiveness_results[0] or all_effectiveness_results[1]:
        global_explain_stats = dict()
        for key, value in all_explain_stats.items():
            print(key)
            if key == 'entities':
                global_explain_stats[f'explained_{key}_size'] = len(value)
            elif key == 'all_neighbours_size':
                global_explain_stats[f'{key}'] = sum(value)
                mean, std = np.mean(value), np.std(value)
                global_explain_stats[f'neighbours_per_entity_mean_(std)'] = f'{round(mean, 3)} ({round(std, 3)})'
            elif key == 'explain_times':
                total_time = np.sum(value)
                ## this was just summing but with multiprocessing this means nothing because it is not real time
                # global_explain_stats[f'{key}_total'] = round(total_time, 3) 
                mean, std = np.mean(value), np.std(value)
                global_explain_stats[f'explain_time_per_entity_mean_(std)'] = f'{round(mean, 3)} ({round(std, 3)})'
            elif key.split('_')[3] == 'size':
                mean, std = np.mean(value), np.std(value)
                # global_explain_stats[f'{key}_mean'] = mean
                # global_explain_stats[f'{key}_std'] = std
                global_explain_stats[f'{key}_mean_(std)'] = f'{round(mean, 3)} ({round(std, 3)})'
                dif = np.array(all_explain_stats['all_neighbours_size']) - np.array(value)
                global_explain_stats[f'{key}_all_neighbours_used'] = round(np.count_nonzero(dif==0) / len(all_explain_stats['entities']), 3)
                ## this was I'm evaluating the sparsity of the global model so entities with few facts don't have so
                ## much weight as with the other version of sparsity
                global_explain_stats[f'{key}_global_sparsity'] = round(sum(value) / sum(all_explain_stats['all_neighbours_size']), 3)
                # using this way I'm calculating sparsity for each and then averaging, this means that entities where
                # there are few facts sparsity will be high and influence more the global sparsity value
                # print(np.array(value))
                # print(np.array(value).shape)
                # print(np.array(all_explain_stats['all_neighbours_size']))
                # print(np.count_nonzero(dif==0))
                # print(len(all_explain_stats['entities']))
                # print(np.array(all_explain_stats['all_neighbours_size']).shape)
                sparsity = np.array(value) / np.array(all_explain_stats['all_neighbours_size'])
                mean, std = np.mean(sparsity), np.std(sparsity)
                # global_explain_stats[f'{key}_sparsity_per_entity_mean'] = mean
                # global_explain_stats[f'{key}_sparsity_per_entity_std'] = std
                global_explain_stats[f'{key}_sparsity_per_entity_mean_(std)'] = f'{round(mean, 3)} ({round(std, 3)})'
                # global_explain_stats[f'{key.split('_')[0:2]}_using_all_neighbours'] = value / 
            elif key.split('_')[2] == 'satisfied':
                racio = value.count(True) / len(value)
                global_explain_stats[f'{key}_racio'] = round(racio, 3)

        print(f'\n##########     Explainer Stats     ##########')
        for key, value in global_explain_stats.items():
            print(f'{key}: {value}')
        print('\n')    
        
    if aproximate_model:
        model_type = 'RAN' ## representation with aggregate neighbours
    else:
        model_type = 'RO' ## representation with original

    df = pd.DataFrame(all_results_summary)
    df.to_csv(os.path.join(model_path, f'all_results_{model_type}.csv'), sep='\t')

    with open(os.path.join(model_path, f'global_results_{model_type}.json'), 'w', encoding ='utf8') as f: 
        json.dump(global_results, f, ensure_ascii = False)
    df = pd.DataFrame([global_results])
    df.to_csv(os.path.join(model_path, f'global_results_{model_type}.csv'), sep='\t')

    # if all_effectiveness_results[0] and all_effectiveness_results[1]:
    #     with open(os.path.join(model_path, f'global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
    #         json.dump(global_effectiveness_results_lenx, f, ensure_ascii = False)
    #     with open(os.path.join(model_path, f'global_effectiveness_results_len1_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
    #         json.dump(global_effectiveness_results_len1, f, ensure_ascii = False)
    #     df = pd.DataFrame([global_effectiveness_results_lenx])
    #     df.to_csv(os.path.join(model_path, f'global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.csv'), sep='\t')
    #     df = pd.DataFrame([global_effectiveness_results_len1])
    #     df.to_csv(os.path.join(model_path, f'global_effectiveness_results_len1_{explanation_limit}_{model_type}.csv'), sep='\t')

    if all_effectiveness_results[0]:
        df = pd.DataFrame(all_effectiveness_results[0])
        df.to_csv(os.path.join(model_path, f'all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.csv'), sep='\t')
        with open(os.path.join(model_path, f'global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
            json.dump(global_effectiveness_results_lenx, f, ensure_ascii = False)
        df = pd.DataFrame([global_effectiveness_results_lenx])
        df.to_csv(os.path.join(model_path, f'global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.csv'), sep='\t')
    if all_effectiveness_results[1]:
        df = pd.DataFrame(all_effectiveness_results[1])
        df.to_csv(os.path.join(model_path, f'all_effectiveness_results_len1_{explanation_limit}_{model_type}.csv'), sep='\t')
        with open(os.path.join(model_path, f'global_effectiveness_results_len1_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
            json.dump(global_effectiveness_results_len1, f, ensure_ascii = False)
        df = pd.DataFrame([global_effectiveness_results_len1])
        df.to_csv(os.path.join(model_path, f'global_effectiveness_results_len1_{explanation_limit}_{model_type}.csv'), sep='\t')
    if all_effectiveness_results[0] or all_effectiveness_results[1]:
        with open(os.path.join(model_path, f'all_explain_stats_{explanation_limit}.json'), 'w', encoding ='utf8') as f: 
            json.dump(all_explain_stats, f, ensure_ascii = False)
        df = pd.DataFrame([global_explain_stats])
        df.to_csv(os.path.join(model_path, f'global_explain_stats_{explanation_limit}.csv'), sep='\t')
        with open(os.path.join(model_path, f'global_explain_stats_{explanation_limit}.json'), 'w', encoding ='utf8') as f: 
            json.dump(global_explain_stats, f, ensure_ascii = False)

    if all_random_effectiveness_results[0]:
        df = pd.DataFrame(all_random_effectiveness_results[0])
        df.to_csv(os.path.join(model_path, f'all_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.csv'), sep='\t')
        with open(os.path.join(model_path, f'global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
            json.dump(global_random_effectiveness_results_lenx, f, ensure_ascii = False)
        df = pd.DataFrame([global_random_effectiveness_results_lenx])
        df.to_csv(os.path.join(model_path, f'global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_{model_type}.csv'), sep='\t')
    if all_random_effectiveness_results[1]:
        df = pd.DataFrame(all_random_effectiveness_results[1])
        df.to_csv(os.path.join(model_path, f'all_random_effectiveness_results_len1_{explanation_limit}_{model_type}.csv'), sep='\t')
        with open(os.path.join(model_path, f'global_random_effectiveness_results_len1_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
            json.dump(global_random_effectiveness_results_len1, f, ensure_ascii = False)
        df = pd.DataFrame([global_random_effectiveness_results_len1])
        df.to_csv(os.path.join(model_path, f'global_random_effectiveness_results_len1_{explanation_limit}_{model_type}.csv'), sep='\t')

    if effectiveness_results_len1_global_explanations:
        with open(os.path.join(model_path, f'global_explainer_global_effectiveness_results_len1_{explanation_limit}_{model_type}.json'), 'w', encoding ='utf8') as f: 
            json.dump(effectiveness_results_len1_global_explanations, f, ensure_ascii = False)
        df = pd.DataFrame([effectiveness_results_len1_global_explanations])
        df.to_csv(os.path.join(model_path, f'global_explainer_global_effectiveness_results_len1_{explanation_limit}_{model_type}.csv'), sep='\t')

    return global_results


# explanation_limit='threshold'

# aproximate_model = False
# all_results_summary, all_effectiveness_results, all_random_effectiveness_results, all_explain_stats, \
#     effectiveness_results_len1_global_explanations = run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
#                      train_index_files_designation, test_index_files_designation, aproximate_model, RANDOM_STATE,
#                      max_len_explanations, explanation_limit, n_jobs,
#                      n_partitions=n_splits)

# save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_random_effectiveness_results,
#                     all_explain_stats, max_len_explanations, explanation_limit,
#                     effectiveness_results_len1_global_explanations)

explanation_limit='threshold'

aproximate_model = True
all_results_summary, all_effectiveness_results, all_random_effectiveness_results, all_explain_stats, \
     effectiveness_results_len1_global_explanations = run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                     train_index_files_designation, test_index_files_designation, aproximate_model, RANDOM_STATE,
                     max_len_explanations, explanation_limit, n_jobs,
                     n_partitions=n_splits, overwrite_invidivual_explanations=True)

save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_random_effectiveness_results,
                    all_explain_stats, max_len_explanations, explanation_limit,
                    effectiveness_results_len1_global_explanations)

explanation_limit='class_change'

aproximate_model = True
all_results_summary, all_effectiveness_results, all_random_effectiveness_results, all_explain_stats, \
     effectiveness_results_len1_global_explanations = run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                     train_index_files_designation, test_index_files_designation, aproximate_model, RANDOM_STATE,
                     max_len_explanations, explanation_limit, n_jobs,
                     n_partitions=n_splits, overwrite_invidivual_explanations=False)

save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_random_effectiveness_results,
                    all_explain_stats, max_len_explanations, explanation_limit,
                    effectiveness_results_len1_global_explanations)

toc_total_script_time = time.perf_counter()
print(f"\nTotal script time in ({toc_total_script_time - tic_total_script_time:0.4f}s)\n")

shutil.move('node_classifier/tmp/train_models.log', os.path.join(model_path, 'train_models.log'))