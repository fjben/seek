

import argparse
import json
import os
import random
import shutil
import sys
import time
import warnings

from multiprocessing import cpu_count
from collections import defaultdict
from collections import OrderedDict

from joblib import dump
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from pyrdf2vec import RDF2VecTransformer

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
from utils.logger import Logger
from run_seek_explanations import compute_effectiveness_kelpie


tic_total_script_time = time.perf_counter()


############################################################################### arguments

parser = argparse.ArgumentParser(description="description")
parser.add_argument("--dataset",
                    type=str,
                    choices=['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE'],
                    help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")
args = parser.parse_args()
dataset = args.dataset

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
    data_partitions_path = os.path.join(filename_output, 'data_partitions')
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


def setup_random_seeds(current_model_models_path, cpu_num):
    try:
        with open(os.path.join(current_model_models_path, 'reproducibility_parameters.txt')) as f:
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
model_path = f'node_classifier/cv_model/{dataset}'
transformer_model_path = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/models/RDF2Vec_{dataset}'
entity_to_neighbours_path = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/trained/entity_to_neighbours.json'
path_embedding_classes = f'node_classifier/model/{dataset}/{dataset}_model_0_RAN/trained/neighbours_embeddings.json'
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

RANDOM_STATE, workers = setup_random_seeds(model_path, cpu_num)

# n_jobs = 2 ## original specs found in online-learning.py
n_jobs = cpu_num

print('RANDOM_STATE:\t\t', RANDOM_STATE)
print('workers:\t\t', workers)
print("Number of used cpu:\t", n_jobs, '\n')

train_index_files_designation, test_index_files_designation = run_partition(entities, labels, model_path, n_splits, RANDOM_STATE)

## I don't think I need this, I can load this information using the /trained data directly from the dict
transformer = RDF2VecTransformer().load(transformer_model_path)
all_embeddings = transformer._embeddings
all_entities = transformer._entities

with open(entity_to_neighbours_path, 'r') as f:
    entity_to_neighbours = json.load(f)

with open(path_embedding_classes, 'r') as f:
    dic_emb_classes = json.load(f)

def run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                         train_index_files_designation, test_index_files_designation,
                         aproximate_model, RANDOM_STATE, max_len_explanations, explanation_limit, n_jobs,
                         n_partitions):
    
    all_results_summary = []
    all_effectiveness_results_lenx = []
    all_effectiveness_results_len1 = []
    all_explain_stats = defaultdict(list)
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

        clf, results_summary, effectiveness_results, explain_stats = train_classifier(train_embeddings, train_labels, test_embeddings,
                                                                       test_labels, test_entities, aproximate_model,
                                                                       entity_to_neighbours, dic_emb_classes,
                                                                       max_len_explanations, explanation_limit,
                                                                       current_model_path, n_jobs, RANDOM_STATE)

        save_model_results(dataset, clf, current_model_models_path, current_model_models_results_path, current_model_trained_path,
             results_summary)
        
        all_results_summary.append(results_summary)
        all_effectiveness_results_lenx.append(effectiveness_results[0])
        all_effectiveness_results_len1.append(effectiveness_results[1])
        if explain_stats:
            for key, _ in explain_stats.items():
                all_explain_stats[key].extend(explain_stats[key])

    if not any(all_effectiveness_results_lenx) and not any(all_effectiveness_results_len1):
        all_effectiveness_results_lenx = False
        all_effectiveness_results_len1 = False

    return all_results_summary, [all_effectiveness_results_lenx, all_effectiveness_results_len1], all_explain_stats

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


def save_explanation(path_entity_explanations, len_explanations, explanation_limit, single_expl_dict, expl_type):
    save_path = os.path.join(path_entity_explanations, f'{expl_type}_len{len_explanations}_{explanation_limit}')
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


def train_classifier(train_embeddings, train_labels, test_embeddings, test_labels, test_entities, aproximate_model,
                     entity_to_neighbours, dic_emb_classes, max_len_explanations, explanation_limit,
                     current_model_path, n_jobs, RANDOM_STATE):
    # Fit a Support Vector Machine on train embeddings and pick the best
    # C-parameters (regularization strength).
    param_grid = {"max_depth": [2, 4, 6, 8, 10]}
    scoring=['accuracy',
            'f1_weighted',
            'f1_macro',
            'precision_weighted',
            'recall_weighted',
            'precision_macro',
            'recall_macro'
            ]
    clf = GridSearchCV(
        # SVC(random_state=RANDOM_STATE), {"C": [10**i for i in range(-3, 4)]}
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        scoring=scoring,
        refit='f1_weighted'
    )

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

    if aproximate_model:
        dataset_labels = list(zip(test_entities, test_labels))
        path_explanations = os.path.join(current_model_path, 'explanations')
        ensure_dir(path_explanations, option='make_if_not_exists')
        path_individual_explanations = os.path.join(path_explanations, 'individual_explanations')
        ensure_dir(path_individual_explanations, option='make_if_not_exists')

        # compute_effectiveness_kelpie(dataset_labels, path_embedding_classes, entity_to_neighbours_path,
        #                              path_file_model, model_stats_path, path_explanations, max_len_explanations)
        effectiveness_results, explanations_dicts, explain_stats = compute_effectiveness_kelpie(dataset_labels, dic_emb_classes,
                                                                  entity_to_neighbours, clf, results_summary,
                                                                  path_explanations, max_len_explanations,
                                                                  explanation_limit, n_jobs)
        
        with open(os.path.join(path_explanations, f'effectiveness_results_len{max_len_explanations}.json'), 'w', encoding ='utf8') as f: 
            json.dump(effectiveness_results[0], f, ensure_ascii = False)
        df = pd.DataFrame([effectiveness_results[0]])
        df.to_csv(os.path.join(path_explanations, f'effectiveness_results_len{max_len_explanations}.csv'), sep='\t')
        with open(os.path.join(path_explanations, f'effectiveness_results_len1.json'), 'w', encoding ='utf8') as f: 
            json.dump(effectiveness_results[1], f, ensure_ascii = False)
        df = pd.DataFrame([effectiveness_results[1]])
        df.to_csv(os.path.join(path_explanations, f'effectiveness_results_len1.csv'), sep='\t')

        explanations_dict_lenx, explanations_dict_len1 = explanations_dicts
        for key, [nec_len, suf_len] in explanations_dict_lenx.items():
            path_entity_explanations = os.path.join(path_individual_explanations, f'{key.split("/")[-1]}')
            ensure_dir(path_entity_explanations, option='make_if_not_exists')
            save_explanation(path_entity_explanations, max_len_explanations, explanation_limit, nec_len, 'necessary')
            save_explanation(path_entity_explanations, max_len_explanations, explanation_limit, suf_len, 'sufficient')

        for key, [nec_len, suf_len] in explanations_dict_len1.items():
            path_entity_explanations = os.path.join(path_individual_explanations, f'{key.split("/")[-1]}')
            ensure_dir(path_entity_explanations, option='make_if_not_exists')
            len_explanations = 1
            save_explanation(path_entity_explanations, len_explanations, explanation_limit, nec_len, 'necessary')
            save_explanation(path_entity_explanations, len_explanations, explanation_limit, suf_len, 'sufficient')
    else:
        effectiveness_results = [False, False]
        explain_stats = False

    return clf, results_summary, effectiveness_results, explain_stats


def save_model_results(dataset, clf, current_model_models_path, current_model_models_results_path, current_model_trained_path,
             results_summary):
    # transformer.save(os.path.join(current_model_models_path, f'RDF2Vec_{dataset}')) ## save transformer model

    dump(clf, os.path.join(current_model_models_path, f'classifier_{dataset}')) ## save node classification model

    ## save grid search cv results although they are also saved with the joblib.dump
    df = pd.DataFrame(clf.cv_results_)
    df.to_csv(os.path.join(current_model_models_results_path, 'classifier_cv_results_.csv'), sep='\t')

    ## save grid search cv best estimator although it is also saved with the joblib.dump
    with open(os.path.join(current_model_models_results_path, 'classifier_best_estimator_.json'), 'w', encoding ='utf8') as f: 
            json.dump(str(clf.best_estimator_), f, ensure_ascii = False)

    ## save results summary for test set
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
    

def save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_explain_stats,
                        max_len_explanations, explanation_limit):
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
        print(f'\n##########     Global Explainer Results for Maximum Length of Explanation of {max_len_explanations}     ##########')
        for key, value in global_effectiveness_results_lenx.items():
            print(f'{key}: {value}')
        print('\n')

    if all_effectiveness_results[1]:
        print(f'\n##########     Global Explainer Results for Maximum Length of Explanation of 1     ##########')
        for key, value in global_effectiveness_results_len1.items():
            print(f'{key}: {value}')
        print('\n')

    if all_effectiveness_results[0] or all_effectiveness_results[1]:
        global_explain_stats = dict()
        for key, value in all_explain_stats.items():
            if key == 'explain_times':
                total_time = np.sum(value)
                global_explain_stats[f'{key}_total'] = total_time
            if key != 'entities':
                mean, std = np.mean(value), np.std(value)
                global_explain_stats[f'{key}_mean'] = mean
                global_explain_stats[f'{key}_std'] = std

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
        df = pd.DataFrame([global_explain_stats])
        df.to_csv(os.path.join(model_path, f'global_explain_stats_{explanation_limit}.csv'), sep='\t')
        with open(os.path.join(model_path, f'global_explain_stats_{explanation_limit}.json'), 'w', encoding ='utf8') as f: 
            json.dump(global_explain_stats, f, ensure_ascii = False)

    return global_results


explanation_limit='threshold'

aproximate_model = False
all_results_summary, all_effectiveness_results, all_explain_stats = run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                     train_index_files_designation, test_index_files_designation, aproximate_model, RANDOM_STATE,
                     max_len_explanations, explanation_limit, n_jobs, n_partitions=n_splits)

save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_explain_stats, max_len_explanations, explanation_limit)

explanation_limit='threshold'

aproximate_model = True
all_results_summary, all_effectiveness_results, all_explain_stats = run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                     train_index_files_designation, test_index_files_designation, aproximate_model, RANDOM_STATE,
                     max_len_explanations, explanation_limit, n_jobs, n_partitions=n_splits)

save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_explain_stats, max_len_explanations, explanation_limit)

explanation_limit='class_change'

aproximate_model = True
all_results_summary, all_effectiveness_results, all_explain_stats = run_cross_validation(all_embeddings, all_entities, entity_to_neighbours, dic_emb_classes, entities, labels,
                     train_index_files_designation, test_index_files_designation, aproximate_model, RANDOM_STATE,
                     max_len_explanations, explanation_limit, n_jobs, n_partitions=n_splits)

save_global_results(aproximate_model, all_results_summary, all_effectiveness_results, all_explain_stats, max_len_explanations, explanation_limit)

toc_total_script_time = time.perf_counter()
print(f"\nTotal script time in ({toc_total_script_time - tic_total_script_time:0.4f}s)\n")

shutil.move('node_classifier/tmp/train_models.log', os.path.join(model_path, 'train_models.log'))