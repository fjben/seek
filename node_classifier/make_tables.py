import argparse
import json
import os

from collections import defaultdict
from collections import OrderedDict

import shutil

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


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
args = parser.parse_args()
dataset = args.dataset
kge_model = args.kge_model

# dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
# dataset = 'MDGENRE'

# main_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model'
# main_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_backup_latest'
# main_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_xgb_local_final'
main_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_mlp_local_final'


############################################################################### functions

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


def create_df_from_global_effectiveness_results(load_path, dataset, max_len_explanations, explanation_limit,
                                                random_results=False):
    # load_path = f'{main_path}/{dataset}_{kge_model}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')
    df_std_values = df.copy(deep=True)

    columns_to_keep = [
        'mean_original_accuracy_score_',
        'mean_necessary_accuracy_score_',
        'mean_delta_necessary_accuracy_score_',
        'mean_sufficient_accuracy_score_',
        'mean_delta_sufficient_accuracy_score_',
        'mean_original_f1_score_weighted',
        'mean_necessary_f1_score_weighted',
        'mean_delta_necessary_f1_score_weighted',
        'mean_sufficient_f1_score_weighted',
        'mean_delta_sufficient_f1_score_weighted',
        'mean_original_precision_score_weighted',
        'mean_necessary_precision_score_weighted',
        'mean_delta_necessary_precision_score_weighted',
        'mean_sufficient_precision_score_weighted',
        'mean_delta_sufficient_precision_score_weighted',
        'mean_original_recall_score_weighted',
        'mean_necessary_recall_score_weighted',
        'mean_delta_necessary_recall_score_weighted',
        'mean_sufficient_recall_score_weighted',
        'mean_delta_sufficient_recall_score_weighted',
    ]
    df = df[columns_to_keep]
    std_columns_to_keep = [
        'std_original_accuracy_score_',
        'std_necessary_accuracy_score_',
        'std_delta_necessary_accuracy_score_',
        'std_sufficient_accuracy_score_',
        'std_delta_sufficient_accuracy_score_',
        'std_original_f1_score_weighted',
        'std_necessary_f1_score_weighted',
        'std_delta_necessary_f1_score_weighted',
        'std_sufficient_f1_score_weighted',
        'std_delta_sufficient_f1_score_weighted',
        'std_original_precision_score_weighted',
        'std_necessary_precision_score_weighted',
        'std_delta_necessary_precision_score_weighted',
        'std_sufficient_precision_score_weighted',
        'std_delta_sufficient_precision_score_weighted',
        'std_original_recall_score_weighted',
        'std_necessary_recall_score_weighted',
        'std_delta_necessary_recall_score_weighted',
        'std_sufficient_recall_score_weighted',
        'std_delta_sufficient_recall_score_weighted',
    ]
    df_std_values = df_std_values[std_columns_to_keep]

    indexes = [0, 5, 10, 15]
    rows = []
    # for i in range(5):
    #     df.iloc[:, indexes]
    #     rows.append(df.iloc[:, indexes].to_numpy()[0].tolist())
    #     indexes = [index + 1 for index in indexes]
    # for i in range(5):
    #     df.iloc[:, indexes]
    #     rows.append(f'{df.iloc[:, indexes].to_numpy()[0].tolist()} ({df_std_values.iloc[:, indexes].to_numpy()[0].tolist()})')
    #     indexes = [index + 1 for index in indexes]
    for i in range(5):
        row = []
        for index in indexes:
            # print('here')
            # print(index)
            # print('here')
            # print(df.iloc[0, index])
            # print(df_std_values.iloc[:, index])
            # print('here')
            # print(f'{df.iloc[0, index]} ({df_std_values.iloc[0, index]})')
            # print('here')
            row.append(f'{round(df.iloc[0, index], 3)} ({round(df_std_values.iloc[0, index], 3)})')
        # rows.append(df.iloc[:, indexes].to_numpy()[0].tolist())
        rows.append(row)
        indexes = [index + 1 for index in indexes]

    rows = np.array(rows)
    # print(rows)
    columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
    df_final = pd.DataFrame(rows, columns=columns_names)

    if random_results:
        indexes_names = ['original', 'random necessary', 'delta random necessary', 'random sufficient', 'delta random sufficient']
    else:
        indexes_names = ['original', 'explainer necessary', 'delta explainer necessary', 'explainer sufficient', 'delta explainer sufficient']
    df_final = df_final.set_index([indexes_names])

    empty_data = {col: ['' for _ in range(1)] for col in df_final.columns}
    df_empty = pd.DataFrame(empty_data)
    df_empty = df_empty.set_index([['']])

    df1 = df_final.iloc[:1]
    df2 = df_final.iloc[1:]
    df_final = pd.concat([df1, df_empty, df2])

    df1 = df_final.iloc[:4]
    df2 = df_final.iloc[4:]
    df_final = pd.concat([df1, df_empty, df2])

    return df_final


def create_df_from_all_effectiveness_results(load_path, dataset, max_len_explanations, explanation_limit):
    # load_path = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')

    columns_to_keep = [
        'original_accuracy_score_',
        'necessary_accuracy_score_',
        'delta_necessary_accuracy_score_',
        'sufficient_accuracy_score_',
        'delta_sufficient_accuracy_score_',
        'original_f1_score_weighted',
        'necessary_f1_score_weighted',
        'delta_necessary_f1_score_weighted',
        'sufficient_f1_score_weighted',
        'delta_sufficient_f1_score_weighted',
        'original_precision_score_weighted',
        'necessary_precision_score_weighted',
        'delta_necessary_precision_score_weighted',
        'sufficient_precision_score_weighted',
        'delta_sufficient_precision_score_weighted',
        'original_recall_score_weighted',
        'necessary_recall_score_weighted',
        'delta_necessary_recall_score_weighted',
        'sufficient_recall_score_weighted',
        'delta_sufficient_recall_score_weighted',
    ]
    df = df[columns_to_keep]

    # indexes = [0, 5, 10]
    # rows = []
    # for i in range(5):
    #     df.iloc[:, indexes]
    #     rows.append(df.iloc[:, indexes].to_numpy()[0].tolist())
    #     indexes = [index + 1 for index in indexes]

    # rows = np.array(rows)
    # columns_names = ['f1_score', 'precision', 'recall']
    # df_final = pd.DataFrame(rows, columns=columns_names)

    # indexes_names = ['original', 'necessary', 'delta_necessary', 'sufficient', 'delta_sufficient']
    # df_final = df_final.set_index([indexes_names])

    # empty_data = {col: ['' for _ in range(1)] for col in df_final.columns}
    # df_empty = pd.DataFrame(empty_data)
    # df_empty = df_empty.set_index([['']])

    # df1 = df_final.iloc[:1]
    # df2 = df_final.iloc[1:]
    # df_final = pd.concat([df1, df_empty, df2])

    # df1 = df_final.iloc[:4]
    # df2 = df_final.iloc[4:]
    # df_final = pd.concat([df1, df_empty, df2])

    df_final = df

    return df_final


def create_df_from_global_classifier_results(load_path, dataset, model_type):
    # load_path = f'{main_path}/{dataset}_{kge_model}/global_results_{model_type}'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')
    df_std_values = df.copy(deep=True)


    columns_to_keep = [
        'mean_acc_scr',
        'mean_f1_scr_wei',
        'mean_prec_scr_wei',
        'mean_reca_scr_wei',
        'mean_mean_preds',
        'mean_std_preds',
        'mean_classifier_fit_time'
    ]
    std_columns_to_keep = [
        'std_acc_scr',
        'std_f1_scr_macro',
        'std_prec_scr_macro',
        'std_reca_scr_macro',
        'std_mean_preds',
        'std_std_preds',
        'std_classifier_fit_time'
    ]
    df = df[columns_to_keep]
    df_std_values = df_std_values[std_columns_to_keep]

    row = []
    for index in range(len(df.columns)):
        row.append(f'{round(df.iloc[0, index], 3)} ({round(df_std_values.iloc[0, index], 3)})')

    row = np.array(row)
    # print(len(df.columns))
    # for r in row:
    #     print(r)

    columns_names = ['accuracy', 'f1_score', 'precision', 'recall', 'pred_proba_in_pred_classes', 'threshold', 'classifier_fit_time']
    # print(len(columns_names))
    df_final = pd.DataFrame([row], columns=columns_names)

    # df_final = df ## .drop(['Unnamed: 0'],axis=1)

    return df_final


def create_df_from_classifier_results(load_path, dataset, model_type):
    # load_path = f'{main_path}/{dataset}_{kge_model}/all_results_{model_type}'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')

    columns_to_keep = [
        'acc_scr',
        'f1_scr_wei',
        'prec_scr_wei',
        'reca_scr_wei',
        'mean_preds',
        'std_preds',
        'classifier_fit_time',
    ]
    other_columns = [
        'f1_scr_macro',
        'prec_scr_macro',
        'reca_scr_macro',
        'mean_preds',
        'std_preds'
    ]
    df = df[columns_to_keep]

    df_final = df ## .drop(['Unnamed: 0'],axis=1)

    return df_final


def wilcoxon_test(original_scores, necessary_scores, sufficient_scores):
    results = []
    for score in [necessary_scores, sufficient_scores]:
        if np.sum(original_scores - score) == 0:
            ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
            results.append('x[i]-y[i]=0')
        else:
            ## always gives the same result if all the values from one sample are larger then the values of the other
            res = wilcoxon(np.array(original_scores), np.array(score)) # , alternative='greater')
            # results.append(res[1])
            results.append("{:.2e}".format(res[1]))
            ## print to check why p-values all equal
            # print('\n')
            # print(original_scores)
            # print(score)
            # print(res)
            # print('\n')

    return results

def compute_final_stats_tests(load_path, dataset, max_len_explanations, explanation_limit):
    accuracy_labels = ['original_accuracy_score_', 'necessary_accuracy_score_', 'sufficient_accuracy_score_']
    f1_weighted_labels = ['original_f1_score_weighted', 'necessary_f1_score_weighted', 'sufficient_f1_score_weighted']
    precision_weighted_labels = ['original_precision_score_weighted', 'necessary_precision_score_weighted', 'sufficient_precision_score_weighted',]
    recall_weighted_labels = ['original_recall_score_weighted', 'necessary_recall_score_weighted', 'sufficient_recall_score_weighted',]

    # load_path = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    df_effectiveness_stats_tests = create_df_from_all_effectiveness_results(load_path=load_path, dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
    # all_stats_tests_results = OrderedDict()
    all_stats_test_results_by_explan_type = defaultdict(list)
    necessary_scores_all_metrics = []
    sufficient_scores_all_metrics = []
    for met_labels in [accuracy_labels, f1_weighted_labels, precision_weighted_labels, recall_weighted_labels]:
        original_scores_label, necessary_scores_label, sufficient_scores_label = met_labels
        description = f'{original_scores_label.split("_")[1]} p-values'
        original_scores = np.array(df_effectiveness_stats_tests[original_scores_label])
        necessary_scores = np.array(df_effectiveness_stats_tests[necessary_scores_label])
        sufficient_scores = np.array(df_effectiveness_stats_tests[sufficient_scores_label])
        necessary_scores_all_metrics.append(necessary_scores)
        sufficient_scores_all_metrics.append(sufficient_scores)

        # print('\n')
        # print(original_scores_label)
        # print(original_scores)
        # print(necessary_scores)
        # # print(sufficient_scores)
        # print('\n')
        [necessary_p_value, sufficient_p_value] = wilcoxon_test(original_scores, necessary_scores, sufficient_scores)
        # all_stats_tests_results[description] = {'necessary_p_value': necessary_p_value,
        #                                         'sufficient_p_value': sufficient_p_value}
        all_stats_test_results_by_explan_type['necessary'].append(necessary_p_value)
        all_stats_test_results_by_explan_type['sufficient'].append(sufficient_p_value)

    # for score_metric in all_stats_tests_results.keys():
    #     all_stats_tests_results[score_metric]['necessary_p_value']
    #     new_col = ['',
    #             '',
    #             all_stats_tests_results[score_metric]['necessary_p_value'],
    #             '',
    #             '',
    #             all_stats_tests_results[score_metric]['sufficient_p_value'],
    #             ''
    #             ]
    #     df_final[f'{score_metric}'] = new_col
        
    # print(all_stats_test_results_by_explan_type)
    # raise
        
    return all_stats_test_results_by_explan_type, [necessary_scores_all_metrics, sufficient_scores_all_metrics]


def create_final_effectiveness_results(load_path_global, load_path_all, dataset, max_len_explanations,
                                       explanation_limit, random_results=False):
    # load_path = f'{main_path}/{dataset}_{kge_model}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    df_final = create_df_from_global_effectiveness_results(load_path=load_path_global, dataset=dataset, max_len_explanations=max_len_explanations,
                                                           explanation_limit=explanation_limit, random_results=random_results)
    # print('\n', df_final)
    # df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')
    
    # load_path = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    all_stats_test_results_by_explan_type, individual_scores_for_all_metrics = compute_final_stats_tests(load_path=load_path_all, dataset=dataset, max_len_explanations=max_len_explanations,
                                                           explanation_limit=explanation_limit)
    
    columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
    if random_results:
        index_names = ['p-values random vs original']
    else:
        index_names = ['p-values explainer vs original']
    df_nec = pd.DataFrame([all_stats_test_results_by_explan_type['necessary']], columns=columns_names)
    df_nec = df_nec.set_index([index_names])
    df1 = df_final.iloc[:4]
    df2 = df_final.iloc[4:]
    df_final = pd.concat([df1, df_nec, df2])
    df_suf = pd.DataFrame([all_stats_test_results_by_explan_type['sufficient']], columns=columns_names)
    df_suf = df_suf.set_index([index_names])
    # df1 = df_final.iloc[:4]
    # df2 = df_final.iloc[4:]
    df_final = pd.concat([df_final, df_suf])
    # print(df_final)


    # load_path = f'{main_path}/{dataset}_{kge_model}/global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    # df_final = create_df_from_global_effectiveness_results(load_path=load_path, dataset=dataset, max_len_explanations=max_len_explanations,
    #                                                        explanation_limit=explanation_limit)
    # # print('\n', df_final)
    # # df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')
    
    # load_path = f'{main_path}/{dataset}_{kge_model}/all_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    # all_stats_test_results_by_explan_type = compute_final_stats_tests(load_path=load_path, dataset=dataset, max_len_explanations=max_len_explanations,
    #                                                        explanation_limit=explanation_limit)

    # columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
    # df_nec = pd.DataFrame([all_stats_test_results_by_explan_type['necessary']], columns=columns_names)
    # df_nec = df_nec.set_index([['p-values']])
    # df1 = df_final.iloc[:4]
    # df2 = df_final.iloc[4:]
    # df_final = pd.concat([df1, df_nec, df2])
    # df_suf = pd.DataFrame([all_stats_test_results_by_explan_type['sufficient']], columns=columns_names)
    # df_suf = df_suf.set_index([['p-values']])
    # # df1 = df_final.iloc[:4]
    # # df2 = df_final.iloc[4:]
    # df_final = pd.concat([df_final, df_suf])
    # print(df_final)
    # raise


        
        
    

    # print(df_final)
    # raise

    return df_final, individual_scores_for_all_metrics


############################################################################### script

save_path = f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}_{kge_model}'

ensure_dir(save_path, option='overwrite')

## these tables are already final so just copy them
try:
    shutil.copy(f'{main_path}/{dataset}_{kge_model}/global_explain_stats_class_change.csv', f'{save_path}/global_explain_stats_class_change.csv')
    shutil.copy(f'{main_path}/{dataset}_{kge_model}/global_explain_stats_threshold.csv', f'{save_path}/global_explain_stats_threshold.csv')
except FileNotFoundError:
    with open(f'{save_path}/warnings.log', 'a') as f:
        f.write(f'file not found: {main_path}/{dataset}_{kge_model}/global_explain_stats_class_change.csv')

    
max_len_explanations = 1
explanation_limit = 'threshold'
load_path_global = f'{main_path}/{dataset}_{kge_model}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_final, individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit)
# print('\n', df_final)

load_path_global = f'{main_path}/{dataset}_{kge_model}/global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_random_final, random_individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit, random_results=True)
# print('\n', df_random_final)

columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
index_names = ['p-values explainer vs random']
[necessary_scores, sufficient_scores] = individual_scores_for_all_metrics
[random_necessary_scores, random_sufficient_scores] = random_individual_scores_for_all_metrics
all_stats_test_results_by_explan_type = defaultdict(list)
for i in range(len(columns_names)):
    # print(necessary_scores)
    current_necessary_scores = necessary_scores[i]
    current_random_necessary_scores = random_necessary_scores[i]
    if np.sum(current_necessary_scores - current_random_necessary_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['necessary'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_necessary_scores, current_random_necessary_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['necessary'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

    current_sufficient_scores = sufficient_scores[i]
    current_random_sufficient_scores = random_sufficient_scores[i]
    if np.sum(current_sufficient_scores - current_random_sufficient_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['sufficient'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_sufficient_scores, current_random_sufficient_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['sufficient'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

print(all_stats_test_results_by_explan_type)
df_nec_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['necessary']], columns=columns_names)
df_nec_to_random = df_nec_to_random.set_index([index_names])
df_suf_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['sufficient']], columns=columns_names)
df_suf_to_random = df_suf_to_random.set_index([index_names])

df1 = df_final.iloc[:2]
df2 = df_random_final.iloc[2:5]
df3 = df_final.iloc[2:5]
df4 = df_nec_to_random.iloc[:]
df5 = df_random_final.iloc[5:]
df6 = df_final.iloc[6:]
df7 = df_suf_to_random.iloc[:]
df_final_final = pd.concat([df1, df2, df3, df4, df5, df6, df7])
print(df_final_final)
df_final_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}_{kge_model}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')




max_len_explanations = 5
explanation_limit = 'threshold'
load_path_global = f'{main_path}/{dataset}_{kge_model}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_final, individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit)
# print('\n', df_final)

load_path_global = f'{main_path}/{dataset}_{kge_model}/global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_random_final, random_individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit, random_results=True)
# print('\n', df_random_final)

columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
index_names = ['p-values explainer vs random']
[necessary_scores, sufficient_scores] = individual_scores_for_all_metrics
[random_necessary_scores, random_sufficient_scores] = random_individual_scores_for_all_metrics
all_stats_test_results_by_explan_type = defaultdict(list)
for i in range(len(columns_names)):
    # print(necessary_scores)
    current_necessary_scores = necessary_scores[i]
    current_random_necessary_scores = random_necessary_scores[i]
    if np.sum(current_necessary_scores - current_random_necessary_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['necessary'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_necessary_scores, current_random_necessary_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['necessary'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

    current_sufficient_scores = sufficient_scores[i]
    current_random_sufficient_scores = random_sufficient_scores[i]
    if np.sum(current_sufficient_scores - current_random_sufficient_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['sufficient'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_sufficient_scores, current_random_sufficient_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['sufficient'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

print(all_stats_test_results_by_explan_type)
df_nec_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['necessary']], columns=columns_names)
df_nec_to_random = df_nec_to_random.set_index([index_names])
df_suf_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['sufficient']], columns=columns_names)
df_suf_to_random = df_suf_to_random.set_index([index_names])

df1 = df_final.iloc[:2]
df2 = df_random_final.iloc[2:5]
df3 = df_final.iloc[2:5]
df4 = df_nec_to_random.iloc[:]
df5 = df_random_final.iloc[5:]
df6 = df_final.iloc[6:]
df7 = df_suf_to_random.iloc[:]
df_final_final = pd.concat([df1, df2, df3, df4, df5, df6, df7])
print(df_final_final)
df_final_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}_{kge_model}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')




max_len_explanations = 1
explanation_limit = 'class_change'
load_path_global = f'{main_path}/{dataset}_{kge_model}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_final, individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit)
# print('\n', df_final)

load_path_global = f'{main_path}/{dataset}_{kge_model}/global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_random_final, random_individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit, random_results=True)
# print('\n', df_random_final)

columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
index_names = ['p-values explainer vs random']
[necessary_scores, sufficient_scores] = individual_scores_for_all_metrics
[random_necessary_scores, random_sufficient_scores] = random_individual_scores_for_all_metrics
all_stats_test_results_by_explan_type = defaultdict(list)
for i in range(len(columns_names)):
    # print(necessary_scores)
    current_necessary_scores = necessary_scores[i]
    current_random_necessary_scores = random_necessary_scores[i]
    if np.sum(current_necessary_scores - current_random_necessary_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['necessary'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_necessary_scores, current_random_necessary_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['necessary'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

    current_sufficient_scores = sufficient_scores[i]
    current_random_sufficient_scores = random_sufficient_scores[i]
    if np.sum(current_sufficient_scores - current_random_sufficient_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['sufficient'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_sufficient_scores, current_random_sufficient_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['sufficient'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

print(all_stats_test_results_by_explan_type)
df_nec_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['necessary']], columns=columns_names)
df_nec_to_random = df_nec_to_random.set_index([index_names])
df_suf_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['sufficient']], columns=columns_names)
df_suf_to_random = df_suf_to_random.set_index([index_names])

df1 = df_final.iloc[:2]
df2 = df_random_final.iloc[2:5]
df3 = df_final.iloc[2:5]
df4 = df_nec_to_random.iloc[:]
df5 = df_random_final.iloc[5:]
df6 = df_final.iloc[6:]
df7 = df_suf_to_random.iloc[:]
df_final_final = pd.concat([df1, df2, df3, df4, df5, df6, df7])
print(df_final_final)
df_final_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}_{kge_model}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')





max_len_explanations = 5
explanation_limit = 'class_change'
load_path_global = f'{main_path}/{dataset}_{kge_model}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_final, individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit)
# print('\n', df_final)

load_path_global = f'{main_path}/{dataset}_{kge_model}/global_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
load_path_all = f'{main_path}/{dataset}_{kge_model}/all_random_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
df_random_final, random_individual_scores_for_all_metrics = create_final_effectiveness_results(load_path_global=load_path_global, load_path_all=load_path_all,
                                              dataset=dataset, max_len_explanations=max_len_explanations,
                                              explanation_limit=explanation_limit, random_results=True)
# print('\n', df_random_final)

columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
index_names = ['p-values explainer vs random']
[necessary_scores, sufficient_scores] = individual_scores_for_all_metrics
[random_necessary_scores, random_sufficient_scores] = random_individual_scores_for_all_metrics
all_stats_test_results_by_explan_type = defaultdict(list)
for i in range(len(columns_names)):
    # print(necessary_scores)
    current_necessary_scores = necessary_scores[i]
    current_random_necessary_scores = random_necessary_scores[i]
    if np.sum(current_necessary_scores - current_random_necessary_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['necessary'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_necessary_scores, current_random_necessary_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['necessary'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

    current_sufficient_scores = sufficient_scores[i]
    current_random_sufficient_scores = random_sufficient_scores[i]
    if np.sum(current_sufficient_scores - current_random_sufficient_scores) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        all_stats_test_results_by_explan_type['sufficient'].append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(current_sufficient_scores, current_random_sufficient_scores) #, alternative='greater')
        # print('RESULTS', res)
        all_stats_test_results_by_explan_type['sufficient'].append("{:.2e}".format(res[1]))
        # results.append(res[1])

print(all_stats_test_results_by_explan_type)
df_nec_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['necessary']], columns=columns_names)
df_nec_to_random = df_nec_to_random.set_index([index_names])
df_suf_to_random = pd.DataFrame([all_stats_test_results_by_explan_type['sufficient']], columns=columns_names)
df_suf_to_random = df_suf_to_random.set_index([index_names])

df1 = df_final.iloc[:2]
df2 = df_random_final.iloc[2:5]
df3 = df_final.iloc[2:5]
df4 = df_nec_to_random.iloc[:]
df5 = df_random_final.iloc[5:]
df6 = df_final.iloc[6:]
df7 = df_suf_to_random.iloc[:]
df_final_final = pd.concat([df1, df2, df3, df4, df5, df6, df7])
print(df_final_final)
df_final_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}_{kge_model}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')




## be careful because model type is name if dataframe manually
model_type = 'RO'
load_path = f'{main_path}/{dataset}_{kge_model}/global_results_{model_type}'
df_final_RO = create_df_from_global_classifier_results(load_path=load_path, dataset=dataset, model_type='RO')
# print('\n', df_final_RO)
# df_final_RO.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')

model_type = 'RAN'
load_path = f'{main_path}/{dataset}_{kge_model}/global_results_{model_type}'
df_final_RAN = create_df_from_global_classifier_results(load_path=load_path, dataset=dataset, model_type='RAN')
# print('\n', df_final_RAN)
# df_final_RAN.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')


model_type = 'RO'
load_path = f'{main_path}/{dataset}_{kge_model}/all_results_{model_type}'
df_RO = create_df_from_classifier_results(load_path, dataset, 'RO')
print('\n', df_RO)
model_type = 'RAN'
load_path = f'{main_path}/{dataset}_{kge_model}/all_results_{model_type}'
df_RAN = create_df_from_classifier_results(load_path, dataset, 'RAN')
print('\n', df_RAN)

print('HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
results = []
for column in df_RO.columns:
    print(column)
    print(df_RO[column].values)
    print(df_RAN[column].values)
    if np.sum(df_RO[column].values - df_RAN[column].values) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        results.append('x[i]-y[i]=0')
        print('RESULTS', 'x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(df_RO[column].values, df_RAN[column].values) #, alternative='greater')
        print('RESULTS', res)
        results.append("{:.2e}".format(res[1]))
        # results.append(res[1])
print(results)

df_final_RO_RAN = pd.concat([df_final_RO, df_final_RAN])

df_final_RO_RAN.loc[len(df_final_RO_RAN)] = results
# df_final_RO_RAN = df_final_RO_RAN.append(results, ignore_index=True)

indexes_names = ['entities model', 'all neighb model', 'p-value']
df_final_RO_RAN = df_final_RO_RAN.set_index([indexes_names])
print(df_final_RO_RAN)
df_final_RO_RAN.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}_{kge_model}/processed_global_results_RO_RAN.csv', index=True, sep='\t')




# original_accuracy_scores = np.array(df_effectiveness_stats_tests['original_accuracy_score_'])
# necessary_accuracy_scores = np.array(df_effectiveness_stats_tests['necessary_accuracy_score_'])
# sufficient_accuracy_scores = np.array(df_effectiveness_stats_tests['sufficient_accuracy_score_'])

# res = wilcoxon(original_accuracy_scores, necessary_accuracy_scores, alternative='greater')

# if np.sum(original_accuracy_scores - sufficient_accuracy_scores) == 0:
#     ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
#     res = '∀ elem i, x[i]-y[i] = 0'
#     print('\n', res)
# else:
#     res = wilcoxon(original_accuracy_scores, sufficient_accuracy_scores, alternative='greater')
#     print('\n', res)




# max_len_explanations = 5
# explanation_limit = 'threshold'
# df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

# max_len_explanations = 1
# explanation_limit = 'class_change'
# df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

# max_len_explanations = 5
# explanation_limit = 'class_change'
# df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')



