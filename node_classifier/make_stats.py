import json

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


############################################################################### arguments

dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
# dataset = 'MDGENRE'


############################################################################### functions

def create_df_from_all_effectiveness_results(dataset, max_len_explanations, explanation_limit):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/all_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    
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


def create_df_from_global_classifier_results(dataset, model_type):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/global_results_{model_type}'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')

    columns_to_keep = [
        'acc_scr',
        'f1_scr_wei',
        'prec_scr_wei',
        'reca_scr_wei',
    ]
    other_columns = [
        'f1_scr_macro',
        'prec_scr_macro',
        'reca_scr_macro',
        'mean_preds',
        'std_preds'
        'classifier_fit_time',
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

    df_final = df ## .drop(['Unnamed: 0'],axis=1)

    return df_final


############################################################################### script
    
max_len_explanations = 1
explanation_limit = 'threshold'
df_final = create_df_from_all_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

original_accuracy_scores = np.array(df_final['original_accuracy_score_'])
necessary_accuracy_scores = np.array(df_final['necessary_accuracy_score_'])
sufficient_accuracy_scores = np.array(df_final['sufficient_accuracy_score_'])

d = np.around(original_accuracy_scores - necessary_accuracy_scores, decimals=3)
# print(d)
## 'greater': the distribution underlying ``d`` is stochastically greater than a distribution symmetric about zero.
## I think this means that it will check if the original is higher than the other which is what we want
# res = wilcoxon(d, alternative='greater')
# print('original_accuracy_scores')
# print(original_accuracy_scores)
# print('necessary_accuracy_scores')
# print(necessary_accuracy_scores)

res = wilcoxon(original_accuracy_scores, necessary_accuracy_scores, alternative='greater')
print('\n', res)

# res = wilcoxon(original_accuracy_scores, sufficient_accuracy_scores, alternative='greater')

if np.sum(original_accuracy_scores - sufficient_accuracy_scores) == 0:
    ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
    res = '∀ elem i, x[i]-y[i] = 0'
    print('\n', res)
else:
    res = wilcoxon(original_accuracy_scores, sufficient_accuracy_scores, alternative='greater')
    print('\n', res)

# try:
#     res = wilcoxon(original_accuracy_scores, sufficient_accuracy_scores, alternative='greater')
# except ValueError:
#     print('here')
#     if np.sum(original_accuracy_scores - necessary_accuracy_scores) == 0:
#         print('here')
#         res = 'x-y = 0'
#         print('\n', res)




max_len_explanations = 5
explanation_limit = 'threshold'
df_final = create_df_from_all_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

max_len_explanations = 1
explanation_limit = 'class_change'
df_final = create_df_from_all_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

max_len_explanations = 5
explanation_limit = 'class_change'
df_final = create_df_from_all_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

# model_type = 'RO'
# df_final = create_df_from_global_classifier_results(dataset=dataset, model_type=model_type)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')

# model_type = 'RAN'
# df_final = create_df_from_global_classifier_results(dataset=dataset, model_type=model_type)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')