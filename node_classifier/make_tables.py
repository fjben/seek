import json

import numpy as np
import pandas as pd


############################################################################### arguments

# dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
dataset = 'MDGENRE'


############################################################################### functions

def create_df_from_global_effectiveness_results(dataset, max_len_explanations, explanation_limit):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')

    columns_to_keep = [
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

    indexes = [0, 5, 10]
    rows = []
    for i in range(5):
        df.iloc[:, indexes]
        rows.append(df.iloc[:, indexes].to_numpy()[0].tolist())
        indexes = [index + 1 for index in indexes]

    rows = np.array(rows)
    columns_names = ['f1_score', 'precision', 'recall']
    df_final = pd.DataFrame(rows, columns=columns_names)

    indexes_names = ['original', 'necessary', 'delta_necessary', 'sufficient', 'delta_sufficient']
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


def create_df_from_global_classifier_results(dataset, model_type):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/global_results_{model_type}'
    
    # with open(load_path + '.json', 'r') as f:
    #     results = json.load(f)
    # df = pd.DataFrame([results])
    df = pd.read_csv(load_path + '.csv', sep='\t')

    columns_to_keep = [
        'mean_f1_scr_wei',
        'std_f1_scr_wei',
        'mean_prec_scr_wei',
        'std_prec_scr_wei',
        'mean_reca_scr_wei',
        'std_reca_scr_wei',
        'mean_mean_preds',
        'mean_std_preds',
        'mean_classifier_fit_time',
        'std_classifier_fit_time',
    ]
    other_columns = [
        'mean_acc_scr',
        'std_acc_scr',
        'mean_f1_scr_macro',
        'std_f1_scr_macro',
        'mean_prec_scr_macro',
        'std_prec_scr_macro',
        'mean_reca_scr_macro',
        'std_reca_scr_macro',
        'std_mean_preds',
        'std_std_preds'
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
df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

max_len_explanations = 5
explanation_limit = 'threshold'
df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

max_len_explanations = 1
explanation_limit = 'class_change'
df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

max_len_explanations = 5
explanation_limit = 'class_change'
df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

model_type = 'RO'
df_final = create_df_from_global_classifier_results(dataset=dataset, model_type=model_type)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')

model_type = 'RAN'
df_final = create_df_from_global_classifier_results(dataset=dataset, model_type=model_type)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')