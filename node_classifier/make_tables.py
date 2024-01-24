import json

import numpy as np
import pandas as pd


############################################################################### arguments

dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
# dataset = 'MDGENRE'


############################################################################### functions

def create_df(dataset, max_len_explanations):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/global_effectiveness_results_len{max_len_explanations}_RAN'
    
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

############################################################################### script
    
max_len_explanations = 1
df_final = create_df(dataset=dataset, max_len_explanations=max_len_explanations)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_RAN.csv', sep='\t')

max_len_explanations = 5
df_final = create_df(dataset=dataset, max_len_explanations=max_len_explanations)
print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_RAN.csv', sep='\t')
