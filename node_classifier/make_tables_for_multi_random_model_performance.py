


import json
import os

import pandas as pd

dataset = ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']

path = '/home/fpaulino/SEEK/seek/node_classifier/1_multi_random_single_model/'

# metric = 'f1'
metric = 'accuracy'

if metric == 'f1':
    metric_name = 'f1'
    metric_name_alt = 'F1'
    extra = 'weighted'
elif metric == 'accuracy':
    metric_name = 'accuracy'
    metric_name_alt = 'Acc'

    extra = ''

df_f1 = pd.DataFrame()
df_acc = pd.DataFrame()
df_single_columns = pd.DataFrame()
single_column_nec = []
single_column_suf = []
for ds in dataset:
    global_results_f1_RAN = []
    global_results_f1_RO = []
    global_results_acc_RAN = []
    global_results_acc_RO = []
    for idx in range(10):
        current_model_path = os.path.join(path, f'{ds}_RDF2Vec', f'{ds}_RDF2Vec_{idx}')
        with open(os.path.join(current_model_path, 'global_results_RAN.json'), 'r') as f:
            global_results_RAN = json.load(f)
        with open(os.path.join(current_model_path, 'global_results_RO.json'), 'r') as f:
            global_results_RO = json.load(f)
        global_results_f1_RAN.append(global_results_RAN[f'mean_f1_scr_wei'])
        global_results_acc_RAN.append(global_results_RAN[f'mean_acc_scr'])
        global_results_f1_RO.append(global_results_RO[f'mean_f1_scr_wei'])
        global_results_acc_RO.append(global_results_RO[f'mean_acc_scr'])
    df_f1[f'{ds}_F1_RAN'] = global_results_f1_RAN
    df_f1[f'{ds}_F1_RO'] = global_results_f1_RO
    df_acc[f'{ds}_Acc_RAN'] = global_results_acc_RAN
    df_acc[f'{ds}_Acc_RO'] = global_results_acc_RO

    # single_column_nec.extend(global_random_effectiveness_metric_nec)
    # single_column_nec.extend(global_effectiveness_metric_nec)
    # single_column_suf.extend(global_random_effectiveness_metric_suf)
    # single_column_suf.extend(global_effectiveness_metric_suf)

df_f1.to_csv(f'node_classifier/results_processed_multi_random/f1_RAN_RO.csv', index=False)
df_acc.to_csv(f'node_classifier/results_processed_multi_random/acc_RAN_RO.csv', index=False)

# df_single_columns[f'Necessary_{metric_name_alt}'] = single_column_nec
# df_single_columns[f'Sufficient_{metric_name_alt}'] = single_column_suf
# df_single_columns.to_csv(f'node_classifier/results_processed_multi_random/single_columns_{metric_name}.csv', index=False)