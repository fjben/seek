


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

df_nec = pd.DataFrame()
df_suf = pd.DataFrame()
df_single_columns = pd.DataFrame()
single_column_nec = []
single_column_suf = []
for ds in dataset:
    global_effectiveness_metric_nec = []
    global_random_effectiveness_metric_nec = []
    global_effectiveness_metric_suf = []
    global_random_effectiveness_metric_suf = []
    for idx in range(10):
        current_model_path = os.path.join(path, f'{ds}_RDF2Vec', f'{ds}_RDF2Vec_{idx}')
        with open(os.path.join(current_model_path, 'global_effectiveness_results_len5_threshold_RAN.json'), 'r') as f:
            global_effectiveness = json.load(f)
        with open(os.path.join(current_model_path, 'global_random_effectiveness_results_len5_threshold_RAN.json'), 'r') as f:
            global_random_effectiveness = json.load(f)
        global_effectiveness_metric_nec.append(global_effectiveness[f'mean_delta_necessary_{metric_name}_score_{extra}'])
        global_random_effectiveness_metric_nec.append(global_random_effectiveness[f'mean_delta_necessary_{metric_name}_score_{extra}'])
        global_effectiveness_metric_suf.append(global_effectiveness[f'mean_delta_sufficient_{metric_name}_score_{extra}'])
        global_random_effectiveness_metric_suf.append(global_random_effectiveness[f'mean_delta_sufficient_{metric_name}_score_{extra}'])
    df_nec[f'{ds}_Nec_Rand_{metric_name_alt}'] = global_random_effectiveness_metric_nec
    df_nec[f'{ds}_Nec_Expl_{metric_name_alt}'] = global_effectiveness_metric_nec
    df_suf[f'{ds}_Suf_Rand_{metric_name_alt}'] = global_random_effectiveness_metric_suf
    df_suf[f'{ds}_Suf_Expl_{metric_name_alt}'] = global_effectiveness_metric_suf

    single_column_nec.extend(global_random_effectiveness_metric_nec)
    single_column_nec.extend(global_effectiveness_metric_nec)
    single_column_suf.extend(global_random_effectiveness_metric_suf)
    single_column_suf.extend(global_effectiveness_metric_suf)

df_nec.to_csv(f'node_classifier/results_processed_multi_random/{metric_name}_nec.csv', index=False)
df_suf.to_csv(f'node_classifier/results_processed_multi_random/{metric_name}_suf.csv', index=False)

df_single_columns[f'Necessary_{metric_name_alt}'] = single_column_nec
df_single_columns[f'Sufficient_{metric_name_alt}'] = single_column_suf
df_single_columns.to_csv(f'node_classifier/results_processed_multi_random/single_columns_{metric_name}.csv', index=False)