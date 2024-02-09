

import os

import pandas as pd


## args #######################################################################

# results_path = '/home/fpaulino/SEEK/seek/node_classifier/results_processed_rf_local_final'
# results_path = '/home/fpaulino/SEEK/seek/node_classifier/results_processed_xgb_local_final'
results_path = '/home/fpaulino/SEEK/seek/node_classifier/results_processed_mlp_local_final'

# explanation_limit = 'class_change'
explanation_limit = 'threshold'

# kge_model = 'RDF2Vec'
# kge_model = 'ComplEx'
# kge_model = 'distMult'
# kge_model = 'TransH'
# kge_model = 'TransE'

# explan_type = 'necessary'
explan_type = 'sufficient'


## script #####################################################################

datasets = ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']

metrics = ['precision',	'recall', 'f1_score', 'accuracy']

results_files = [f'processed_global_effectiveness_results_len1_{explanation_limit}_RAN.csv',
                 f'processed_global_effectiveness_results_len5_{explanation_limit}_RAN.csv']

def get_met_code(metric_name):
    if metric_name == 'precision':
        met_code = 'Pr'
    elif metric_name == 'recall':
        met_code = 'Re'
    elif metric_name == 'f1_score':
        met_code = 'F1'
    elif metric_name == 'accuracy':
        met_code = 'Ac'
    else:
        raise Exception('invalid metric name')
    return met_code

# base_txt = "Hello, {}{}!".format(*["Johnson", "Another Johnson"])
# insert_list = [met_code, ]
# base_txt = '& $\Delta${} & {} & {} & {} & {} && {} & {} & {} & {} && {} & {} & {} & {} && {} & {} & {} & {} \\'.format(
#     *insert_list
# )
# print(base_txt)
# raise

def fill_value(value):
    while len(value) < 6:
        value = value + ' '
    return value

def read_value_in_df(row_name, column_name, df, value_part='all'):
    if isinstance(row_name, int):
        if value_part == 'all':
            return df.loc[row_name, column_name]
        elif value_part == 'mean':
            return df[df['Unnamed: 0'] == row_name][column_name].values[0].split(' ')[0]
    else:
        if value_part == 'all':
            return df[df['Unnamed: 0'] == row_name][column_name].values[0]
        elif value_part == 'mean':
            return df[df['Unnamed: 0'] == row_name][column_name].values[0].split(' ')[0]

def metrics_for_kge_model(kge_model, metrics, datasets, results_files, results_path, explan_type):
    all_metrics_txt = ''
    for met in metrics:
        met_code = get_met_code(met)
        results_lst = []
        for dataset in datasets:
            for results_file in results_files:
                df = pd.read_csv(os.path.join(results_path, f'{dataset}_{kge_model}', results_file), sep='\t')
                # print(df)
                value_delta_explainer = read_value_in_df(f'delta explainer {explan_type}', met, df, 'mean')
                value_delta_explainer_original = value_delta_explainer
                value_delta_explainer = fill_value(value_delta_explainer)
                value_delta_random = read_value_in_df(f'delta random {explan_type}', met, df, 'mean')
                value_delta_random_original = value_delta_random
                value_delta_random = fill_value(value_delta_random)
                value_explainer_original = read_value_in_df(f'explainer {explan_type}', met, df, 'mean')
                value_random_original = read_value_in_df(f'random {explan_type}', met, df, 'mean')

                ## bold and italicize for latex
                if explan_type == 'necessary':
                    row_idx = 7
                elif explan_type == 'sufficient':
                    row_idx = 15
                p_value_explainer_original = read_value_in_df(row_idx, met, df, 'all')
                # print(p_value_explainer_original)
                if p_value_explainer_original == 'x[i]-y[i]=0' and explan_type == 'sufficient':
                    value_delta_explainer = f'\\textbf{{{value_delta_explainer}}}'
                    # print(value_delta_explainer)
                else:
                    p_value_explainer_original = eval(p_value_explainer_original)
                    if p_value_explainer_original <= 0.05 and explan_type == 'necessary' and eval(value_delta_explainer_original) < 0:
                        value_delta_explainer = f'\\textbf{{{value_delta_explainer}}}'
                        # print(value_delta_explainer)
                    if p_value_explainer_original > 0.05 and explan_type == 'sufficient':
                        value_delta_explainer = f'\\textbf{{{value_delta_explainer}}}'

                if explan_type == 'necessary':
                    row_idx = 8
                elif explan_type == 'sufficient':
                    row_idx = 16
                p_value_explainer_random = read_value_in_df(row_idx, met, df, 'all')
                # print(p_value_explainer_random)
                if p_value_explainer_random == 'x[i]-y[i]=0' and explan_type == 'sufficient':
                    pass
                else:
                    p_value_explainer_random = eval(p_value_explainer_random)
                    if p_value_explainer_random <= 0.05 and explan_type == 'necessary' and eval(value_explainer_original) - eval(value_random_original) < 0:
                        value_delta_explainer = f'\\textit{{{value_delta_explainer}}}'
                        # print(value_delta_explainer)
                    if p_value_explainer_random <= 0.05 and explan_type == 'sufficient' and eval(value_random_original) - eval(value_explainer_original) < 0:
                        value_delta_explainer = f'\\textit{{{value_delta_explainer}}}'

                results_lst.extend([value_delta_explainer, value_delta_random])
        # print(results_lst)
        insert_list = [met_code]
        insert_list.extend(results_lst)
        metric_txt = '& $\Delta${} & {} & {} & {} & {} && {} & {} & {} & {} && {} & {} & {} & {} && {} & {} & {} & {} \\\\\n'.format(
        *insert_list
        )
        # print(metric_txt)
        all_metrics_txt += metric_txt
    if kge_model == 'TransE':
        all_metrics_txt = all_metrics_txt[:-1] + ' \\bottomrule\n' ## trim the last '\n' and add the '\\bottomrule'
    else:
        all_metrics_txt = all_metrics_txt[:-1] + ' \\midrule\n' ## trim the last '\n' and add the '\\midrule'
    # print(all_metrics_txt)

    return all_metrics_txt

kge_models = ['RDF2Vec', 'ComplEx', 'distMult', 'TransH', 'TransE']

complete_table_txt = ''
for kge_model in kge_models:
    complete_table_txt = complete_table_txt + f'\\multirow{{3}}{{*}}{{{kge_model}}}\n'
    # print(complete_table_txt)
    # raise
    all_metrics_txt = metrics_for_kge_model(kge_model, metrics, datasets, results_files, results_path, explan_type)
    complete_table_txt = complete_table_txt + all_metrics_txt
print(complete_table_txt)