import json

from collections import OrderedDict

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


############################################################################### arguments

dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
# dataset = 'MDGENRE'


############################################################################### functions

def create_df_from_global_effectiveness_results(dataset, max_len_explanations, explanation_limit):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN'
    
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
            print('here')
            print(index)
            print('here')
            print(df.iloc[0, index])
            print(df_std_values.iloc[:, index])
            print('here')
            print(f'{df.iloc[0, index]} ({df_std_values.iloc[0, index]})')
            print('here')
            row.append(f'{round(df.iloc[0, index], 3)} ({round(df_std_values.iloc[0, index], 3)})')
        # rows.append(df.iloc[:, indexes].to_numpy()[0].tolist())
        rows.append(row)
        indexes = [index + 1 for index in indexes]

    rows = np.array(rows)
    print(rows)
    columns_names = ['accuracy', 'f1_score', 'precision', 'recall']
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


def create_df_from_classifier_results(dataset, model_type):
    load_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}/all_results_{model_type}'
    
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

            res = wilcoxon(original_scores, score) # , alternative='greater')
            # results.append(res[1])
            results.append("{:.2e}".format(res[1]))
            ## print to check why p-values all equal
            # print('\n')
            # print('here')
            # print(original_scores)
            # print(score)
            # print(res)
            # print('\n')

    return results


############################################################################### script
    
accuracy_labels = ['original_accuracy_score_', 'necessary_accuracy_score_', 'sufficient_accuracy_score_']
f1_weighted_labels = ['original_f1_score_weighted', 'necessary_f1_score_weighted', 'sufficient_f1_score_weighted']
precision_weighted_labels = ['original_precision_score_weighted', 'necessary_precision_score_weighted', 'sufficient_precision_score_weighted',]
recall_weighted_labels = ['original_recall_score_weighted', 'necessary_recall_score_weighted', 'sufficient_recall_score_weighted',]


max_len_explanations = 1
explanation_limit = 'threshold'
df_final = create_df_from_global_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
# print('\n', df_final)
# df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')


df_effectiveness_stats_tests = create_df_from_all_effectiveness_results(dataset=dataset, max_len_explanations=max_len_explanations, explanation_limit=explanation_limit)
all_stats_tests_results = OrderedDict()
for met_labels in [accuracy_labels, f1_weighted_labels, precision_weighted_labels, recall_weighted_labels]:
    original_scores_label, necessary_scores_label, sufficient_scores_label = met_labels
    description = f'{original_scores_label.split("_")[1]} p-values'
    original_scores = np.array(df_effectiveness_stats_tests[original_scores_label])
    necessary_scores = np.array(df_effectiveness_stats_tests[necessary_scores_label])
    sufficient_scores = np.array(df_effectiveness_stats_tests[sufficient_scores_label])

    # print('\n')
    # print(original_scores_label)
    # print(original_scores)
    # print(necessary_scores)
    # # print(sufficient_scores)
    # print('\n')
    [necessary_p_value, sufficient_p_value] = wilcoxon_test(original_scores, necessary_scores, sufficient_scores)
    all_stats_tests_results[description] = {'necessary_p_value': necessary_p_value,
                                            'sufficient_p_value': sufficient_p_value}


# print(all_stats_tests_results)

for score_metric in all_stats_tests_results.keys():
    all_stats_tests_results[score_metric]['necessary_p_value']
    new_col = ['',
               '',
               all_stats_tests_results[score_metric]['necessary_p_value'],
               '',
               '',
               all_stats_tests_results[score_metric]['sufficient_p_value'],
               ''
               ]
    df_final[f'{score_metric}'] = new_col

print('\n', df_final)
df_final.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_effectiveness_results_len{max_len_explanations}_{explanation_limit}_RAN.csv', sep='\t')

    


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




## be careful because model type is name if dataframe manually
model_type = 'RO'
df_final_RO = create_df_from_global_classifier_results(dataset=dataset, model_type='RO')
print('\n', df_final_RO)
df_final_RO.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')

model_type = 'RAN'
df_final_RAN = create_df_from_global_classifier_results(dataset=dataset, model_type='RAN')
print('\n', df_final_RAN)
df_final_RAN.to_csv(f'/home/fpaulino/SEEK/seek/node_classifier/results_processed/{dataset}/processed_global_results_{model_type}.csv', index=False, sep='\t')




df_RO = create_df_from_classifier_results(dataset, 'RO')
# print('\n', df_RO)

df_RAN = create_df_from_classifier_results(dataset, 'RAN')
# print('\n', df_RAN)

results = []
for column in df_RO.columns:
    # print(column)
    if np.sum(df_RO[column].values - df_RAN[column].values) == 0:
        ## ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
        results.append('x[i]-y[i]=0')
    else:
        # print(df_RO[column].values)
        res = wilcoxon(df_RO[column].values, df_RAN[column].values) #, alternative='greater')
        # print('RESULTS', res)
        # results.append("{:.2e}".format(res[1]))
        results.append(res[1])
print(results)

df_final_RO_RAN = pd.concat([df_final_RO, df_final_RAN])

df_final_RO_RAN.loc[len(df_final_RO_RAN)] = results
# df_final_RO_RAN = df_final_RO_RAN.append(results, ignore_index=True)

print(df_final_RO_RAN)




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