import csv
import os

from collections import Counter, defaultdict
from itertools import product

import pandas as pd

dataset = 'AIFB'
kge_model = 'RDF2Vec'

# results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}_{kge_model}'
# results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model_backup_latest/{dataset}_{kge_model}'
results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model_rf_local_final/{dataset}_{kge_model}'

individual_explanations_paths = []
for i in range(10):
    sub_path = f'{dataset}_model_{i}_RAN/explanations/individual_explanations'
    individual_explanations_folders = os.listdir(os.path.join(results_path, sub_path))
    for individual_folder in individual_explanations_folders:
        individual_explanations_paths.append(os.path.join(results_path, sub_path, individual_folder))
# print(individual_explanations_paths)

explanation_type = ['necessary', 'sufficient']
explanation_len = ['len1', 'len5']
acceptance_condition = ['threshold', 'class_change']

combinations = list(product(*[explanation_type, explanation_len, acceptance_condition]))
# print(combinations)

# print(len(individual_explanations_paths))
# raise

one_property_per_entity = True
# one_property_per_entity = False

all_explanations_properties = defaultdict(list)
# combinations = [('necessary', 'len5', 'threshold')]
# combinations = [('sufficient', 'len1', 'class_change')]
# for p in combinations[:1]:
# for p in combinations[-1:]:
for p in combinations:
    print(p)
    file_name = '_'.join(p) + '.csv'
    # explanations_dict = dict()
    for individual_expl_path in individual_explanations_paths:
        load_path = os.path.join(individual_expl_path, file_name)
        # print(load_path)
        explanation_rows = []
        exists_in_explanation_rows = []
        with open(load_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            original = next(reader)
            for row in reader:
                # print(row)
                # row = row.remove('') ## this is not working do not understand why
                row = row[:-1]
                row = row[5:]
                # print('\n', row)
                explanation_rows.append(row)

                row = [row[i] for i in range(1, len(row), 2)]
                # print(row)
                # raise
                # print('\n', row)
                [exists_in_explanation_rows.append(prop) for prop in row if prop not in exists_in_explanation_rows]
                if not one_property_per_entity:
                    all_explanations_properties[p].extend(row)
            if one_property_per_entity:
                all_explanations_properties[p].extend(exists_in_explanation_rows)
            # print('\n', explanation_rows)
# print(list(all_explanations_properties.values())[0])
# all_explanations_properties = [all_explanations_properties[key] = Counter(all_explanations_properties[key]) for key in all_explanations_properties.keys()]
# all_explanations_properties['explained_entities_size'] = len(individual_explanations_paths)

all_explanations_properties_counter = dict()
all_explanations_properties_stats = defaultdict(dict)
all_explanations_properties_counter['explained_entities_size'] = len(individual_explanations_paths)
for key_main in all_explanations_properties.keys():
    all_explanations_properties_counter[key_main] = Counter(all_explanations_properties[key_main])
    for key_property, value_property in all_explanations_properties_counter[key_main].items():
        all_explanations_properties_stats[key_main].update({key_property: round(value_property / all_explanations_properties_counter['explained_entities_size'], 3)})

print(all_explanations_properties_counter)
print(all_explanations_properties_stats)

df = pd.DataFrame(all_explanations_properties_stats)
print(df)
df.to_csv(os.path.join('node_classifier/more_results', f'property_stats_local_explanations_{dataset}_{kge_model}_one_per_entity.csv'), sep='\t')



        # print('\n', header)
        # print('\n', original)
        # print('\n', explanation_rows)
        # print(explanation_rows[-1][1])
        # print(explanation_rows[-1])
                
        # if p[0] == 'necessary':
        # elif p[0] == 'sufficient':

        # # if condition1 and condition2 and condition3:
        # ## JUST FOR SUFFICIENT TO TRY AND FIND ONE WITH MORE THAN ONE EXPLANATION
        # if condition1 and condition2 and condition3 and condition4:
        #     dif = float(original[0]) - float(explanation_rows[-1][0])
        #     explanations_dict[individual_expl_path.split('/')[-1]] = [float(dif), original, explanation_rows, load_path.split('/')[-5]]

    # try:
    #     values, keys = zip(*sorted(zip(list(explanations_dict.values()), list(explanations_dict.keys()))))
    #     # print(values)

    #     ## the lists are sorted ascending so for necessary we want the largest values (the last x) and for sufficient we
    #     ## want the smallers (the first x)
    #     number_of_top_explanations = 1
    #     if p[0] == 'necessary':
    #         sl = slice(len(keys)-number_of_top_explanations, len(keys))
    #         for key, value in reversed(list(zip(keys, values))[sl]):
    #             print(f"'{key}', {{'location': '{value[3]}'}}")
    #     elif p[0] == 'sufficient':
    #         sl = slice(0, number_of_top_explanations)
    #         for key, value in list(zip(keys, values))[sl]:
    #             print(f"'{key}', {{'location': '{value[3]}'}}")
    # except:
    #     print('no explanations found that respect the search conditions')