import csv
import os

from itertools import product

import pandas as pd

dataset = 'AIFB'
kge_model = 'RDF2Vec'

results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}_{kge_model}'

individual_explanations_paths = []
for i in range(10):
    sub_path = f'AIFB_model_{i}_RAN/explanations/individual_explanations'
    individual_explanations_folders = os.listdir(os.path.join(results_path, sub_path))
    for individual_folder in individual_explanations_folders:
        individual_explanations_paths.append(os.path.join(results_path, sub_path, individual_folder))
# print(individual_explanations_paths)

explanation_type = ['necessary', 'sufficient']
explanation_len = ['len1', 'len5']
acceptance_condition = ['threshold', 'class_change']

combinations = list(product(*[explanation_type, explanation_len, acceptance_condition]))

for p in combinations[:1]:
    file_name = '_'.join(p) + '.csv'
    explanations_dict = dict()
    for individual_expl_path in individual_explanations_paths:
        load_path = os.path.join(individual_expl_path, file_name)
        # print(load_path)
        explanation_rows = []
        with open(load_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            original = next(reader)
            for row in reader:
                explanation_rows.append(row)
        # print('\n', header)
        # print('\n', original)
        # print('\n', explanation_rows)
        # print(explanation_rows[-1][1])
        condition1 = eval(explanation_rows[-1][1]) == True
        # print(condition1)
        condition2 = explanation_rows[-1][2] == explanation_rows[-1][3]
        # print(condition2)
        if condition1 and condition2:
            dif = float(original[0]) - float(explanation_rows[-1][0])
            explanations_dict[individual_expl_path.split('/')[-1]] = [float(dif), original, explanation_rows]

# print(sorted(explanations_dict.values()))

values, keys = zip(*sorted(zip(list(explanations_dict.values()), list(explanations_dict.keys()))))

print(keys[:5])