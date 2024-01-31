


import csv
import json
import os

from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph, rdflib_to_networkx_graph


# dataset = 'AIFB'
dataset = 'AM_FROM_DGL'
kge_model = 'RDF2Vec'

# explanation_type = ('necessary', 'len1', 'threshold')
# instance_to_explore = ['id4instance', {'location': 'AIFB_model_7_RAN'}]

# explanation_type = ('necessary', 'len5', 'class_change')
# instance_to_explore = ['id2056instance', {'location': 'AIFB_model_4_RAN'}]

# explanation_type = ('sufficient', 'len5', 'threshold')
# instance_to_explore = ['id2133instance', {'location': 'AIFB_model_2_RAN'}]

explanation_type = ('sufficient', 'len5', 'class_change')
instance_to_explore = ['proxy-28901', {'location': 'AM_FROM_DGL_model_0_RAN'}]


data_path = f'node_classifier/data/{dataset}'
results_path = f'/home/fpaulino/SEEK/seek/node_classifier/cv_model/{dataset}_{kge_model}'
sub_path = f"{instance_to_explore[1]['location']}/explanations/individual_explanations"
sub_sub_path = instance_to_explore[0]
file_name = '_'.join(explanation_type) + '.csv'

load_path = os.path.join(results_path, sub_path, sub_sub_path, file_name)
# print(load_path)

explanation_rows = []
with open(load_path, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    original = next(reader)
    for row in reader:
        row.remove('') ## to remove empty string at the end of the list
        explanation_rows.append(row)

all_neighb_predicted_class = explanation_rows[-1][3]
if explanation_type[0] == 'necessary':
    best_explanation = explanation_rows[-1][5:]
if explanation_type[0] == 'sufficient':
    best_explanation = explanation_rows[0][5:]
print('\nall_neighb_predicted_class: ', all_neighb_predicted_class)
# print(best_explanation)
# raise

with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
    ds_metadata = json.load(f)

test_data = pd.read_csv(os.path.join(data_path, "testSet.tsv"), sep="\t")
train_data = pd.read_csv(os.path.join(data_path, "trainingSet.tsv"), sep="\t")

train_entities = [entity for entity in train_data[ds_metadata['entities_name']]]
train_labels = list(train_data[ds_metadata['labels_name']])

test_entities = [entity for entity in test_data[ds_metadata['entities_name']]]
test_labels = list(test_data[ds_metadata['labels_name']])

entities = train_entities + test_entities
labels = train_labels + test_labels

location = os.path.join(data_path, ds_metadata['rdf_file'])
skip_predicates = set(ds_metadata['skip_predicates'])

target_predicate = ds_metadata['target_predicate']


##### load rdf graph and convert to networkx
grph = rdflib.Graph().parse(location)
# Grph = rdflib_to_networkx_multidigraph(grph)
Grph = rdflib_to_networkx_graph(grph)
# print(Grph.nodes())

# for train_entity, train_label in zip(train_entities, train_labels):
for entity, label in zip(entities, labels):
    # Grph.add_edge(
    #     # rdflib.term.URIRef(train_entity),
    #     # rdflib.term.URIRef(train_label),
    #     rdflib.term.URIRef(entity),
    #     rdflib.term.URIRef(label),
    #     key=rdflib.term.URIRef(target_predicate)        
    # )
    Grph.add_edge(
        # rdflib.term.URIRef(train_entity),
        # rdflib.term.URIRef(train_label),
        rdflib.term.URIRef(entity),
        rdflib.term.URIRef(label),       
    )

tail_list = []
print('\nbest explanation: ', best_explanation)
for idx, item in enumerate(best_explanation):
    if idx % 2 == 0:
        tail_list.append(item)
print('\ntail_list: ', tail_list)
# raise

rdf_tail_list = [rdflib.term.URIRef(tail) for tail in tail_list]
rdf_label_list = [rdflib.term.URIRef(label) for label in sorted(list(set(labels)))]


##### using shortest paths
all_shortest_paths_list = []
# print(tail_list)
# raise
for tail in rdf_tail_list:
# for tail in tail_list[1:]:
    # print('\ntail', tail)
    for label in rdf_label_list:
        # print('\nlabel', label)
        # print([p for p in nx.all_shortest_paths(Grph, source=rdflib.term.URIRef(tail), target=label)])
        shortest_paths_list = [p for p in nx.all_shortest_paths(Grph, source=tail, target=label)]
        for path in shortest_paths_list:
            all_shortest_paths_list.append([str('/'.join(item.split('/')[-2:])) for item in path])
# print('\nall_shortest_paths_list')
# [print('\n', path) for path in all_shortest_paths_list]

# # final_trimmed_list = []
# #     for path in all_shortest_paths_list:
# #         if len(final_trimmed_list) <=5:
# #             final_trimmed_list.append(all_shortest_paths_list)
# #         else:
# #             for idx, path_in_trimmed in enumerate(final_trimmed_list):
# #                 if len(path) < len(path_in_trimmed):

from collections import defaultdict

all_shortest_paths_dict = defaultdict(list)
for path in all_shortest_paths_list:
    # print('path', path, '\n')
    all_shortest_paths_dict[len(path)].append(path)
all_shortest_paths_dict = OrderedDict(sorted(all_shortest_paths_dict.items()))
# print('\nall_shortest_paths_dict:')
for key, value in all_shortest_paths_dict.items():
    print('\n\n\n\n')
    print(f'label is {key} hops away')
    current_val = None
    for val in value:
        if val[0] == current_val:
            current_val = val[0]
            print('\n', val)
        else:
            current_val = val[0]
            print('\n\n\t\tNEIGHBOUR:', current_val)
            print('\n', val)
    print('\n\n')

# limit_paths = 5

# print('(len(value_list), key) for key, value_list in all_shortest_paths_dict.items()')
# print([(len(value_list), key) for key, value_list in all_shortest_paths_dict.items()], '\n')

# trimmed_shortest_paths_dict = defaultdict(list)
# for key in sorted(all_shortest_paths_dict.keys()):
#     current_trimmed_dict_len = sum([len(value_list) for key, value_list in trimmed_shortest_paths_dict.items()])
#     print(current_trimmed_dict_len)
#     if current_trimmed_dict_len < 5:
#         trimmed_shortest_paths_dict[key] = all_shortest_paths_dict[key]
#     else:
#         break
# print('trimmed_shortest_paths_dict')
# print(trimmed_shortest_paths_dict, '\n')
    



# # Plot Networkx instance of RDF Graph
# pos = nx.spring_layout(Grph, scale=2)
# edge_labels = nx.get_edge_attributes(Grph, 'r')
# nx.draw_networkx_edge_labels(Grph, pos, edge_labels=edge_labels)
# nx.draw(Grph, with_labels=True)

# plt.savefig("obtain_context.png")

# #if not in interactive mode for 
# plt.show()


# identic_count = 0
# rule_in_bulk_count = 0
# total_rules = 0
# for fact_to_explain in fact_to_explain_2_best_rule_no_bulk.keys():
#     best_rule_no_bulk = fact_to_explain_2_best_rule_no_bulk[fact_to_explain]