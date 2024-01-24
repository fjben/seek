import json
import random
import os
import sys
import argparse
import numpy as np
import time
import copy
import gc
import pandas as pd

from multiprocessing import Pool

import joblib
from tqdm import tqdm
import rdflib
from rdflib.namespace import RDF, OWL, RDFS

import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_array
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat
from sklearn import metrics
from operator import itemgetter

import time
import pickle

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

# def _identity(x): return x

# def _rdflib_to_networkx_graph(
#         graph,
#         nxgraph,
#         calc_weights,
#         edge_attrs,
#         transform_s=_identity, transform_o=_identity):
#     """Helper method for multidigraph, digraph and graph.
#     Modifies nxgraph in-place!
#     Arguments:
#         graph: an rdflib.Graph.
#         nxgraph: a networkx.Graph/DiGraph/MultiDigraph.
#         calc_weights: If True adds a 'weight' attribute to each edge according
#             to the count of s,p,o triples between s and o, which is meaningful
#             for Graph/DiGraph.
#         edge_attrs: Callable to construct edge data from s, p, o.
#            'triples' attribute is handled specially to be merged.
#            'weight' should not be generated if calc_weights==True.
#            (see invokers below!)
#         transform_s: Callable to transform node generated from s.
#         transform_o: Callable to transform node generated from o.
#     """
#     assert callable(edge_attrs)
#     assert callable(transform_s)
#     assert callable(transform_o)
#     import networkx as nx
#     for s, p, o in graph:
#         if p == RDFS.subClassOf or p==rdflib.term.URIRef('http://hasAnnotation'):
#             ts, to = transform_s(s), transform_o(o)  # apply possible transformations
#             data = nxgraph.get_edge_data(ts, to)
#             if data is None or isinstance(nxgraph, nx.MultiDiGraph):
#                 # no edge yet, set defaults
#                 data = edge_attrs(s, p, o)
#                 if calc_weights:
#                     data['weight'] = 1
#                 nxgraph.add_edge(ts, to, **data)
#             else:
#                 # already have an edge, just update attributes
#                 if calc_weights:
#                     data['weight'] += 1
#                 if 'triples' in data:
#                     d = edge_attrs(s, p, o)
#                     data['triples'].extend(d['triples'])

# def process_GO_annotations(annotations_file_path):

#     file_annot = open(annotations_file_path, 'r')
#     file_annot.readline()
#     dic_annotations = {}
#     for annot in file_annot:
#         list_annot = annot.split('\t')
#         id_prot, GO_term = list_annot[1], list_annot[4]

#         url_GO_term = "http://purl.obolibrary.org/obo/GO_" + GO_term.split(':')[1]

#         if url_GO_term == 'http://purl.obolibrary.org/obo/GO_0044212':
#             print(annot)

#         url_prot = id_prot

#         if url_prot not in dic_annotations:
#             dic_annotations[url_prot] = [url_GO_term]
#         else:
#             dic_annotations[url_prot] = dic_annotations[url_prot] + [url_GO_term]
#     file_annot.close()
#     return dic_annotations


# def process_HP_annotations(annotations_file_path):
#     file_annot = open(annotations_file_path, 'r')
#     file_annot.readline()
#     dic_annotations = {}
#     for annot in file_annot:
#         list_annot = annot[:-1].split('\t')
#         id_ent, HPO_term = list_annot[0], list_annot[1]

#         url_HPO_term = 'http://purl.obolibrary.org/obo/HP_' + HPO_term
#         url_ent = id_ent

#         if url_ent not in dic_annotations:
#             dic_annotations[url_ent] = [url_HPO_term]
#         else:
#             dic_annotations[url_ent] = dic_annotations[url_ent] + [url_HPO_term]

#     file_annot.close()
#     return dic_annotations


# def add_annotations(g, dic_annotations):
#     for ent in dic_annotations:
#         for a in dic_annotations[ent]:
#             g.add((rdflib.term.URIRef(ent), rdflib.term.URIRef('http://hasAnnotation'),rdflib.term.URIRef(a)))
#     return g


# def construct_kg(ontology_file_path, annotations_file_path, type_dataset='PPI'):
#     if type_dataset == "PPI":
#         dic_annotations = process_GO_annotations(annotations_file_path)
#     elif type_dataset == "GDA":
#         dic_annotations = process_HP_annotations(annotations_file_path)
#     g_ontology = rdflib.Graph()
#     g_ontology.parse(ontology_file_path, format="xml")
#     dic_labels_classes = {}
#     for (sub, pred, obj) in g_ontology.triples((None, RDFS.label, None)):
#         dic_labels_classes[str(sub)] = str(obj)
#     return add_annotations(g_ontology, dic_annotations), dic_labels_classes


# def process_indexes_partition(file_partition):
#     """
#     Process the partition file and returns a list of indexes.
#     :param file_partition: partition file path (each line is a index);
#     :return: list of indexes.
#     """
#     file_partitions = open(file_partition, 'r')
#     indexes_partition = []
#     for line in file_partitions:
#         indexes_partition.append(int(line[:-1]))
#     file_partitions.close()
#     return indexes_partition


# def process_dataset(path_dataset_file):
#     """
#     """
#     list_labels = []
#     with open(path_dataset_file, 'r') as dataset:
#         for line in dataset:
#             split1 = line.split('\t')
#             ent1, ent2 = split1[0], split1[1]
#             label = int(split1[2][:-1])
#             list_labels.append([(ent1, ent2), label])
#     return list_labels


# def bar_plot(df, df_without, df_only, n_ancestors, prob_class, num_classes, output):
#     baseline = 1/num_classes

#     cm = 1 / 2.54
#     fig, ax = plt.subplots()
#     #plt.figure(figsize=((1 * n_ancestors + 1.5) * cm, 9 * cm))
#     fig.set_size_inches(10, 8)
#     #plt.figure(figsize=(5, n_ancestors*2+3))
#     sns.set_color_codes("pastel")

#     df_without['value'] = [x-baseline if x>baseline else -(1-x)+baseline for x in df_without[prob_class]]
#     list_colors_without=[]
#     for x in df_without['value']:
#         if x < 0:
#             z='#FF5050'
#         elif x==0:
#             z="#FFFFFF"
#         else:
#             z='#70BB83'
#         list_colors_without.append(z)
#     df_without['colors'] = list_colors_without

#     df_only['value'] = [float(x)-baseline if x > baseline else -(1-x)+baseline for x in df_only[prob_class]]
#     list_colors_only = []
#     for x in df_only['value']:
#         if x < 0:
#             z = '#FF5050'
#         elif x== 0:
#             z = "#FFFFFF"
#         else:
#             z = '#70BB83'
#         list_colors_only.append(z)
#     df_only['colors'] = list_colors_only

#     df['value'] = [float(x)-baseline if x > baseline else -(1-x)+baseline for x in df[prob_class]]
#     df['colors'] = ['#FF5050' if x < 0 else '#70BB83' for x in df['value']]

#     ax.hlines(y=df.neighbour, xmin=0, xmax=df.value, color=df['colors'], alpha=0.8, linewidth=2.5)
#     for x, y, tex in zip(df.value, df.neighbour, df.value):
#         if len(y) > 2:
#             t = ax.text(x, y, round(abs(tex) + baseline, 2), horizontalalignment='right' if x < 0 else 'left',
#                         verticalalignment='center',
#                         fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

#     ax.hlines(y=df_without.neighbour, xmin=0, xmax=df_without.value, color=df_without['colors'], alpha=0.8, linewidth=2.5)
#     for x, y, tex in zip(df_without.value, df_without.neighbour, df_without.value):
#         if len(y)>2:
#             t = ax.text(x, y, round(abs(tex) + baseline, 2), horizontalalignment='right' if x < 0 else 'left',
#                         verticalalignment='center',
#                         fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

#     ax.hlines(y=df_only.neighbour, xmin=0, xmax=df_only.value, color=df_only['colors'], alpha=0.8, linewidth=2.5)
#     for x, y, tex in zip(df_only.value, df_only.neighbour, df_only.value):
#         if len(y)>2:
#             t = ax.text(x, y, round(abs(tex) + baseline, 2), horizontalalignment='right' if x < 0 else 'left',
#                         verticalalignment='center',
#                         fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

#     ax.set_xlim(-0.7,0.7)
#     ax.set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
#     ax.set_xticklabels(("1","0.75", "0.5", "0.75", "1"))

#     fig.show()
#     fig.savefig(output, bbox_inches='tight')
#     plt.close('all')


# def process_representation_file(path_file_representation):
#     """
#     """
#     dict_representation = {}
#     with open(path_file_representation, 'r') as file_representation:
#         for line in file_representation:
#             line = line[:-1]
#             split1 = line.split('\t')
#             ent1 = split1[0].split('/')[-1]
#             ent2 = split1[1].split('/')[-1]
#             feats = split1[2:]
#             feats_floats = [float(i) for i in feats]
#             dict_representation[(ent1, ent2)] =  feats_floats
#     return dict_representation


# def read_representation_dataset_file(path_file_representation, path_dataset_file):
#     """
#     """
#     list_representation, labels, list_ents= [], [], []
#     dict_representation = process_representation_file(path_file_representation)
#     list_labels = process_dataset(path_dataset_file)
#     for (ent1, ent2), label in list_labels:
#         representation_floats = dict_representation[(ent1, ent2)]
#         list_ents.append([ent1, ent2])
#         list_representation.append(representation_floats)
#         labels.append(label)
#     return list_ents, list_representation , labels


# def run_save_model(path_file_representation, path_dataset_file, path_file_model, algorithms):

#     pairs, X_train, y_train = read_representation_dataset_file(path_file_representation, path_dataset_file)

#     for alg in algorithms:

#         ensure_dir(path_file_model + alg + '/')
        
#         if alg == "XGB":
#             ml_model = xgb.XGBClassifier()
#             ml_model.fit(np.array(X_train), np.array(y_train))
#         if alg == "RF":
#             ml_model = RandomForestClassifier()
#             ml_model.fit(X_train, y_train)

#         if alg == "MLP":
#             ml_model = MLPClassifier()
#             ml_model.fit(X_train, y_train)

#         pickle.dump(ml_model, open(path_file_model + alg + "/Model_" + alg + ".pickle", "wb"))


# def run_save_graph(ontology_file_path, annotations_file_path, path_graph, path_label_classes, type='PPI'):

#     g, dic_labels_classes = construct_kg(ontology_file_path, annotations_file_path, type)
#     G = nx.DiGraph()
#     _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})

#     nx.write_gpickle(G, path_graph)
#     with open(path_label_classes, 'wb') as path_file:
#         pickle.dump(dic_labels_classes, path_file)



def getExplanations(path_graph, path_label_classes, path_embedding_classes, entity_to_neighbours_path, target_entity,
                    alg, path_file_model, model_stats_path, path_explanations, n_embeddings=100, type='PPI'):
    
    raise('development - need to include threshold to generate explanations using model_stats_path')

    # g, dic_labels_classes = construct_kg(ontology_file_path, annotations_file_path, type)
    # G = nx.DiGraph()
    # _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})
    # G = nx.read_gpickle(path_graph) ## read KG, not needed, loading nwighbours directly
    ## read dictionary for go terms
    ## not needed for AIFB because meaning is explicit in node name
    # with open(path_label_classes, 'rb') as label_classes:
    #     dic_labels_classes = pickle.load(label_classes)

    ## embeddings for each go term
    ## replace with embeddings for each neighbour (set of neighbours found for all labelled nodes)
    # dic_emb_classes = eval(open(path_embedding_classes, 'r').read())
    with open(path_embedding_classes, 'r') as f:
        dic_emb_classes = json.load(f)

    ## read ML model
    # ml_model = pickle.load(open(path_file_model + alg + "/Model_" + alg + ".pickle", "rb"))
    ml_model = joblib.load(path_file_model)
    num_classes = len(ml_model.classes_)

    ## unpack entities in entity pair, replace with target_entity to explain
    # ent1, ent2 = target_pair

    start = time.time()

    ensure_dir(path_explanations + "/")
    file_predictions = open(path_explanations + "/" + target_entity.split('/')[-1] + ".txt", 'w') ## replace ent1-ent2 with ent
    ## replace DCA with Neighbour, replace binary with multiclass
    # file_predictions.write('DCA\tRemoving\tPredicted-label\tProb-class0\tProb-class1\n')
    file_predictions.write('Neighbour\tRemoving\tPredicted-label')
    for i in range(num_classes):
        file_predictions.write(f'\tProb-class{i}')
    file_predictions.write('\n')


    ## extract all common ancestors
    ## replace with all_neighbours
    # all_common_ancestors = list(nx.descendants(G, rdflib.term.URIRef(ent1)) & nx.descendants(G, rdflib.term.URIRef(ent2)))
    # ## ignore, no need to find disjoint_common_ancestors for entity to explain
    # disjoint_common_ancestors, parents = [], {}
    # for anc in all_common_ancestors:
    #     parents[anc] = list(nx.descendants(G, anc))
    # for ancestor in all_common_ancestors:
    #     parent = False
    #     for anc2 in all_common_ancestors:
    #         if anc2 != ancestor:
    #             if ancestor in parents[anc2]:
    #                 parent = True
    #     if parent == False:
    #         if str(ancestor) in dic_emb_classes:
    #             disjoint_common_ancestors.append(ancestor)
    with open(entity_to_neighbours_path, 'r') as f:
        entity_to_neighbours = json.load(f)
    # print(entity_to_neighbours)
    all_neighbours = entity_to_neighbours[target_entity]

    necessary_explan, sufficient_explan = [], []
    results, results_without, results_only = [], [], []

    ## get all dcas embeddings and obtain the average
    # all_vectors = []
    # for dca in disjoint_common_ancestors:
    #     ## embeddings of the dcas, replace with embeddings of the neighbours
    #     all_vectors.append(dic_emb_classes[str(dca)])
    # if len(all_vectors) == 0:
    #     all_avg_vectors = np.array([0 for i in range(n_embeddings)])
    # elif len(all_vectors) == 1:
    #     all_avg_vectors = np.array(all_vectors[0])
    # else:
    #     all_array_vectors = np.array(all_vectors)
    #     all_avg_vectors = np.average(all_array_vectors, 0)
    all_vectors = []
    for neighbour in all_neighbours:
        ## embeddings of the dcas, replace with embeddings of the neighbours
        all_vectors.append(dic_emb_classes[neighbour])
    if len(all_vectors) == 0:
        all_avg_vectors = np.array([0 for i in range(n_embeddings)])
    elif len(all_vectors) == 1:
        all_avg_vectors = np.array(all_vectors[0])
    else:
        all_array_vectors = np.array(all_vectors)
        all_avg_vectors = np.average(all_array_vectors, 0)

    ## obtain the prediction for the dcas embeddings average
    X_test_original = [all_avg_vectors.tolist()]

    pred_original = ml_model.predict(X_test_original).tolist()[0]
    proba_pred_original = ml_model.predict_proba(X_test_original).tolist()[0]
    ## replace binary with multiclass
    file_predictions.write('All' + '\t' + 'NA' + '\t' + str(pred_original))
    for i in range(num_classes):
        file_predictions.write('\t' + str(proba_pred_original[i]))
    file_predictions.write('\n')

    print(proba_pred_original)

    ## for each dca, obtain the prediction for all dcas except the dca and obtain the prediction for the dca only
    ## replace disjoint_common_ancestors with neighbours
    # for dca in disjoint_common_ancestors:
    for neighbour in all_neighbours:

        vectors = []
        # for dca2 in disjoint_common_ancestors:
        for neighbour2 in all_neighbours:
            # if dca2 != dca:
            if neighbour2 != neighbour:
                # vectors.append(dic_emb_classes[str(dca2)])
                vectors.append(dic_emb_classes[neighbour2])

        if len(vectors) == 0:
            avg_vectors = np.array([0 for i in range(n_embeddings)])
        elif len(vectors) == 1:
            avg_vectors = np.array(vectors[0])
        else:
            array_vectors = np.array(vectors)
            avg_vectors = np.average(array_vectors, 0)


        X_test_without_dca = [avg_vectors.tolist()]
        pred_without_dca = ml_model.predict(X_test_without_dca).tolist()[0]
        proba_pred_without_dca = ml_model.predict_proba(X_test_without_dca).tolist()[0]

        ## replace binary with multiclass
        file_predictions.write(neighbour + '\t' + 'True' + '\t' + str(pred_without_dca))
        for i in range(num_classes):
            file_predictions.write('\t' + str(proba_pred_without_dca[i]))
        file_predictions.write('\n')

        ## only save results where prediction is changed with necessary
        ## replace with threshold or save every explanation?
        ## replace binary with multiclass
        if pred_original != pred_without_dca:
            necessary_explan.append(neighbour)
            ## dic_labels_classes not needed for AIFB because meaning is explicit in node name
            ## dic_labels_classes[str(dca)] -> str(dca) ?
            results_list = ["w/o '" + neighbour + "'"]
            for i in reversed(range(num_classes)):
                results_list.append(str(proba_pred_without_dca[i]))
            results_without.append(results_list)

        X_test_only_dca = [dic_emb_classes[neighbour]]
        pred_only_dca = ml_model.predict(X_test_only_dca).tolist()[0]
        proba_pred_only_dca = ml_model.predict_proba(X_test_only_dca).tolist()[0]

        ## replace binary with multiclass
        file_predictions.write(neighbour + '\t' + 'False' + '\t' + str(pred_only_dca))
        for i in range(num_classes):
            file_predictions.write('\t' + str(proba_pred_only_dca[i]))
        file_predictions.write('\n')

        ## only save results where prediction is maintained with sufficient
        ## replace with threshold or save every explanation?
        ## replace binary with multiclass
        if pred_original == pred_only_dca:
            sufficient_explan.append(neighbour)
            ## dic_labels_classes not needed for AIFB because meaning is explicit in node name
            ## dic_labels_classes[str(dca)] -> str(dca) ?
            results_list = ["only '" + neighbour + "'"]
            for i in reversed(range(num_classes)):
                results_list.append(proba_pred_only_dca[i])
            results_only.append(results_list)

    ## obtain the prediction for all the sufficient dcas
    vectors_withsufficient = []
    for suf in sufficient_explan:
        vectors_withsufficient.append(dic_emb_classes[suf])
    if len(sufficient_explan) == 0:
        x_withsufficient = np.array([0 for j in range(n_embeddings)])
    elif len(sufficient_explan) == 1:
        x_withsufficient = np.array(vectors_withsufficient[0])
    else:
        x_withsufficient = np.average(np.array(vectors_withsufficient), 0)
    predicted_label_withsufficient = ml_model.predict(x_withsufficient.reshape(1, -1))[0]
    proba_withsufficient = ml_model.predict_proba(x_withsufficient.reshape(1, -1))[0]

    ## obtain the prediction for all the necessary dcas
    vectors_withoutnecessary = []
    for not_nec in all_neighbours:
        if not_nec not in necessary_explan:
            vectors_withoutnecessary.append(dic_emb_classes[not_nec])
    if len(necessary_explan) == len(all_neighbours):
        x_withoutnecessary = np.array([0 for j in range(n_embeddings)])
    elif len(vectors_withoutnecessary) == 1:
        x_withoutnecessary = np.array(vectors_withoutnecessary[0])
    else:
        x_withoutnecessary = np.average(np.array(vectors_withoutnecessary), 0)
    predicted_label_withoutnecessary = ml_model.predict(x_withoutnecessary.reshape(1, -1))[0]
    proba_withoutnecessary = ml_model.predict_proba(x_withoutnecessary.reshape(1, -1))[0]

    file_predictions.close()

    end = time.time()
    print(end - start)


    ## section for the bar_plot, commented
    # # print(results_only, '\n\n')

    # # results_without.sort(key=itemgetter(2))
    # # results_only.sort(key=itemgetter(2))
    # idx_pred_original = ml_model.classes_.tolist().index(pred_original)
    # results_without.sort(key=itemgetter(idx_pred_original+1))
    # results_only.sort(key=itemgetter(idx_pred_original+1))

    # # print(results_only)

    # # results_without.append(['  ', 0.5, 0.5]) ## replace with multiclass
    # results_list = ['  ']
    # for i in range(num_classes):
    #     results_list.append(1/num_classes)
    # results_without.append(results_list)

    # # results.append(['global', proba_pred_original[1], proba_pred_original[0]])
    # # results.append(['w/o necessary', proba_withoutnecessary[1], proba_withoutnecessary[0]])
    # # results.append(['only sufficient', proba_withsufficient[1], proba_withsufficient[0]])
    # # results.append([' ', 0.5, 0.5])
    # proba_pred_original_list = [proba_pred_original[i] for i in reversed(range(num_classes))]
    # proba_withoutnecessary_list = [proba_withoutnecessary[i] for i in reversed(range(num_classes))]
    # proba_withsufficient_list = [proba_withsufficient[i] for i in reversed(range(num_classes))]
    # proba_pred_original_list.insert(0, 'global')
    # proba_withoutnecessary_list.insert(0, 'w/o necessary')
    # proba_withsufficient_list.insert(0, 'only sufficient')
    # results.append(proba_pred_original_list)
    # results.append(proba_withoutnecessary_list)
    # results.append(proba_withsufficient_list)
    # results_list = [' ']
    # for i in range(num_classes):
    #     results_list.append(1/num_classes)
    # results.append(results_list)

    # # df_without = pd.DataFrame(results_without, columns=["dca", "prob class 1", "prob class 0"])
    # # df_only = pd.DataFrame(results_only, columns=["dca", "prob class 1", "prob class 0"])
    # # df = pd.DataFrame(results, columns=["dca", "prob class 1", "prob class 0"])
    # # cols_only = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results_only]
    # # cols_without = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results_without]
    # # cols = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results]
    # columns = [f'prob class {i}' for i in reversed(range(num_classes))]
    # columns.insert(0, 'neighbour')
    # df_without = pd.DataFrame(results_without, columns=columns)
    # df_only = pd.DataFrame(results_only, columns=columns)
    # df = pd.DataFrame(results, columns=columns)
    # # cols_only = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results_only]
    # # cols_without = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results_without]
    # # cols = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results]

    # print(df, '\n\n')
    # print(df_without, '\n\n')
    # print(df_only, '\n\n')
    # print(len(all_neighbours), '\n\n')

    # bar_plot(df, df_without, df_only, len(all_neighbours), prob_class, num_classes, path_explanations + "/Plot_" + target_entity.split('/')[-1] + ".png")


def compute_performance_metrics(predicted_labels, list_labels):
    waf = f1_score(list_labels, predicted_labels, average='weighted')
    pr = precision_score(list_labels, predicted_labels, average='weighted')
    re = recall_score(list_labels, predicted_labels, average='weighted')
    return waf, pr, re

## get the wrapper method necessary (backward) and sufficient (forward) explanations
def get_new_score(ml_model, X_test, predicted_class_original):
    X_test = [X_test.tolist()]
    pred = ml_model.predict(X_test).tolist()[0]
    proba_pred = ml_model.predict_proba(X_test).tolist()[0]
    pred_proba_predicted_class = proba_pred[predicted_class_original]

    return pred_proba_predicted_class


def get_avg_vectors(vectors, n_embeddings):
    if len(vectors) == 0:
        avg_vectors = np.array([0 for i in range(n_embeddings)])
    elif len(vectors) == 1:
        avg_vectors = np.array(vectors[0])
    else:
        array_vectors = np.array(vectors)
        avg_vectors = np.average(array_vectors, 0)

    return avg_vectors


def get_list_of_vectors_with_some_neighbours(dic_emb_classes, all_neighbours, some_neighbours, explan_type):
    vectors = []
    for neighbour in all_neighbours:
        if explan_type == 'necessary':
            if neighbour not in some_neighbours:
                vectors.append(dic_emb_classes[neighbour])
        elif explan_type == 'sufficient':
            if neighbour in some_neighbours:
                vectors.append(dic_emb_classes[neighbour])
        else:
            raise Exception('must have a explan_type of explanation')

    return vectors


def compute_one_round_of_candidate_neighbours(ml_model, predicted_class_original,
                                                pred_proba_predicted_class_original, threshold, dic_emb_classes,
                                                n_embeddings, all_neighbours, current_neighbours_in_explanation,
                                                candidate_neighbours, explan_type):
    
    explanation_found = False
    early_stop = False

    candidate_neighbours_results = dict()
    current_neighbours_in_explanation = list(current_neighbours_in_explanation)
    candidate_neighbours = list(candidate_neighbours)
    # print(type(candidate_neighbours))
    # print('current_neighbours_in_explanation')
    # print(current_neighbours_in_explanation)
    for candidate_neighb in candidate_neighbours:
        # print('candidate_neighb')
        # print(type(candidate_neighb))
        some_neighbours = current_neighbours_in_explanation + [candidate_neighb]
        # print('some_neighbours')
        # print(some_neighbours)
        avg_vectors = get_list_of_vectors_with_some_neighbours(dic_emb_classes, all_neighbours,
                                                                some_neighbours, explan_type=explan_type)
        
        X_test = get_avg_vectors(avg_vectors, n_embeddings)

        pred_proba_predicted_class = get_new_score(ml_model, X_test, predicted_class_original)
        # print('some_neighbours')
        # print(some_neighbours)
        candidate_neighbours_results[tuple(some_neighbours)] = pred_proba_predicted_class
        if early_stop:
            if explan_type == 'necessary':
                if pred_proba_predicted_class_original - pred_proba_predicted_class >= threshold:
                    explanation_found = True
                    current_best_neighbours = set(min(candidate_neighbours_results.items(), key=lambda x: x[1])[0])
                    return explanation_found, current_best_neighbours, candidate_neighbours_results
            elif explan_type == 'sufficient':
                if pred_proba_predicted_class_original - pred_proba_predicted_class <= threshold:
                    explanation_found = True
                    current_best_neighbours = set(max(candidate_neighbours_results.items(), key=lambda x: x[1])[0])
                    # print('here')
                    # print(pred_proba_predicted_class_original, pred_proba_predicted_class)
                    return explanation_found, current_best_neighbours, candidate_neighbours_results
            else:
                raise Exception('must have a explan_type of explanation')

    if explan_type == 'necessary':
        current_best_neighbours = set(min(candidate_neighbours_results.items(), key=lambda x: x[1])[0])
        explanation_found = True if pred_proba_predicted_class_original - pred_proba_predicted_class >= threshold else False
    elif explan_type == 'sufficient':
        # print('candidate_neighbours_results')
        # print(candidate_neighbours_results.items())
        current_best_neighbours = set(max(candidate_neighbours_results.items(), key=lambda x: x[1])[0])
        explanation_found = True if pred_proba_predicted_class_original - pred_proba_predicted_class <= threshold else False
    else:
        raise Exception('must have a explan_type of explanation')

    return explanation_found, current_best_neighbours, candidate_neighbours_results


## necessary (backward)
def wrapper_method_for_explanation_selection(
        ml_model, predicted_class_original, pred_proba_predicted_class_original, threshold, dic_emb_classes,
        n_embeddings, all_neighbours, max_len_explanations, explan_type):
    
    all_neigbours_set = set(all_neighbours)

    wrapper_explan = dict()
    wrapper_explan_len1 = dict()
    
    current_neighbours_in_explanation = set()
    candidate_neighbours = set(all_neighbours)
    explan_len1 = True
    while candidate_neighbours:
        # print('here')
        # print('current_neighbours_in_explanation')
        # print(current_neighbours_in_explanation)

        explanation_found, \
        current_neighbours_in_explanation, \
        current_explan = compute_one_round_of_candidate_neighbours(ml_model, predicted_class_original,
                                                    pred_proba_predicted_class_original, threshold, dic_emb_classes,
                                                    n_embeddings, all_neighbours,
                                                    current_neighbours_in_explanation, candidate_neighbours,
                                                    explan_type=explan_type)
        wrapper_explan.update(current_explan)
        if explan_len1:
            wrapper_explan_len1.update(current_explan)
            explan_len1 = False
        # print(len(list(current_explan.keys())[0]))
        # print(list(current_explan.keys())[0])
        if explanation_found or len(list(current_explan.keys())[0]) == max_len_explanations:
            break

        candidate_neighbours = all_neigbours_set.difference(current_neighbours_in_explanation)

        # i += 1
        # if i == 5:
        #     raise

    # print(explanation_found)

    # print(pred_proba_predicted_class_original)
    # print('necessary_explan')
    # print(necessary_explan, '\n\n')
    if explan_type == 'necessary':
        # wrapper_explan_numbers = min(wrapper_explan.items(), key=lambda x: x[1])
        wrapper_explan = list(min(wrapper_explan.items(), key=lambda x: x[1])[0])
        wrapper_explan_len1 = list(min(wrapper_explan_len1.items(), key=lambda x: x[1])[0])
    elif explan_type == 'sufficient':
        # wrapper_explan_numbers = max(wrapper_explan.items(), key=lambda x: x[1])
        wrapper_explan = list(max(wrapper_explan.items(), key=lambda x: x[1])[0])
        wrapper_explan_len1 = list(max(wrapper_explan_len1.items(), key=lambda x: x[1])[0])
    # print(necessary_explan)
    # raise

    return wrapper_explan, wrapper_explan_len1 # , wrapper_explan_numbers


def compute_nec_suf_delta(met, test_labels, predictions, predictions_necessary, predictions_sufficient, average_type=''):
    names = ['original', 'necessary', 'delta_necessary', 'sufficient', 'delta_sufficient']
    keys_names = ['_'.join([name, met.__name__, average_type]) for name in names]
    preds_list = [predictions, predictions_necessary, predictions_sufficient]
    scores = []
    for pred in preds_list:
        if met == accuracy_score:
            scores.append(met(test_labels, pred))
        else:
            scores.append(met(test_labels, pred, average=average_type))
    delta_necessary = scores[1] - scores[0]
    delta_sufficient = scores[2] - scores[0]
    scores.insert(2, delta_necessary)
    scores.append(delta_sufficient)
    scores_dict = dict(zip(keys_names, scores))

    return scores_dict

def compute_performance_metrics_v2(test_labels, predictions, predictions_necessary, predictions_sufficient):
    effectiveness_results = compute_nec_suf_delta(accuracy_score, test_labels, predictions, predictions_necessary, predictions_sufficient)

    metrics_list = [f1_score, precision_score, recall_score]
    average_type = ['weighted', 'macro']
    for aver_type in average_type:
        for met in metrics_list:
            effectiveness_results.update(compute_nec_suf_delta(met, test_labels, predictions,
                                                                predictions_necessary, predictions_sufficient, aver_type))


    return effectiveness_results


def get_pred_withoutnecessary(ml_model, all_neighbours, necessary_explan, dic_emb_classes, n_embeddings):
    vectors_withoutnecessary = []
    for not_nec in all_neighbours:
        if not_nec not in necessary_explan:
            vectors_withoutnecessary.append(dic_emb_classes[not_nec])
    if len(necessary_explan) == len(all_neighbours):
        x_withoutnecessary = np.array([0 for j in range(n_embeddings)])
    elif len(vectors_withoutnecessary) == 1:
        x_withoutnecessary = np.array(vectors_withoutnecessary[0])
    else:
        x_withoutnecessary = np.average(np.array(vectors_withoutnecessary), 0)
    pred_withoutnecessary = ml_model.predict(x_withoutnecessary.reshape(1, -1))[0]

    return pred_withoutnecessary

def get_pred_withsufficient(ml_model, all_neighbours, sufficient_explan, dic_emb_classes, n_embeddings):
    vectors_withsufficient = []
    for suf in all_neighbours:
        if suf in sufficient_explan:
            vectors_withsufficient.append(dic_emb_classes[suf])

    if len(sufficient_explan) == 0:
        x_withsufficient = np.array([0 for j in range(n_embeddings)])
    elif len(sufficient_explan) == 1:
        x_withsufficient = np.array(vectors_withsufficient[0])
    else:
        x_withsufficient = np.average(np.array(vectors_withsufficient), 0)
    pred_withsufficient = ml_model.predict(x_withsufficient.reshape(1, -1))[0]

    return pred_withsufficient


def explain(input_data):
    entity, label, entity_to_neighbours, dic_emb_classes, n_embeddings, ml_model, threshold, max_len_explanations = \
        input_data
    # print(entity, label)

    # all_common_ancestors = list(nx.descendants(G, rdflib.term.URIRef(ent1)) & nx.descendants(G, rdflib.term.URIRef(ent2)))
    # disjoint_common_ancestors, parents = [], {}
    # for anc in all_common_ancestors:
    #     parents[anc] = list(nx.descendants(G, anc))
    # for ancestor in all_common_ancestors:
    #     parent = False
    #     for anc2 in all_common_ancestors:
    #         if anc2 != ancestor:
    #             if ancestor in parents[anc2]:
    #                 parent = True
    #     if parent == False:
    #         if str(ancestor) in dic_emb_classes:
    #             disjoint_common_ancestors.append(ancestor)

    # with open(entity_to_neighbours_path, 'r') as f:
    #     entity_to_neighbours = json.load(f)
    # print(entity_to_neighbours)

    all_neighbours = entity_to_neighbours[entity]

    all_vectors = []
    for neighbour in all_neighbours:
        all_vectors.append(dic_emb_classes[neighbour])
    if len(all_vectors) == 0:
        all_avg_vectors = np.array([0 for i in range(n_embeddings)])
    elif len(all_vectors) == 1:
        all_avg_vectors = np.array(all_vectors[0])
    else:
        all_array_vectors = np.array(all_vectors)
        all_avg_vectors = np.average(all_array_vectors, 0)

    X_test_original = [all_avg_vectors.tolist()]
    pred_original = ml_model.predict(X_test_original).tolist()[0]
    pred_proba_original = ml_model.predict_proba(X_test_original).tolist()[0]
    predicted_class_original = np.argmax(pred_proba_original)
    pred_proba_predicted_class_original = pred_proba_original[predicted_class_original]




    ## get the best single necessary and single sufficient explanations
    # necessary_explan, sufficient_explan = [], []
    necessary_explan, sufficient_explan = dict(), dict()

    for neighbour in all_neighbours:

        vectors = []
        for neighbour2 in all_neighbours:
            if neighbour2 != neighbour:
                vectors.append(dic_emb_classes[neighbour2])

        if len(vectors) == 0:
            avg_vectors = np.array([0 for i in range(n_embeddings)])
        elif len(vectors) == 1:
            avg_vectors = np.array(vectors[0])
        else:
            array_vectors = np.array(vectors)
            avg_vectors = np.average(array_vectors, 0)

        X_test_without_dca = [avg_vectors.tolist()]
        pred_without_dca = ml_model.predict(X_test_without_dca).tolist()[0]
        proba_pred_without_dca = ml_model.predict_proba(X_test_without_dca).tolist()[0]
        pred_proba_predicted_class_without_dca = proba_pred_without_dca[predicted_class_original]
        necessary_explan[neighbour] = pred_proba_predicted_class_without_dca
        # if explanation_limit == 'threshold':
        #     if pred_proba_predicted_class_original - pred_proba_predicted_class_without_dca >= threshold:
        #         necessary_explan.append(neighbour)
        # elif explanation_limit == 'class_change':
        #     if pred_original != pred_without_dca:
        #         necessary_explan.append(neighbour)
        # else:
        #     raise Exception('invalid choice of explanation limit')

        X_test_only_dca = [dic_emb_classes[neighbour]]
        pred_only_dca = ml_model.predict(X_test_only_dca).tolist()[0]
        proba_pred_only_dca = ml_model.predict_proba(X_test_only_dca).tolist()[0]
        pred_proba_predicted_class_only_dca = proba_pred_only_dca[predicted_class_original]
        sufficient_explan[neighbour] = pred_proba_predicted_class_only_dca
        # if explanation_limit == 'threshold':
        #     if pred_proba_predicted_class_original - pred_proba_predicted_class_without_dca <= threshold:
        #         sufficient_explan.append(neighbour)
        # elif explanation_limit == 'class_change':
        #     if pred_original == pred_only_dca:
        #         sufficient_explan.append(neighbour)
        # else:
        #     raise Exception('invalid choice of explanation limit')

    # print(necessary_explan)
    # necessary_explan_old = min(necessary_explan.items(), key=lambda x: x[1])
    necessary_explan = [min(necessary_explan.items(), key=lambda x: x[1])[0]]

    # print(sufficient_explan)
    # sufficient_explan_old = max(sufficient_explan.items(), key=lambda x: x[1])
    sufficient_explan = [max(sufficient_explan.items(), key=lambda x: x[1])[0]]

    ## this was used to calculate nec and suf for SEEK
    # if pred_original == label:

    #     vectors_withoutnecessary = []
    #     for not_nec in all_neighbours:
    #         if not_nec not in necessary_explan:
    #             vectors_withoutnecessary.append(dic_emb_classes[not_nec])
    #     if len(necessary_explan) == len(all_neighbours):
    #         x_withoutnecessary = np.array([0 for j in range(n_embeddings)])
    #     elif len(vectors_withoutnecessary) == 1:
    #         x_withoutnecessary = np.array(vectors_withoutnecessary[0])
    #     else:
    #         x_withoutnecessary = np.average(np.array(vectors_withoutnecessary), 0)
    #     pred_withoutnecessary = ml_model.predict(x_withoutnecessary.reshape(1, -1))[0]

    #     pred_eva_necessary.append(pred_withoutnecessary)
    #     original_pred_eva_necessary.append(pred_original)
    #     y_eva_necessary.append(label)

    # if pred_original != label:
        # vectors_withsufficient = []
        # for suf in sufficient_explan:
        #     vectors_withsufficient.append(dic_emb_classes[suf])
        # if len(sufficient_explan) == 0:
        #     x_withsufficient = np.array([0 for j in range(n_embeddings)])
        # elif len(sufficient_explan) == 1:
        #     x_withsufficient = np.array(vectors_withsufficient[0])
        # else:
        #     x_withsufficient = np.average(np.array(vectors_withsufficient), 0)
        # pred_withsufficient = ml_model.predict(x_withsufficient.reshape(1, -1))[0]

        # pred_eva_sufficient.append(pred_withsufficient)
        # original_pred_eva_sufficient.append(pred_original)
        # y_eva_sufficient.append(label)

    necessary_explan, necessary_explan_len1 = wrapper_method_for_explanation_selection(
        ml_model, predicted_class_original, pred_proba_predicted_class_original, threshold, dic_emb_classes,
        n_embeddings, all_neighbours, max_len_explanations, explan_type='necessary')
    
    sufficient_explan, sufficient_explan_len1 = wrapper_method_for_explanation_selection(
        ml_model, predicted_class_original, pred_proba_predicted_class_original, threshold, dic_emb_classes,
        n_embeddings, all_neighbours, max_len_explanations, explan_type='sufficient')
    
    # all_necessary_explan.append(necessary_explan)
    # all_sufficient_explan.append(sufficient_explan)

    # if necessary_explan_old[0] != necessary_explan_numbers[0][0]:
    #     print('difference found in necessary')
    #     print(entity, '\n', necessary_explan_old, '\n', necessary_explan_numbers, '\n\n')

    # if sufficient_explan_old[0] != sufficient_explan_numbers[0][0]:
    #     print('difference found in sufficient')
    #     print(entity, '\n', sufficient_explan_old, '\n', sufficient_explan_numbers, '\n\n')




    ## necessary and sufficient:
    ##     for all test entities instead of just the correcly predicted
    ##     with just the top explanation instead of all the necessary explanations
    ##     with threshold instead of changing class
    ## necessary
    pred_withoutnecessary = get_pred_withoutnecessary(ml_model, all_neighbours, necessary_explan, dic_emb_classes,
                                                      n_embeddings)
    pred_withoutnecessary_len1 = get_pred_withoutnecessary(ml_model, all_neighbours, necessary_explan_len1,
                                                           dic_emb_classes, n_embeddings)

    # pred_eva_necessary.append(pred_withoutnecessary)
    # original_pred_eva_necessary.append(pred_original)
    # y_eva_necessary.append(label)

    ## sufficient before
    # vectors_withsufficient = []
    # for suf in sufficient_explan:
    #     vectors_withsufficient.append(dic_emb_classes[suf])

    # sufficient_explan = random.choice(all_neighbours)

    pred_withsufficient = get_pred_withsufficient(ml_model, all_neighbours, sufficient_explan, dic_emb_classes,
                                                  n_embeddings)
    pred_withsufficient_len1 = get_pred_withsufficient(ml_model, all_neighbours, sufficient_explan_len1,
                                                       dic_emb_classes, n_embeddings)


    # pred_eva_sufficient.append(pred_withsufficient)
    # original_pred_eva_sufficient.append(pred_original)
    # y_eva_sufficient.append(label)

    return label, pred_original, pred_withoutnecessary, pred_withsufficient, pred_withoutnecessary_len1, pred_withsufficient_len1

def pool_handler(pool_size, input_data):
    with Pool(pool_size) as p:
        res = list(
                p.map(explain,
                        input_data,
                        # chunksize=1
                        )
        )

    return res


def pool_handler_tqdm(pool_size, input_data, items, verbose):
    with Pool(pool_size) as p:
        res = list(
            tqdm(
                p.imap(explain,
                        input_data,
                        # chunksize=1
                        ),
                total=len(items),
                disable=True if verbose == 0 else False,
            )
        )
    return res


# def compute_effectiveness_kelpie(dataset_labels, path_graph, path_label_classes, path_embedding_classes,
#                                  entity_to_neighbours_path, path_file_model, model_stats_path, path_explanations,
#                                  max_len_explanations, n_embeddings=100):
# def compute_effectiveness_kelpie(dataset_labels, path_embedding_classes,
#                                  entity_to_neighbours_path, path_file_model, model_stats_path, path_explanations,
#                                  max_len_explanations, n_embeddings=100):
def compute_effectiveness_kelpie(dataset_labels, dic_emb_classes,
                                 entity_to_neighbours, ml_model, results_summary, path_explanations,
                                 max_len_explanations, n_jobs, n_embeddings=100):
    
    explanation_limit='threshold'
    # explanation_limit='class_change'

    multiproc = True
    # multiproc = False

    # all_necessary_explan, all_sufficient_explan = [], []

    # G = nx.read_gpickle(path_graph)
    # with open(path_label_classes, 'rb') as label_classes:
    #     dic_labels_classes = pickle.load(label_classes)

    ## for loading std dev for threshold
    # with open(model_stats_path, 'r') as f:
    #     results_summary = json.load(f)

    threshold = results_summary['std_preds']

    # with open(path_embedding_classes, 'r') as f:
    #     dic_emb_classes = json.load(f)

    n_embeddings = len(list(dic_emb_classes.values())[0])

    # ml_model = joblib.load(path_file_model)

    pred_eva_necessary, pred_eva_necessary_len1, original_pred_eva_necessary, y_eva_necessary = [], [], [], []
    pred_eva_sufficient, pred_eva_sufficient_len1, original_pred_eva_sufficient, y_eva_sufficient = [], [], [], []
    if multiproc:
        num_items = len(dataset_labels)
        entities = [entity for (entity, _) in dataset_labels]
        labels = [label for (_, label) in dataset_labels]
        input_data = zip(
            [entity for (entity, _) in dataset_labels],
            [label for (_, label) in dataset_labels],
            [entity_to_neighbours] * num_items,
            [dic_emb_classes] * num_items,
            [n_embeddings] * num_items,
            [ml_model] * num_items,
            [threshold] * num_items,
            [max_len_explanations] * num_items,
        )
        input_data = tuple(list(map(list, input_data)))
        # res = pool_handler(n_jobs, input_data)
        print(f'Parallelizing explanations:')
        res = pool_handler_tqdm(n_jobs, input_data, dataset_labels, verbose=True)

        for item in res:
            label, pred_original, pred_withoutnecessary, pred_withsufficient, pred_withoutnecessary_len1, \
                pred_withsufficient_len1 = item
            y_eva_necessary.append(label)
            y_eva_sufficient.append(label)
            original_pred_eva_necessary.append(pred_original)
            original_pred_eva_sufficient.append(pred_original)
            pred_eva_necessary.append(pred_withoutnecessary)
            pred_eva_sufficient.append(pred_withsufficient)
            pred_eva_necessary_len1.append(pred_withoutnecessary_len1)
            pred_eva_sufficient_len1.append(pred_withsufficient_len1)
    else:
        # for (entity, label) in dataset_labels[33:34]:
        # for (entity, label) in dataset_labels[0:1]:
        # for (entity, label) in dataset_labels[2:3]:
        for (entity, label) in dataset_labels:
            input_data = [entity, label, entity_to_neighbours, dic_emb_classes, n_embeddings, ml_model, threshold,
                          max_len_explanations]
            label, \
            pred_original, \
            pred_withoutnecessary, \
            pred_withsufficient, \
            pred_withoutnecessary_len1, \
            pred_withsufficient_len1 = explain(input_data)
            y_eva_necessary.append(label)
            y_eva_sufficient.append(label)
            original_pred_eva_necessary.append(pred_original)
            original_pred_eva_sufficient.append(pred_original)
            pred_eva_necessary.append(pred_withoutnecessary)
            pred_eva_sufficient.append(pred_withsufficient)
            pred_eva_necessary_len1.append(pred_withoutnecessary_len1)
            pred_eva_sufficient_len1.append(pred_withsufficient_len1)


    # print('all_necessary_explan')
    # print(all_necessary_explan)

    # print('all_sufficient_explan')
    # print(all_sufficient_explan)

    # print(len(original_pred_eva_necessary), '\n')
    # print(len(y_eva_necessary))
    
    effectiveness_results_lenx = compute_performance_metrics_v2(y_eva_necessary, original_pred_eva_necessary,
                                                           pred_eva_necessary, pred_eva_sufficient)
    
    effectiveness_results_len1 = compute_performance_metrics_v2(y_eva_necessary, original_pred_eva_necessary,
                                                           pred_eva_necessary_len1, pred_eva_sufficient_len1)

    original_waf_necc, original_pr_necc, original_re_necc = compute_performance_metrics(original_pred_eva_necessary,y_eva_necessary)
    waf_necc, pr_necc, re_necc = compute_performance_metrics(pred_eva_necessary, y_eva_necessary)

    original_waf_suf, original_pr_suf, original_re_suf = compute_performance_metrics(original_pred_eva_sufficient,y_eva_sufficient)
    waf_suf, pr_suf, re_suf = compute_performance_metrics(pred_eva_sufficient, y_eva_sufficient)


    with open(path_explanations + "/ExplanationEfectivenessKelpie.txt", 'w') as file_output:
        file_output.write('TypeExplanation\twaf\tprecision\trecall\n')
        file_output.write('Original\t' + str(original_waf_necc) + '\t' + str(original_pr_necc) + '\t' + str(
            original_re_necc) + '\n')
        file_output.write('Necessary\t' + str(waf_necc) + '\t' + str(pr_necc) + '\t' + str(re_necc) + '\n')
        file_output.write('DeltaNecessary\t' + str(waf_necc - original_waf_necc) + '\t' + str(pr_necc - original_pr_necc) + '\t' + str(re_necc - original_re_necc) + '\n')
        file_output.write('\n')
        file_output.write('Original\t' + str(original_waf_suf) + '\t' + str(original_pr_suf) + '\t' + str(original_re_suf) + '\n')
        file_output.write('Sufficient\t' + str(waf_suf) + '\t' + str(pr_suf) + '\t' + str(re_suf) + '\n')
        file_output.write('DeltaSufficient\t' + str(waf_suf - original_waf_suf) + '\t' + str(pr_suf - original_pr_suf) + '\t' + str(re_suf - original_re_suf) + '\n')
    
    return [effectiveness_results_lenx, effectiveness_results_len1]

if __name__== '__main__':

    ####################################### PPI prediction

    # path_file_representation = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"
    # # path_file_model = "./PPI/Models/RF/Model_RF.pickle"
    # path_file_model = "./PPI/Models/"
    # alg = "RF"
    # # alg = "MLP"
    # path_graph = "./PPI/KG.gpickle"
    # path_label_classes = "PPI/Labelclasses.pkl"
    # path_explanations = "./PPI/Explanations/"
    # path_embedding_classes = "PPI/Embeddings/Emb_classes_maxdepth4_nwalks100_disjointcommonancestor.txt"
    # target_pair = ('P25398','P46783')
    # getExplanations(path_graph, path_label_classes, path_embedding_classes, target_pair, alg, path_file_model, path_explanations)

    # dataset = 'AIFB'
    # dataset = 'MUTAG'
    # dataset = 'AM_FROM_DGL'
    dataset = 'MDGENRE'

    # max_len_explanations=1
    max_len_explanations=5

    data_path = f'node_classifier/data/{dataset}'

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

    model_path = f'node_classifier/model/{dataset}'

    saved_models = os.listdir(model_path)
    if not saved_models:
        sys.exit('there are no saved models, directory is empty')

    ## sorts the models first to last using the name ending digits
    saved_models.sort(key=lambda x: int(x.split('_')[-2]))
    saved_models = saved_models[-2:]
    saved_models.sort(key=lambda x: x.split('_')[-1], reverse=True)
    last_saved_model = saved_models[-1]
    current_model_path = os.path.join(model_path, last_saved_model)

    # mode='explain'
    mode='evaluate'

    path_explanations = os.path.join(current_model_path, 'explanations')

    if mode == 'explain':
        if not os.path.exists(path_explanations):
            os.mkdir(path_explanations)
        else:
            if os.listdir(path_explanations):
                raise Exception('explanation already exists for the current model')
    elif mode == 'evaluate':
        if not os.path.exists(path_explanations):
            os.mkdir(path_explanations)
            # raise Exception('there are no explanations to evaluate')

    # path_file_representation = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"
            
    path_file_model = os.path.join(current_model_path, f'models/classifier_{dataset}')
    clf = joblib.load(path_file_model)


    alg = None
    path_graph = os.path.join(data_path, 'KG.gpickle')
    path_label_classes = None

    path_embedding_classes = os.path.join(current_model_path, 'trained/neighbours_embeddings.json')
    with open(path_embedding_classes, 'r') as f:
        dic_emb_classes = json.load(f)

    entity_to_neighbours_path = os.path.join(current_model_path, 'trained/entity_to_neighbours.json')
    with open(entity_to_neighbours_path, 'r') as f:
        entity_to_neighbours = json.load(f)

    model_stats_path = os.path.join(current_model_path, 'models_results/results_summary.json')
    with open(model_stats_path, 'r') as f:
        results_summary = json.load(f)

    # target_entity = 'http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id1909instance'
    # target_entity = 'http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id2040instance'
    # target_entity = 'http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id2119instance'
    target_entity = 'http://www.aifb.uni-karlsruhe.de/Personen/viewPersonOWL/id2121instance'

    # getExplanations(path_graph, path_label_classes, path_embedding_classes, entity_to_neighbours_path,
    #                 target_entity, alg, path_file_model, model_stats_path, path_explanations)


    dataset_labels = list(zip(test_entities, test_labels))
    n_jobs = 1

    # compute_effectiveness_kelpie(dataset_labels, path_embedding_classes, entity_to_neighbours_path, path_file_model,
    #                              model_stats_path, path_explanations, max_len_explanations)
    compute_effectiveness_kelpie(dataset_labels, dic_emb_classes, entity_to_neighbours, clf,
                                 model_stats_path, path_explanations, max_len_explanations, n_jobs)