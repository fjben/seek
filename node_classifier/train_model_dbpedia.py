## based on https://github.com/IBCNServices/pyRDF2Vec/tree/main/examples


import argparse

from collections import OrderedDict
import json
import logging
from multiprocessing import cpu_count
import os
import random
import shutil
import sys
import time
import warnings

from joblib import dump
# import networkx as nx
import numpy as np
import pandas as pd
import rdflib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.graphs.vertex import Vertex
from pyrdf2vec.walkers import RandomWalker

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
from utils.logger import Logger
# from run_seek_explanations import _rdflib_to_networkx_graph

# os.environ['PYTHONHASHSEED'] = str(123)
# os.execv(sys.executable, ['python3'] + sys.argv)


############################################################################### arguments

# dataset = 'AIFB'
# dataset = 'MUTAG'
# dataset = 'AM_FROM_DGL'
# dataset = 'MDGENRE'

# aproximate_model=True
# aproximate_model=False

# best_embeddings_params = True
best_embeddings_params = False

embeddings_for_all_entities = True
# embeddings_for_all_entities = False

verbose = 1


############################################################################### logging

sys.stdout = Logger()


############################################################################### functions

def create_save_dirs(model_path, dataset, model_type, current_model_num):
    current_model_path = os.path.join(model_path, f'{dataset}_model_{current_model_num}_{model_type}')
    current_model_models_path = os.path.join(current_model_path, 'models')
    current_model_models_results_path = os.path.join(current_model_path, 'models_results')
    current_model_trained_path = os.path.join(current_model_path, 'trained')
    os.makedirs(current_model_models_path)
    os.makedirs(current_model_models_results_path)
    os.makedirs(current_model_trained_path)

    return current_model_models_path, current_model_models_results_path, current_model_trained_path

## this could alternatively be accomplished using the pyRDF2Vec kg object with the get_neighbors method, one advantage
## is that it only needs to load the graph once for the kg object
## or this code should go to some dataset object or folder the first time the dataset is used because the neighbours
## are always the same
def find_neighbours(graph, entities):        
    # Print the number of "triples" in the Graph
    print(f"Graph graph has {len(graph)} statements.")
    # Prints: Graph g has 86 statements.
    all_neighbours = []
    entity_to_neighbours = OrderedDict()
    for entity in entities:
        # kg._get_hops(Vertex(str(entity)))
        entity = rdflib.term.URIRef(entity)
        ## this only lists outgoing entities, not ingoing, but I think it's okay because when generating walks only
        ## outgoing entities are used
        if (entity, None, None) in graph:

            # create a graph
            entitygraph = rdflib.Graph()
            # add all triples with subject 'bob'
            entitygraph += graph.triples((entity, None, None))
            entity_neighbours = []
            entity_neighbour_relation = []
            for s, p, o in entitygraph:
                if isinstance(o, (rdflib.Literal, rdflib.BNode)): ## ignoring literals
                    continue
                all_neighbours.append(str(o))
                entity_neighbours.append(str(o))
                entity_neighbour_relation.append(str(p))
            entity_to_neighbours[str(s)] = [entity_neighbours, entity_neighbour_relation]

    all_neighbours = list(set(all_neighbours))

    return all_neighbours, entity_to_neighbours


def get_all_entities(graph):
    # all_entities = [entity for entity in graph.subjects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))]
    all_subjects = {str(entity) for entity in graph.subjects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))}
    all_objects = {str(entity) for entity in graph.objects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))}
    all_entities = all_subjects.union(all_objects)
    print(len(all_entities))

    return list(all_entities)


def stats_for_preds(predictions_proba):
        maxs_list = [np.ndarray.max(single_pred) for single_pred in predictions_proba]
        # for single_pred in predictions_proba:
        #     print(numpy.ndarray.max(single_pred))
        mean_preds = np.mean(maxs_list)
        std_preds = np.std(maxs_list)

        return mean_preds, std_preds


def transformer_fit_transform_with_times(kg, entities):
    tic = time.time()
    walks = transformer.get_walks(kg, entities)
    toc = time.time()
    walks_time = toc - tic
    tic = time.time()
    transformer.fit(walks)
    toc = time.time()
    embeddings_fit_time = toc - tic
    embeddings, _ = transformer.transform(kg, entities)

    return embeddings, walks_time, embeddings_fit_time


############################################################################### script

parser = argparse.ArgumentParser(description="description")
parser.add_argument("--dataset",
                    type=str,
                    choices=['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE', 'BGS_FROM_DGL', 'CITIES'],
                    help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")
parser.add_argument("--aproximate_model",
                    action="store_true",
                    help="To use the aggregate representation model.")
args = parser.parse_args()
dataset = args.dataset
aproximate_model = args.aproximate_model

cpu_num = cpu_count()

data_path = f'node_classifier/data/{dataset}'
model_path = f'node_classifier/model/{dataset}'

if aproximate_model:
    model_type = 'RAN' ## representation with aggregate neighbours
else:
    model_type = 'RO' ## representation with original

saved_models = os.listdir(model_path)
if not saved_models:
    # current_model_path = os.path.join(model_path, f'{dataset}_model_{model_type}_0')
    # current_model_models_path = os.path.join(current_model_path, 'models')
    # current_model_models_results_path = os.path.join(current_model_path, 'models_results')
    # current_model_trained_path = os.path.join(current_model_path, 'trained')
    # os.makedirs(current_model_models_path)
    # os.makedirs(current_model_models_results_path)
    # os.makedirs(current_model_trained_path)
    last_saved_model_num = -1
else:
    ## sorts the models first to last using the name ending digits
    last_saved_model = saved_models.sort(key=lambda x: int(x.split('_')[-2]))
    last_saved_model_num = int(saved_models[-1].split('_')[-2])
if os.path.exists('node_classifier/tmp/reproducibility_parameters_train_model.txt'):
    if aproximate_model:
        current_model_num = last_saved_model_num
        current_model_models_path, \
        current_model_models_results_path, \
        current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)
        # current_model_path = os.path.join(model_path, f'{dataset}_model_{model_type}_{current_model_num}')
        # current_model_models_path = os.path.join(current_model_path, 'models')
        # current_model_models_results_path = os.path.join(current_model_path, 'models_results')
        # current_model_trained_path = os.path.join(current_model_path, 'trained')
        # os.makedirs(current_model_models_path)
        # os.makedirs(current_model_models_results_path)
        # os.makedirs(current_model_trained_path)
        shutil.move('node_classifier/tmp/reproducibility_parameters_train_model.txt', os.path.join(current_model_models_path, 'reproducibility_parameters_train_model.txt'))
    else:
        current_model_num = last_saved_model_num + 1
        current_model_models_path, \
        current_model_models_results_path, \
        current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)
        # current_model_path = os.path.join(model_path, f'{dataset}_model_{model_type}_{current_model_num}')
        # current_model_models_path = os.path.join(current_model_path, 'models')
        # current_model_models_results_path = os.path.join(current_model_path, 'models_results')
        # current_model_trained_path = os.path.join(current_model_path, 'trained')
        # os.makedirs(current_model_models_path)
        # os.makedirs(current_model_models_results_path)
        # os.makedirs(current_model_trained_path)
        shutil.copy('node_classifier/tmp/reproducibility_parameters_train_model.txt', os.path.join(current_model_models_path, 'reproducibility_parameters_train_model.txt'))
else:
    current_model_num = last_saved_model_num + 1
    current_model_models_path, \
    current_model_models_results_path, \
    current_model_trained_path = create_save_dirs(model_path, dataset, model_type, current_model_num)
    # current_model_path = os.path.join(model_path, f'{dataset}_model_{model_type}_{current_model_num}')
    # current_model_models_path = os.path.join(current_model_path, 'models')
    # current_model_models_results_path = os.path.join(current_model_path, 'models_results')
    # current_model_trained_path = os.path.join(current_model_path, 'trained')
    # os.makedirs(current_model_models_path)
    # os.makedirs(current_model_models_results_path)
    # os.makedirs(current_model_trained_path)

try:
    with open(os.path.join(current_model_models_path, 'reproducibility_parameters_train_model.txt')) as f:
        lines = f.readlines()
    RANDOM_STATE = int(lines[5])
    workers = int(lines[7])
    if workers != 1:
        warnings.warn('workers parameter is not equal to 1, so the results are not reproducible')
except:
    warnings.warn('no reproducibility parameters available, so the results are not reproducible')
    # RANDOM_STATE = random.randrange(0, 4294967295)
    # workers = cpu_num
    # # for debugging purposes
    RANDOM_STATE = 22
    workers = 1

shutil.move('node_classifier/tmp/train_models.log', os.path.join(current_model_models_results_path, 'train_models.log'))

# n_jobs = 2 ## original specs found in online-learning.py
n_jobs = cpu_num

print('RANDOM_STATE:\t\t', RANDOM_STATE)
print('workers:\t\t', workers)
print("Number of used cpu:\t", n_jobs, '\n')

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

## original specs found in online-learning.py used when best_embeddings_params is False
vector_size = 100
sg=0
max_depth=2
max_walks=None
## specs from LoFI used when best_embeddings_params is True
if best_embeddings_params:
    match dataset:
        case 'AIFB':
            vector_size=500
            sg=1
            max_depth=4
            max_walks=500
        case 'MUTAG':
            vector_size=50
            sg=1
            max_depth=4
            max_walks=500
        case 'AM_FROM_DGL':
            vector_size=500
            sg=1
            max_depth=2
            max_walks=500
        case 'MDGENRE':
            vector_size=500
            sg=1
            max_depth=2
            max_walks=500

# Defines the KG with the predicates to be skipped.
tic = time.perf_counter()
# kg = KG(
#     location = location,
#     skip_predicates=skip_predicates,
# )
kg = KG(
    "https://dbpedia.org/sparql",
    skip_predicates=skip_predicates
)
toc = time.perf_counter()
kg_init_time = toc - tic
print(f"KG initialization time ({kg_init_time:0.4f}s)")

transformer = RDF2VecTransformer(
    # Ensure random determinism for Word2Vec.
    # Must be used with PYTHONHASHSEED.
    Word2Vec(workers=workers, vector_size=vector_size, sg=sg),
    # Extract all walks with a maximum depth of 2 for each entity by using two
    # processes and a random state to ensure that the same walks are generated
    # for the entities.
    walkers=[RandomWalker(max_depth=max_depth, max_walks=max_walks, n_jobs=n_jobs, random_state=RANDOM_STATE)],
    verbose=verbose,
)




# print(entities)
# raise

# tic = time.perf_counter()
# graph = rdflib.Graph().parse(location)
# toc = time.perf_counter()
# print(f"Graph parse time ({toc - tic:0.4f}s)")
# all_entities = get_all_entities(graph)
def find_neighbours_dbpedia(entities):        
    all_neighbours = []
    entity_to_neighbours = OrderedDict()
    for entity in entities[:1]:
        # kg._get_hops(Vertex(str(entity)))
        ## this only lists outgoing entities, not ingoing, but I think it's okay because when generating walks only
        ## outgoing entities are used
        entity_neighbours, entity_neighbour_relation = get_predicate_object_pairs_from_bdpedia(entity)
        all_neighbours.extend(entity_neighbours)
        entity_to_neighbours[entity] = [entity_neighbours, entity_neighbour_relation]

    all_neighbours = list(set(all_neighbours))

    return all_neighbours, entity_to_neighbours

def get_predicate_object_pairs_from_bdpedia(entity):
    neighbours, neighbours_relations = [], []
    graph = rdflib.Graph()
    qres = graph.query(
        """
        SELECT ?p ?n
        WHERE {
            SERVICE <https://dbpedia.org/sparql> {
                <entity> ?p ?n.
            }
        }
        """.replace('entity', entity)
    )
    for row in qres:
        neighbours.append(str(row[1]))
        neighbours_relations.append(str(row[0]))
    
    return neighbours, neighbours_relations

all_neighbours, entity_to_neighbours = find_neighbours_dbpedia(entities)

print(len(list(entity_to_neighbours.values())[0][1]))
raise


if embeddings_for_all_entities:
    all_embeddings, walks_time, embeddings_fit_time = transformer_fit_transform_with_times(kg, all_entities)

    dic_emb_classes = dict()
    for entity, embeddings in zip(all_entities, all_embeddings):
        dic_emb_classes[entity] = embeddings.tolist()

if aproximate_model:
    all_neighbours, entity_to_neighbours = find_neighbours(graph, entities)

    if not embeddings_for_all_entities:
        ## uses transformer object
        print('here')
        neighbours_embeddings, walks_time, embeddings_fit_time = transformer_fit_transform_with_times(kg, all_neighbours)

        dic_emb_classes = dict()
        for neighbour, neighbour_embeddings in zip(all_neighbours, neighbours_embeddings):
            dic_emb_classes[neighbour] = neighbour_embeddings.tolist()

        all_embeddings = neighbours_embeddings

    embeddings = []
    for key, (entity, [neighbours, _]) in enumerate(entity_to_neighbours.items()):
        entity_neighbours_embeddings = []
        for neighbour in neighbours:
            idx = transformer._entities.index(neighbour)
            # entity_neighbours_embeddings.append(neighbours_embeddings[idx])
            entity_neighbours_embeddings.append(all_embeddings[idx])
        entity_neighbours_embeddings = np.array(entity_neighbours_embeddings)
        entity_neighbours_embeddings_avg = np.average(entity_neighbours_embeddings, 0)
        embeddings.append(entity_neighbours_embeddings_avg.tolist())


else:
    if not embeddings_for_all_entities:
        ## uses transformer object
        entities_embeddings, walks_time, embeddings_fit_time = transformer_fit_transform_with_times(kg, entities)

        all_embeddings = entities_embeddings

    embeddings = []
    for entity in entities:
        idx = transformer._entities.index(entity)
        embeddings.append(all_embeddings[idx])
    
train_embeddings = embeddings[: len(train_entities)]
test_embeddings = embeddings[len(train_entities) :]

# Fit a Support Vector Machine on train embeddings and pick the best
# C-parameters (regularization strength).
param_grid = {"max_depth": [2, 4, 6, 8, 10]}
scoring=['accuracy',
         'f1_weighted',
         'f1_macro',
         'precision_weighted',
         'recall_weighted',
         'precision_macro',
         'recall_macro'
         ]
clf = GridSearchCV(
    # SVC(random_state=RANDOM_STATE), {"C": [10**i for i in range(-3, 4)]}
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid,
    scoring=scoring,
    refit='f1_weighted'
)

tic = time.perf_counter()
clf.fit(train_embeddings, train_labels)
toc = time.perf_counter()
classifier_fit_time = toc - tic
print(f"Fitted classifier model in ({classifier_fit_time:0.4f}s)\n")

# Evaluate the Support Vector Machine on test embeddings.
predictions = clf.predict(test_embeddings)
predictions_proba = clf.predict_proba(test_embeddings)

acc_scr = accuracy_score(test_labels, predictions)
f1_scr_wei = f1_score(test_labels, predictions, average='weighted')
prec_scr_wei = precision_score(test_labels, predictions, average='weighted')
reca_scr_wei = recall_score(test_labels, predictions, average='weighted')
f1_scr_macro = f1_score(test_labels, predictions, average='macro')
prec_scr_macro = precision_score(test_labels, predictions, average='macro')
reca_scr_macro = recall_score(test_labels, predictions, average='macro')

print(
    f"Predicted {len(test_entities)} entities with\n"
    + f"\t{acc_scr * 100 :.4f}% ACCURACY\n"
    + f"\t{f1_scr_wei * 100 :.4f}% F1-WEIGHTED\n"
    + f"\t{prec_scr_wei * 100 :.4f}% PRECISION-WEIGHTED\n"
    + f"\t{reca_scr_wei * 100 :.4f}% RECALL-WEIGHTED\n"
    + f"\t{f1_scr_macro * 100 :.4f}% F1-MACRO\n"
    + f"\t{prec_scr_macro * 100 :.4f}% PRECISION-MACRO\n"
    + f"\t{reca_scr_macro * 100 :.4f}% RECALL-MACRO"
)
print("Confusion Matrix ([[TN, FP], [FN, TP]]):")
print(confusion_matrix(test_labels, predictions))

mean_preds, std_preds = stats_for_preds(predictions_proba)
print("Mean in probability of predicted class:\t\t\t", mean_preds)
print("Standard deviation in probability of predicted class:\t", std_preds)

results_summary = {
    'kg_init_time': kg_init_time,
    'walks_time': walks_time,
    'embeddings_fit_time': embeddings_fit_time,
    'classifier_fit_time': classifier_fit_time,
    'acc_scr': acc_scr,
    'f1_scr_wei': f1_scr_wei,
    'prec_scr_wei': prec_scr_wei,
    'reca_scr_wei': reca_scr_wei,
    'f1_scr_macro': f1_scr_macro,
    'prec_scr_macro': prec_scr_macro,
    'reca_scr_macro': reca_scr_macro,
    'mean_preds': mean_preds,
    'std_preds': std_preds
}


############################################################################### save

transformer.save(os.path.join(current_model_models_path, f'RDF2Vec_{dataset}')) ## save transformer model

dump(clf, os.path.join(current_model_models_path, f'classifier_{dataset}')) ## save node classification model

## save grid search cv results although they are also saved with the joblib.dump
df = pd.DataFrame(clf.cv_results_)
df.to_csv(os.path.join(current_model_models_results_path, 'classifier_cv_results_.csv'))

## save grid search cv best estimator although it is also saved with the joblib.dump
with open(os.path.join(current_model_models_results_path, 'classifier_best_estimator_.json'), 'w', encoding ='utf8') as f: 
        json.dump(str(clf.best_estimator_), f, ensure_ascii = False)

## save results summary for test set
with open(os.path.join(current_model_models_results_path, 'results_summary.json'), 'w', encoding ='utf8') as f: 
        json.dump(results_summary, f, ensure_ascii = False)

## save dictionary with embeddings for each neighbour, save dicionary with neighbours for each entity
if aproximate_model:
    with open(os.path.join(current_model_trained_path, 'neighbours_embeddings.json'), 'w', encoding ='utf8') as f: 
        json.dump(dic_emb_classes, f, ensure_ascii = False)
    with open(os.path.join(current_model_trained_path, 'entity_to_neighbours.json'), 'w', encoding ='utf8') as f: 
        json.dump(entity_to_neighbours, f, ensure_ascii = False)


# from rdflib.namespace import RDFS

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
#         print(p)
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

# G = nx.DiGraph()
# _rdflib_to_networkx_graph(graph, G, calc_weights=False, edge_attrs=lambda s, p, o: {})

# nx.write_gpickle(G, os.path.join(data_path, 'KG.gpickle'))


# G = nx.read_gpickle(os.path.join(data_path, 'KG.gpickle')) ## read KG
# print(list(G.nodes))