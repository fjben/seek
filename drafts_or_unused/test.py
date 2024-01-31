

import json
import os

from collections import Counter, OrderedDict

import pandas as pd

import rdflib


dataset='AIFB'

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

def get_all_entities(graph):
    # all_entities = [entity for entity in graph.subjects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))]
    all_subjects = {str(entity) for entity in graph.subjects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))}
    all_objects = {str(entity) for entity in graph.objects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))}
    all_entities = all_subjects.union(all_objects)
    print(len(all_entities))

    return list(all_entities)

graph = rdflib.Graph().parse(location)

all_entities = get_all_entities(graph)

all_entities_except_entities = list(set(all_entities).difference(set(entities)))
print(len(all_entities_except_entities))

def find_neighbours(graph, entities):        
    # Print the number of "triples" in the Graph
    print(f"Graph graph has {len(graph)} statements.")
    # Prints: Graph g has 86 statements.
    all_neighbours = []
    entity_to_neighbours = OrderedDict()
    total_count = []
    single_neighbours = []
    relations_left_out = []
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
            count = 0
            for s, p, o in entitygraph:
                if isinstance(o, (rdflib.Literal, rdflib.BNode)): ## ignoring literals
                    # print('------------------------------NOT IN NEIGHBOURS', o)
                    # print(o, str(p))
                    relations_left_out.append(str(p))
                    continue
                # print('IN NEIGHBOURS', o)
                count += 1
                all_neighbours.append(str(o))
                entity_neighbours.append(str(o))
                entity_neighbour_relation.append(str(p))
            if count <= 1:
                single_neighbours.append(entity_neighbours)
            entity_to_neighbours[str(s)] = [entity_neighbours, entity_neighbour_relation]
            total_count.append(count)

    all_neighbours = list(set(all_neighbours))

    return all_neighbours, entity_to_neighbours, total_count, single_neighbours, relations_left_out

graph = rdflib.Graph().parse(location)

## all train test entities
all_neighbours, entity_to_neighbours, total_count, single_neighbours, relations_left_out = find_neighbours(graph, entities)

# print(total_count)
# print(single_neighbours)

all_relations = []
for key, [entity_neighb, entity_neighb_rel] in entity_to_neighbours.items():
    all_relations.extend(entity_neighb_rel)

cnt = Counter(all_relations)
print(cnt)

cnt_out = Counter(relations_left_out)
print(cnt_out)

## all the other entities
all_neighbours, entity_to_neighbours, total_count, single_neighbours, relations_left_out = find_neighbours(graph, all_entities_except_entities)

all_relations = []
for key, [entity_neighb, entity_neighb_rel] in entity_to_neighbours.items():
    all_relations.extend(entity_neighb_rel)

cnt = Counter(all_relations)
print(cnt)

cnt_out = Counter(relations_left_out)
print(cnt_out)