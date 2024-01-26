import argparse
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import os
import json
import rdflib
from rdflib.namespace import RDF, OWL, RDFS
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))
from OpenKE import config
from OpenKE import models




#####################
##    Functions    ##
#####################

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: path-like object representing a file system path;
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)



def construct_model(dic_nodes, dic_relations, list_triples, path_output, model_embedding):
    """
    Construct and embedding model and compute embeddings.
    :param dic_nodes: dictionary with KG nodes and respective ids;
    :param dic_relations: dictionary with type of relations in the KG and respective ids;
    :param list_triples: list with triples of the KG;
    :param path_output: OpenKE path;
    :param n_embeddings: dimension of embedding vectors;
    :param models_embeddings: list of embedding models;
    :return: write a json file with the embeddings for all nodes and relations.
    """
    entity2id_file = open(path_output + "entity2id.txt", "w")
    entity2id_file.write(str(len(dic_nodes))+ '\n')
    for entity , id in dic_nodes.items():
        entity = entity.replace('\n' , ' ')
        entity = entity.replace(' ' , '__' )
        entity = entity.encode('utf8')
        entity2id_file.write(str(entity) + '\t' + str(id)+ '\n')
    entity2id_file.close()

    relations2id_file = open(path_output + "relation2id.txt", "w")
    relations2id_file.write(str(len(dic_relations)) + '\n')
    for relation , id in dic_relations.items():
        relation  = relation.replace('\n' , ' ')
        relation = relation.replace(' ', '__')
        relation = relation.encode('utf8')
        relations2id_file.write(str(relation) + '\t' + str(id) + '\n')
    relations2id_file.close()

    train2id_file = open(path_output + "train2id.txt", "w")
    train2id_file.write(str(len(list_triples)) + '\n')
    for triple in list_triples:
        train2id_file.write(str(triple[0]) + '\t' + str(triple[2]) + '\t' + str(triple[1]) + '\n')
    train2id_file.close()

    # Input training files from data folder.
    con = config.Config()
    con.set_in_path(path_output)
    con.set_dimension(vector_size)

    print('--------------------------------------------------------------------------------------------------------------------')
    print('MODEL: ' + model_embedding)

    # Models will be exported via tf.Saver() automatically.
    con.set_export_files(path_output + model_embedding + "/model.vec.tf", 0)
    # Model parameters will be exported to json files automatically.
    con.set_out_files(path_output + model_embedding + "/embedding.vec.json")

    if model_embedding == 'ComplEx':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        #Set the knowledge embedding model
        con.set_model(models.ComplEx)

    elif model_embedding == 'distMult':
        con.set_work_threads(8)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.5)
        con.set_lmbda(0.05)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.DistMult)

    elif model_embedding == 'HOLE':
        con.set_work_threads(4)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(0.2)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.HolE)

    elif model_embedding == 'RESCAL':
        con.set_work_threads(4)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.1)
        con.set_bern(0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("Adagrad")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.RESCAL)

    elif model_embedding == 'TransD':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_margin(4.0)
        con.set_bern(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransD)

    elif model_embedding == 'TransE':
        con.set_work_threads(8)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransE)

    elif model_embedding == 'TransH':
        con.set_work_threads(8)
        con.set_train_times(500)
        con.set_nbatches(100)
        con.set_alpha(0.001)
        con.set_margin(1.0)
        con.set_bern(0)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransH)

    elif model_embedding == 'TransR':
        con.set_work_threads(8)
        con.set_train_times(1000)
        con.set_nbatches(100)
        con.set_alpha(1.0)
        con.set_lmbda(4.0)
        con.set_margin(1)
        con.set_ent_neg_rate(1)
        con.set_rel_neg_rate(0)
        con.set_opt_method("SGD")
        # Initialize experimental settings.
        con.init()
        # Set the knowledge embedding model
        con.set_model(models.TransR)

    # Train the model.
    con.run()



def write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes):
    """
    Writing embeddings.
    :param path_model_json: json file with the embeddings for all nodes and relations;
    :param path_embeddings_output: embedding file path;
    :param ents: list of entities for which embeddings will be saved;
    :param dic_nodes: dictionary with KG nodes and respective ids;
    :return: writes an embedding file with format "{ent1:[...], ent2:[...]}".
    """
    with open(path_model_json, 'r') as embeddings_file:
        data = embeddings_file.read()
    embeddings = json.loads(data)
    embeddings_file.close()

    ensure_dir(path_embeddings_output)
    # with open(path_embeddings_output, 'w') as file_output:
    #     file_output.write("{")
    #     first = False
    #     for i in range(len(ents)):
    #         ent = ents[i]
    #         if first:
    #             if "ent_embeddings" in embeddings:
    #                 file_output.write(", '%s':%s" % (str(ent), str(embeddings["ent_embeddings"][dic_nodes[str(ent)]])))
    #             else:
    #                 file_output.write(
    #                     ", '%s':%s" % (str(ent), str(embeddings["ent_re_embeddings"][dic_nodes[str(ent)]])))
    #         else:
    #             if "ent_embeddings" in embeddings:
    #                 file_output.write("'%s':%s" % (str(ent), str(embeddings["ent_embeddings"][dic_nodes[str(ent)]])))
    #             else:
    #                 file_output.write(
    #                     "'%s':%s" % (str(ent), str(embeddings["ent_re_embeddings"][dic_nodes[str(ent)]])))
    #             first = True
    #     file_output.write("}")
    # file_output.close()

    dic_emb_classes = dict()
    for ent in ents:
        if "ent_embeddings" in embeddings:
            dic_emb_classes[ent] = embeddings["ent_embeddings"][dic_nodes[str(ent)]]
        else:
            dic_emb_classes[ent] = embeddings["ent_re_embeddings"][dic_nodes[str(ent)]]
    with open(path_embeddings_output + '.json', 'w', encoding ='utf8') as f: 
        json.dump(dic_emb_classes, f, ensure_ascii = False)




########################################################################################################################
##############################################        Call Embeddings       ############################################
########################################################################################################################

def construct_kg(kg_file_path):
    g = rdflib.Graph()
    g.parse(kg_file_path) ## , format='xml')
    return g

def buildIds(g):
    """
    Assigns ids to KG nodes and KG relations.
    :param g: knowledge graph;
    :return: 2 dictionaries and one list. "dic_nodes" is a dictionary with KG nodes and respective ids. "dic_relations" is a dictionary with type of relations in the KG and respective ids. "list_triples" is a list with triples of the KG.
    """
    dic_nodes = {}
    id_node = 0
    id_relation = 0
    dic_relations = {}
    list_triples = []

    for (subj, predicate, obj) in g:
        if str(subj) not in dic_nodes:
            dic_nodes[str(subj)] = id_node
            id_node = id_node + 1
        if str(obj) not in dic_nodes:
            dic_nodes[str(obj)] = id_node
            id_node = id_node + 1
        if str(predicate) not in dic_relations:
            dic_relations[str(predicate)] = id_relation
            id_relation = id_relation + 1
        list_triples.append([dic_nodes[str(subj)], dic_relations[str(predicate)], dic_nodes[str(obj)]])

    return dic_nodes, dic_relations, list_triples

def run_embedddings(dataset, kg_file_path, entities_file_path, entities, vector_size, path_output, embedding_models, path_openke):

    g = construct_kg(kg_file_path)
    dic_nodes, dic_relations, list_triples = buildIds(g)

    # ents = [line.strip() for line in open(entities_file_path).readlines()]
    ents = entities

    for embedding_model in embedding_models:
        construct_model(dic_nodes, dic_relations, list_triples, path_openke, embedding_model)
        path_model_json = path_openke + embedding_model + "/embedding.vec.json"
        path_embeddings_output = path_output + '/' + embedding_model + '/' + dataset + '_' + embedding_model + '_' + str(vector_size)
        write_embeddings(path_model_json, path_embeddings_output, ents, dic_nodes)


def get_all_entities(graph):
    # all_entities = [entity for entity in graph.subjects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))]
    all_subjects = {str(entity) for entity in graph.subjects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))}
    all_objects = {str(entity) for entity in graph.objects(unique=True) if not isinstance(entity, (rdflib.Literal, rdflib.BNode))}
    all_entities = all_subjects.union(all_objects)
    print(len(all_entities))

    return list(all_entities)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="description")
    parser.add_argument("--dataset",
                        type=str,
                        choices=['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE'],
                        help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")
    args = parser.parse_args()
    dataset = args.dataset

    #################################### Parameters ####################################
    embedding_models = ['TransE', 'TransH', 'distMult', 'ComplEx']
    vector_size = 100

    # dataset = 'AIFB'
    # dataset = 'MUTAG'
    # dataset = 'AM_FROM_DGL'
    # dataset = 'MDGENRE'
    
    data_path = f'node_classifier/data/{dataset}'
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        ds_metadata = json.load(f)
    location = os.path.join(data_path, ds_metadata['rdf_file'])

    graph = rdflib.Graph().parse(location)
    all_entities = get_all_entities(graph)



    # kg_file_path = "Data/PPI/go-basic-annots.owl" # kg file
    kg_file_path = location
    entities_file_path = "Data/PPI/Prots_v11(score950).txt" # file with entities for wich we wanna save the embeddings
    path_output = "Embeddings/node_classification" # folder where embeddings will be saved
    # path_openke = "./OpenKE/" # folder of OpenKE code
    path_openke = "/home/fpaulino/SEEK/OpenKE" # folder of OpenKE code
    run_embedddings(dataset, kg_file_path, entities_file_path, all_entities, vector_size, path_output, embedding_models, path_openke)
