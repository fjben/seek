##### to improve to a script with arguments in the command line


import os

import rdflib


## this was not working for this BGS dataset
rdf_data_path = '/home/fpaulino/SEEK/seek/node_classifier/data/BGS'

g = rdflib.Graph()

g.parse(os.path.join(rdf_data_path, 'completeDataset.nt'), format='nt')

haslithogenesis = rdflib.term.URIRef("http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis")

rels = set(g.predicates())

g.remove((None, haslithogenesis, None))

g.serialize(destination=os.path.join(rdf_data_path, 'completeDataset_stripped.nt'), format='nt')

g.close()