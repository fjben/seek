

from collections import OrderedDict

import rdflib

entity = 'http://dbpedia.org/resource/Vancouver'
g = rdflib.Graph()
qres = g.query(
    # """
    # SELECT ?s
    # WHERE {
    #   SERVICE <https://dbpedia.org/sparql> {
    #     ?s a ?o .
    #   }
    # }
    # LIMIT 3
    # """
    """
    # SELECT ?p ?n
    # WHERE {
    #   SERVICE <https://dbpedia.org/sparql> {
    #     <http://dbpedia.org/resource/Vancouver> ?p ?n.
    #   }
    # }
    # LIMIT 3
    # """
    """
    SELECT ?p ?n
    WHERE {
      SERVICE <https://dbpedia.org/sparql> {
        <entity> ?p ?n.
      }
    }
    LIMIT 3
    """.replace('entity', entity)
)

for row in qres:
    print(row)