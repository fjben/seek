##### new

##### description: main purpose is to convert .nt files to .n3, because with .n3 the results are deterministic as
##### expected when taking all the steps that ensure determinism. Even with those steps in place, with .nt the results
##### are still not deterministic

##### example use in the command line:
##### python3 dataset_utils/convert_rdf_format.py --rdf_file_path data/AIFB/rdf_data/aifb_fixed_complete_stripped.nt --rdf_target_format .n3

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--rdf_file_path",
                    type=str,
                    help="Path where to find the rdf model to convert")

parser.add_argument("--rdf_target_format",
                    type=str,
                    help="Format to which to convert the file")

args = parser.parse_args()
rdf_file_path = args.rdf_file_path
rdf_target_format = args.rdf_target_format


from rdflib import Graph

g = Graph()
g.parse(rdf_file_path)

parent_filename = rdf_file_path.split('.')[0]
filename = parent_filename + rdf_target_format
g.serialize(destination=filename)