import argparse
import json
from multiprocessing import cpu_count
import random
import sys
import time
import os

from datetime import datetime


# parser = argparse.ArgumentParser(description="description")
# parser.add_argument("--set_workers_for_reproducibility",
#                     action="store_true",
#                     help="To set workers parameter equal to 1 so the results are reproducible.")
# args = parser.parse_args()
# set_workers_for_reproducibility = args.set_workers_for_reproducibility


# if set_workers_for_reproducibility:
#     workers = 1
# else:
#     workers = cpu_count()

state = random.getstate()

reproducibility_parameters_file = 'node_classifier/tmp/reproducibility_parameters.txt'

rand_parameters = ['ranHashSeed', 'ranSeed', 'ranState']

with open(reproducibility_parameters_file, "w") as f:
    for rand_param in rand_parameters:
        random.seed(datetime.now().timestamp())
        f.write(f'{rand_param}\n' + str(random.randrange(0, 4294967295)) + '\n')
    f.write('workers\n' + str(1) + '\n')
    # f.write('workers\n' + str(workers) + '\n')

random.setstate(state)
