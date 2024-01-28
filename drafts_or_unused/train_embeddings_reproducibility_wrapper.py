import random
from subprocess import call, run

run("conda activate seek_train_embeddings_lis5", shell=True, check=True)

cmd = ['python3', 'node_classifier/train_embeddings.py']

hashseed = str(0)
print('\nhashseed', hashseed)
call(cmd, env={'PYTHONHASHSEED': hashseed})