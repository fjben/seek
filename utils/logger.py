import sys

class Logger(object):
    def __init__(self):
        self.console = sys.stdout
        self.log = open('node_classifier/tmp/train_embeddings.log', 'w')


    def write(self, message):
        self.console.write(message)
        self.log.write(message)


    def flush(self):
        pass
