import numpy as np
import argparse
import random
import sys

VERBOSE = False
ERROR = 'ERROR'
INFO = 'INFO'
PROG = 'PROG'
WARN = 'WARN'


def read_context_embedding(path):
    """
    Reads context embeddings of nodes. Each node can have one or more embedding

    :param path: A path to context embedding of nodes
    :return: A dictionary of node->context_embeddings
    """
    log('Reading context embedding from {}'.format(path))
    embeddings = {}
    with open(path, 'rb') as f:
        for line in f:
            ln = line.strip().split()
            node = int(ln[0])
            emb = [float(val) for val in ln[1:]]
            if node in embeddings:
                embeddings[node].append(emb)
            else:
                embeddings[node] = [emb]
                
    for node in embeddings:
        embeddings[node] = np.vstack(embeddings[node])
    return embeddings
            
            
def read_global_embedding(path):
    """
    Reads global embeddings of nodes

    :param path:
    :return:
    """
    log('Reading embedding from {}'.format(path))
    embeddings = {}
    with open(path, 'rb') as f:
        for line in f:
            ln = line.strip().split()
            node = int(ln[0])
            embeddings[node] = list(map(float, ln[1:]))
    return embeddings


def log(msg, cr=False, level=INFO):
    """
    Message logger

    :param msg: The message
    :param cr: Whether to carriage return or not
    :param level: log level
    :return:
    """
    global VERBOSE
    if VERBOSE:
        if cr:
            sys.stdout.write(f'\r{level}: {msg}')
            sys.stdout.flush()
        else:
            print(f"{level}: {msg}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default='./data/cora/graph.txt', type=str,
                        help='Path to the graph file')
    parser.add_argument('--fmt', type=str, default='edgelist',
                        help="Format ('edgelist-Default', 'adjlist') of the input graph file. ")
    parser.add_argument('--output-dir', type=str, default="./data/cora/outputs/",
                        help='Path to save outputs, mainly the embedding file.')
    parser.add_argument('--dim', type=int, default=200,
                        help='Embedding dimension. Default is 200')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of training epochs. Default is 100')
    parser.add_argument('--tr-rate', type=float, default=.15,
                        help='Use only tr-rate fraction of the edges for train-set, a value in (0, 1]. Default is 0.15')
    parser.add_argument('--dev-rate', type=float, default=0.2,
                        help='Use dev-rate fraction of the training set for dev-set, a value in [0, 1). Default is 0.2')
    parser.add_argument('--directed', type=int, default=1,
                        help='Whether the graph is directed. Default is 1')
    parser.add_argument('--learning-rate', type=float, default=.0001,
                        help='The learning rate. Default is 0.0001')
    parser.add_argument('--dropout-rate', type=float, default=.5,
                        help='The dropout rate. Default is 0.5')
    parser.add_argument('--nbr-size', type=int, default=100,
                        help='The maximum neighborhood size. Default is 100')
    parser.add_argument('--workers', type=int,
                        default=8, help="Turn logging on or off")
    parser.add_argument('--verbose', type=int,
                        default=1, help="Turn logging on or off")
    return parser.parse_args()
