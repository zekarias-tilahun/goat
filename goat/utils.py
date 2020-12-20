"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull, Flickr, Yelp

from datasets import LocalDatasets

from urllib import request

import argparse
import time
import sys
import os
import os.path as osp

import numpy as np

import torch


def log(msg="", cr=False, verbose=True):
    """
    Logs a message to a console

    :param msg: The message
    :param cr: If true carriage return will be used
    :param verbose: If false nothing will be logged
    :return:
    """
    if verbose:
        if msg == "":
            print()
        else:
            if cr:
                sys.stdout.write(f"\r{msg}")
                sys.stdout.flush()
            else:
                print(msg)


def decide_config(root, name):
    """
    Create a configuration to download datasets

    :param root: A path to a root directory where data will be stored
    :param name: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    if name == 'cora' or name == 'citeseer' or name == "pubmed":
        root = osp.join(root, "planetoid")
        dataset_class = Planetoid
        params = {"root": root, "name": name}
    elif name == 'computers' or name == "photo":
        root = osp.join(root, name)
        dataset_class = Amazon
        params = {"root": root, "name": name}
    elif name == 'cs' or name == 'physics':
        root = osp.join(root, name)
        dataset_class = Coauthor
        params = {"root": root, "name": name}
    elif name == 'dblp' or name == 'cora-full':
        root = osp.join(root, "full")
        dataset_class = CitationFull
        params = {"root": root, "name": name.split("-")[0]}
    elif name == 'flickr':
        root = osp.join(root, name)
        dataset_class = Flickr
        params = {"root": root}
    elif name == 'yelp':
        root = osp.join(root, name)
        dataset_class = Yelp
        params = {"root": root}
    else:
        dataset_class = LocalDatasets
        params = {"root": root, "name": name}
    return dataset_class, params


def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 100 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        log(s, cr=True)
        if readsofar >= totalsize:
            log()
    else:
        log("read %d\n" % (readsofar,))
        
    

def sample_negative_nodes(graph, dist, source, target, size=1):
    """
    Samples a number of negative nodes from a distribution for the edge (source, target). A negative node is a node
    which is different from both the source and target node and is not connected to neither.

    :param graph: The graph
    :param dist: The node sample distribution table
    :param source: The source node
    :param target: The target node
    :param size: The number of negative nodes
    :return: A list of negative nodes
    """
    negatives = []
    while len(negatives) < size:
        rnd_idx = np.random.randint(0, len(dist), size)[0]
        rnd_node = dist[rnd_idx]
        if source != rnd_node != target and not graph.has_edge(source, rnd_node) and \
                not graph.has_edge(target, rnd_node):
            negatives.append(rnd_node)
    return torch.tensor(negatives, dtype=torch.long)


def sample_negative_pairs(graph, size):
    """
    Samples a specified number of pair of nodes that are not connected by an edge in a graph
    
    :param graph: The graph
    :param size: The number of pairs to sample
    """
    log(f"Sampling {size} negative pairs")
    negative_edges = []
    nodes = list(graph.nodes())
    while len(negative_edges) < size:
        u = np.random.randint(low=0, high=len(nodes), size=1)[0]
        v = np.random.randint(low=0, high=len(nodes), size=1)[0]
        if u != v and not graph.has_edge(u, v):
            negative_edges.append((u, v))
        log(f"{len(negative_edges)}/{size}", cr=True)
    log()
    return torch.tensor(negative_edges, dtype=torch.long)


def collate_function(batch):
    """
    Collates a batch containing a list of input tensors as dictionary

    :param batch: An tuple batch
    :return: A dictionary batch
    """
    batch_ = {
        "source": batch[0],
        "target": batch[1],
        "source_neighborhood": batch[2],
        "target_neighborhood": batch[3],
        "source_mask": batch[4],
        "target_mask": batch[5],
    }
    if len(batch) > 6:
        batch_["negative"] = batch[6]
        batch_["negative_neighborhood"] = batch[7]
        batch_["negative_mask"] = batch[8]
    return batch_


def to(dict_of_tensors, device):
    """
    Places tensors stored in a dictionary on a device
    
    :param dict_of_tensors: A dictionary containing tensors
    :param device: The device
    """
    for key, data in dict_of_tensors.items():
        dict_of_tensors[key] = data.to(device)
    return dict_of_tensors

def init_dirs(dirs):
    """
    Initializes the necessary directory structure
    
    :param dirs: A list of directories to be created
    """
    log("Initializing directories ...")
    for dir_ in dirs:
        dirs_ = dir_.split("/")
        path = ''
        for d in dirs_:
            if d == '.' or d == '..':
                path = d
            else:
                path = osp.join(path, d)
                os.makedirs(path, exist_ok=True)
                
                
def files_exist_in(files, in_dir):
    """
    Checks if all the files are in a directory
    
    :param files: The files
    :param in_dir: The directory:
    """
    return all([osp.exists(osp.join(in_dir, file)) for file in files])


def scale_min_max(x, new_max, new_min):
    """
    Utility for scaling a number to a new range

    :param x: The number
    :param new_max: The maximum value of the new range
    :param new_min: The minimum value of the new range
    :return:
    """
    mn = x.min()
    mx = x.max()
    x_std = (x - mn) / (mx - mn + 0.000000001)
    return x_std * (new_max - new_min) + new_min


def expand_if(arr, dim, cond):
    """
    Conditionally expands a numpy array along a given dim (axis)
    
    :param arr: The array
    :param dim: The dim (axis) along which the array will be expanded
    :param cond: The condition
    :return a numpy array
    """
    if cond:
        return np.expand_dims(arr, dim)
    return arr


def visualize_attention_weights(path, neighborhood, weights, node_id, node_id2name=None):
    """
    Utility to visualize attention weights associated with the neighborhood of a given node

    :param path: A path to save the visualization
    :param neighborhood: The neighborhood of the given node
    :param weights: The attention weights of the neighbors
    :param node_id: The id of the given node
    :param node_id2name: a node id to name mapping, if any.
    :return:
    """
    weights_ = scale_min_max(weights, new_max=1, new_min=0)
    
    def format_output(j):
        nbr = neighborhood[j] if node_id2name is None else node_id2name[neighborhood[j]]
        if (j + 1) % 15 == 0:
            return f"<br><span style='background-color: rgba(255, 0, 0, {weights_[j]})'>{nbr}</span>"
        else:
            return f"<span style='background-color: rgba(255, 0, 0, {weights_[j]})'>{nbr}</span>"
        
    with open(path, 'w') as f:
        nodes = set()
        output = '<p>Attention weight: </p> ' + ' '.join(format_output(i) for i in range(neighborhood.shape[0]))
        node_label = f'<br>Node: {node_id if node_id2name is None else node_id2name[node_id]}'
        f.write(f"{node_label}{output}<br><br>")


def parse_train_args():
    """
    Parses command line arguments required for training GOAT
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="../data", help="A path to a directory to put datasets. Default is ./data")
    parser.add_argument("--name", "-n", type=str, default="cora", help="The name of a dataset. Default is cora")
    parser.add_argument("--dim", "-d", type=int, default=200, help="The size of the embedding dimension. Default is 200")
    parser.add_argument("--num-nbrs", "-nbr", type=int, default=100, help="The number of neighbors (N in the paper). Default is 100.")
    parser.add_argument("--num-neg", "-neg", type=int, default=2, help="The number of negative samples for contrastive loss. Default is 2")
    parser.add_argument("--lr", '-l', type=float, default=0.0001, help="The learning rate. Default is .0001")
    parser.add_argument("--dropout", "-do", type=float, default=0.5, help="The dropout rate. Default is .5")
    parser.add_argument("--epoch", '-e', type=int, default=10, help="The number of epochs. Default is 10.")
    parser.add_argument("--batch", '-b', type=int, default=512, help="Batch size. Default is 512")
    parser.add_argument("--train-prop", "-tp", type=float, default=.5, help="The fraction of edges to be used for training. Default is .5")
    return parser.parse_args()


def parse_eval_args():
    """
    Parses command line arguments required for evaluating GOAT
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="../data", help="A path to a directory to put datasets")
    parser.add_argument("--name", "-n", type=str, default="cora", help="The name of a dataset")
    parser.add_argument("--task", "-t", type=str, default="link_prediction", help="The type of evaluation task. Default is link_prediction")
    parser.add_argument("--nodes", '-nd', type=int, nargs="*", help="Two node ids (source and target) for visualization")
    parser.add_argument("--epoch", '-e', type=int, default=10, help="The particular epoch to be evaluated. Default is 10")
    parser.add_argument("--batch", '-b', type=int, default=512, help="Batch size. Default is 512.")
    parser.add_argument("--train-prop", "-tp", type=float, default=.5, help="The particular training proportion to be evaluated. Default is .5")
    return parser.parse_args()


class Trainer:

    def __init__(self, config):
        """
        A utility class for training GOAT. Expects all the necessary configurations provided via 
        the command line arguments.
        
        :param config: Configurations (arguments)
        """
        self._config = config

    def fit(self, model, train_loader, dev_loader=None):
        """
        Trains a model using a data provided from a training loader

        :param model: The model
        :param train_loader: Training data loader
        :param dev_loader: Validation data loader
        :return: None
        """
        config = self._config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _, params = decide_config(root=config.root, name=config.name)
        root = params['root']
        path = osp.join(root, params['name']) if "name" in params else root
        model_dir = osp.join(path, "models")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
        model.to(device)
        model.train()
        for epoch in range(config.epoch):
            start = time.time()
            for batch_counter, batch in train_loader:
                
                _, _, _, loss = model(**to(batch, device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                msg = "Epoch: {:03d}/{:03d} batch: {:04d}/{:04d} training loss: {:.4f} "
                log(msg.format(epoch + 1, config.epoch, batch_counter + 1, len(train_loader), loss.data), cr=True)

            if dev_loader is not None:
                val_auc = []
                val_loss = []
                for batch_counter, batch in dev_loader:
                    src_ctx_rep, trg_ctx_rep, neg_ctx_rep, loss = model(**to(batch, device))
                    auc = model.evaluate(src_ctx_rep, trg_ctx_rep, neg_ctx_rep)
                    val_loss.append(float(loss.data))
                    val_auc.append(float(auc))

                delta = time.time() - start
                msg = "validation loss: {:.4f} validation auc {:.4f}"
                log(msg.format(np.mean(val_loss), np.mean(val_auc)))
            path = osp.join(model_dir, f"goat.ep{epoch + 1}.pt")
            torch.save(model.state_dict(), path)
