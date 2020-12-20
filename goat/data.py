"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""

from utils import sample_negative_nodes, sample_negative_pairs, collate_function, log, decide_config, init_dirs

from torch_geometric.utils import to_networkx
from torch.utils import data

from multiprocessing import cpu_count
from urllib import request

import os.path as osp

import networkx as nx
import numpy as np

import torch


class Data:

    def __init__(self, args):
        """
        A data object used to store the actual content or data of an InputDataset object.

        :param args: The necessary arguments required to create the data (root and name are the expected ones).
        """
        self._args = args
        self._process()
        
    def _init(self):
        self._dataset_class, self._params = decide_config(root=self._args.root, name=self._args.name)
        root = self._params["root"]
        self._args.name = self._args.name.split("-")[0]
        if "name" in self._params:
            dirs = [osp.join(root, self._args.name, "raw"),
                    osp.join(root, self._args.name, "processed"),
                    osp.join(root, self._args.name, "results"), 
                    osp.join(root, self._args.name, "models")]
        else:
            dirs = [osp.join(root, "raw"),
                    osp.join(root, "processed"),
                    osp.join(root, "results"),
                    osp.join(root, "models")]
        init_dirs(dirs)
        
    def _process(self):
        """
        Either processes or loads the processed file

        :return:
        """
        self._init()
        args = self._args
        path = osp.join(self._params["root"], self._params["name"]) if "name" in self._params else self._params["root"]
        path = osp.join(path, "processed", self.processed_files[0])
        if not osp.exists(path):
            log("Processing ...")
            self.process()
        else:
            log(f"Loading processed file from {path} ...")
            processed_data = torch.load(path)
            self.x = processed_data[0]
            self.edges = processed_data[1]
            self.graph = processed_data[2]
            self.neighborhood = processed_data[3]
            self.neighborhood_mask = processed_data[4]
            self.train_partition = processed_data[5]
            self.dev_partition = processed_data[6]
            self.test_partition = processed_data[7]
            self.negative_pairs = processed_data[8]
            self.y = processed_data[9]
            self.dim = processed_data[10]
            self.num_neg = processed_data[11]
            self.num_nodes = self.graph.number_of_nodes()
            self.num_edges = self.graph.number_of_edges()
            self.node_dist = [node for node, degree in self.graph.degree for _ in range(int(degree * .75))]

    def _build_neighborhood(self, graph):
        """
        Builds a neighborhood sample for each node in a graph

        :param graph: The graph
        :return:
        """
        log("Building neighborhood")
        args = self._args
        self.neighborhood = torch.zeros(self.num_nodes, args.num_nbrs, dtype=torch.long)
        neg_inf = -99999999.
        self.neighborhood_mask = torch.tensor([[neg_inf] * args.num_nbrs] * self.num_nodes)
        for node in graph.nodes():
            neighbors = list(nx.all_neighbors(graph, node))
            if len(neighbors) < args.num_nbrs:
                neighbors = neighbors
            elif len(neighbors) > args.num_nbrs:
                neighbors = np.random.choice(neighbors, size=args.num_nbrs)
            self.neighborhood[node, :len(neighbors)] = torch.tensor(neighbors)
            self.neighborhood_mask[node, :len(neighbors)] = torch.zeros(len(neighbors))

    @property
    def processed_files(self):
        return [f"data.prop.{int(self._args.train_prop * 100)}.pt"]

    def process(self):
        """
        Processes a data

        :return:
        """
        args = self._args

        dataset = self._dataset_class(**self._params)

        data_ = dataset.data
        self.graph = to_networkx(data_).to_undirected()
        log(f"{args.name} dataset has {self.graph.number_of_nodes()} number of nodes")
        log(f"{args.name} dataset has {self.graph.number_of_edges()} number of edges")
        
        self.x = None
        self.y = None
        if hasattr(data_, "x"):
            self.x = data_.x
        if hasattr(data_, "y"):
            self.y = data_.y
            
        self.edges = data_.edge_index.transpose(0, 1)
        self.num_edges = self.edges.shape[0]
        self.num_nodes = data_.num_nodes
        self.num_nodes = self.num_nodes[0] if isinstance(self.num_nodes, list) else self.num_nodes

        perm = torch.randperm(self.edges.shape[0])
        self.edges = self.edges[perm]
        train_size = int(self.edges.shape[0] * args.train_prop)
        test_size = int(self.edges.shape[0] * (1 - args.train_prop))
        self.negative_pairs = sample_negative_pairs(graph=self.graph, size=test_size)
        self.train_partition = np.arange(train_size)
        self.test_partition = np.arange(train_size, self.edges.shape[0])
        dev_size = int(self.train_partition.shape[0] * .2)
        self.train_partition = np.arange(train_size - dev_size)
        self.dev_partition = np.arange(train_size - dev_size, train_size)

        log(f"Number of training points: {self.train_partition.shape[0]}")
        log(f"Number of dev points: {self.dev_partition.shape[0]}")
        log(f"Number of test points: {self.test_partition.shape[0]}")

        
        self._build_neighborhood(self.graph)
        self.node_dist = [node for node, degree in self.graph.degree for _ in range(int(degree * .75))]
        
        path = osp.join(self._params["root"], self._params["name"]) if "name" in self._params else self._params["root"]
        
        path = osp.join(path, "processed", self.processed_files[0])

        log(f"Saving processed file to {path} ...")
        torch.save((self.x, self.edges, self.graph, self.neighborhood, self.neighborhood_mask, self.train_partition, 
                    self.dev_partition, self.test_partition, self.negative_pairs, self.y, args.dim, args.num_neg), path)


class InputDataset(data.Dataset):

    def __init__(self, input_data, num_neg=None, partition=None):
        """
        A Dataset object that will be directly fed to the GOAT model, and it is indexable.
        data points are indexed from a particular partition of the input data depending on
        the parition name.

        :param input_data:  The input data
        :param num_neg: The number of negative samples to be generated for each pair or edge indexed 
                        from a specified partition. If 0 or None, sampling will be skipped
        :param partition: The partition name ("train", "dev", "test", None). Defaul None, no partition
        """
        self._data = input_data
        self._num_neg = num_neg
        if partition == "train":
            self._data_partition = input_data.train_partition
        elif partition == "dev":
            self._data_partition = input_data.dev_partition
        elif partition == "test":
            self._data_partition = input_data.test_partition
        else:
            self._data_partition = np.arange(self._data.num_edges)

    def __len__(self):
        return self._data_partition.shape[0]

    def __getitem__(self, index):
        """
        Indexes the edge associated with the index and builds training input containing
        the source, target and negative node ids, the neighborhood and mask assiated with
        the node ids.
        
        :param index: The index associated to the
        :return:
        """
        edge_index = self._data_partition[index]
        source, target = self._data.edges[edge_index]
        
        source_neighborhood = self._data.neighborhood[source]
        target_neighborhood = self._data.neighborhood[target]

        source_mask = self._data.neighborhood_mask[source]
        target_mask = self._data.neighborhood_mask[target]
        if self._num_neg:
            negative = sample_negative_nodes(graph=self._data.graph, dist=self._data.node_dist,
                                        source=source, target=target, size=self._num_neg)
            negative_neighborhood = self._data.neighborhood[negative]
            negative_mask = self._data.neighborhood_mask[negative]
            return (source, target, source_neighborhood, target_neighborhood, source_mask, target_mask, 
                    negative, negative_neighborhood, negative_mask) 
        
        return source, target, source_neighborhood, target_neighborhood, source_mask, target_mask
            

class DataLoader:

    def __init__(self, dataset, batch_size, num_workers=-1):
        """
        A wrapper class for the PyTorch DataLoader class

        :param dataset: The dataset
        :param batch_size: The number of training points to be grouped in a batch.
        :param num_workers: The number of parallel workers. Default -1, uses all the available cores
        """
        num_workers = cpu_count() if num_workers == -1 else num_workers
        self._loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    def __iter__(self):
        for batch_counter, batch in enumerate(self._loader):
            yield batch_counter, collate_function(batch)

    def __len__(self):
        return len(self._loader)
