# coding=utf-8
from collections import namedtuple
import networkx as nx
import numpy as np
import random
import os

import torch
import torch.utils.data as td


import helper

random.seed(40)
np.random.seed(40)


def source_targets(edge_list):
    """
    Splits (unzip) an edge list into a tuple of list of sources and list of targets.

    :param edge_list: Edge list

    :return: A tuple
    """
    return [list(nodes) for nodes in zip(*edge_list)]


def _generate_mask(tensor):
    """
    Generates a mask for a tensor.  The mask will be generated according to the following rule

            mask[i, j] = 0 if tensor[i, j] != 0
            mask[i, j] = -∞ if tensor[i, j] == 0

    :param tensor: Input tensor

    :return: A mask tensor
    """

    neg_inf = -99999999.
    mask = (tensor != 0)
    mask = mask.astype('float')
    mask[mask == 0] = neg_inf
    mask[mask != neg_inf] = 0
    return mask


def _sample_neighborhood(nbr_size, graph):
    """
    Samples a fixed neighborhood per node, according to a neighborhood size criteria.
    For nodes with less or equal neighbors to the criteria, all will be considered, otherwise
    a fixed number of nodes will be sampled at random.

    :param nbr_size: The neighborhood size
    :param graph: The graph
    :return: A N-by-nbr_size neighborhood matrix, where N is the number of nodes.
    """
    nodes = list(graph.nodes())
    neighborhood_matrix = np.zeros(shape=(len(nodes), nbr_size))
    for n in nodes:
        neighbors = list(set(nx.all_neighbors(graph, n)) | {n})

        # In the following line, we increase the indices of the neighborhood nodes for the purpose of masking.
        # This is because we use index 0 as a padding index of global embedding lookup inside Goat Model.
        arr = np.array(neighbors) + 1
        if arr.shape[0] > nbr_size:
            arr = np.random.choice(arr, size=nbr_size, replace=False)
        neighborhood_matrix[n, :arr.shape[0]] = arr
    return neighborhood_matrix
    
    
def split_graph(g, train_rate):
    """
    Splits the edges of a graph into training and test sets according to a given ratio

    :param g: The graph
    :param train_rate: The training ratio
    :return: A tuple of training and test edge sets, where each set by itself is a tuple
    """
    edges = list(g.edges())
    train_size = int(len(edges) * train_rate)
    random.shuffle(edges)
    train_edges = edges[:train_size]
    test_edges = edges[train_size:]
    return source_targets(train_edges), source_targets(test_edges)


def _generate_input_batches(neighborhood_matrix, mask_matrix, edges, batch_size=64):
    """
    Prepares model inputs and organizes them into batches

    :param neighborhood_matrix:
    :param mask_matrix:
    :param edges:
    :param batch_size:
    :return:
    """
    batches = []
    size = edges.shape[0]
    for start in range(0, size, batch_size):
        end = start + batch_size if size - start > batch_size else size
        src, trg, neg = edges[start:end, 0], edges[start:end, 1], edges[start:end, 2]
        batch = {'source': src, 'target': trg, 'negative': neg,
                 'source_neighborhood': neighborhood_matrix[src], 'target_neighborhood': neighborhood_matrix[trg],
                 'negative_neighborhood': neighborhood_matrix[neg], 'source_mask': mask_matrix[src],
                 'target_mask': mask_matrix[trg],
                 'negative_mask': mask_matrix[neg]}
        batches.append(namedtuple('Batch', batch.keys())(*batch.values()))
    return batches


class RawData:

    def __init__(self, args, train_graph=None, test_graph=None):
        self.train_graph = train_graph
        self.test_graph = test_graph
        self._args = args
        self._hold_out = args.tr_rate < 1.
        self.use_dev = args.dev_rate > 0
        self._read_graph()
        self._train_test_split()
        self._sample_neighborhood()
        self._process_edges()

    def _read_graph(self):
        """
        Reads or initializes a graph.

        :return:
        """
        args = self._args
        if self.train_graph is None or self.test_graph is None:
            helper.log(f'Reading graph from {args.input}')
            self._reader = nx.read_adjlist if args.fmt == 'adjlist' else nx.read_edgelist
            self._creator = nx.DiGraph if args.directed else nx.Graph
            self.graph = self._reader(path=args.input, create_using=self._creator, nodetype=int)
            self.num_nodes = self.graph.number_of_nodes()
            self.num_edges = self.graph.number_of_edges()
        else:
            self.graph = nx.union(self.train_graph, self.test_graph)

    def _train_test_split(self):
        """
        Splits the data into training and test sets

        :return:
        """
        helper.log('Splitting data into train and test sets')
        args = self._args
        self.test_nodes = []
        if self.train_graph is not None:
            self.train_sources, self.train_targets = source_targets(list(self.train_graph.edges()))
            test_sources, test_targets = source_targets(list(self.test_graph.edges()))
            self.train_nodes = set(self.train_sources) | set(self.train_targets)
            self.test_nodes = set(test_sources) | set(test_targets)
            self._hold_out = True
            return

        if self._hold_out:
            splits = split_graph(g=self.graph, train_rate=args.tr_rate)
            self.train_sources, self.train_targets = splits[0]
            if args.output_dir != '':
                test_sources, test_targets = splits[1]
                self.test_nodes = set(test_sources) | set(test_targets)

                path = os.path.join(args.output_dir, f'train_graph_{int(args.tr_rate * 100)}.txt')
                helper.log(
                    f"Persisting train graph to {path} and the number of training points is  {len(self.train_sources)}")
                nx.write_edgelist(self._creator(list(zip(self.train_sources, self.train_targets))), path=path,
                                  data=False)

                path = os.path.join(args.output_dir, f'test_graph_{int(args.tr_rate * 100)}.txt')
                helper.log(f"Persisting test graph to {path} and the number of test points is  {len(test_sources)}")
                nx.write_edgelist(self._creator(list(zip(test_sources, test_targets))), path=path, data=False)
            else:
                helper.log('No output directory is provided! Hence the test data is discarded',
                           level=helper.WARN)
        else:
            helper.log('Test data is not persisted', level=helper.WARN)
            self.train_sources, self.train_targets = source_targets(self.graph.edges())

        self.train_nodes = set(self.train_sources) | set(self.train_targets)
        helper.log(f'Number of nodes {self.num_nodes}')
        helper.log(f'Number of edges {self.num_edges}')

    def _sample_neighborhood(self):
        """
        A wrapper for the module level neighborhood sampling function (_sample_neighborhood)

        :return:
        """
        args = self._args
        self.neighborhood_matrix = _sample_neighborhood(nbr_size=args.nbr_size, graph=self.graph)

        if self._hold_out:
            """
            Masking out test nodes from the neighborhood matrix, to ensure that they are not sampled
            as a neighbor of a node during training.
            """
            msk = np.isin(self.neighborhood_matrix, self.test_nodes)
            self.neighborhood_matrix[msk.reshape(self.neighborhood_matrix.shape)] = 0

        self.mask_matrix = torch.FloatTensor(_generate_mask(self.neighborhood_matrix))
        self.neighborhood_matrix = torch.LongTensor(self.neighborhood_matrix)

    def _process_edges(self):
        """
        Process edges so as to associate negative connections

        :return:
        """
        def draw_negative_node(u, v):
            while True:
                rand_index = np.random.randint(0, len(self.node_dist_table))
                rand_node = self.node_dist_table[rand_index]
                if rand_node != u and rand_node != v:
                    return rand_node
        helper.log('Sampling negative nodes')

        degree = {node: int(1 + self.graph.degree(node) ** 0.75) for node in self.graph.nodes()}
        # node_dist_table is equivalent of the uni-gram distribution raised to the power of .75
        # table in the word2vec implementation
        self.node_dist_table = [node for node, new_degree in degree.items() for _ in range(new_degree)]
        edges = []
        for i in range(len(self.train_sources)):
            negative_node = draw_negative_node(self.train_sources[i], self.train_targets[i])
            edges.append((self.train_sources[i], self.train_targets[i], negative_node))
            if (i + 1) % 50000 == 0:
                helper.log(f"{i}/{len(self.train_sources)}", cr=True, level=helper.PROG)
        helper.log()
        self.edges = np.array(edges)


class EagerDataset:

    """
    A dataset for training GOAT. It prepares all the data necessary for training as a list of
    batches for training and validating the model. This is used when the data is fairly small,
    i.e. when the number of edges is <100K. However, GOAT runs faster with this alternative as
    there is no processing while training,.
    """

    def __init__(self, args, data=None):
        self._args = args
        self._data = data
        self._processes_train_dev_data()

    def _processes_train_dev_data(self):
        """
        Builds training and validation batches

        :return:
        """
        args = self._args
        self.dev_batches = []
        train_edges = self._data.edges
        if self._data.use_dev:
            dev_size = int(len(self._data.train_sources) * args.dev_rate)
            helper.log(f'Number of dev points: {dev_size}')

            dev_edges = self._data.edges[:dev_size]
            train_edges = self._data.edges[dev_size:]
            self.dev_batches = _generate_input_batches(
                neighborhood_matrix=self._data.neighborhood_matrix, mask_matrix=self._data.mask_matrix, edges=dev_edges)

        self.train_batches = _generate_input_batches(
            neighborhood_matrix=self._data.neighborhood_matrix, mask_matrix=self._data.mask_matrix, edges=train_edges)
        self.total_batches = len(self.train_batches)
        helper.log(f'Number of training points: {len(train_edges)}')


class LazyDataset(td.Dataset):

    """
    A dataset for training GOAT. It prepares all the data necessary for training using PyTorch
    Dataset and DataLoader. This enables GOAT to handle larger datasets, and slightly slower
    than the EagerDataset.
    """

    def __init__(self, data, partition=None):
        self._data = data
        self._partition = partition
        super(LazyDataset, self).__init__()

    def __len__(self):
        return len(self._partition)

    def __getitem__(self, index):
        edge = self._data.edges[self._partition[index]]
        source, target, negative = edge[0], edge[1], edge[1]
        source_nh = self._data.neighborhood_matrix[source, :]
        target_nh = self._data.neighborhood_matrix[target, :]
        negative_nh = self._data.neighborhood_matrix[negative, :]
        source_mask = self._data.mask_matrix[source, :]
        target_mask = self._data.mask_matrix[target, :]
        negative_mask = self._data.mask_matrix[negative, :]
        return source, target, negative, source_nh, target_nh, negative_nh, source_mask, target_mask, negative_mask


def compile_training_data(args, train_graph=None, test_graph=None):
    """
    Builds training and validation batches as a list of batches or iterator.
    For small datasets list will be used otherwise iterator.

    :param args: Input arguments
    :param train_graph: If specified, a graph will be initialized instead of reading
    :param test_graph: If specified, a graph will be initialized instead of reading
    :return: A dictionary containing all the relevant data for training GOAT
    """
    raw_data = RawData(args, train_graph=train_graph, test_graph=test_graph)
    data = {"num_nodes": raw_data.num_nodes, "num_edges": raw_data.num_edges, "dev_batches": None}
    if raw_data.num_edges < 20000000:
        # This code is tested with a synthetic dataset upto 20M edges, using a 100GB RAM machine.
        # The alternative condition should be used for large graphs.
        dataset = EagerDataset(args=args, data=raw_data)
        data["train_batches"] = dataset.train_batches
        if len(dataset.dev_batches) > 1:
            data["dev_batches"] = dataset.dev_batches
        data["total_batches"] = len(data["train_batches"])
        data['from_td'] = False
    else:
        # This branch has a bug
        # TODO: Find out the bug and fix it
        train_partition = np.random.permutation(len(raw_data.edges))
        dev_partition = []
        if raw_data.use_dev:
            dev_size = int(train_partition.size * args.dev_rate)
            dev_partition = train_partition[:dev_size]
            train_partition = train_partition[dev_size:]

        train_dataset = LazyDataset(data=raw_data, partition=train_partition)
        train_loader = td.DataLoader(dataset=train_dataset, batch_size=64, num_workers=args.workers)
        data["train_batches"] = train_loader
        if len(dev_partition) > 0:
            dev_dataset = LazyDataset(data=raw_data, partition=dev_partition)
            dev_loader = td.DataLoader(dataset=dev_dataset, batch_size=64, num_workers=args.workers)
            data["dev_batches"] = dev_loader
        data["total_batches"] = len(train_partition) // 64
        data['from_td'] = True
    return namedtuple("TrainData", data.keys())(*data.values())
