"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""


from urllib import request

from torch_geometric.data import Data

import networkx as nx
import pandas as pd

import os.path as osp

import torch

import utils




class LocalDatasets:
    
    def __init__(self, root, name):
        """
        A class for creating specfic datasets. The current implementation can be used to create instances of
        Zhihu and Email datasets used in the paper. However, it can be used to create other datasets just by 
        overloading the url_file_mappings property. This property should return a dictionary as
        return {
            <DATASET_NAME_1>: {
                "urls": {
                    "edge": <url to white space separated online edge list file>
                    <ANOTHER_KEY_1> : <url to a white space separated file related to the key ANOTHER_KEY_1>
                    ...
                    <ANOTHER_KEY_K> : <url to a white space separated file related to the key ANOTHER_KEY_K>
                }
                "files": {
                    "edge": <the raw edge file name>
                    <ANOTHER_KEY_1> : <the raw file name for ANOTHER_KEY_1>
                    ...
                    <ANOTHER_KEY_K> : <the raw file name for ANOTHER_KEY_K>
                }
            }
            ...
            <DATASET_NAME_N>: {
                <similar internal structure as DATASET_NAME_1>
            }
        }
        """
        
        self._root = root
        self._name = name
        self._build_dataset()
        
    @property
    def url_file_mappings(self):
        return {
            "email": {
                "urls": {
                    "edge": "http://snap.stanford.edu/data/email-Eu-core.txt.gz", 
                    "label": "http://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
                },
                "files": {
                    "edge": "edges.txt.gz",
                    "label": "labels.txt.gz"
                }
            },
            "zhihu": {
                "urls": {
                    "edge": "https://raw.githubusercontent.com/thunlp/CANE/master/datasets/zhihu/graph.txt", 
                },
                "files": {
                    "edge": "edges.txt",
                }
            }
        }
    
    @property
    def processed_files_names(self):
        return ["data.pt"]
    
    def _build_dataset(self):
        self._download()
        self._process()
        
    def _download(self):
        self.download()
    
    def _process(self):
        self.process()
            
    def download(self):
        """
        Downloads the necssary files for the current dataset according the keys
        specified in the url_file_mapping property

        """
        root, name = self._root, self._name
        raw_dir = osp.join(root, name, "raw")
        raw_files = list(self.url_file_mappings[name]['files'].values())
        file_keys = list(self.url_file_mappings[name]['files'].keys())
        if not utils.files_exist_in(files=raw_files, in_dir=raw_dir):
            for key in file_keys:
                url = self.url_file_mappings[name]['urls'][key]
                file = self.url_file_mappings[name]['files'][key]
                path = osp.join(raw_dir, file)
                utils.log(f"Downloading {url}")
                request.urlretrieve(url, path)
            
    def _read_if(self, file_name_key):
        """
        Reads file specified by a key if the key exists in the current dataset 
        
        :param file_name_key: The key
        """
        if file_name_key in self.url_file_mappings[self._name]['files']:
            file_name = self.url_file_mappings[self._name]['files'][file_name_key]
            path = osp.join(self._root, self._name, "raw", file_name)
            return pd.read_csv(path, comment="#", header=None, sep=f"\s+")
        
    def _map_if(self, nodes):
        """
        Generates node to id mapping if the nodes are not properly identified.
        A proper identification should have node ids starting from 0 to the 
        number of nodes minus 1 without intermediate jumps.
        
        :param nodes: The nodes
        :return: A mapping if not proper otherwise None
        
        """
        nodes = sorted(nodes)
        relabel = False
        for i in range(len(nodes)):
            if i != nodes[i]:
                relabel = True
                
        if relabel:
            ids = range(len(nodes))
            return dict(zip(nodes, ids))
            
    def process(self):
        """
        Processes a downloaded file to make it ready for training and evaluating the GOAT model.
        
        """
        root, name = self._root, self._name
        processed_dir = osp.join(root, name, "processed")
        processed_path = osp.join(processed_dir, self.processed_files_names[0])
        if not utils.files_exist_in(files=self.processed_files_names, in_dir=processed_dir):
            utils.log("Processing ...")
            # Graph
            edge_file_name = self.url_file_mappings[name]['files']['edge']
            path = osp.join(root, name, "raw", edge_file_name)
            df = pd.read_csv(path, sep=r"\s+", names=['source', 'target'], comment="#", header=None)
            graph = nx.from_pandas_edgelist(df)

            # Features
            features = self._read_if(file_name_key="feature")
            

            # Labels
            labels = self._read_if(file_name_key="label")

            node2id = self._map_if(graph.nodes())
            if node2id is not None:
                graph = nx.relabel_nodes(graph, node2id)

            # TODO: Handle features
            
            x = y = None
            if labels is not None:
                if labels.shape == 1:
                    labels = labels.reset_index()
                labels.columns = ['node', 'label']
                if node2id is not None:
                    labels.node = labels.node.apply(lambda l: node2id[l])
                y = labels.sort_values("node").label.values

            adj = nx.to_scipy_sparse_matrix(graph, sorted(graph.nodes()))
            edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

            self.data = Data(edge_index=edge_index)
            if x is not None:
                self.data.x = torch.tensor(x, dtype=torch.float32)

            if y is not None:
                self.data.y = torch.tensor(y, dtype=torch.long)
            self.data.num_nodes = graph.number_of_nodes()
            
            torch.save(self.data, processed_path)
        else:
            utils.log("Loading ...")
            self.data = torch.load(processed_path)
