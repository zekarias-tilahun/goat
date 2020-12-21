"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""

from sklearn import metrics, cluster

import os.path as osp

import networkx as nx
import pandas as pd
import numpy as np
import argparse

import torch

import model
import utils
import data


class Evaluate:
    
    def __init__(self, args):
        """
        Evaluates GOAT on three types of workloads, which are link prediction, node clustering 
        and attention weight visualization. The type of workload should be specified in a given
        argument. 
        
        :param args: The arguments necessary for evaluation.
        """
        self._args = args
        self._load()
    
    def _load(self):
        """
        Loads the data and model objects
        
        """
        args = self._args
        _, self._params = utils.decide_config(args.root, args.name)
        self._args.name = self._args.name.split("-")[0]
        
        data_ = data.Data(args=args)
        self.num_dev_samples = data_.dev_partition.shape[0]
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataset = data.InputDataset(input_data=data_, num_neg=data_.num_neg)
        dev_dataset = data.InputDataset(input_data=data_, num_neg=data_.num_neg, partition="dev")
        test_dataset = data.InputDataset(input_data=data_, num_neg=data_.num_neg, partition="test")
        self._loader = data.DataLoader(dataset=dataset, batch_size=self._args.batch) # used for node clustering
        self._dev_loader = data.DataLoader(dataset=dev_dataset, batch_size=self._args.batch) # used for link prediction
        self._test_loader = data.DataLoader(dataset=test_dataset, batch_size=self._args.batch) # used for link prediction and attention visualization
        self._negative_pairs = data_.negative_pairs
        self._neighborhood = data_.neighborhood
        self._mask = data_.neighborhood_mask
        self._y = None
        if hasattr(data_, "y"):
            self._y = data_.y
        self._model = model.Goat(num_nodes=data_.num_nodes, dim=data_.dim)
        self._model.to(self._device)
        
    def _compute_auc(self, true_source_feature, true_target_feature, false_source_feature, false_target_feature):
        """
        Computes the area under the receiver operating characteristic curve (ROC AUC) score based on true edges and false pairs.
        A good evaluation should produce higher scores (dot product) between the features of the incident nodes of each true edge 
        and lower for each incident node in the false pairs.
        
        :param true_source_feature: The features for the source nodes of the true edges
        :param true_target_feature: The features for the target nodes of the true edges
        :param false_source_feature: The features for the source nodes of the false pairs
        :param false_target_feature: The features for the target nodes of the false pairs
        :return: ROC AUC score 
        """
        true_scores = np.multiply(true_source_feature, true_target_feature).sum(-1)
        true_labels = np.ones(true_scores.shape[0])
        
        false_scores = np.multiply(false_source_feature, false_target_feature).sum(-1)
        false_labels = np.zeros(false_scores.shape[0])
        
        y_hat = np.concatenate([true_scores, false_scores])
        y_true = np.concatenate([true_labels, false_labels])
        
        return metrics.roc_auc_score(y_true, y_hat)
        
    def _fit_false_pairs(self, num_false_samples=None):
        """
        Inference using false pairs. Useful for validating and testing the quality of link prediction.
        
        :param num_false_samples: The number of false pairs.
        """
        if num_false_samples:
            false_pairs = self._negative_pairs[:num_false_samples]
        else:
            false_pairs = self._negative_pairs
            num_false_samples = false_pairs.shape[0]
            
        source_features, target_features = [], []
        batch_size = self._args.batch
        for i in range(0, num_false_samples, batch_size):
            end = i + batch_size if num_false_samples - batch_size > i else num_false_samples
            false_data = {
                "source_neighborhood": self._neighborhood[self._negative_pairs[i:end, 0]],
                "target_neighborhood": self._neighborhood[self._negative_pairs[i:end, 1]],
                "source_mask": self._mask[self._negative_pairs[i:end, 0]],
                "target_mask": self._mask[self._negative_pairs[i:end, 1]],
            }
            false_source_feature, false_target_feature, _, _ = self._model(**utils.to(false_data, self._device))
            source_features.append(false_source_feature.detach().cpu().numpy())
            target_features.append(false_target_feature.detach().cpu().numpy())
            
        false_source_feature = np.concatenate([utils.expand_if(f, 0, len(f.shape)==1) for f in source_features])
        false_target_feature = np.concatenate([utils.expand_if(f, 0, len(f.shape)==1) for f in target_features])
        return false_source_feature, false_target_feature
        
    def _fit(self, loader):
        """
        Fits the GOAT model using data from a specified loader
        
        :param loader: The loader
        """
        sources, targets, source_atn, target_atn, source_features, target_features = [], [], [], [], [], []
        for bc, batch in self._test_loader:
            sources.append(batch["source"].cpu().numpy())
            targets.append(batch["target"].cpu().numpy())
            
            true_source_feature, true_target_feature, _, _ = self._model(**utils.to(batch, self._device))
            
            source_atn.append(self._model.source_atn.squeeze().detach().cpu().numpy())
            target_atn.append(self._model.target_atn.squeeze().detach().cpu().numpy())
            source_features.append(true_source_feature.detach().cpu().numpy())
            target_features.append(true_target_feature.detach().cpu().numpy())
            
            
        sources = np.concatenate(sources)
        targets = np.concatenate(targets)
        source_atn = np.concatenate([utils.expand_if(atn_v, 0, len(atn_v.shape)==1) for atn_v in source_atn])
        target_atn = np.concatenate([utils.expand_if(atn_v, 0, len(atn_v.shape)==1) for atn_v in target_atn])
        true_source_feature = np.concatenate([utils.expand_if(f, 0, len(f.shape)==1) for f in source_features])
        true_target_feature = np.concatenate([utils.expand_if(f, 0, len(f.shape)==1) for f in target_features])
        
        return sources, targets, source_atn, target_atn, true_source_feature, true_target_feature
    
    def _load_best_model(self, best_epoch):
        """
        Loads the model identified as best model
        
        :param best_epoch: The epoch which produced the best model.
        """
        params = self._params
        data_dir = osp.join(params["root"], params["name"]) if "name" in params else params["root"]
        model_path = osp.join(data_dir, "models", f"goat.ep{best_epoch}.pt")
        utils.log(f"Loading model from {model_path}")
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()
        
    def identify_best_epoch(self):
        """
        Selects the best model using the validation set
        
        """
        utils.log("Identifying the best epoch using the validation set")
        params = self._params
        data_dir = osp.join(params["root"], params["name"]) if "name" in params else params["root"]
        epoch_scores = {}
        for i in range(self._args.epoch):
            model_path = osp.join(data_dir, "models", f"goat.ep{i + 1}.pt")
            self._model.load_state_dict(torch.load(model_path))
            self._model.eval()
            _, _, _, _, true_source_feature, true_target_feature = self._fit(self._dev_loader)
            false_source_feature, false_target_feature = self._fit_false_pairs(
                num_false_samples=self.num_dev_samples)
            
            auc = self._compute_auc(true_source_feature, true_target_feature, 
                                    false_source_feature, false_target_feature)
            self._model.embedding.reset_parameters()
            utils.log("Epoch {:04d} validation auc: {:.4f}".format(i + 1, auc))
            
            epoch_scores[i + 1] = auc
        best_epoch = max(epoch_scores.items(), key=lambda l: l[1])
        utils.log("Best epoch: {:04d} with validation auc: {:.4f}".format(best_epoch[0], best_epoch[1]))
        return best_epoch[0]
    
    def link_prediction(self, epoch):
        """
        Evaluate model performance over the link prediction task using the saved model of a specified epoch
        
        :param epoch: The epoch
        """
        self._load_best_model(best_epoch=epoch)
        utils.log("Predicting links ...")
        _, _, _, _, true_source_feature, true_target_feature = self._fit(self._test_loader)
        false_source_feature, false_target_feature = self._fit_false_pairs()
        auc = self._compute_auc(true_source_feature, true_target_feature,
                                false_source_feature, false_target_feature)
        utils.log("Test auc {:.4f}".format(auc))
    
    def node_clustering(self, epoch=5):
        """
        Evaluate model performance over the node clustering task using the saved model of a specified epoch
        
        :param epoch: The epoch
        """
        self._load_best_model(best_epoch=epoch)
        utils.log("Clustering nodes ...")
        sources, targets, _, _, source_features, target_features = self._fit(self._loader)
        nodes = sorted(set(sources) | set(targets))
        
        # Group features by node id
        def group_features(emb_dict, n, e):
            if n in emb_dict:
                emb_dict[n].append(e)
            else:
                emb_dict[n] = [e]
            
        source_grouped_f, target_grouped_f = {}, {}
        for i in range(sources.shape[0]):
            group_features(source_grouped_f, sources[i], source_features[i])
            group_features(target_grouped_f, targets[i], target_features[i])
            
        # Aggregate grouped features
        aggregated_features = np.zeros((self._y.shape[0], source_features.shape[1]))
        for node in nodes:
            se = te = None
            if node in source_grouped_f:
                se = np.stack(source_grouped_f[node]).mean(0)
            if node in target_grouped_f:
                te = np.stack(target_grouped_f[node]).mean(0)
                
            if se is not None and te is not None:
                aggregated_features[node] = np.stack([se, te]).mean(0)
            elif se is not None:
                aggregated_features[node] = se
            elif te is not None:
                aggregated_features[node] = te

        kmeans = cluster.KMeans(n_clusters=42, random_state=0)
        clusters = kmeans.fit(aggregated_features).labels_
        
        nmi = metrics.normalized_mutual_info_score(self._y, clusters, average_method='arithmetic')
        ami = metrics.adjusted_mutual_info_score(self._y, clusters, average_method='arithmetic')
        utils.log(f"Clustering results: NMI = {nmi} and AMI = {ami}")
        return nmi, ami
        
    
    def visualize_attention_weights(self, epoch):
        """
        Visualize attention of the model saved during an epoch
        
        :param epoch: The epoch
        """

        #TODO: Add a node name mapping for easily readable visualization
        
        nodes = self._args.nodes
        assert len(nodes) == 2
        src, trg = nodes
        utils.log(f"Generating attention weight visualization between source = {src} and target = {trg} nodes")
        self._load_best_model(best_epoch=epoch)
        
        batch = {
            "source_neighborhood": self._neighborhood[src].unsqueeze(0),
            "target_neighborhood": self._neighborhood[trg].unsqueeze(0),
            "source_mask": self._mask[src].unsqueeze(0),
            "target_mask": self._mask[trg].unsqueeze(0),
        }
        self._model(**utils.to(batch, self._device))
        src_mask = self._mask[src] == 0
        trg_mask = self._mask[trg] == 0
        source_atn = self._model.source_atn.squeeze().detach().cpu().numpy()[src_mask]
        target_atn = self._model.target_atn.squeeze().detach().cpu().numpy()[trg_mask]
        source_nbh = batch["source_neighborhood"].squeeze()[src_mask]
        target_nbh = batch["source_neighborhood"].squeeze()[trg_mask]
 
        params = self._params
        data_dir = osp.join(params["root"], params["name"]) if "name" in params else params["root"]
        src_path = osp.join(data_dir, "results", f"source_atn_vis_{src}.html")
        trg_path = osp.join(data_dir, "results", f"target_atn_vis_{trg}.html")
        utils.visualize_attention_weights(path=src_path, neighborhood=source_nbh, weights=source_atn, node_id=src)
        utils.visualize_attention_weights(path=trg_path, neighborhood=target_nbh, weights=target_atn, node_id=trg)
        
        
if __name__ == '__main__':
    args_ = utils.parse_eval_args()
    print(args_.identify)
    evaluator = Evaluate(args=args_)
    best_epoch = evaluator.identify_best_epoch() if args_.identify else args_.epoch
    if args_.task == "link_prediction":
        evaluator.link_prediction(best_epoch)
    elif args_.task == "node_clustering":
        evaluator.node_clustering(best_epoch)
    elif args_.task == "visualization":
        evaluator.visualize_attention_weights(best_epoch)
    