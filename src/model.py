"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""

import torch as t
import torch.nn as nn


class LogLoss:

    def __init__(self, model, weights=None, context=True):
        self.model = model
        self.weights = weights
        if context:
            self._compute_loss(source_rep=self.model.source_rep, target_rep=self.model.target_rep,
                               negative_rep=self.model.negative_rep)
        else:
            # Used for training the GLOBAL model
            self._compute_loss(source_rep=self.model.source_emb, target_rep=self.model.target_emb,
                               negative_rep=self.model.negative_emb)

    def _compute_loss(self, source_rep, target_rep, negative_rep):
        """
        Loss function used for training GOAT
        :return:
        """
        model_score = source_rep.mul(target_rep).sum(1).sigmoid().log()
        noise_score = source_rep.mul(-negative_rep).sum(1).sigmoid().log()
        score = model_score + noise_score
        self.loss = -t.mean(score if self.weights is None else score * self.weights)


class GlobalEmbedding(nn.Module):

    """
    Global embedding of nodes
    """

    def __init__(self, in_dim, out_dim, rate):
        super(GlobalEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rate = rate
        self._init()

    def _init(self):
        self.embedding = nn.Embedding(self.in_dim, self.out_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.dropout = nn.Dropout(self.rate)

    def forward(self, x, training):
        emb = self.dropout(self.embedding(x)) if training else self.embedding(x)
        return emb.transpose(1, 2) if len(emb.shape) > 2 else emb


class GOAT(nn.Module):

    """
    GOAT model
    """

    def __init__(self, num_nodes, emb_dim, rate=0.5):
        super(GOAT, self).__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.rate = rate
        self._init()

    def _init(self):
        self.global_embedding = GlobalEmbedding(in_dim=self.num_nodes + 1, out_dim=self.emb_dim, rate=self.rate)

    def _model_forward(self, source, target, source_nh, target_nh, source_mask, target_mask, training):
        self.source_emb = self.global_embedding(source, training=training)
        if target is None:
            return
        self.target_emb = self.global_embedding(target, training=training)
        
        self.source_nh_emb = self.global_embedding(source_nh, training=training)
        self.target_nh_emb = self.global_embedding(target_nh, training=training)
        self.source_target_sim = self.source_nh_emb.transpose(1, 2).matmul(self.target_nh_emb)
        self.source_target_sim = t.tanh(self.source_target_sim)

        source_attention_vec = t.mean(self.source_target_sim, dim=-1, keepdim=True)
        source_attention_vec = source_attention_vec + source_mask.unsqueeze(-1)
        self.source_attention_vec = t.softmax(source_attention_vec, dim=1)

        target_attention_vec = t.mean(self.source_target_sim, dim=1, keepdim=True).transpose(1, 2)
        target_attention_vec = target_attention_vec + target_mask.unsqueeze(-1)
        self.target_attention_vec = t.softmax(target_attention_vec, dim=1)

        self.source_rep = self.source_nh_emb.matmul(self.source_attention_vec).squeeze()
        self.target_rep = self.target_nh_emb.matmul(self.target_attention_vec).squeeze()

    def _noise_forward(self, negative, negative_nh, negative_mask):
        self.negative_emb = self.global_embedding(negative, training=True)
        self.negative_nh_emb = self.global_embedding(negative_nh, training=True)

        self.source_neg_sim = self.source_nh_emb.transpose(1, 2).matmul(self.negative_nh_emb)
        self.source_neg_sim = t.tanh(self.source_neg_sim)

        neg_attention_vec = t.mean(self.source_neg_sim, dim=1, keepdim=True).transpose(1, 2)
        neg_attention_vec = neg_attention_vec + negative_mask.unsqueeze(-1)
        self.neg_attention_vec = t.softmax(neg_attention_vec, dim=1)

        self.negative_rep = self.negative_nh_emb.matmul(self.neg_attention_vec).squeeze()

    def forward(self, training, source, target=None, negative=None, source_neighborhood=None, target_neighborhood=None,
                negative_neighborhood=None, source_mask=None, target_mask=None, negative_mask=None):
        self._model_forward(source=source, target=target, source_nh=source_neighborhood, target_nh=target_neighborhood,
                            source_mask=source_mask, target_mask=target_mask, training=training)
        if negative is not None:
            self._noise_forward(negative, negative_neighborhood, negative_mask)
