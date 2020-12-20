"""
Author: Zekarias Tilahun Kefato <zekarias@kth.se>


This module implements the core of the GOAT model.
GOAT is an acronym for the paper titled

    Gossip and Attend: A Context-sensitive Graph Representation Learning

"""

import torch.nn.functional as F
import torch.nn as nn
import torch


class Goat(nn.Module):

    def __init__(self, num_nodes, dim, dropout=0.):
        """
        The Goat model.

        :param num_nodes: The number of nodes
        :param dim: The size of the embedding dimension
        :param dropout: The dropout rate
        """
        super().__init__()
        self._num_nodes = num_nodes
        self._dim = dim
        self._dropout = dropout
        self._init_params()

    def _init_params(self):
        self.embedding = nn.Embedding(self._num_nodes, self._dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    @staticmethod
    def _loss_fn(source, target, negative):
        model_score = source.mul(target).sum(-1).sigmoid().log()
        noise_score = source.unsqueeze(-2).mul(-negative).sum(-1).sigmoid().log().mean(-1)
        return -torch.mean(model_score + noise_score)

    def embed(self, x):
        x = self.embedding(x).transpose(-1, -2)
        x = F.dropout(x, p=self._dropout, training=self.training)
        return x

    def _forward(self, source, target, source_mask, target_mask):
        source_emb = self.embed(source)
        target_emb = self.embed(target)

        if len(target_emb.shape) == 4:
            source_emb = source_emb.unsqueeze(-3)

        alignment = torch.tanh(source_emb.transpose(-1, -2).matmul(target_emb))
        source_atn = torch.mean(alignment, dim=-1, keepdim=True)
        target_atn = torch.mean(alignment, dim=-2, keepdim=True).transpose(-1, -2)

        if len(target_atn.shape) == 4:
            source_mask = source_mask.unsqueeze(-2)

        self.source_atn = torch.softmax(source_atn + source_mask.unsqueeze(-1), dim=-2)
        self.target_atn = torch.softmax(target_atn + target_mask.unsqueeze(-1), dim=-2)

        source_context_rep = source_emb.matmul(self.source_atn).squeeze()
        target_context_rep = target_emb.matmul(self.target_atn).squeeze()

        return source_context_rep, target_context_rep

    def forward(self, source_neighborhood, target_neighborhood, source_mask, target_mask,
                negative_neighborhood=None, negative_mask=None, **kwargs):
        src_ctx_rep, trg_ctx_rep = self._forward(source_neighborhood, target_neighborhood, source_mask, target_mask)
        loss = 0
        neg_ctx_rep = None
        if self.training:
            _, neg_ctx_rep = self._forward(source_neighborhood, negative_neighborhood, source_mask, negative_mask)
            loss = self._loss_fn(src_ctx_rep, trg_ctx_rep, neg_ctx_rep)
        return src_ctx_rep, trg_ctx_rep, neg_ctx_rep, loss
    
    @staticmethod
    def evaluate(source, target, negative):
        """
        Computes an AUC score between a between positive pair(s) (source, target) and 
        simple negative pair(s) (source, negative)

        :param source: Source features
        :param target: Target features
        :param negative: Negative features
        :return: An AUC score
        """
        pos_score = source.mul(target).sum(-1)
        neg_score = source.unsqueeze(-2).mul(negative).sum(-1).mean(-1)
        hit = (pos_score > neg_score).sum()
        tie = (pos_score == neg_score).sum() * 0.5
        return (hit + tie) / pos_score.shape[0]
