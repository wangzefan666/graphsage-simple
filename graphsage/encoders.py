import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator, device,
                 base_model=None, num_sample=10, gcn=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model is not None:
            self.base_model = base_model
        self.gcn = gcn
        self.deivce = device

        self.weight = nn.Parameter(torch.FloatTensor(
            self.feat_dim if self.gcn else 2 * self.feat_dim, embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        # call aggregator, pay attention to the num_sample
        # (batch, feat_dim)
        neigh_feats = self.aggregator.forward(
            nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)

        if not self.gcn:
            self_feats = self.features(torch.LongTensor(nodes).to(self.device))
            combined = torch.cat([self_feats, neigh_feats], dim=1)  # (batch, 2 * feat_dim)
        # use gcn as aggregator, not concat
        else:
            combined = neigh_feats

        # (batch, feat_dim) x (feat_dim, embed_dim) -> (batch, embed_dim)
        combined = F.relu(combined.mm(self.weight))
        return combined
