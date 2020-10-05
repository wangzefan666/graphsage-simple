import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, device, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.device = device
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        聚合邻接点特征作为目标节点特征

        output:
            nodes --- list of nodes in a batch
            to_neighs --- list of sets, each set is the set of neighbors for node in batch
            num_sample --- number of neighbors to sample. No sampling if None.
        """
        # 从目标节点的邻接点集合中作固定数目的采样
        # 若邻接点数量小于要求,则直接将邻接点集加入
        # ~原论文中若邻接点数量小于要求,则进行可放回的采样满足要求
        _set = set
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, num_sample))
                       if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]

        if self.gcn:
            samp_neighs = [samp_neigh + {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]

        # 构建局部邻接矩阵,并进行归一化
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(nodes)) for j in range(len(samp_neighs[i]))]
        mask = torch.zeros(len(nodes), len(unique_nodes_list))
        mask[row_indices, column_indices] = 1
        mask = mask.to(self.device)
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        # 构建局部特征矩阵
        # (num_nodes&neighs, ) -> (num_nodes&neighs, num_feat)
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))

        # 将邻接点的平均特征向量作为目标节点的特征
        # (num_nodes, num_nodes&neighs) * (num_nodes&neighs, num_feat) -> (num_nodes, num_feat)
        to_feats = mask.mm(embed_matrix)
        return to_feats
