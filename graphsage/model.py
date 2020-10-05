import os

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(enc.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)  # (num_nodes, embed_dim)
        # (num_nodes, embed_dim) x (embed_dim, num_classes) -> (num_nodes, num_classes)
        scores = embeds.mm(self.weight)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    """
    output:
        feat_data - 样本特征， ndarray, shape (num_nodes, num_feat)
        labels - 样本标签， ndarray, shape (num_nodes, 1)
        adj_lists - 邻接表， defaultdict(set), shape (num_nodes, len(n(v)) )
    """
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("../cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("../cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置随机种子,固定结果
    seed = 424
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    num_nodes = 2708
    num_feat = 1433
    num_embed = 128
    num_classes = 7
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(num_nodes, num_feat)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.to(device)

    agg1 = MeanAggregator(features, device)
    enc1 = Encoder(features, num_feat, num_embed, adj_lists, agg1, device, gcn=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes), device)
    enc2 = Encoder(lambda nodes: enc1(nodes), num_embed, num_embed, adj_lists, agg2, device,
                   gcn=True)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(num_classes, enc2)
    graphsage.to(device)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    rand_indices = np.random.permutation(num_nodes)  # 随机排列一个下标序列
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        # 重新打乱顺序，达到随机批抽样的目的
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        # 1. batch_nodes -> result (num_nodes, num_classes)
        # 2. loss(result, labels)
        loss = graphsage.loss(batch_nodes, torch.LongTensor(labels[np.array(batch_nodes)]).to(device))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    # average：
    #   micro  直接求整体的F1-score
    #   macro  求出各类别的F1-score后再作平均
    print("Validation F1:", f1_score(labels[val], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Test F1:", f1_score(labels[test], test_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))  # numpy 可以直接作用于 list


def load_pubmed():
    # hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("../pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1]) - 1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("../pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_pubmed():
    # 设置随机种子,固定结果
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    num_nodes = 19717
    num_feat = 500
    num_embed = 128
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(num_nodes, num_feat)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    features.cuda()

    # 使用了 gcn 作为 aggregator
    agg1 = MeanAggregator(features)
    enc1 = Encoder(features, num_feat, num_embed, adj_lists, agg1, gcn=True, cuda=True)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes), cuda=True)
    enc2 = Encoder(lambda nodes: enc1(nodes), num_embed, num_embed, adj_lists, agg2,
                   gcn=True, cuda=True)
    # 对于不同层次使用了不同的采样数目
    # 采样数目与论文相符
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                              torch.LongTensor(labels[np.array(batch_nodes)]).cuda())
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    print("Validation F1:", f1_score(labels[val], val_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Test F1:", f1_score(labels[test], test_output.cpu().data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    torch.cuda.set_device(0)
    run_cora()
    # run_pubmed()
