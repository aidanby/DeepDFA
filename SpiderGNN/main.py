"""
Based on 5_graph_classification.py
"""

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from linevul.util import load_model
from model import SpiderGNN
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent/"LineVul"))
sys.path.append(str(Path(__file__).parent.parent))

##### LOAD DATASET

# Generate a synthetic dataset with 10000 graphs, ranging from 10 to 500 nodes.
dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)
# TODO: Load instead a dataset of programs where each node is a function with tokenized text as "input_ids"

print("Node feature dimensionality:", dataset.dim_nfeats)
print("Number of graph categories:", dataset.gclasses)


from dgl.dataloading import GraphDataLoader

from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False
)

it = iter(train_dataloader)
batch = next(it)
print(batch)

##### LOAD MODEL

# Default Graph Convolutional Network
# from dgl.nn import GraphConv

# class GCN(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, num_classes)

#     def forward(self, g, in_feat):
#         h = self.conv1(g, in_feat)
#         h = F.relu(h)
#         h = self.conv2(g, h)
#         g.ndata["h"] = h
#         return dgl.mean_nodes(g, "h")

# Use GGNN to bridge among function encodings
model_name_or_path = "codellama/CodeLlama-7b-hf"
model, config, tokenizer = load_model(model_name_or_path)
model = SpiderGNN(model, config, gnn_out_dim=4096)

##### TRAIN
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    losses = []
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata["attr"].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy())
    print("Epoch:", epoch, "Loss:", np.average(losses))

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata["attr"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print("Test accuracy:", num_correct / num_tests)
