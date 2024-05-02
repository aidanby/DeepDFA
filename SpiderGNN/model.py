from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
from torch import nn
import torch as th
from transformers import AutoModel, AutoConfig

from linevul.linevul_model import HfClassificationHead

class SpiderGNN(nn.Module):
    def __init__(self, encoder: AutoModel, config: AutoConfig, gnn_out_dim: int=32):
        super(SpiderGNN, self).__init__()
        self.encoder = encoder
        self.gnn = GatedGraphConv(in_feats=config.hidden_size,
                                  out_feats=gnn_out_dim,
                                  n_steps=5,
                                  n_etypes=1)
        pooling_gate_nn = nn.Linear(gnn_out_dim, 1)
        self.gnn_readout = GlobalAttentionPooling(pooling_gate_nn)
        self.classifier = HfClassificationHead(config, gnn_out_dim)

    def forward(
        self,
        graph=None,  # Call graph - each node is a function
        labels=None, # Supply if you want loss to be calculated
    ):
        # Encode each function using a transformer
        # Transformer encoding for each function
        input_ids = graph.ndata["attr"]
        input_ids = input_ids.int() # NOTE: For DEBUG ONLY with dummy data
        hf_out_node = self.encoder(
            input_ids,
            attention_mask=input_ids.ne(1),
        )[0]
        hf_out_node = hf_out_node[:,-1,:] # TODO: Extract a single representation for each sentence. [CLS]?
            
        # Use GNN to aggregate encodings among functions
        gnn_nodes = self.gnn(graph, hf_out_node)
        gnn_graph = self.gnn_readout(graph, gnn_nodes)

        # Classifier layer
        logits = self.classifier(gnn_graph)
        prob = th.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
