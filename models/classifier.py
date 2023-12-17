from utils.config import *
from models.nnLayer import *
from torch import nn as nn
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F
from dgl.nn import GlobalAttentionPooling

params = config()

# The model used in training has no output attention score
class GraphLncLoc2(nn.Module):
    def __init__(self, embedding, embDropout=0.2, fcDropout=0.2, in_dim=params.d,
                 hidden_dim=params.hidden_dim, device=params.device):
        super(GraphLncLoc2, self).__init__()
        self.nodeEmbedding = NodeEmbedding(embedding, dropout=embDropout).to(device)
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none').to(device)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none').to(device)
        self.gap = GlobalAttentionPooling(nn.Linear(hidden_dim, 1)).to(device)
        self.lin = MLP(hidden_dim, hidden_dim, [], fcDropout).to(device)
        self.classify = MLP(hidden_dim, 1, [], fcDropout).to(device)

        self.nodeidx = torch.arange(4 ** params.kmer).to(device)

    def forward(self, g):
        h = self.nodeEmbedding(torch.cat([self.nodeidx] * g.batch_size, dim=0))
        h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
        x = self.gap(g, h)
        x1 = torch.squeeze(torch.sigmoid(self.classify(x)), 1)
        x2 = self.lin(x)
        return x1, x2

# The model used in the prediction will output the attention score
class GraphLncLoc2_alpha(nn.Module):
    def __init__(self, embedding, embDropout=0.2, fcDropout=0.2, in_dim=params.d,
                 hidden_dim=params.hidden_dim, device=params.device):
        super(GraphLncLoc2_alpha, self).__init__()
        self.nodeEmbedding = NodeEmbedding(embedding, dropout=embDropout).to(device)
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm='none').to(device)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm='none').to(device)
        self.gap = GlobalAttentionPooling(nn.Linear(hidden_dim, 1)).to(device)
        self.lin = MLP(hidden_dim, hidden_dim, [], fcDropout).to(device)
        self.classify = MLP(hidden_dim, 1, [], fcDropout).to(device)

        self.nodeidx = torch.arange(4 ** params.kmer).to(device)

    def forward(self, g):
        h = self.nodeEmbedding(torch.cat([self.nodeidx] * g.batch_size, dim=0))
        h = F.relu(self.conv1(g, h, edge_weight=g.edata['weight']))
        h = F.relu(self.conv2(g, h, edge_weight=g.edata['weight']))
        x, alpha = self.gap(g, h, get_attention=True)
        x1 = torch.squeeze(torch.sigmoid(self.classify(x)), 1)
        x2 = self.lin(x)
        return x1, x2, alpha
