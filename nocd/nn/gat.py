import numpy as np
import scipy.sparse  as sp
from nocd.utils import to_sparse_tensor
from torch_geometric.nn import GATConv,GATv2Conv
from nocd.nn import  gcn
import torch.nn.functional as F
import torch
import torch.nn as nn

def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.cuda.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)

class GAT(nn.Module):
    # def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
    #     super().__init__()
    #     # super().__init__(input_dim, hidden_dims, output_dim,dropout,batch_norm)
    #     # super(GAT, self).__init__(input_dim, hidden_dims, output_dim)
    #     self.dropout = dropout
    #     self.layers = nn.ModuleList([GATConv(input_dim, hidden_dims,heads = 6,add_self_loops=False,concat = False ),GATConv(hidden_dims , output_dim, heads = 6,add_self_loops=False, concat = False )])
    #     # self.batch_norm  = [nn.BatchNorm1d(128, affine=False, track_running_stats=False)]
    #     if batch_norm:
    #         # self.batch_norm = [
    #         #     nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
    #         # ]
    #         self.batch_norm  = [nn.BatchNorm1d(128, affine=False, track_running_stats=False)]
    #     else:
    #         self.batch_norm = None
    #     #---corect

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.5, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        layer_dims = np.concatenate([hidden_dims, [output_dim]]).astype(np.int32)
        self.layers = nn.ModuleList([GATConv(input_dim, layer_dims[0],heads = 2,add_self_loops=True,concat = False)])
        self.layers.append(GATv2Conv(hidden_dims[0] ,output_dim , heads = 2,add_self_loops=True, concat = False ))#heads was 6

        # for idx in range(len(layer_dims) - 1):
        #     print(layer_dims[idx], layer_dims[idx + 1])
        #     # self.layers.append(gcn.GraphConvolution(layer_dims[idx], layer_dims[idx + 1]))
        #     self.layers.append(GATConv(layer_dims[idx], layer_dims[idx + 1], heads = 6, add_self_loops=False, concat = False))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None
        # print('hidden dims', hidden_dims)
        # print('batchnorm', batch_norm)

    # def forward(self, x, edge_index, edge_weight=None):
    #     # x = F.dropout(x, p=self.dropout, training=self.training)
    #     # if self.dropout != 0:
    #     #     x = gcn.sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
    #     x = self.layers[0](x, edge_index, edge_weight).relu()
        
    #     x = self.batch_norm[0](x)
    #     # if self.dropout != 0:
    #     #     x = gcn.sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.layers[1](x, edge_index, edge_weight).relu()
    #     return x
    
    def forward(self, x, adj):
        for idx, gat in enumerate(self.layers):
            if self.dropout != 0:
                x = sparse_or_dense_dropout(x, p=self.dropout, training=self.training)
            x = gat(x, adj)
            if idx != len(self.layers) - 1:
                x = F.relu(x)
                if self.batch_norm is not None:
                    x = self.batch_norm[idx](x)
        return x

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]


    @staticmethod
    def get_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            # adj.setdiag(1)
            adj = adj.tocsr()
        return to_sparse_tensor(adj)

    @staticmethod
    def normalize_adj(adj : sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        return to_sparse_tensor(adj_norm)


    @staticmethod
    def nor_edge(adj : sp.csr_matrix):
        """Norm adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
        adj = adj_norm.tocoo()
        newedge_index = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
        return (newedge_index)