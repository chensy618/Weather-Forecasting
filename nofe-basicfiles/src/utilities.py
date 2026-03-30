"""
Training Utilities for Neural Operator Models
==============================================

This module provides utility classes and functions for training neural operator models
on ERA5 weather data. It includes model architectures, loss functions, data loading,
and helper functions.

All dependencies are self-contained - no imports from other project directories.
"""

import os
import pickle
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch_geometric.data import Data


# ============================================================================
# Custom Neural Network Convolution Layer
# ============================================================================

class NNConv_old(MessagePassing):
    """
    Edge-conditioned convolution operator for graph neural networks.
    
    From "Neural Message Passing for Quantum Chemistry" and 
    "Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs".
    
    The convolution uses a neural network to generate edge-specific filters.
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr
        
        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass."""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)
    
    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
    
    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
    
    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# ============================================================================
# Model Architectures
# ============================================================================

class DenseNet(torch.nn.Module):
    """
    Fully connected neural network with configurable depth and nonlinearity.
    Used as edge kernel in the NNConv layer.
    """
    
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()
        
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        
        self.layers = nn.ModuleList()
        
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                
                self.layers.append(nonlinearity())
        
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())
    
    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class KernelNN(nn.Module):
    """
    Graph Neural Network with edge-conditioned convolutions.
    
    Architecture:
        1. Lifting layer (optional, if in_width != width)
        2. Multiple edge-conditioned graph convolution layers
        3. Final projection to output dimension
    
    Args:
        width: Number of features per node after lifting
        ker_width: Dimension of single kernel layer
        depth: Number of graph convolution layers
        ker_in: Number of edge features
        in_width: Input node feature dimension (default: 1)
        out_width: Output dimension (default: 2)
    """
    
    def __init__(
        self,
        width,
        ker_width,
        depth,
        ker_in,
        in_width=1,
        out_width=2,
    ):
        super(KernelNN, self).__init__()
        
        self.depth = depth
        self.out_width = out_width
        
        self.fc1 = nn.Linear(in_width, width)
        
        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        
        self.fc2 = nn.Linear(width, out_width)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        x = self.fc2(x)
        return x


class KernelNN_subgraph(nn.Module):
    def __init__(
        self,
        width,  # number features per node after lifting
        ker_width,  # dimension of single kernel layer
        depth,  # number of graph layers / message passings
        ker_in,  # number of edge features
        known_node_dim=1,  # only needed when lifting is applied within model
        unknown_node_dim=1,
        out_width=2,  # dimension of output
    ):
        super(KernelNN_subgraph, self).__init__()

        self.depth = depth
        self.out_width = (
            out_width  # added to use out_width | probably not necessary to do this
        )

        self.L1 = nn.Linear(known_node_dim, width)
        self.L2 = nn.Linear(unknown_node_dim, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr="mean")
        
        # Separate kernel for cross-edge convolutions (cross_edges have dimension 8)
        kernel_cross = DenseNet([ker_in, ker_width, ker_width, width**2], nn.ReLU)
        self.conv_cross = NNConv_old(width, width, kernel_cross, aggr="mean")

        self.Proj = nn.Linear(width, out_width)  # nn.Linear(width, 1)

    def forward(self, data):
        # subgraph of known inputs
        features_a, edge_idx_a, edges_a = data.source.x, data.source.edge_index, data.source.edge_attr# OLD: data.vert_in_features, data.edge_idx_in, data.edges_in
        # subgraph of prediction nodes
        features_b, edge_idx_b, edges_b = data.target.x, data.target.edge_index, data.target.edge_attr # OLD: data.vert_out_features_constructed, data.edge_idx_out, data.edges_out
        # cross graph components
        cross_edges, cross_edge_idx = data.cross.edges, data.cross.edge_index# OLD: data.cross_edges, data.cross_edge_idx

        # lifting 
        features_a = self.L1(features_a)
        features_b = self.L2(features_b)


        for k in range(self.depth):
            # original code: x = F.relu(self.conv1(x, edge_index, edge_attr))
            # 1. decision: pass subgraph messages first and cross graph messages later
            # 2. decision: share the same kernel matrix network for all types of message passing
            features_a = F.relu(self.conv1(features_a, edge_idx_a, edges_a)) # message passing A -> A
            features_b = F.relu(self.conv1(features_b, edge_idx_b, edges_b)) # MP B -> B
            #
            # Correctly offset only the target indices (which are in B)
            # Source indices (in A) remain 0..N_a-1
            # Target indices (in B) become N_a..N_a+N_b-1
            cross_edge_index_cat = torch.stack([
                cross_edge_idx[0],
                cross_edge_idx[1] + features_a.shape[0]
            ], dim=0)
            
            features_b = F.relu(self.conv_cross(torch.cat([features_a, features_b], dim=0), 
                                            cross_edge_index_cat, 
                                            cross_edges)[features_a.shape[0]:]
                                   ) # MP A -> B (using cross kernel)
            # no messages B -> A
        prediction = self.Proj(features_b)
        return prediction 


# ============================================================================
# Loss Functions
# ============================================================================

def transition_prob_t_dist(a, b):
    """Compute t-distribution based transition probabilities."""
    return 1 / (1 + torch.sum((a - b) ** 2, dim=-1))

def get_output_affinities_local_norm(output, edges):
    """
    Compute output affinities using t-distribution.
    
    Args:
        output: Model output embeddings [N, D]
        edges: Edge index [2, E]
        
    Returns:
        Normalized affinities [N, k]
    """
    eps = 1e-6
    w = transition_prob_t_dist(output[edges[0,:]], output[edges[1,:]])
    w = w.view(output.shape[0], -1)  # reshape to [N, k]
    q = w / (w.sum(dim=1, keepdim=True) + eps)
    return q



def student_t_kernel(x, y):
    """
    Student-t kernel similarity between two sets of points.
    x, y: [E, D]
    returns: [E]
    """
    return 1.0 / (1.0 + ((x - y) ** 2).sum(dim=1))


def estimate_global_Z(output, num_samples=4096):
    """
    Monte Carlo estimate of Z = sum_{i!=j} 1/(1 + ||y_i - y_j||^2)
    """
    N = output.shape[0]
    device = output.device

    idx_i = torch.randint(0, N, (num_samples,), device=device)
    idx_j = torch.randint(0, N, (num_samples,), device=device)

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    diff = output[idx_i] - output[idx_j]
    w = 1.0 / (1.0 + (diff ** 2).sum(dim=1))

    Z_est = w.mean() * (N * (N - 1))
    return Z_est

def exact_global_Z(output, eps=1e-8):
    """
    Exact computation of
    Z = sum_{i != j} 1 / (1 + ||y_i - y_j||^2)

    Args:
        output: Tensor [N, D]
    Returns:
        scalar tensor Z
    """
    # pairwise squared distances: [N, N]
    dist2 = torch.cdist(output, output, p=2.0) ** 2

    # student-t kernel
    w = 1.0 / (1.0 + dist2)

    # remove diagonal (i == j)
    w.fill_diagonal_(0.0)

    Z = w.sum() + eps
    return Z



def get_output_affinities_global_norm(output, edges, num_mc_samples=4096, eps=1e-8):
    """
    Compute low-D t-SNE affinities for neighbors with global Monte Carlo normalization.
    
    Args:
        output: Tensor [N, D]
        edges: LongTensor [2, E] (neighbor index pairs)
        num_mc_samples: samples for global normalization
        eps: numerical stability
        
    Returns:
        q: Tensor [N, k] same shape as original method
    """
    N = output.shape[0]
    k = edges.shape[1] // N  # assuming edges are ordered per node as in your original method

    # ---- estimate global normalization ----
    #Z = estimate_global_Z(output, num_mc_samples) + eps
    
    # ---- exact global normalization ----
    Z = exact_global_Z(output, eps=eps)

    # ---- compute neighbor similarities ----
    w = student_t_kernel(output[edges[0]], output[edges[1]])

    # ---- reshape to [N, k] ----
    q = w.view(N, k)

    # ---- normalize globally ----
    q = q / Z

    return q


get_output_affinities = get_output_affinities_local_norm


def custom_KL_loss(affinities_in, affinities_out, eps=1e-6):
    """
    KL divergence loss between input and output affinities.
    
    Args:
        affinities_in: Target affinities (from data) [N, k]
        affinities_out: Predicted affinities [N, k]
        eps: Small constant for numerical stability
        
    Returns:
        KL divergence loss
    """
    p = affinities_in
    q = affinities_out
    
    loss = (p * torch.log((p + eps) / (q + eps))).sum()
    return loss


# ============================================================================
# Data Loading
# ============================================================================

class GraphLMDBReader:
    """
    Reader for loading graph data from LMDB databases.
    
    Usage:
        reader = GraphLMDBReader('datasets/train.lmdb')
        sample = reader[0]  # Load first sample
    """
    
    def __init__(self, db_path):
        self.env = lmdb.open(db_path, readonly=True, lock=False, map_size=10 * 1024**3)
        with self.env.begin() as txn:
            self.length = int(txn.get(b'__len__').decode())
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            byte_data = txn.get(str(idx).encode())
            return pickle.loads(byte_data)


# ============================================================================
# Device Setup
# ============================================================================

def setup_device():
    """
    Detect and return the best available device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
