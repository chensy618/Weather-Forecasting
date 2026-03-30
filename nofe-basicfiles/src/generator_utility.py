#%%
import os
import torch
from rasterio.windows import from_bounds
import numpy as np
from pyproj import Transformer
import faiss
from torch_geometric.data import Data
from torch_scatter import scatter


# %% 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f'using Device: {device}')


import time
import functools
from collections import defaultdict

# shared registry
_function_timings = defaultdict(lambda: {"total": 0.0, "calls": 0})


def track_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        stats = _function_timings[func.__name__]
        stats["total"] += elapsed
        stats["calls"] += 1

        return result

    return wrapper

# %% define graph_generator
class Graph_Generator:
    """
    This class is a parent class for the actual generator calsses.
    Actual classes will need to extend this e.g. by loading
    """
    def __init__(self, datafolder:str):
        self.directory= os.path.realpath(datafolder)
    
    @track_time
    def load_data(self):
        # should be implemented by subclass
        pass
    
    @track_time
    def generate_normed_sample(self, N:int, method:str):
        # should be implemented by subclass
        pass
        
        #self.config = sample_config.copy()
        # Dynamically create attributes for easy access
        #for key, value in sample_config.items():
        #    setattr(self, key, value)


    @track_time
    def get_edges(self, source:np.ndarray=None, target:np.ndarray=None, k:int=None, perplexity:int=None, mode:str="self_search"):
        """
        Searches for k nearest neighbors of 'target' in 'source'.
        
        :param source: dataset to find neighbors in (coordinates)
        :type source: np.ndarray
        :param target: dataset with points that neighbors are searched for (coordinates)
        :type target: np.ndarray
        :param k: number of neighbors
        :type k: int
        :param cross_search: determines operations
        :type cross_search: bool
        """

        if perplexity is None:
            perplexity = 0
        
        N_per = 3*perplexity
        
        if mode=="self_search":
            NN = max(k+1, N_per+1)
        else:
            NN = max(k, N_per+1)


        
        N, D = source.shape
        index = faiss.IndexFlatL2(D)  # L2 distance index (Euclidean)
        index.add(source)
        distance_tab, indices_tab = index.search(target, NN) 


        # get long table of edge indices and edge distances
        if mode == "cross_search":
            dist_tab_k, idx_tab_k = distance_tab[:, :k], indices_tab[:, :k]       
        elif mode == "self_search": 
            dist_tab_k, idx_tab_k = distance_tab[:, 1:k+1], indices_tab[:, 1:k+1]       
        else:
            print("no valid mode was provided!")
        
        source_idx = idx_tab_k.reshape(-1)
        target_idx = np.repeat(np.arange(target.shape[0]),k)
        edge_indices = torch.from_numpy(np.vstack([source_idx, target_idx]))
        
        edge_dist = dist_tab_k.reshape(-1, 1)
       
        # get neighborhood index table for sigma calulation
        idx_tab_p = indices_tab[:, 1:N_per+1]


        return edge_indices, edge_dist, idx_tab_p


    @track_time
    def feature_distances(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distances for neighboring node pairs.

        Args:
        indices: LongTensor of shape (N, p) specifying neighbor indices for each node.
        values: FloatTensor of shape (N, f) with coordinates for each node.

        Returns:
        distances: FloatTensor of shape (N, p) with Euclidean distances to neighbors.
        """
        # Gather neighbor coordinates
        # indices: (N, p), values: (N, f) -> neighbor_values: (N, p, f)
        neighbor_values = values[indices] # advanced indexing in PyTorch

        # Expand original values to (N, 1, f) to broadcast
        node_values = values.unsqueeze(1) # (N, 1, f)

        # Compute Euclidean distance along last axis (f)
        distances = torch.norm(node_values - neighbor_values, dim=-1) # (N, p)

        return distances



    @track_time
    def get_edge_attributes(
        self,
        source_coords: torch.Tensor,    # (N, f)
        target_coords: torch.Tensor,    # (N, f)
        edge_indices: torch.Tensor,     # (2, E), dtype=torch.long
        edge_dist: torch.Tensor,        # (E, 1)
        k: int,
        include_feat_diff: bool = False
    ):
        """
        Returns:
            edge_attributes: (E, 3f + 1)
        """
        
        # Node positions (target)
        node_pos = target_coords[edge_indices[1]]  # (E, f)

        # ---- Regularized distances ----
        # max dist per source node (repeat k times)
        # edge_indices[0] tells us which node each edge belongs to
        max_dist_per_node = torch.zeros(
            source_coords.shape[0], device=edge_dist.device
        )

        max_dist_per_node.scatter_reduce_(
            0, edge_indices[0], edge_dist.squeeze(), reduce="amax"
        )

        regularized_distances = (
            edge_dist.squeeze() /
            max_dist_per_node[edge_indices[0]]
        ).unsqueeze(1)

        # ---- Displacement vectors ----
        source_nodes = source_coords[edge_indices[0]]
        target_nodes = target_coords[edge_indices[1]]
        absolute_displacement = source_nodes - target_nodes

        norm = torch.norm(absolute_displacement, dim=1, keepdim=True)
        normed_displacement = absolute_displacement / (norm + 1e-12)

        if include_feat_diff:
            print("including feature difference not yet implemented. Continuing without")

        # ---- Stack features ----
        edge_attributes = torch.cat(
            [node_pos, normed_displacement, regularized_distances],
            dim=1
        )

        return edge_attributes


    @track_time
    def get_node_features(self, normed_coords:np.ndarray=None, normed_values:torch.Tensor=None, mode:str="channel_only"):
        if type(normed_coords) != torch.Tensor:
            normed_coords = torch.as_tensor(normed_coords, dtype=torch.float32)
        
        if mode=="channel_only":
            node_features = normed_values
        elif mode=="position_only":
            node_features=normed_coords
        elif mode=="channel_and_position":
            node_features = torch.cat([normed_values, normed_coords], dim=1)
        else:
            print('invalid node feature mode')
            
        return node_features


    @track_time
    def inverse_distance_interpolation(self,
        normed_values_source: torch.Tensor,   # (N, F)
        normed_coords_source: torch.Tensor,   # (N, D)
        normed_coords_target: torch.Tensor,   # (M, D)
        cross_edge_index: torch.Tensor,         # (2, E)
        eps: float = 1e-12
    ):
        """
        Interpolate source features to target points using inverse distance weighting.
        
        Returns:
            v_target: (M, F) interpolated features at target points
        """
        # 1. Extract source and target indices
        source_idx = cross_edge_index[0]
        target_idx = cross_edge_index[1]
        
        # 2. Compute distances for each edge
        diff = normed_coords_target[target_idx] - normed_coords_source[source_idx]  # (E, D)
        dist = torch.norm(diff, dim=1) + eps  # avoid division by zero, shape (E,)
        
        # 3. Compute weights
        weights = 1.0 / dist  # (E,)
        
        # 4. Weighted sum of source features
        weighted_values = normed_values_source[source_idx] * weights[:, None]  # (E, F)
        v_target = scatter(weighted_values, target_idx, dim=0, reduce='sum')  # sum over neighbors
        
        # 5. Normalize by sum of weights
        sum_weights = scatter(weights, target_idx, dim=0, reduce='sum')  # (M,)
        v_target /= sum_weights[:, None]
        
        return v_target


    @track_time
    def optimize_sigma(self, distances, perplexity, tol=1e-5, max_iter=50):
        """
        Compute sigma (bandwidth) for each point based on target perplexity using binary search.
        
        Parameters:
        - distances: np.ndarray of shape (N, f), distances from each point to its f nearest neighbors
        - perplexity: float, target perplexity
        - tol: float, tolerance for convergence
        - max_iter: int, maximum number of iterations in binary search
        
        Returns:
        - sigmas: np.ndarray of shape (N,), the sigma for each point
        """
        distances = distances.cpu().numpy()
        N, f = distances.shape
        sigmas = np.zeros(N)
        log_perplexity = np.log(perplexity)
        
        for i in range(N):
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0  # initial guess
            Di = distances[i]  # distances of point i to its neighbors
            
            # Binary search for beta (1 / (2*sigma^2))
            for _ in range(max_iter):
                # Compute Gaussian affinities
                P = np.exp(-Di * beta)
                sumP = np.sum(P)
                if sumP == 0:
                    sumP = 1e-10  # avoid division by zero
                P /= sumP
                
                # Compute entropy H(P)
                H = -np.sum(P * np.log(P + 1e-10))
                H_diff = H - log_perplexity
                
                if np.abs(H_diff) < tol:
                    break
                
                # Update beta using binary search
                if H_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf or beta_max == -np.inf:
                        beta *= 2.0
                    else:
                        beta = (beta + beta_max) / 2.0
                else:
                    beta_max = beta
                    if beta_min == -np.inf or beta_min == np.inf:
                        beta /= 2.0
                    else:
                        beta = (beta + beta_min) / 2.0
            
            # Convert beta to sigma
            sigmas[i] = np.sqrt(1 / (2 * beta))
        
        return torch.tensor(sigmas, dtype=torch.float32)

#TODO: does not work properly:
    def optimize_sigma_gpu(self, distances: torch.Tensor, perplexity: float,
        tol=1e-5, max_iter=50) -> torch.Tensor:
        """
        Optimize sigma for each point to match target perplexity.

        Args:
        distances: FloatTensor of shape (N, p) with squared distances to neighbors.
        perplexity: desired perplexity value (scalar).
        tol: tolerance for perplexity matching.
        max_iter: max binary search iterations.

        Returns:
        sigmas: FloatTensor of shape (N,) with optimized sigma values.
        """
        N, p = distances.shape
        # Convert distances to float32 for stability
        distances = distances.float()

        # Compute log2(perplexity)
        log_perplexity = torch.log2(torch.tensor(perplexity, device=distances.device))

        # Initialize sigma bounds
        sigma_min = torch.zeros(N, device=distances.device)
        sigma_max = torch.full((N,), 1e5, device=distances.device)
        sigma = torch.full((N,), 1.0, device=distances.device)

        for _ in range(max_iter):
            # Compute conditional probabilities
            # p_ij ~ exp(-d_ij^2 / 2σ^2)
            # sigma: (N,) -> (N,1) for broadcasting
            p_ij = torch.exp(-distances / (2 * sigma.unsqueeze(1)**2))
            # Normalize
            p_ij_sum = p_ij.sum(dim=1, keepdim=True)
            p_ij = p_ij / p_ij_sum

            # Compute entropy H(P_i)
            H = -(p_ij * torch.log2(p_ij + 1e-12)).sum(dim=1) # shape (N,)

            # Binary search update
            mask = H > log_perplexity # need larger sigma
            sigma_min = torch.where(mask, sigma, sigma_min)
            sigma_max = torch.where(mask, sigma_max, sigma)
            # Avoid sigma_max == 1e5 for the first iteration
            sigma = (sigma_min + sigma_max) / 2

            # Early stopping if all entropies within tolerance
            if torch.all(torch.abs(H - log_perplexity) < tol):
                break

        return sigma




    @track_time
    def get_affinities(self, normed_values:torch.Tensor=None, edge_indices:torch.Tensor=None, sigmas:np.ndarray=None):
        """
        Compute local affinity matrix using per-point sigma values.

        Args:
            var_tens (Tensor): [N, D] input features
            edges (LongTensor): [2, E] edge list, edges[:,0] = source, edges[:,1] = target
            sigmas (Tensor): [N] per-point sigma values for each source node

        Returns:
            Tensor: [N, k] affinity values (p_j|i) normalized per source node
        """
        
        eps = 1e-6
        
        # Get node embeddings for source and target
        x_i = normed_values[edge_indices[0, :]]  # [E, D]
        x_j = normed_values[edge_indices[1, :]]  # [E, D]
        sigma_i = sigmas[edge_indices[0, :]]  # [E]
        
        # Compute squared distances
        sqdist = torch.sum((x_i - x_j) ** 2, dim=1)
        
        # Compute affinities using Gaussian kernel with per-point sigma
        denom = 2 * sigma_i ** 2 + eps
        v = torch.exp(-sqdist / denom)  # [E]
        
        # Group v into rows of shape [N, k]
        N = normed_values.shape[0]
        k = v.numel() // N
        v = v.view(N, k)
        
        # Normalize to get conditional probabilities p_j|i
        p = v / (v.sum(dim=1, keepdim=True) + eps)  # [N, k]
        p = p.to(torch.float32)
        
        return p #input affinities


    def sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random", gen_feature_mode:str="channel_only"):
        #normed_values, gps = self.generate_normed_sample(N=N, method=sampling_method) # sample subset 
        #cartesian = self.cartesian_projection(lat=gps[:, 0], lon=gps[:, 1]) # get 3 dim. cartesian coordinates for sample locations
        #normed_coords = self.norm_coords(cartesian=cartesian) # norm coordinates based on subregion
        normed_values, normed_coords, meta = self.generate_normed_sample(N=N, method=sampling_method)

        edge_indices, edge_dist, idx_tab_p = self.get_edges(source=normed_coords, target=normed_coords, k=k, perplexity=perplexity, mode="self_search")

        # --- GPU transition ---
        edge_indices = torch.as_tensor(edge_indices, device=device, dtype=torch.long) 
        edge_dist = torch.as_tensor(edge_dist, device=device, dtype=torch.float32)
        idx_tab_p = torch.as_tensor(idx_tab_p, device=device, dtype=torch.long)
        normed_values = torch.as_tensor(normed_values, device=device, dtype=torch.float32)
        normed_coords = torch.tensor(normed_coords, device=device, dtype=torch.float32)

        node_features = self.get_node_features(normed_coords=normed_coords, normed_values=normed_values, mode=gen_feature_mode)

        edge_attributes = self.get_edge_attributes(source_coords=normed_coords, 
                            target_coords=normed_coords, 
                            edge_indices=edge_indices, 
                            edge_dist=edge_dist,
                            k=k,
                            include_feat_diff=False)

        if perplexity is not None and perplexity != 0:
            feature_distances = self.feature_distances(indices=idx_tab_p, values=normed_values)
            meta['feature_distances'] = feature_distances
            sigmas = self.optimize_sigma(distances=feature_distances, perplexity=perplexity)
            sigmas = sigmas.to(device=device)
            input_affinities = self.get_affinities(normed_values=normed_values, edge_indices=edge_indices, sigmas=sigmas)
        else:
            input_affinities, sigmas = None, None

        return Data(x=node_features, edge_index=edge_indices, edge_attr=edge_attributes, edge_dist=edge_dist, input_affinities=input_affinities, sigmas=sigmas, normed_coords=normed_coords, normed_values=normed_values, meta=meta)



    @track_time
    def connect_graphs(self, in_graph:Data=None, out_graph:Data=None, k_cross:int=None):
        """
        This method creates a cross graph structure.
        the output is something like
        out_graph = sample_graph(self, N:int=None, k:int=None, perplexity:int=None, sampling_method:str="random")
        
        returns list of Data objects
        !!! Careful when writing data as it is in form of a list !!!
        """
        cross_edge_indices, cross_edge_dist, _  = self.get_edges(source=in_graph.normed_coords.cpu().numpy(), 
                                                                                 target=out_graph.normed_coords.cpu().numpy(), 
                                                                                 k=k_cross, mode="cross_search", perplexity=0)
        cross_edge_indices = torch.as_tensor(cross_edge_indices, device=device, dtype=torch.long)
        cross_edge_dist = torch.as_tensor(cross_edge_dist, device=device, dtype=torch.float32)
        

        
        cross_edge_attributes = self.get_edge_attributes(source_coords=in_graph.normed_coords, 
                            target_coords=out_graph.normed_coords, 
                            edge_indices=cross_edge_indices, 
                            edge_dist=cross_edge_dist,
                            k=k_cross,
                            include_feat_diff=False,
                            )
        
        cross_graph = Data(source = in_graph,
                           target = out_graph,
                           cross = Data(
                               edges=cross_edge_attributes,
                               edge_index=cross_edge_indices
                               )
                           )
        
        
        return cross_graph
    
    
    @track_time
    def construct_output_node_features(self, cross_graph:Data, interpolate:bool=True, gen_feature_mode:str="channel_only"):
        """
        construct node features for the output graph
        
        interpolate: if true: interpolates channel values for target graph based on neighboring values from source graph
        gen_feature_mode: determines whether features contain channel values, position or both
        """
        normed_values_source=cross_graph.source.x
        # more efficient. previously: normed_coords_source=torch.tensor(cross_graph.vert_in)
        normed_coords_source=torch.as_tensor(cross_graph.source.normed_coords)
        # more efficient. previously: normed_coords_target=torch.tensor(cross_graph.vert_out)
        normed_coords_target=torch.as_tensor(cross_graph.target.normed_coords)
        cross_edge_index=cross_graph.cross.edge_index
        
        if interpolate and (gen_feature_mode != "position_only"):
            interpolated_values = self.inverse_distance_interpolation(
                normed_values_source=normed_values_source,   # (N, F)
                normed_coords_source=normed_coords_source,   # (N, D)
                normed_coords_target=normed_coords_target,   # (M, D)
                cross_edge_index=cross_edge_index,         # (2, E)
                eps = 1e-12)
            constructed_features=self.get_node_features(normed_coords=normed_coords_target, normed_values=interpolated_values, mode=gen_feature_mode)
            
        else:
            constructed_features=self.get_node_features(normed_coords=normed_coords_target, mode=gen_feature_mode)
        
        return constructed_features.to(torch.float32)
    
    
    @track_time
    def cross_graph_sampling(self, n_samples:int=None, perplexity:int=None, k_cross:int=None, interpolate_graph_features:bool=True, gen_feature_mode:str="channel_only",
                             N_in:int=None, k_in:int=None, input_sampling_method:str="random",
                             N_out:int=None, k_out:int=None, output_sampling_method:str="regular_subset"):
        graph_list = []
        
        out_graph=self.sample_graph(N=N_out, k=k_out, perplexity=perplexity, sampling_method=output_sampling_method)
        print('output graph done!')
        for s in range(n_samples):
            print(f'{s+1}/{n_samples}')
            in_graph = self.sample_graph(N=N_in, k=k_in, perplexity=0, sampling_method=input_sampling_method)
            cross_graph = self.connect_graphs(in_graph=in_graph, out_graph=out_graph, k_cross=k_cross)
            cross_graph.cross["x"] = self.construct_output_node_features(cross_graph=cross_graph, interpolate=interpolate_graph_features, gen_feature_mode=gen_feature_mode)
            graph_list.append(cross_graph)
        return graph_list

