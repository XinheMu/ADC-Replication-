from __future__ import division
import torch
import sys
import ast
from fractions import Fraction
torch.set_num_threads(36) 
torch.set_grad_enabled(False)
import torch.nn as nn
import os
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from scipy.stats import qmc
from scipy.stats import norm
from typing import Tuple, Union


class SingleAttributeHistogram:
    def __init__(self, num_bins: int, num_mcvs: int):
        if num_bins <= 0 or num_mcvs < 0:
            raise ValueError("num_bins must be positive and num_mcvs must be non-negative.")
        self.num_bins = num_bins
        self.num_mcvs = num_mcvs
        # Placeholders for loaded data
        self.histograms = None
        self.mcv_info = None
        self.metadata = None

    def train(self, data: np.ndarray, hist_path='histograms.npy', mcv_path='mcv_info.npz', meta_path='metadata.npz'):
        """
        Trains the histograms and MCV lists on the given data and saves them to files.
        """
        print("Starting training...")
        data_torch = torch.from_numpy(data).float()
        num_rows, num_dimensions = data_torch.shape
        # --- 1. Calculate overall metadata ---
        min_vals = torch.min(data_torch, dim=0).values
        max_vals = torch.max(data_torch, dim=0).values
        self.metadata = {
            'min_vals': min_vals.numpy(),
            'max_vals': max_vals.numpy(),
            'total_rows': num_rows
        }
        # --- 2. Initialize storage for histograms and MCVs ---
        hist_counts_tensor = torch.zeros((num_dimensions, self.num_bins), dtype=torch.float32)
        mcv_values_tensor = torch.full((num_dimensions, self.num_mcvs), float('nan'), dtype=torch.float32)
        mcv_counts_tensor = torch.zeros((num_dimensions, self.num_mcvs), dtype=torch.float32)
        # --- 3. Process each dimension (attribute) separately ---
        for d in range(num_dimensions):
            print(f"  Processing dimension {d+1}/{num_dimensions}...")
            column_data = data_torch[:, d]
            # --- 4. Identify and separate MCVs ---
            if self.num_mcvs > 0:
                unique_vals, counts = torch.unique(column_data, return_counts=True)
                sorted_indices = torch.argsort(counts, descending=True)
                num_actual_mcvs = min(self.num_mcvs, len(unique_vals))
                top_indices = sorted_indices[:num_actual_mcvs]
                mcv_values = unique_vals[top_indices]
                mcv_counts = counts[top_indices]
                mcv_values_tensor[d, :num_actual_mcvs] = mcv_values
                mcv_counts_tensor[d, :num_actual_mcvs] = mcv_counts
                # Create a mask to filter out MCVs from the column data for histogramming
                is_mcv = torch.isin(column_data, mcv_values)
                non_mcv_data = column_data[~is_mcv]
            else:
                non_mcv_data = column_data
            # --- 5. Build histogram on the remaining (non-MCV) data ---
            # Use original min/max for consistent binning across all data
            col_min = min_vals[d].item()
            col_max = max_vals[d].item()
            if col_min < col_max and len(non_mcv_data) > 0:
                hist = torch.histc(non_mcv_data, bins=self.num_bins, min=col_min, max=col_max)
                hist_counts_tensor[d, :] = hist
        self.histograms = hist_counts_tensor.numpy()
        self.mcv_info = {'values': mcv_values_tensor.numpy(), 'counts': mcv_counts_tensor.numpy()}
        # --- 6. Save all computed information to files ---
        print(f"Saving histogram data to {hist_path}")
        np.save(hist_path, self.histograms)
        print(f"Saving MCV data to {mcv_path}")
        np.savez(mcv_path, **self.mcv_info)
        print(f"Saving metadata to {meta_path}")
        np.savez(meta_path, **self.metadata)
        print("Training complete.")
    def load(self, hist_path='histograms.npy', mcv_path='mcv_info.npz', meta_path='metadata.npz'):
        """
        Loads pre-trained histogram and MCV data from files.
        """
        print("Loading pre-trained model...")
        if not all(os.path.exists(p) for p in [hist_path, mcv_path, meta_path]):
            raise FileNotFoundError("One or more required model files are missing.")
            
        self.histograms = np.load(hist_path)
        self.mcv_info = np.load(mcv_path)
        self.metadata = np.load(meta_path)
        
        # Verify loaded data matches instance config
        assert self.histograms.shape[1] == self.num_bins, "Loaded histogram has mismatched num_bins."
        assert self.mcv_info['values'].shape[1] == self.num_mcvs, "Loaded MCVs have mismatched num_mcvs."
        print("Model loaded successfully.")

    def estimate(self, dimension: int, lower_bound: float, upper_bound: float) -> float:
        """
        Estimates the cardinality for a single-attribute range query.
        Query: SELECT COUNT(*) FROM table WHERE lower_bound <= attribute[dimension] <= upper_bound.

        Args:
            dimension (int): The index of the attribute (column) to query.
            lower_bound (float): The lower bound of the query range (inclusive).
            upper_bound (float): The upper bound of the query range (inclusive).

        Returns:
            float: The estimated cardinality.
        """
        if self.histograms is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        if not (0 <= dimension < self.histograms.shape[0]):
            raise ValueError(f"Dimension must be between 0 and {self.histograms.shape[0]-1}.")
        estimated_cardinality = 0.0
        # --- 1. Add counts from MCVs that fall within the query range ---
        if self.num_mcvs > 0:
            mcv_vals = self.mcv_info['values'][dimension]
            mcv_counts = self.mcv_info['counts'][dimension]
            # Create a boolean mask for MCVs within the range
            in_range_mask = (mcv_vals >= lower_bound) & (mcv_vals <= upper_bound)
            estimated_cardinality += mcv_counts[in_range_mask].sum()
        # --- 2. Estimate cardinality from the histogram ---
        hist_counts = self.histograms[dimension]
        min_val = self.metadata['min_vals'][dimension]
        max_val = self.metadata['max_vals'][dimension]
        if min_val >= max_val: # All values are the same, already handled by MCVs if frequent
            return estimated_cardinality
        bin_width = (max_val - min_val) / self.num_bins
        # Clamp query bounds to the data's actual min/max
        query_start = max(lower_bound, min_val)
        query_end = min(upper_bound, max_val)
        if query_start > query_end: # Query range is outside the data's range
            return estimated_cardinality
        # Find which bins the query range touches
        start_bin = int((query_start - min_val) / bin_width)
        end_bin = int((query_end - min_val) / bin_width)
        # Clamp bin indices to be safe
        start_bin = max(0, min(start_bin, self.num_bins - 1))
        end_bin = max(0, min(end_bin, self.num_bins - 1))
        if start_bin == end_bin:
            # Query is contained within a single bin
            bin_start_val = min_val + start_bin * bin_width
            overlap = query_end - query_start
            fraction = overlap / bin_width
            estimated_cardinality += fraction * hist_counts[start_bin]
        else:
            # Query spans multiple bins
            # a) Partial contribution from the start bin
            bin_end_val = min_val + (start_bin + 1) * bin_width
            overlap = bin_end_val - query_start
            fraction = overlap / bin_width
            estimated_cardinality += fraction * hist_counts[start_bin]
            # b) Full contribution from intermediate bins
            estimated_cardinality += hist_counts[start_bin + 1 : end_bin].sum()
            # c) Partial contribution from the end bin
            bin_start_val = min_val + end_bin * bin_width
            overlap = query_end - bin_start_val
            fraction = overlap / bin_width
            estimated_cardinality += fraction * hist_counts[end_bin]
        return estimated_cardinality
    

class BayesnetEstimator:
    def __init__(self, bayes_called_attribute, bayes_source_attribute, bayes_assist_attribute, 
                 S1, S2, bin_val, mcv_val):
        self.idx_y = bayes_called_attribute
        self.idx_x1 = bayes_source_attribute
        self.idx_x2 = bayes_assist_attribute
        self.S1 = S1
        self.S2 = S2
        self.bin_val = bin_val
        self.mcv_val = mcv_val
        
        self.filename = f"FD3_{self.idx_y}{self.idx_x1}{self.idx_x2}_{dataset_name}.npy"
        
        # Storage for model parameters (loaded later)
        # Grid boundaries for x1 and x2
        self.grid_x1 = None 
        self.grid_x2 = None
        
        # Per-cell stats: shape (S1, S2, ...)
        self.cell_y_min = None
        self.cell_y_max = None
        self.hist_weights = None # shape (S1, S2, bin_val)
        self.mcv_values = None   # shape (S1, S2, mcv_val)
        self.mcv_probs = None    # shape (S1, S2, mcv_val)
        self.device = torch.device("cpu") # Default to CPU

    def train(self, data_used):
        """
        Constructs the multidimensional histogram and saves to .npy.
        data_used: Tensor (size, dimension)
        """
        N = data_used.shape[0]
        X1 = data_used[:, self.idx_x1]
        X2 = data_used[:, self.idx_x2]
        Y  = data_used[:, self.idx_y]

        # 1. Define Grid for x1, x2 (Equal Width)
        x1_min, x1_max = X1.min().item(), X1.max().item()
        x2_min, x2_max = X2.min().item(), X2.max().item()
        
        # Epsilon to ensure max value falls in last bin
        eps = 1e-6
        self.grid_x1 = torch.linspace(x1_min, x1_max + eps, self.S1 + 1)
        self.grid_x2 = torch.linspace(x2_min, x2_max + eps, self.S2 + 1)

        # 2. Assign points to cells
        # bucket indices 0 to S-1
        idx_1 = torch.bucketize(X1, self.grid_x1, right=True) - 1
        idx_2 = torch.bucketize(X2, self.grid_x2, right=True) - 1
        
        # Clamp indices just in case
        idx_1 = idx_1.clamp(0, self.S1 - 1)
        idx_2 = idx_2.clamp(0, self.S2 - 1)

        # Initialize storage
        self.cell_y_min = torch.zeros((self.S1, self.S2))
        self.cell_y_max = torch.zeros((self.S1, self.S2))
        self.hist_weights = torch.zeros((self.S1, self.S2, self.bin_val))
        self.mcv_values = torch.zeros((self.S1, self.S2, self.mcv_val))
        self.mcv_probs = torch.zeros((self.S1, self.S2, self.mcv_val))
        
        filled_mask = torch.zeros((self.S1, self.S2), dtype=torch.bool)

        # 3. Populate Cells
        # Note: Python loops here are acceptable during training (offline)
        for i in range(self.S1):
            for j in range(self.S2):
                mask = (idx_1 == i) & (idx_2 == j)
                y_vals = Y[mask]
                
                if len(y_vals) == 0:
                    continue
                
                filled_mask[i, j] = True
                
                # Determine MCVs
                vals, counts = torch.unique(y_vals, return_counts=True)
                
                # Sort by count descending
                sorted_indices = torch.argsort(counts, descending=True)
                vals = vals[sorted_indices]
                counts = counts[sorted_indices]
                
                # Take top MCVs
                n_mcv = min(len(vals), self.mcv_val)
                self.mcv_values[i, j, :n_mcv] = vals[:n_mcv]
                prob_mcv_total = 0.0
                
                for k in range(n_mcv):
                    prob = counts[k].item() / len(y_vals)
                    self.mcv_probs[i, j, k] = prob
                    prob_mcv_total += prob
                
                # Build Histogram for remainder
                # Filter out MCV data from y_vals to build histogram on 'remainder'
                # Optimization: If MCV covers everything, skip hist
                remaining_prob = 1.0 - prob_mcv_total
                
                if remaining_prob > 1e-9:
                    # Identify non-MCV values
                    mcv_set = vals[:n_mcv]
                    # Create a mask for values NOT in mcv_set
                    # (Simple way for 1D tensor)
                    non_mcv_mask = torch.isin(y_vals, mcv_set, invert=True)
                    y_rem = y_vals[non_mcv_mask]
                    
                    if len(y_rem) > 0:
                        local_min, local_max = y_rem.min().item(), y_rem.max().item()
                        self.cell_y_min[i, j] = local_min
                        self.cell_y_max[i, j] = local_max
                        
                        # Equal width histogram
                        hist = torch.histc(y_rem, bins=self.bin_val, min=local_min, max=local_max)
                        # Normalize so sum of hist * weights + sum MCV = 1
                        # current sum(hist) is count of remainder.
                        # We want sum(hist_weights) = remaining_prob
                        self.hist_weights[i, j] = (hist / len(y_rem)) * remaining_prob
                    else:
                        # Remainder exists conceptually but empty due to precision? Just set min/max
                        self.cell_y_min[i, j] = vals[0]
                        self.cell_y_max[i, j] = vals[0]
                else:
                    # All mass in MCV. Set range to MCV range just for safety
                    self.cell_y_min[i, j] = vals[0]
                    self.cell_y_max[i, j] = vals[0]

        # 4. Handle Empty Cells (Nearest Neighbor Imputation)
        # Create coordinate grids
        ii, jj = torch.meshgrid(torch.arange(self.S1), torch.arange(self.S2), indexing='ij')
        
        # Coordinates of filled cells
        filled_coords = torch.stack([ii[filled_mask], jj[filled_mask]], dim=1).float()
        
        # Iterate over empty cells
        empty_indices = torch.nonzero(~filled_mask, as_tuple=False)
        
        if filled_coords.shape[0] > 0:
            for idx in empty_indices:
                r, c = idx[0].item(), idx[1].item()
                target = torch.tensor([r, c], dtype=torch.float)
                
                # Compute differences
                diffs = torch.abs(filled_coords - target)
                diff_x1 = diffs[:, 0]
                diff_x2 = diffs[:, 1]
                
                score = (torch.clamp(diff_x1-0.8,min=0) * (self.S2 + 0.1))**2 + (torch.clamp(diff_x2-0.8,min=0) * self.S1)**2
                nearest_idx = torch.argmin(score)
                
                nr, nc = filled_coords[nearest_idx].long().tolist()
                
                # Copy parameters
                self.cell_y_min[r, c] = self.cell_y_min[nr, nc]
                self.cell_y_max[r, c] = self.cell_y_max[nr, nc]
                self.hist_weights[r, c] = self.hist_weights[nr, nc]
                self.mcv_values[r, c] = self.mcv_values[nr, nc]
                self.mcv_probs[r, c] = self.mcv_probs[nr, nc]

        # Save
        state = {
            'grid_x1': self.grid_x1,
            'grid_x2': self.grid_x2,
            'cell_y_min': self.cell_y_min,
            'cell_y_max': self.cell_y_max,
            'hist_weights': self.hist_weights,
            'mcv_values': self.mcv_values,
            'mcv_probs': self.mcv_probs
        }
        np.save(self.filename, state)

    def load(self):
        if not os.path.exists(dataset_name+'/'+self.filename):
            raise FileNotFoundError(f"Bayesnet model file not found.")
        state = np.load(dataset_name+'/'+self.filename, allow_pickle=True).item()
        self.grid_x1 = state['grid_x1']
        self.grid_x2 = state['grid_x2']
        self.cell_y_min = state['cell_y_min']
        self.cell_y_max = state['cell_y_max']
        self.hist_weights = state['hist_weights']
        self.mcv_values = state['mcv_values']
        self.mcv_probs = state['mcv_probs']

    def get_all_slice_cond_probs(self, query_range):
        """
        query_range: Tensor shape (2) -> [p1, q1]
        Returns: Tensor shape (S1, S2) containing P(y in [p1,q1] | cell_jk)
        Uses vectorization (no loops).
        """
        p1, q1 = query_range[0], query_range[1]
        
        # 1. MCV Contribution
        # Mask: (S1, S2, mcv_val)
        mcv_hit = (self.mcv_values >= p1) & (self.mcv_values <= q1)
        # Sum prob where hit occurs
        prob_mcv = (mcv_hit * self.mcv_probs).sum(dim=-1)
        
        # 2. Histogram Contribution
        # Expand dims for broadcasting against bins
        # cell_y_min: (S1, S2) -> (S1, S2, 1)
        mins = self.cell_y_min.unsqueeze(-1)
        maxs = self.cell_y_max.unsqueeze(-1)
        
        # Calculate bin edges for all cells simultaneously
        # Bin width: (S1, S2, 1)
        widths = (maxs - mins) / self.bin_val
        # Handle case where max == min (width 0), set to 1 to avoid NaN (prob will be 0 anyway)
        widths[widths == 0] = 1.0 
        
        # Create bin indices [0, ..., bin_val-1]
        k = torch.arange(self.bin_val, device=self.device) # (B)
        
        # Calculate low and high for every bin in every cell
        # shape (S1, S2, B)
        bin_lows = mins + k * widths
        bin_highs = bin_lows + widths
        # Clamp last bin high to max to handle precision
        # (Optional, but good for consistency)
        
        # Calculate overlap of [p1, q1] with [bin_low, bin_high]
        # Overlap = max(0, min(q1, bin_high) - max(p1, bin_low))
        # Broadcasting p1, q1 against (S1, S2, B)
        inter_low = torch.maximum(bin_lows, p1)
        inter_high = torch.minimum(bin_highs, q1)
        overlap = (inter_high - inter_low).clamp(min=0)
        
        # Fraction of bin covered
        frac = overlap / widths
        
        # Weighted sum of fractions
        prob_hist = (frac * self.hist_weights).sum(dim=-1)
        
        total_prob = prob_mcv + prob_hist
        return total_prob.to(torch.float32)


class GMM_Estimator:
    def __init__(self, dimension_init, kernels_matrix, 
                 bayes_called_attributes, bayes_source_attributes, bayes_assist_attributes, 
                 S1, S2, bin_val, mcv_val):
        """
        kernels_matrix: shape (2n+1, kernels_num). 
        """
        self.dim_init = dimension_init
        self.non_bayes_indexes=[i not in bayes_called_attributes for i in range(0,dimension_init)]
        self.bayes_called = bayes_called_attributes
        self.bayes_source = bayes_source_attributes
        self.bayes_assist = bayes_assist_attributes
        fullrange_query=torch.tensor([[1.5001 for i in range(0,dimension_init)],[-1.5001 for i in range(0,dimension_init)]])
        self.edge_values=fullrange_query-0.05*(fullrange_query<reg_const_two[0:1,:])-0.05*(fullrange_query<reg_const_two[1:2,:])+0.05*(fullrange_query>reg_const_two[2:3,:])+0.05*(fullrange_query>reg_const_two[3:4,:])
 
        # Device handling (assuming CPU for now, but prepared for cuda)
        self.device = kernels_matrix.device
        
        self.kernels_num = kernels_matrix.shape[1]
        n_gmm = (kernels_matrix.shape[0] - 1) // 2
        self.means = kernels_matrix[:n_gmm, :]                         
        kernels_std_unperturbed=torch.clone(kernels_matrix[dimension:2*dimension]).to(torch.float32)
        kernels_std=[kernels_std_unperturbed]
        for i in range(0,7):
            kernels_std.append(torch.sqrt(kernels_matrix[dimension:2*dimension]**2+Var_min[i]).to(torch.float32))
        self.stds = torch.clone(kernels_std[0])
        self.stds_perturbed = [torch.clone(kernels_std[i]) for i in range(1,8)]
        self.weights = (kernels_matrix[-1, :]).to(torch.float64)/torch.sum((kernels_matrix[-1, :]).to(torch.float64))
        self.weights = self.weights.to(torch.float32)        
        self.gaussian_coefficient=[torch.unsqueeze(torch.prod(self.stds_perturbed[i],0),1)*(ma.sqrt(2*3.14159265))**dimension for i in range(0,7)]

        # Initialize Bayesnet Estimators
        self.bayes_estimators = []
        for y, x1, x2 in zip(self.bayes_called, self.bayes_source, self.bayes_assist):
            bn = BayesnetEstimator(y, x1, x2, S1, S2, bin_val, mcv_val)
            bn.load() 
            self.bayes_estimators.append(bn)
            
        # --- 1. Map Attributes and Identify "Pure" vs "Hybrid" ---
        
        # Map: Original Attribute Index -> GMM Row Index (-1 if not in GMM)
        self.orig_to_gmm_map = [-1] * dimension_init
        gmm_counter = 0
        for i in range(dimension_init):
            if i not in self.bayes_called:
                self.orig_to_gmm_map[i] = gmm_counter
                gmm_counter += 1
        
        # Identify GMM indices used by Bayesnets (x1, x2)
        self.hybrid_gmm_indices = set()
        for bn in self.bayes_estimators:
            self.hybrid_gmm_indices.add(self.orig_to_gmm_map[bn.idx_x1])
            self.hybrid_gmm_indices.add(self.orig_to_gmm_map[bn.idx_x2])
            
        # Identify "Pure" GMM indices (Attributes in GMM but NOT x1 or x2)
        pure_gmm_list = []
        pure_orig_list = []
        
        for orig_idx, gmm_idx in enumerate(self.orig_to_gmm_map):
            if gmm_idx != -1 and gmm_idx not in self.hybrid_gmm_indices:
                pure_gmm_list.append(gmm_idx)
                pure_orig_list.append(orig_idx)
                
        # Convert to tensors for fast indexing during inference
        self.pure_gmm_indices = torch.tensor(pure_gmm_list, dtype=torch.long, device=self.device)
        self.pure_orig_indices = torch.tensor(pure_orig_list, dtype=torch.long, device=self.device)
        
        # --- 2. Warmup Hybrid Integrals ---
        self.precomputed_integrals = [] 
        self._warmup()
        
    def _cdf(self, x, mean=0, std=1):
        return 0.5 * (1 + torch.erf((x - mean) / (std * 1.41421356)))

    def _integrate_gaussian(self, lower, upper, mean, std):
        # generic integrator supporting broadcasting
        return self._cdf(upper, mean, std) - self._cdf(lower, mean, std)

    def _warmup(self):
        """
        Pre-calculates integrals of Gaussian kernels over the fixed grid slices
        defined by the Bayesnet estimators.
        """
        for i, bn in enumerate(self.bayes_estimators):
            gmm_idx_x1 = self.orig_to_gmm_map[bn.idx_x1]
            gmm_idx_x2 = self.orig_to_gmm_map[bn.idx_x2]
            
            m1, s1 = self.means[gmm_idx_x1, :], self.stds[gmm_idx_x1, :]
            m2, s2 = self.means[gmm_idx_x2, :], self.stds[gmm_idx_x2, :]
            
            grid1 = bn.grid_x1.to(self.device)
            grid2 = bn.grid_x2.to(self.device)
            
            # X1 Integrals (S1, K)
            lows1 = grid1[:-1].unsqueeze(1)
            highs1 = grid1[1:].unsqueeze(1)
            int_x1 = self._integrate_gaussian(lows1, highs1, m1.unsqueeze(0), s1.unsqueeze(0))
            
            # X2 Integrals (S2, K)
            lows2 = grid2[:-1].unsqueeze(1)
            highs2 = grid2[1:].unsqueeze(1)
            int_x2 = self._integrate_gaussian(lows2, highs2, m2.unsqueeze(0), s2.unsqueeze(0))
            
            self.precomputed_integrals.append({
                'x1': int_x1, 
                'x2': int_x2, 
                'gmm_idx_x1': gmm_idx_x1,
                'gmm_idx_x2': gmm_idx_x2
            })

    def predict_analytical(self, queried_rectangle):
        """
        queried_rectangle: Tensor (2, dimension_init)
        """
        q_min = queried_rectangle[0]
        q_max = queried_rectangle[1]
        
        # 1. Optimized Pure GMM Probability Calculation
        if len(self.pure_gmm_indices) > 0:
            # Gather bounds for all pure dimensions: Shape (Num_Pure)
            p_mins = q_min[self.pure_orig_indices]
            p_maxs = q_max[self.pure_orig_indices]
            
            # Gather means/stds for all pure dimensions: Shape (Num_Pure, K)
            means_pure = self.means[self.pure_gmm_indices]
            stds_pure = self.stds[self.pure_gmm_indices]
            
            # Reshape bounds for broadcasting: (Num_Pure, 1)
            p_mins = p_mins.unsqueeze(1)
            p_maxs = p_maxs.unsqueeze(1)
            
            # Calculate Integrals: Shape (Num_Pure, K)
            probs_pure_matrix = self._integrate_gaussian(p_mins, p_maxs, means_pure, stds_pure)
            
            # Collapse dimensions via product to get probability per kernel: Shape (K)
            pure_gmm_prob = probs_pure_matrix.prod(dim=0)
        else:
            # If no pure dimensions exist, start with 1.0
            pure_gmm_prob = torch.ones(self.kernels_num, device=self.device)
                
        # 2. Calculate Hybrid Contributions
        hybrid_probs = torch.ones(self.kernels_num, device=self.device)
        P_matrices=[] 
        
        for i, bn in enumerate(self.bayes_estimators):
            cache = self.precomputed_integrals[i]
            
            a1, b1 = q_min[bn.idx_x1], q_max[bn.idx_x1]
            a2, b2 = q_min[bn.idx_x2], q_max[bn.idx_x2]
            p1, q1_y = q_min[bn.idx_y], q_max[bn.idx_y]
            
            # Get P matrix (S1, S2)
            P_matrix = bn.get_all_slice_cond_probs(torch.tensor([p1, q1_y],dtype=torch.float16,device=self.device))
            P_matrices.append(P_matrix)
            
            # --- Construct v1 (x1) ---
            idx_start_1 = (torch.bucketize(a1, bn.grid_x1, right=True) - 1).clamp(0, bn.S1 - 1)
            idx_end_1   = (torch.bucketize(b1, bn.grid_x1, right=True) - 1).clamp(0, bn.S1 - 1)
            
            m1, s1 = self.means[cache['gmm_idx_x1']], self.stds[cache['gmm_idx_x1']]
            
            # Start with precomputed body
            v1 = cache['x1'].T.clone() # (K, S1)
            
            # Mask body
            seq_1 = torch.arange(bn.S1, device=self.device).unsqueeze(0)
            mask_1 = (seq_1 > idx_start_1) & (seq_1 < idx_end_1)
            v1 = v1 * mask_1.float()
            
            # Handle Head/Tail edges
            if idx_start_1 == idx_end_1:
                v1[:, idx_start_1] = self._integrate_gaussian(a1, b1, m1, s1)
            else:
                head_high = bn.grid_x1[idx_start_1 + 1]
                tail_low = bn.grid_x1[idx_end_1]
                v1[:, idx_start_1] = self._integrate_gaussian(a1, torch.min(head_high, b1), m1, s1)
                v1[:, idx_end_1] = self._integrate_gaussian(torch.max(tail_low, a1), b1, m1, s1)

            # --- Construct v2 (x2) ---
            idx_start_2 = (torch.bucketize(a2, bn.grid_x2, right=True) - 1).clamp(0, bn.S2 - 1)
            idx_end_2   = (torch.bucketize(b2, bn.grid_x2, right=True) - 1).clamp(0, bn.S2 - 1)
            
            m2, s2 = self.means[cache['gmm_idx_x2']], self.stds[cache['gmm_idx_x2']]
            
            v2 = cache['x2'].T.clone() # (K, S2)
            
            seq_2 = torch.arange(bn.S2, device=self.device).unsqueeze(0)
            mask_2 = (seq_2 > idx_start_2) & (seq_2 < idx_end_2)
            v2 = v2 * mask_2.float()
            
            if idx_start_2 == idx_end_2:
                v2[:, idx_start_2] = self._integrate_gaussian(a2, b2, m2, s2)
            else:
                head_high = bn.grid_x2[idx_start_2 + 1]
                tail_low = bn.grid_x2[idx_end_2]
                v2[:, idx_start_2] = self._integrate_gaussian(a2, torch.min(head_high, b2), m2, s2)
                v2[:, idx_end_2] = self._integrate_gaussian(torch.max(tail_low, a2), b2, m2, s2)

            # --- Combine ---
            # v1 (K, S1) @ P (S1, S2) -> (K, S2)
            # (Result * v2).sum -> (K)
            term_1 = torch.matmul(v1, P_matrix) 
            term_final = (term_1 * v2).sum(dim=1)
            
            hybrid_probs *= term_final
            
        # 3. Final Weighted Sum
        final_kernel_probs = pure_gmm_prob * hybrid_probs
        total_prob = (final_kernel_probs * self.weights).sum()
        
        return total_prob, P_matrices

    def predict_and_sample(self,
                        target_rectangle_init,
                        bayes_source_attributes=None,
                        bayes_called_attributes=None,
                        bayes_assist_attributes=None,
                        working_mode='ADC'):
        result=-1
        attribute_exceeded_above=(maxvals[0,:]<=target_rectangle_init[1,:])
        attribute_exceeded_below=(minvals[0,:]>=target_rectangle_init[0,:])
        attributes_not_covered=attribute_exceeded_above*attribute_exceeded_below*1
        inside_intervalone=(target_rectangle_init>reg_const_zero[0:1,:])*(target_rectangle_init<reg_const_zero[1:2,:])
        inside_intervaltwo=(target_rectangle_init>reg_const_zero[3:4,:])*(target_rectangle_init<reg_const_zero[4:5,:])
        target_rectangle_init=target_rectangle_init-(target_rectangle_init-goto_value_one)*inside_intervalone-(deducted_value_one)*(target_rectangle_init>reg_const_zero[1:2,:])-(target_rectangle_init-goto_value_two)*inside_intervaltwo-(deducted_value_two)*(target_rectangle_init>reg_const_zero[4:5,:])
        center=(target_rectangle_init[0:1,:]+target_rectangle_init[1:2,:])/2
        interval_of_center=torch.clamp(torch.floor((center-reg_const_one[1,:])*64/(3*reg_const_one[2,:])),min=0,max=63)
        modified_target_rectangle=(target_rectangle_init-reg_const_one[1:2,:])/reg_const_one[2:3,:]-1.5
        modified_target_rectangle=modified_target_rectangle-0.05*(modified_target_rectangle<reg_const_two[0:1,:])-0.05*(modified_target_rectangle<reg_const_two[1:2,:])+0.05*(modified_target_rectangle>reg_const_two[2:3,:])+0.05*(modified_target_rectangle>reg_const_two[3:4,:])
        modified_target_rectangle[0:1,attribute_exceeded_below]=2*self.edge_values[1:2,attribute_exceeded_below]-modified_target_rectangle[1:2,attribute_exceeded_below]
        modified_target_rectangle[1:2,attribute_exceeded_above]=2*self.edge_values[0:1,attribute_exceeded_above]-modified_target_rectangle[0:1,attribute_exceeded_above]
        modified_target_rectangle=modified_target_rectangle-reg_const_three
        center=(modified_target_rectangle[0,:]+modified_target_rectangle[1,:])/2
        mcvs_inside=(minimum_radius_mcvs[:,:,0]>modified_target_rectangle[0:1,:])*(minimum_radius_mcvs[:,:,0]<modified_target_rectangle[1:2,:])
        upperbound_mcvs=torch.amax(minimum_radius_mcvs[:,:,0]+minimum_radius_mcvs[:,:,1]*0.05-1000000000.0*(torch.logical_not(mcvs_inside)),0)
        lowerbound_mcvs=torch.amin(minimum_radius_mcvs[:,:,0]-minimum_radius_mcvs[:,:,1]*0.05+1000000000.0*(torch.logical_not(mcvs_inside)),0)
        modified_target_rectangle[0,indexes]=torch.minimum(modified_target_rectangle[0,indexes],lowerbound_mcvs[indexes])
        modified_target_rectangle[1,indexes]=torch.maximum(modified_target_rectangle[1,indexes],upperbound_mcvs[indexes])
        minrad_common=minimum_radius[torch.arange(dimension_init),interval_of_center.to(torch.int)]/(2*reg_const_one[2,:])
        modified_target_rectangle[0,indexes]=torch.minimum(modified_target_rectangle[0,indexes],(center-minrad_common)[0,indexes])
        modified_target_rectangle[1,indexes]=torch.maximum(modified_target_rectangle[1,indexes],(center+minrad_common)[0,indexes])
        if bayes_source_attributes is not None and torch.sum((1-attributes_not_covered)[bayes_called_attributes])>0:
            result, P_matrices=self.predict_analytical(modified_target_rectangle)
            if result<1/(20*size):
                return result, 0, True, [], 0
        bayes_used=(result>-1)*1
        if bayes_used:
            for i, bn in enumerate(self.bayes_estimators):
                nonzero_indices=torch.nonzero(P_matrices[i])
                modified_target_rectangle[0,bayes_source_attributes[i]]=max(modified_target_rectangle[0,bayes_source_attributes[i]],bn.grid_x1[torch.min(nonzero_indices[:,0])])
                modified_target_rectangle[1,bayes_source_attributes[i]]=min(modified_target_rectangle[1,bayes_source_attributes[i]],bn.grid_x1[torch.max(nonzero_indices[:,0])+1])
                modified_target_rectangle[0,bayes_assist_attributes[i]]=max(modified_target_rectangle[0,bayes_assist_attributes[i]],bn.grid_x2[torch.min(nonzero_indices[:,1])])
                modified_target_rectangle[1,bayes_assist_attributes[i]]=min(modified_target_rectangle[1,bayes_assist_attributes[i]],bn.grid_x2[torch.max(nonzero_indices[:,1])+1])
        query_edge_length=modified_target_rectangle[1,:]-modified_target_rectangle[0,:]
        values, _=torch.topk(query_edge_length[indexes],k=3,largest=False)
        volume=torch.prod(values)
        pertubation_level=torch.sum(volume>(15*torch.sqrt(Var_min[0:6]))**3)
        tempmat=(torch.transpose(modified_target_rectangle[:,self.non_bayes_indexes],0,1).unsqueeze(1)-self.means.unsqueeze(2))
        tempval_unpert=self._cdf(tempmat/(self.stds).unsqueeze(2))
        kernelprob_nobayes=torch.clamp(torch.prod(tempval_unpert[:,:,1]-tempval_unpert[:,:,0],0),min=0)
        kernelprob_nobayes=(kernelprob_nobayes*self.weights).to(torch.float64)
        result_nobayes=torch.sum(kernelprob_nobayes)
        kernelprob_nobayes=kernelprob_nobayes/result_nobayes
        result_nobayes=result_nobayes.to(torch.float32)
        query_volume[current_query], KDE_estimated_selectivity[current_query]=volume, max(result_nobayes,1e-8)
        if working_mode=='ADC+':
            if (result*bayes_used+result_nobayes*(1-bayes_used))<1/(20*size) or classifier.predict(np.array([[ma.log(volume),ma.log(result_nobayes.item())]]))==0:
                return (result*(bayes_used)+result_nobayes*(1-bayes_used)), 0, True, [], 0
        if working_mode=='ADC':
            if (result*bayes_used+result_nobayes*(1-bayes_used))<1/(20*size):
                return (result*(bayes_used)+result_nobayes*(1-bayes_used)), 0, True, [], 0
        tempval_pert=self._cdf(tempmat/(self.stds_perturbed[pertubation_level]).unsqueeze(2))
        kernelprob_nobayes_pert=torch.prod(tempval_pert[:,:,1]-tempval_pert[:,:,0],0)
        kernelprob_nobayes_pert=(torch.clamp(kernelprob_nobayes_pert,min=0)*self.weights).to(torch.float64)
        sumprob_nobayes_pert=torch.sum(kernelprob_nobayes_pert)
        if sumprob_nobayes_pert<5e-9:
            kernelprob_nobayes_pert=(kernelprob_nobayes_pert+1e-11)/torch.sum(kernelprob_nobayes_pert+1e-11)
        else:
            kernelprob_nobayes_pert=(kernelprob_nobayes_pert)/sumprob_nobayes_pert
        choose_kernel=np.random.choice(kernels_num,draws,True,kernelprob_nobayes_pert)
        position=torch.rand((dimension,draws))
        position=position*(tempval_pert[:,choose_kernel,1]-tempval_pert[:,choose_kernel,0])+(tempval_pert[:,choose_kernel,0])
        position=normal.icdf(position)*(self.stds_perturbed[pertubation_level])[:,choose_kernel]+self.means[:,choose_kernel]
        prob_est_list=torch.ones((draws))
        if bayes_used:
            for i, bn in enumerate(self.bayes_estimators):
                prob_est_list=prob_est_list*(P_matrices[i])[torch.bucketize(position[self.orig_to_gmm_map[bn.idx_x1],:],bn.grid_x1[1:-1])-1,torch.bucketize(position[self.orig_to_gmm_map[bn.idx_x2],:],bn.grid_x2[1:-1])-1]            
        return (result*(bayes_used)+result_nobayes*(1-bayes_used)), position, False, prob_est_list, pertubation_level
    
    def GMM_OnePointEst(self,
                        position: torch.Tensor,
                        pertubation_level=0):
        calc_matrix=(position-self.means[:dimension,:].unsqueeze(2))/self.stds_perturbed[pertubation_level].unsqueeze(2)
        calc_matrix=0.5*(calc_matrix**2)
        calc_matrix=torch.exp(-1*torch.sum(calc_matrix,0))/(self.gaussian_coefficient[pertubation_level])
        return torch.sum(calc_matrix*self.weights.unsqueeze(1),0)
        
'''神经网络训练时不加时移'''
class net_tail(nn.Module):
    global dimension
    def __init__(self,net_structure):
        super(net_tail,self).__init__()
        layers=[]
        for i in range(0,len(net_structure)-2):
            layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
            layers.append(nn.LeakyReLU(0.3))
        i=len(net_structure)-2
        layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
        self.structure=nn.Sequential(*layers)

    def forward(self,x):
        y=self.structure(torch.transpose(torch.cat((x[0:dimension],torch.sqrt(x[dimension:dimension+1])),0),0,-1))
        return torch.transpose(y,0,-1)*decay_function(x[dimension:dimension+1])
def decay_function(x):
    return 1/(torch.exp(x)-torch.exp(-x))
'''神经网络训练时不考虑时移'''
class net_head(nn.Module):
    global dimension, activated_nodes, activation_array
    def __init__(self,net_structure):
        super(net_head,self).__init__()
        layers=[]
        for i in range(0,len(net_structure)-2):
            layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
            layers.append(nn.Hardtanh(-2,2))
        i=len(net_structure)-2
        layers.append(nn.Linear(net_structure[i],net_structure[i+1]))
        self.structure=nn.Sequential(*layers)
    def forward(self,x):
        y=self.structure(torch.transpose(torch.cat((x[0:dimension]*torch.exp(x[dimension:dimension+1]),(x[dimension+1:dimension+2])*(0.1)),0),0,-1))
        return (torch.transpose(y[:,:,0:dimension],0,-1))*x[dimension+2:dimension+3,:,:]*0.04+(torch.transpose(y[:,:,dimension:2*dimension],0,-1))*x[dimension+1:dimension+2,:,:]*0.2+(torch.transpose(y[:,:,2*dimension:3*dimension],0,-1))

def set_global(Intervals,TimeAllowed_min,batch_size,dim,cutoff,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None):
    global histogram, gmm_estimator, length, BatchSize, TimeMax, TimeChart, TimeChart_InNetwork, SignalDecay, SignalDecay_abbrev, NoiseVar, NoiseVar_inv, NoiseVarSquare, NoiseVarSquare_inv, Weight, Weight_sqrt, Weight_used, Weight_term_additional, PointArray, dimension, PointArrayScore, PointArrayScore_double, cutoff_length, cutoff_time, Var_min, Time_min, residual_tail_adjust_term, normal
    histogram=SingleAttributeHistogram(num_bins=200, num_mcvs=20)
    histogram.load(dataset_name+'/'+dataset_name+'_histogram.npy',dataset_name+'/'+dataset_name+'_mcv.npz',dataset_name+'/'+dataset_name+'_meta.npz')
    cutoff_length=cutoff
    cutoff_time=Intervals[cutoff]
    TimeMax=Intervals[len(Intervals)-1]
    dimension=dim
    BatchSize=batch_size
    length=len(Intervals)-1
    TimeChart=np.zeros((1,(length)*batch_size))
    TimeChart[0,:]=np.repeat([(Intervals[0]+Intervals[1])/2]+[(Intervals[i]+Intervals[i+1])/2 for i in range(1,cutoff_length)]+[(Intervals[i]+Intervals[i+1])/2 for i in range(cutoff_length,length)],batch_size)
    SignalDecay=np.exp(-1*TimeChart)
    SignalDecay_abbrev=(torch.exp(-1*torch.tensor([(Intervals[0]+Intervals[1])/2]+[(Intervals[i]+Intervals[i+1])/2 for i in range(1,length)]))).unsqueeze(0)
    NoiseVarSquare=(1-SignalDecay**2)
    NoiseVar=np.sqrt(NoiseVarSquare)
    Weight=np.zeros((1,(length)*batch_size))
    Weight[0,:]=np.repeat([(Intervals[i+1]-Intervals[i]) for i in range(0,length)],batch_size)
    Weight=torch.tensor(Weight)
    Weight_term_additional=torch.zeros(1,(length)*batch_size)
    Weight_term_additional[0,4*batch_size]=TimeAllowed_min/8
    Weight_sqrt=torch.sqrt(Weight)
    SobolSampler=stats.qmc.Sobol(dim,scramble=True,seed=123)
    UniSobol=np.transpose(SobolSampler.random(2**ma.ceil(ma.log2(batch_size*length))))
    UniSobol=(UniSobol-1*(UniSobol>1))
    NormalSobol=stats.norm.ppf(UniSobol*0.999999+0.0000005)
    PointArray=torch.tensor(NormalSobol[:,0:(length)*batch_size]*NoiseVar)
    PointArrayScore=-1*PointArray/NoiseVarSquare
    PointArrayScore_double=2*PointArrayScore
    TimeChart=torch.tensor(TimeChart)
    SignalDecay=torch.tensor(SignalDecay)
    NoiseVar=torch.tensor(NoiseVar)
    NoiseVarSquare=torch.tensor(NoiseVarSquare)
    Time_min=[TimeAllowed_min,2*TimeAllowed_min,3*TimeAllowed_min,4*TimeAllowed_min,6*TimeAllowed_min, 8*TimeAllowed_min, 10*TimeAllowed_min]
    NoiseVarSquare_inv=[(1/(1-torch.exp(-2*(TimeChart+Time_min[i])))).to(torch.float32) for i in range(0,7)]
    NoiseVar_inv=[(torch.sqrt(NoiseVarSquare_inv[i])).to(torch.float32) for i in range(0,7)]
    '''
    Note that NoiseVar_inv is used inside the noise prediction network, trained without implementing the early stopping time-shift; while NoiseVar is used in the density estimator where the early stopping time-shift was already implemented. Therefore, these two are NOT the inverses of one another due to this discrepancy
    '''
    Var_min=torch.tensor([ma.exp(2*Time_min[i])-1 for i in range(0,7)])
    TimeChart_InNetwork=[(TimeChart+Time_min[i]).to(torch.float32) for i in range(0,7)]
    residual_tail_adjust_term=[(1/NoiseVarSquare[:,BatchSize*cutoff_length:]-(NoiseVarSquare_inv[i])[:,BatchSize*cutoff_length:]).unsqueeze(2).to(torch.float32) for i in range(0,7)]
    Weight_used=torch.transpose(Weight/BatchSize,0,1)
    Weight_term_additional=torch.transpose(Weight_term_additional/BatchSize,0,1)
    TimeChart=TimeChart.to(torch.float32)
    SignalDecay=SignalDecay.to(torch.float32)
    SignalDecay_abbrev=SignalDecay_abbrev.to(torch.float32)
    NoiseVar=NoiseVar.to(torch.float32)
    NoiseVarSquare=NoiseVarSquare.to(torch.float32)
    Weight=Weight.to(torch.float32)
    Weight_sqrt=Weight_sqrt.to(torch.float32)
    Weight_used=Weight_used.to(torch.float32)
    PointArray=PointArray.to(torch.float32)
    PointArrayScore=PointArrayScore.to(torch.float32)
    PointArrayScore_double=PointArrayScore_double.to(torch.float32)
    if bayes_source_attributes is not None:
        bcalled_tensor=torch.tensor(bayes_called_attributes)
        bsource_actual_position=[i-torch.sum(i>bcalled_tensor).item() for i in bayes_source_attributes]
    else:
        bsource_actual_position=[]
    gmm_estimator=GMM_Estimator(dimension_init, kernels_matrix, bayes_called_attributes, bayes_source_attributes, bayes_assist_attributes, 150, 8, 10, 1)
    normal=torch.distributions.Normal(loc=0.0, scale=1.0)

def get_reg_consts():
    global reg_const_one, reg_const_two, reg_const_three, reg_const_zero, maxvals, minvals, goto_value_one, goto_value_two, deducted_value_one, deducted_value_two, size, nan_nums, orisize, minimum_radius, minimum_radius_mcvs
    reg_consts=np.load(dataset_name+'/'+'reg_consts_'+dataset_name+'.npy')
    reg_const_one=torch.tensor(reg_consts[0:3,:]).to(torch.float32)
    reg_const_two=torch.tensor(reg_consts[3:7,:]).to(torch.float32)
    reg_const_three=torch.tensor(reg_consts[7:8,:]).to(torch.float32)
    reg_const_zero=torch.tensor(reg_consts[9:15,:]).to(torch.float32)
    maxvals=reg_const_one[0:1,:]+(reg_const_zero[1:2,:]-reg_const_zero[2:3,:])+(reg_const_zero[4:5,:]-reg_const_zero[5:6,:])
    minvals=reg_const_one[1:2,:]
    goto_value_one=(reg_const_zero[0:1,:]+reg_const_zero[2:3,:])/2
    goto_value_two=(reg_const_zero[3:4,:]+reg_const_zero[5:6,:])/2
    deducted_value_one=reg_const_zero[1:2,:]-reg_const_zero[2:3,:]
    deducted_value_two=reg_const_zero[4:5,:]-reg_const_zero[5:6,:]
    size=reg_consts[8,0].astype(int)
    nan_nums=reg_consts[8,1].astype(int)
    orisize=size+nan_nums
    minimum_radius=torch.tensor(np.load(dataset_name+'/'+'minrad_'+dataset_name+'.npy'))
    minimum_radius_mcvs=torch.tensor(np.load(dataset_name+'/'+'minrad_mcvs_'+dataset_name+'.npy'))

def CardEst_Implement(target_rectangle,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None,attributes_not_covered=None,working_mode='ADC'):
    result,position,predictor_is_more_accurate,prob_est_list,pertubation_level=gmm_estimator.predict_and_sample(target_rectangle,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,working_mode)
    if predictor_is_more_accurate or working_mode=='ADC-':
        return result
    position=position.unsqueeze(1)
    density_KDE=gmm_estimator.GMM_OnePointEst(position,pertubation_level)
    position=position*(ma.exp(-1*Time_min[pertubation_level]))
    eval_point=torch.cat((PointArray,TimeChart_InNetwork[pertubation_level],NoiseVar_inv[pertubation_level],NoiseVarSquare_inv[pertubation_level]),0)
    eval_point=eval_point.unsqueeze(2).repeat(1,1,draws)
    position_used=(position*(SignalDecay_abbrev.unsqueeze(2))).unsqueeze(2).expand(-1,-1,BatchSize,-1)
    position_used=position_used.reshape(dimension,length*BatchSize,draws)
    eval_point[:dimension,:,:]=eval_point[:dimension,:,:]+position_used
    eval_point_head=eval_point[:,:cutoff_length*BatchSize,:]
    eval_point_tail=eval_point[:,cutoff_length*BatchSize:,:]
    '''score_head无需以此方法修正,是因为时移项只作为输入时的提示项,不直接出现在输出中'''
    score_head=evalnet_head(eval_point_head)
    '''神经网络训练时未加时移项训练,故需以此方法人为修正时移项影响,更正residual_tail的取值'''
    residual_tail=evalnet_tail(eval_point_tail)+eval_point_tail[:dimension,:]*(residual_tail_adjust_term[pertubation_level])
    eval_value=torch.cat((score_head*(score_head-(2*PointArrayScore[:,0:cutoff_length*BatchSize]).unsqueeze(2))+1,residual_tail*(residual_tail-2*((PointArrayScore[:,cutoff_length*BatchSize:]).unsqueeze(2)+eval_point[0:dimension,cutoff_length*BatchSize:,:]/(NoiseVarSquare[:,cutoff_length*BatchSize:]).unsqueeze(2)))),1)
    eval_value=torch.sum(eval_value,0)*Weight_used
    eval_value=torch.sum(eval_value,0)
    log_density=dimension*(0.5*ma.log(1/(2*3.14159265))-0.5)-eval_value-ma.log(1-ma.exp(-2*cutoff_time))*(dimension/2)-(torch.sum(position*position,0)).unsqueeze(0)*(1/(2*(1-ma.exp(-2*cutoff_time)))-1/2)
    density_diffusion=(torch.exp(log_density))
    density_diffusion[torch.isnan(density_diffusion)]=0
    adjust_term=(density_diffusion+1e-7)/(density_KDE+1e-7)
    bufferval=torch.mean(prob_est_list)
    z=min(torch.mean(adjust_term*(prob_est_list+bufferval*0.25+1e-7))/(1.25*bufferval+1e-7),40)
    return result*z


def CardEst_Implement_Selective(target_rectangle,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None,nan_to=-100000000,working_mode='ADC'):
    torch.manual_seed(123)
    np.random.seed(123)
    estimate_zero=((torch.sum(minvals>target_rectangle[1:2,:])+torch.sum(maxvals<target_rectangle[0:1,:]))>0)
    nan_queried=torch.prod(target_rectangle[0:1,:]<=nan_to+1e-8)*torch.prod(target_rectangle[1:2,:]>=nan_to)
    if estimate_zero:
        return torch.tensor([nan_queried*nan_nums])
    attribute_exceeded_above=maxvals<=target_rectangle[1:2,:]
    attribute_exceeded_below=minvals>=target_rectangle[0:1,:]
    attributes_not_covered=attribute_exceeded_above*attribute_exceeded_below*1
    target_rectangle[0:1,unit_of_variables[0,:]!=0]=(torch.ceil(target_rectangle[0:1,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]-1e-6)-0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    target_rectangle[1:2,unit_of_variables[0,:]!=0]=(torch.floor(target_rectangle[1:2,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]+1e-6)+0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    edge_length=target_rectangle[1:2,:]-target_rectangle[0:1,:]
    if torch.min(edge_length)<1e-6:
        return torch.tensor([nan_queried*nan_nums])
    if torch.sum(attributes_not_covered)==dimension_init:
        return torch.tensor([size+nan_queried*nan_nums])
    if torch.sum(attributes_not_covered)==dimension_init-1:
        queriedatt=torch.argmin(attributes_not_covered)
        target_rectangle=target_rectangle.numpy()
        return torch.tensor([ma.ceil(histogram.estimate(queriedatt,target_rectangle[0,queriedatt],target_rectangle[1,queriedatt]))])+nan_queried*nan_nums
    return torch.ceil((CardEst_Implement(target_rectangle,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,attributes_not_covered,working_mode)*size+nan_queried*nan_nums))
            
def calculate_metric(q_errors):
    return np.mean(np.log(q_errors)**2)

def run(dname,uvar,dim,kernum,tm,workload_size,bayes_source_attributes=None,bayes_called_attributes=None,bayes_assist_attributes=None,nan_to=-10000000000):
    global data_init, dataset_name, unit_of_variables, dimension, dimension_init, kernels_num, tmin, evalnet_head, evalnet_tail, kernels_matrix, timeused, j, CardEst_Compiled, kde_pointvals, diffusion_pointvals, current_query, predictions, indexes, bayesnet, query_volume, KDE_estimated_selectivity
    dataset_name=dname
    unit_of_variables=(torch.tensor(uvar)).float()
    dimension_init=dim
    if bayes_source_attributes is None:
        dimension=dimension_init
        indexes=[i for i in range(0,dimension_init)]
    else:
        dimension=dimension_init-len(bayes_source_attributes)
        indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    if bayes_source_attributes is None:
        bayes_source_attributes=[]
    if bayes_called_attributes is None:
        bayes_called_attributes=[]
    if bayes_assist_attributes is None:
        bayes_assist_attributes=[]
    kernels_num=kernum
    tmin=tm
    starting_col=0
    bayesnet=[]    
    get_reg_consts()    
    kernels_matrix=torch.tensor(np.load(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy')).float()
    evalnet_head=torch.load(dataset_name+'/'+dataset_name+'_head.pkl')
    evalnet_tail=torch.load(dataset_name+'/'+dataset_name+'_tail.pkl')
    j=0
    min_sel=1/orisize
    timeused=np.zeros(workload_size+1)
    set_global(timestep,tmin,64,dimension,43,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes)
    workloads=pd.read_csv(dataset_name+'training/'+dataset_name+'_trainset.csv', delimiter=',', dtype=np.float32, header=None)
    workloads=workloads.to_numpy()
    workloads=torch.tensor(workloads)
    actual_selectivity=np.load(dataset_name+'training/'+dataset_name+'_real_train.npy')/orisize
    error_with_diffusion=np.zeros((workload_size))
    error_without_diffusion=np.zeros((workload_size))
    query_volume=np.zeros((workload_size))+1e-8
    KDE_estimated_selectivity=np.zeros((workload_size))+1e-8
    timestart=time.time()
    print("Step 1: Testing ADC and ADC- on different query variants")
    for i in range(0,workload_size):
        if i%500==499:
            print('processed '+str(i+1)+' queries in training')
        current_query=i
        target_rectangle=workloads[2*i:2*i+2,:]        
        estimate_selectivity=CardEst_Implement_Selective(target_rectangle,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,nan_to,working_mode='ADC')/orisize
        a=CardEst_Implement_Selective(target_rectangle,bayes_source_attributes,bayes_called_attributes,bayes_assist_attributes,nan_to,working_mode='ADC-')/orisize
        e=actual_selectivity[i]
        error_with_diffusion[i]=(max(e,min_sel))/max(estimate_selectivity,min_sel)
        error_without_diffusion[i]=max(e,min_sel)/max(a,min_sel)
    QError=np.maximum(error_with_diffusion,1/error_with_diffusion)
    QError_without_diffusion=np.maximum(error_without_diffusion,1/error_without_diffusion)
    z=np.sort(QError)
    print('Quickinfo: Current 95th QError for ADC is')
    print(z[ma.ceil(workload_size*0.95)])
    print('Current 99th QError for ADC is')
    print(z[ma.ceil(workload_size*0.99)])
    colors=np.where(QError<=QError_without_diffusion,'red','blue').astype('<U5')
    colors[QError==QError_without_diffusion]='green'
    colors[query_volume<1.1e-8]='white'
    query_volume=np.log(query_volume)
    KDE_estimated_selectivity=np.log(KDE_estimated_selectivity)
    relative_error=np.log(error_with_diffusion)
    log_selectivity=np.log(actual_selectivity+1e-11)
    plt.figure(figsize=(8, 8))  
    plt.scatter(np.exp(KDE_estimated_selectivity), np.exp(query_volume)/((3.4)**3), c=colors, s=1.5 , alpha=0.5, edgecolors='none') 
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Estimated Selectivity (log scale)',fontsize=18)
    plt.ylabel('Query Volume (log scale)',fontsize=18)
    plt.suptitle(dname, y=0.95, fontsize=20)
    plt.savefig('Est-Vol-'+dataset_name+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    q_error_predictor=QError_without_diffusion
    q_error_corrector=QError
    log_q_pred_sq = np.log(q_error_predictor)**2
    log_q_corr_sq = np.log(q_error_corrector)**2
    y_labels = (log_q_corr_sq  < log_q_pred_sq).astype(int)
    sample_weights = np.abs(log_q_pred_sq - log_q_corr_sq)
    X=pd.DataFrame({'vol':query_volume, 'est':KDE_estimated_selectivity})
    X_train, X_test, y_train, y_test, q_pred_train, q_pred_test, q_corr_train, q_corr_test, weights_train, weights_test = train_test_split(X, y_labels, q_error_predictor, q_error_corrector, sample_weights, test_size=0.2, random_state=0)
    print("Training the Decision Tree with sample weights...")
    classifier = DecisionTreeClassifier(max_depth=3, random_state=0, criterion='entropy')
    classifier.fit(X_train, y_train, sample_weight=weights_train)
    print("Training complete.")
    predictions = classifier.predict(X_test)
    metric_predictor_only = calculate_metric(q_pred_test)
    metric_corrector_only = calculate_metric(q_corr_test)
    hybrid_q_errors = np.where(predictions == 0, q_pred_test, q_corr_test)
    metric_hybrid = calculate_metric(hybrid_q_errors)
    print(f"\n--- Overall Performance (lower is better) ---")
    print(f"  Standalone Predictor: {metric_predictor_only:.4f}")
    print(f"  Predictor-Corrector:  {metric_corrector_only:.4f}")
    print(f"  Hybrid (Our Model):   {metric_hybrid:.4f}")
    with open(dataset_name+'/'+'classifier_'+dataset_name+'.pkl','wb') as classifierfile:
        pickle.dump(classifier,classifierfile)

if __name__ == "__main__":
    timestep=(np.load('timestep.npy')).tolist()
    cutoff=43
    gmmker=1280
    params=['Estimator','higgs',"[1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]",'7','1/1280','10000',"[]","[]","[]",'-10000000000',25]
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    uvar=[ast.literal_eval(params[2])]
    dimension_init=int(params[3])
    Time_min=float(Fraction(params[4]))
    workloadsize=int(params[5])
    bsource_attributes=ast.literal_eval(params[6])
    bcalled_attributes=ast.literal_eval(params[7])
    bassist_attributes=ast.literal_eval(params[8])
    nanto=int(params[9])
    draws=int(params[10])
    run(dataset_name,uvar,dimension_init,gmmker,Time_min,workloadsize,bsource_attributes,bcalled_attributes,bassist_attributes,nanto)
   
