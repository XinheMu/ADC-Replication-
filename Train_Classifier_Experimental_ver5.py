from __future__ import division
import torch
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

class BayesnetEstimator:
    """
    An estimator for a near-functional dependency A[i] -> A[j], with a
    user-configurable number of histogram bins for greater flexibility.
    """
    def __init__(self, i: int, j: int, N: int, hist_bins: int = 3):
        """
        Initializes the estimator.
        
        Args:
            i (int): Index of the determinant attribute (a_i).
            j (int): Index of the dependent attribute (a_j).
            N (int): Number of slices for the determinant attribute.
            hist_bins (int): The number of bins to use for the histogram
                             of non-MCV values.
        """
        self.i, self.j, self.N = i, j, N
        self.hist_bins = hist_bins
        
        # Dynamically calculate the width of our statistics tensor
        # 5 base stats + hist_bins counts + (hist_bins+1) edges
        self.stats_dim = 5 + self.hist_bins + (self.hist_bins + 1)
        
        self.filename = f"FD_{self.i}{self.j}_{dataset_name}.pt"
        
        self.slice_edges: torch.Tensor = None
        self.stats_tensor: torch.Tensor = None
        self.redirect_map: torch.Tensor = None
        
        self.load()

    def train(self):
        print(f"Starting training for dependency {self.i} -> {self.j} with {self.hist_bins} histogram bins...")
        
        local_data_used = data_used.to(torch.float32)
        a_i_col = local_data_used[:, self.i]
        min_val, max_val = a_i_col.min(), a_i_col.max()
        
        self.slice_edges = torch.linspace(min_val, max_val, self.N + 1, dtype=torch.float32)
        self.stats_tensor = torch.zeros((self.N, self.stats_dim), dtype=torch.float32)
        populated_slices_mask = torch.zeros(self.N, dtype=torch.bool)
        
        # Define indices for slicing the stats_tensor for clarity
        counts_start, counts_end = 5, 5 + self.hist_bins
        edges_start = counts_end

        for k in range(self.N):
            slice_min, slice_max = self.slice_edges[k], self.slice_edges[k+1]
            mask = (a_i_col >= slice_min) & (a_i_col < slice_max)
            if k == self.N - 1: mask |= (a_i_col == slice_max)

            aj_slice = local_data_used[mask, self.j]
            total_count = len(aj_slice)
            
            self.stats_tensor[k, 0] = total_count
            if total_count > 0:
                populated_slices_mask[k] = True
                
                self.stats_tensor[k, 1:5] = torch.tensor([
                    aj_slice.min(), aj_slice.max(), 0, 0
                ]) # Min, Max, placeholders for mcv
                
                unique_vals, counts = torch.unique(aj_slice, return_counts=True)
                mcv_idx = torch.argmax(counts)
                mcv_val = unique_vals[mcv_idx]
                
                self.stats_tensor[k, 3] = mcv_val
                self.stats_tensor[k, 4] = counts[mcv_idx]
                
                aj_remaining = aj_slice[aj_slice != mcv_val]
                
                if len(aj_remaining) > 1:
                    hist_counts = torch.histc(aj_remaining, bins=self.hist_bins)
                    hist_min, hist_max = aj_remaining.min(), aj_remaining.max()
                    if hist_max == hist_min: hist_max += 1e-6
                    
                    self.stats_tensor[k, counts_start:counts_end] = hist_counts
                    self.stats_tensor[k, edges_start:] = torch.linspace(hist_min, hist_max, self.hist_bins + 1)

        # Build redirect map (logic unchanged)
        self.redirect_map = torch.arange(self.N, dtype=torch.int32)
        populated_indices = torch.where(populated_slices_mask)[0]
        if len(populated_indices) > 0:
            empty_indices = torch.where(~populated_slices_mask)[0]
            for empty_idx in empty_indices:
                distances = torch.abs(populated_indices - empty_idx)
                self.redirect_map[empty_idx] = populated_indices[torch.argmin(distances)]
        
        self.save()
        print(f"Training complete. Model tensors saved to {self.filename}")

    def save(self):
        """Saves essential tensors and model parameters to a .pt file."""
        data_to_save = {
            'i': self.i, 'j': self.j, 'N': self.N, 'hist_bins': self.hist_bins,
            'slice_edges': self.slice_edges,
            'stats_tensor': self.stats_tensor,
            'redirect_map': self.redirect_map
        }
        torch.save(data_to_save, self.filename)
            
    def load(self):
        """Loads and reconstructs the estimator from a .pt file."""
        if os.path.exists(self.filename):
            try:
                loaded_data = torch.load(self.filename)
                
                # Check for consistency and update model parameters from file
                loaded_hist_bins = loaded_data['hist_bins']
                if self.hist_bins != loaded_hist_bins:
                    print(f"WARNING: Model was trained with hist_bins={loaded_hist_bins}, "
                          f"but object was initialized with {self.hist_bins}. Overriding with loaded value.")
                    self.hist_bins = loaded_hist_bins
                    self.stats_dim = 5 + 2 * self.hist_bins + 1
                
                self.slice_edges = loaded_data['slice_edges']
                self.stats_tensor = loaded_data['stats_tensor']
                self.redirect_map = loaded_data['redirect_map']
                print(f"Successfully loaded model tensors from {self.filename}")
            except Exception as e:
                print(f"Could not load model from {self.filename}. Error: {e}.")

    def _get_prob_for_slice(self, slice_stats_tensor: torch.Tensor, query_range: torch.Tensor) -> float:
        """
        Helper to calculate probability from a 1D stats tensor.
        This version is FULLY VECTORIZED to eliminate the Python loop over bins,
        making it significantly faster for a larger number of histogram bins.
        """
        total_count = slice_stats_tensor[0]
        if total_count == 0:
            return 0.0
        
        # --- 1. Handle MCV contribution (this part is scalar) ---
        q_min, q_max = query_range[0], query_range[1]
        mcv_val, mcv_count = slice_stats_tensor[3], slice_stats_tensor[4]
        
        count_in_range = 0.0
        if q_min <= mcv_val <= q_max:
            count_in_range += mcv_count
            
        # --- 2. Handle Histogram contribution (VECTORIZED) ---
        
        # Unpack histogram data
        counts_start, counts_end = 5, 5 + self.hist_bins
        edges_start = counts_end
        hist_counts = slice_stats_tensor[counts_start:counts_end]
        hist_edges = slice_stats_tensor[edges_start:]
        
        # Create vectors for bin boundaries
        bin_mins = hist_edges[:-1]
        bin_maxs = hist_edges[1:]
        
        # Calculate bin widths, avoiding division by zero
        bin_widths = bin_maxs - bin_mins
        valid_mask = bin_widths > 1e-7
        
        # Calculate the overlap of the query range with each bin's range
        # torch.max/min with broadcasting handles this for all bins at once
        overlap_mins = torch.max(q_min, bin_mins)
        overlap_maxs = torch.min(q_max, bin_maxs)
        
        # Calculate the width of the overlap for each bin
        overlap_widths = torch.clamp(overlap_maxs - overlap_mins, min=0.0)
        
        # Calculate the fraction of each bin that falls within the query range
        # We only perform the division for bins with a non-zero width
        fractions = torch.zeros_like(bin_widths)
        fractions[valid_mask] = overlap_widths[valid_mask] / bin_widths[valid_mask]
        
        # Multiply fractions by bin counts and sum to get total histogram contribution
        hist_count_in_range = torch.sum(fractions * hist_counts)
        
        count_in_range += hist_count_in_range

        return (count_in_range / total_count).item()
    
    def get_all_slice_cond_probs(self, query_range_j: Union[Tuple[float, float], torch.Tensor]) -> torch.Tensor:
        """
        Calculates the conditional probability p(y in range | x in slice_k)
        for ALL N slices in a single vectorized operation.
        
        Args:
            query_range_j: The query range [p, q] for the dependent attribute.
            
        Returns:
            A torch.Tensor of shape (N,) where the k-th element is the
            conditional probability for the k-th slice.
        """
        if self.stats_tensor is None:
            raise RuntimeError("Estimator has not been trained or loaded.")
        
        # --- 1. Prepare Inputs ---
        if isinstance(query_range_j, torch.Tensor):
            query_tensor = query_range_j.to(dtype=torch.float32)
        else:
            query_tensor = torch.tensor(query_range_j, dtype=torch.float32)
        
        q_min, q_max = query_tensor[0], query_tensor[1]

        # --- 2. Unpack All Stats in Tensors (Shape: (N, ...)) ---
        total_counts = self.stats_tensor[:, 0]
        mcv_vals = self.stats_tensor[:, 3]
        mcv_counts = self.stats_tensor[:, 4]

        counts_start, counts_end = 5, 5 + self.hist_bins
        edges_start = counts_end
        
        hist_counts = self.stats_tensor[:, counts_start:counts_end]  # Shape: (N, hist_bins)
        hist_edges = self.stats_tensor[:, edges_start:]            # Shape: (N, hist_bins + 1)
        
        bin_mins = hist_edges[:, :-1]                             # Shape: (N, hist_bins)
        bin_maxs = hist_edges[:, 1:]                              # Shape: (N, hist_bins)
        
        # --- 3. Vectorized Calculations ---
        
        # MCV Contribution (Shape: (N,))
        # (q_min <= mcv_vals) & (mcv_vals <= q_max) creates a boolean mask
        mcv_contrib = torch.where((q_min <= mcv_vals) & (mcv_vals <= q_max), mcv_counts, 0.0)

        # Histogram Contribution (all ops are on (N, hist_bins) tensors)
        bin_widths = bin_maxs - bin_mins
        valid_mask = bin_widths > 1e-7
        
        # Broadcast q_min/q_max to match bin_mins/bin_maxs shape
        overlap_mins = torch.max(q_min, bin_mins)
        overlap_maxs = torch.min(q_max, bin_maxs)
        overlap_widths = torch.clamp(overlap_maxs - overlap_mins, min=0.0)
        
        fractions = torch.zeros_like(bin_widths)
        fractions[valid_mask] = overlap_widths[valid_mask] / bin_widths[valid_mask]
        
        # Sum contributions across the bins dimension (dim=1)
        hist_contrib = torch.sum(fractions * hist_counts, dim=1) # Shape: (N,)

        # --- 4. Final Combination ---
        total_counts_in_range = mcv_contrib + hist_contrib
        
        # Avoid division by zero for empty slices
        cond_probs = torch.zeros_like(total_counts)
        populated_mask = total_counts > 0
        cond_probs[populated_mask] = total_counts_in_range[populated_mask] / total_counts[populated_mask]
        
        return cond_probs

    def __call__(self, a_i_samples: torch.Tensor, query_range_j: Union[Tuple[float, float], torch.Tensor]) -> float:
        if self.stats_tensor is None:
            raise RuntimeError("Estimator has not been trained or loaded.")
        
        if isinstance(query_range_j, torch.Tensor):
            query_tensor = query_range_j.to(dtype=torch.float32)
        else:
            query_tensor = torch.tensor(query_range_j, dtype=torch.float32)

        slice_indices = torch.bucketize(a_i_samples, self.slice_edges, right=True)
        slice_indices = torch.clamp(slice_indices - 1, 0, self.N - 1)
        
        corrected_indices = self.redirect_map[slice_indices]
        relevant_stats = self.stats_tensor[corrected_indices]
        
        probs = [self._get_prob_for_slice(row, query_tensor) for row in relevant_stats]        

        if not probs: return 0.0
        return torch.tensor(probs).mean().item()


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
    
class GMM_Estimator:
    def __init__(self, kernels_matrix, bayesnet_estimators, bsource_actual_pos, dimension_init):
        self.kernels_matrix = kernels_matrix
        self.bayesnet = bayesnet_estimators # This is your 'bayesnet' list
        self.bsource_actual_position = bsource_actual_pos
        self.dimension_init = dimension_init
        self.non_bayes_indexes=indexes
        
        # Pre-extract GMM components for efficiency
        gmm_dim = self.kernels_matrix.shape[0] // 2
        self.gmm_means = self.kernels_matrix[:gmm_dim, :]
        kernels_std_unperturbed=torch.clone(kernels_matrix[dimension:2*dimension]).to(torch.float32)
        kernels_std=[kernels_std_unperturbed]
        for i in range(0,7):
            kernels_std.append(torch.sqrt(kernels_matrix[dimension:2*dimension]**2+Var_min[i]).to(torch.float32))
        self.gmm_stds = torch.clone(kernels_std[0])
        self.gmm_stds_perturbed = [torch.clone(kernels_std[i]) for i in range(1,8)]
        self.gmm_weights = (self.kernels_matrix[-1, :]).to(torch.float64)/torch.sum((self.kernels_matrix[-1, :]).to(torch.float64))
        self.gmm_weights = self.gmm_weights.to(torch.float32)
        self.num_kernels = self.kernels_matrix.shape[1]
        self.gaussian_coefficient=[torch.unsqueeze(torch.prod(self.gmm_stds_perturbed[i],0),1)*(ma.sqrt(2*3.14159265))**dimension for i in range(0,7)]
        # --- NEW: PRE-COMPUTATION OF MASS TABLES ---
        self.precomputed_mass_tables = []
        if self.bayesnet:
            print("Performing one-time pre-computation of GMM mass tables...")
            for i in range(len(self.bayesnet)):
                bayes_obj = self.bayesnet[i]
                source_actual_idx = self.bsource_actual_position[i]
                
                source_means = self.gmm_means[source_actual_idx, :] # Shape: (num_kernels,)
                source_stds = self.gmm_stds[source_actual_idx, :]   # Shape: (num_kernels,)

                slice_edges = bayes_obj.slice_edges             # Shape: (N+1,)
                s_starts = slice_edges[:-1]                     # Shape: (N,)
                s_ends = slice_edges[1:]                        # Shape: (N,)

                # Vectorize the CDF calculation across all slices and all kernels
                # Shapes: (N, 1) op (num_kernels,) -> (N, num_kernels)
                z_starts = (s_starts.unsqueeze(1) - source_means) / source_stds
                z_ends = (s_ends.unsqueeze(1) - source_means) / source_stds

                # The final pre-computed table for this dependency
                mass_table = self._cdf(z_ends) - self._cdf(z_starts)
                self.precomputed_mass_tables.append(mass_table)
            print("Pre-computation complete.")

    def _cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Standard Normal CDF using the error function."""
        return 0.5 * (1 + torch.special.erf(x / 1.41421356))
    
    def predict_analytical(self,
                        queried_rectangle: torch.Tensor,
                        bayes_source_attributes=None,
                        bayes_called_attributes=None):
        """
        Calculates cardinality using the analytical integration method.
        This version implements the "nearest-assignment" tactic to accurately
        handle GMM weight bleeding outside the Bayesnet's trained domain.
        """
        
        # --- NO-BAYESNET SCENARIO (Unchanged) ---
        if bayes_source_attributes is None:
            # ... (same as before) ...
            lower_bounds = queried_rectangle[0, :].unsqueeze(1)
            upper_bounds = queried_rectangle[1, :].unsqueeze(1)
            z_lower = (lower_bounds - self.gmm_means) / self.gmm_stds
            z_upper = (upper_bounds - self.gmm_means) / self.gmm_stds
            prob_mass_per_attr = self._cdf(z_upper) - self._cdf(z_lower)
            prob_per_kernel = torch.prod(prob_mass_per_attr, dim=0)
            return torch.sum(self.gmm_weights * prob_per_kernel)

        # --- SETUP & P_other (Unchanged) ---
        num_deps = len(bayes_source_attributes)
        # ... (rest of setup and P_other calculation is the same as before) ...
        gmm_dim = self.gmm_means.shape[0]
        gmm_attr_to_original_idx = list(range(self.dimension_init))
        for called_idx in sorted(bayes_called_attributes, reverse=True): del gmm_attr_to_original_idx[called_idx]
        non_source_gmm_mask = torch.ones(gmm_dim, dtype=torch.bool)
        non_source_gmm_mask[self.bsource_actual_position] = False
        non_source_original_indices = [idx for i, idx in enumerate(gmm_attr_to_original_idx) if non_source_gmm_mask[i]]
        if not non_source_original_indices:
            p_other_per_kernel = torch.ones(self.num_kernels)
        else:
            lower_b_other = queried_rectangle[0, non_source_original_indices].unsqueeze(1)
            upper_b_other = queried_rectangle[1, non_source_original_indices].unsqueeze(1)
            means_other = self.gmm_means[non_source_gmm_mask, :]
            stds_other = self.gmm_stds[non_source_gmm_mask, :]
            z_lower = (lower_b_other - means_other) / stds_other
            z_upper = (upper_b_other - means_other) / stds_other
            prob_mass_per_attr = self._cdf(z_upper) - self._cdf(z_lower)
            p_other_per_kernel = torch.prod(prob_mass_per_attr, dim=0)

        # --- FINAL ROBUST LOGIC FOR BAYESNET CORRECTION FACTOR ---
        correction_factors_per_kernel = torch.ones(self.num_kernels)

        cond_probs=1

        for i in range(num_deps):
            # --- Setup for this dependency ---
            bayes_obj = self.bayesnet[i]
            mass_table = self.precomputed_mass_tables[i]
            source_original_idx = bayes_source_attributes[i]
            called_original_idx = bayes_called_attributes[i]
            a1_orig, b1_orig = queried_rectangle[0, source_original_idx], queried_rectangle[1, source_original_idx]
            query_range_j = queried_rectangle[:, called_original_idx]
            cond_probs = cond_probs * bayes_obj.get_all_slice_cond_probs(query_range_j)

            if i+1==num_deps or bayes_source_attributes[i+1]!=bayes_source_attributes[i]:

                source_actual_idx = self.bsource_actual_position[i]
                source_means = self.gmm_means[source_actual_idx, :]
                source_stds = self.gmm_stds[source_actual_idx, :]
                
                slice_edges = bayes_obj.slice_edges
                min_domain, max_domain = slice_edges[0], slice_edges[-1]
                
                prob_x1_per_kernel = torch.zeros(self.num_kernels, dtype=torch.float32)

                # --- Part 1: Below-Domain Contribution (Nearest-Assignment using first slice) ---
                lower_integration_end = torch.min(b1_orig, min_domain)
                if a1_orig < lower_integration_end:
                    z_start = (a1_orig - source_means) / source_stds
                    z_end = (lower_integration_end - source_means) / source_stds
                    mass_below = self._cdf(z_end) - self._cdf(z_start)
                    prob_x1_per_kernel += mass_below * cond_probs[0]

                # --- Part 2: Within-Domain Contribution (The logic from two versions ago) ---
                within_a1 = torch.max(a1_orig, min_domain)
                within_b1 = torch.min(b1_orig, max_domain)
                if within_a1 < within_b1:
                    k_start = torch.bucketize(within_a1, slice_edges[1:-1])
                    k_end = torch.bucketize(within_b1, slice_edges[1:-1])

                    if k_start == k_end:
                        z_a1 = (within_a1 - source_means) / source_stds
                        z_b1 = (within_b1 - source_means) / source_stds
                        mass = self._cdf(z_b1) - self._cdf(z_a1)
                        prob_x1_per_kernel += mass * cond_probs[k_start]
                    else:
                        s_end_of_start = slice_edges[k_start + 1]
                        z_a1 = (within_a1 - source_means) / source_stds
                        z_s_end = (s_end_of_start - source_means) / source_stds
                        mass_start = self._cdf(z_s_end) - self._cdf(z_a1)
                        prob_x1_per_kernel += mass_start * cond_probs[k_start]

                        if k_start + 1 < k_end:
                            full_slices_mass = mass_table[k_start + 1 : k_end, :]
                            full_slices_cond_probs = cond_probs[k_start + 1 : k_end]
                            prob_x1_per_kernel += torch.matmul(full_slices_cond_probs, full_slices_mass)

                        s_start_of_end = slice_edges[k_end]
                        z_s_start = (s_start_of_end - source_means) / source_stds
                        z_b1 = (within_b1 - source_means) / source_stds
                        mass_end = self._cdf(z_b1) - self._cdf(z_s_start)
                        prob_x1_per_kernel += mass_end * cond_probs[k_end]

                # --- Part 3: Above-Domain Contribution (Nearest-Assignment using last slice) ---
                upper_integration_start = torch.max(a1_orig, max_domain)
                if upper_integration_start < b1_orig:
                    z_start = (upper_integration_start - source_means) / source_stds
                    z_end = (b1_orig - source_means) / source_stds
                    mass_above = self._cdf(z_end) - self._cdf(z_start)
                    prob_x1_per_kernel += mass_above * cond_probs[-1] # Use last slice's prob

                correction_factors_per_kernel *= prob_x1_per_kernel
                cond_probs=1

        # --- FINAL CALCULATION (Unchanged) ---
        total_prob = torch.sum(self.gmm_weights * p_other_per_kernel * correction_factors_per_kernel)
        return total_prob
    
    def predict_and_sample(self,
                        target_rectangle_init: torch.Tensor,
                        bayes_source_attributes=None,
                        bayes_called_attributes=None,
                        pertubation_level=0):
        result=-1
        modified_target_rectangle=(target_rectangle_init-reg_const_one[1:2,:])/reg_const_one[2:3,:]-1.5
        modified_target_rectangle=modified_target_rectangle-0.05*(modified_target_rectangle<reg_const_two[0:1,:])-0.05*(modified_target_rectangle<reg_const_two[1:2,:])+0.05*(modified_target_rectangle>reg_const_two[2:3,:])+0.05*(modified_target_rectangle>reg_const_two[3:4,:])
        clamped_target_rectangle=torch.clamp(modified_target_rectangle,min=-1.7,max=1.7)
        query_edge_length=clamped_target_rectangle[1,:]-clamped_target_rectangle[0,:]
        values, _=torch.topk(query_edge_length[indexes],k=3,largest=False)
        volume=torch.prod(values)
        modified_target_rectangle=modified_target_rectangle-reg_const_three
        mcvs_inside=(minimum_radius_mcvs[:,:,0]>modified_target_rectangle[0:1,:])*(minimum_radius_mcvs[:,:,0]<modified_target_rectangle[1:2,:])
        upperbound_mcvs=torch.amax(minimum_radius_mcvs[:,:,0]+minimum_radius_mcvs[:,:,1]*0.05-1000000000.0*(torch.logical_not(mcvs_inside)),0)
        lowerbound_mcvs=torch.amin(minimum_radius_mcvs[:,:,0]-minimum_radius_mcvs[:,:,1]*0.05+1000000000.0*(torch.logical_not(mcvs_inside)),0)
        modified_target_rectangle[0,indexes]=torch.minimum(modified_target_rectangle[0,indexes],lowerbound_mcvs[indexes])
        modified_target_rectangle[1,indexes]=torch.maximum(modified_target_rectangle[1,indexes],upperbound_mcvs[indexes])
        if bayes_source_attributes is not None:
            result=self.predict_analytical(modified_target_rectangle,bayes_source_attributes,bayes_called_attributes)
            if result<1/(20*size):
                return result, 0, True, []
        tempmat=(torch.transpose(modified_target_rectangle[:,self.non_bayes_indexes],0,1).unsqueeze(1)-self.gmm_means.unsqueeze(2))
        tempval_unpert=self._cdf(tempmat/(self.gmm_stds).unsqueeze(2))
        result_nobayes=torch.clamp(torch.prod(tempval_unpert[:,:,1]-tempval_unpert[:,:,0],0),min=0)
        result_nobayes=torch.sum(result_nobayes*self.gmm_weights)
        if (result*(result>-1)+result_nobayes*(result==-1))<1/(20*size):
            return (result*(result>-1)+result_nobayes*(result==-1)), 0, True, []
        tempval_pert=self._cdf(tempmat/(self.gmm_stds_perturbed[pertubation_level]).unsqueeze(2))
        kernelprob_nobayes_pert=torch.prod(tempval_pert[:,:,1]-tempval_pert[:,:,0],0)
        kernelprob_nobayes_pert=(torch.clamp(kernelprob_nobayes_pert,min=0)*self.gmm_weights).to(torch.float64)
        kernelprob_nobayes_pert=(kernelprob_nobayes_pert+1e-11)/torch.sum(kernelprob_nobayes_pert+1e-11)
        choose_kernel=np.random.choice(kernels_num,draws,True,kernelprob_nobayes_pert)
        position=torch.rand((dimension,draws))
        position=position*(tempval_pert[:,choose_kernel,1]-tempval_pert[:,choose_kernel,0])+(tempval_pert[:,choose_kernel,0])
        position=normal.icdf(position)*(self.gmm_stds_perturbed[pertubation_level])[:,choose_kernel]+self.gmm_means[:,choose_kernel]
        prob_est_list=torch.ones((draws))
        '''if bayes_source_attributes is not None:
            for i in range(0,len(bayes_called_attributes)):
                prob_est_list=prob_est_list*(self.bayesnet[i])(position[bayes_source_attributes[i],:],modified_target_rectangle[:,bayes_called_attributes[i]])'''
        return (result*(result>-1)+result_nobayes*(result==-1)), position, False, prob_est_list
    
    def GMM_OnePointEst(self,
                        position: torch.Tensor,
                        pertubation_level=0):
        calc_matrix=(position-self.gmm_means[:dimension,:].unsqueeze(2))/self.gmm_stds_perturbed[pertubation_level].unsqueeze(2)
        calc_matrix=0.5*(calc_matrix**2)
        calc_matrix=torch.exp(-1*torch.sum(calc_matrix,0))/(self.gaussian_coefficient[pertubation_level])
        return torch.sum(calc_matrix*self.gmm_weights.unsqueeze(1),0)
        
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

def set_global(Intervals,TimeAllowed_min,batch_size,dim,cutoff,bayes_source_attributes=None,bayes_called_attributes=None):
    global histogram, gmm_estimator, length, BatchSize, TimeMax, TimeChart, TimeChart_InNetwork, SignalDecay, SignalDecay_abbrev, NoiseVar, NoiseVar_inv, NoiseVarSquare, NoiseVarSquare_inv, Weight, Weight_sqrt, Weight_used, PointArray, dimension, PointArrayScore, PointArrayScore_double, cutoff_length, cutoff_time, Var_min, Time_min, residual_tail_adjust_term, normal
    histogram=SingleAttributeHistogram(num_bins=200, num_mcvs=20)
    histogram.load(dataset_name+'_histogram.npy',dataset_name+'_mcv.npz',dataset_name+'_meta.npz')
    cutoff_length=cutoff
    cutoff_time=Intervals[cutoff]
    TimeMax=Intervals[len(Intervals)-1]
    dimension=dim
    BatchSize=batch_size
    length=len(Intervals)-1
    TimeChart=np.zeros((1,(length)*batch_size))
    TimeChart[0,:]=np.repeat([(Intervals[i+1]+Intervals[i])/2 for i in range(0,length)],batch_size)
    SignalDecay=np.exp(-1*TimeChart)
    SignalDecay_abbrev=(torch.exp(-1*torch.tensor([(Intervals[i+1]+Intervals[i])/2 for i in range(0,length)]))).unsqueeze(0)
    NoiseVarSquare=(1-SignalDecay**2)
    NoiseVar=np.sqrt(NoiseVarSquare)
    Weight=np.zeros((1,(length)*batch_size))
    Weight[0,:]=np.repeat([(Intervals[i+1]-Intervals[i]) for i in range(0,length)],batch_size)
    Weight=torch.tensor(Weight)
    Weight_sqrt=torch.sqrt(Weight)
    SobolSampler=stats.qmc.Sobol(dim,scramble=True,seed=123)
    UniSobol=np.transpose(SobolSampler.random(2**ma.ceil(ma.log2(batch_size*length))))
    UniSobol=(UniSobol-1*(UniSobol>1))
    NormalSobol=stats.norm.ppf(UniSobol*0.99999+0.000005)
    PointArray=torch.tensor(NormalSobol[:,0:(length)*batch_size]*NoiseVar)
    PointArrayScore=-1*PointArray/NoiseVarSquare
    PointArrayScore_double=2*PointArrayScore
    TimeChart=torch.tensor(TimeChart)
    SignalDecay=torch.tensor(SignalDecay)
    NoiseVar=torch.tensor(NoiseVar)
    NoiseVarSquare=torch.tensor(NoiseVarSquare)
    Time_min=[TimeAllowed_min,1.5*TimeAllowed_min,2*TimeAllowed_min,3*TimeAllowed_min,4*TimeAllowed_min, 6*TimeAllowed_min, 8*TimeAllowed_min]
    NoiseVarSquare_inv=[(1/(1-torch.exp(-2*(TimeChart+Time_min[i])))).to(torch.float32) for i in range(0,7)]
    NoiseVar_inv=[(torch.sqrt(NoiseVarSquare_inv[i])).to(torch.float32) for i in range(0,7)]
    '''注意NoiseVar_inv不是NoiseVar的倒数，因为NoiseVarInv用在神经网络内部，加了时移；NoiseVar用在神经网络外部，没加时移'''
    Var_min=torch.tensor([ma.exp(2*Time_min[i])-1 for i in range(0,7)])
    TimeChart_InNetwork=[(TimeChart+Time_min[i]).to(torch.float32) for i in range(0,7)]
    residual_tail_adjust_term=[(1/NoiseVarSquare[:,BatchSize*cutoff_length:]-(NoiseVarSquare_inv[i])[:,BatchSize*cutoff_length:]).unsqueeze(2).to(torch.float32) for i in range(0,7)]
    Weight_used=torch.transpose(Weight/BatchSize,0,1)
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
    gmm_estimator=GMM_Estimator( kernels_matrix, bayesnet, bsource_actual_position, dimension_init)
    normal=torch.distributions.Normal(loc=0.0, scale=1.0)

def get_reg_consts():
    global reg_const_one, reg_const_two, reg_const_three, size, minimum_radius, minimum_radius_mcvs
    reg_consts=np.load('reg_consts_'+dataset_name+'.npy')
    reg_const_one=torch.tensor(reg_consts[0:3,:]).to(torch.float32)
    reg_const_two=torch.tensor(reg_consts[3:7,:]).to(torch.float32)
    reg_const_three=torch.tensor(reg_consts[7:8,:]).to(torch.float32)
    size=reg_consts[8,0].astype(int)
    minimum_radius=torch.tensor(np.load('minrad_'+dataset_name+'.npy'))
    minimum_radius_mcvs=torch.tensor(np.load('minrad_mcvs_'+dataset_name+'.npy'))

def CardEst_Implement(target_rectangle,pertubation_level=0,perturbation_sample=False,perturbation_int=False,perturbation_calc=False,bayes_source_attributes=None,bayes_called_attributes=None,attributes_not_covered=None,KDE_only=False):
    result,position,predictor_is_more_accurate,prob_est_list=gmm_estimator.predict_and_sample(target_rectangle,bayes_source_attributes,bayes_called_attributes,pertubation_level)
    if predictor_is_more_accurate or KDE_only:
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
    z=min(torch.mean(adjust_term*(prob_est_list+1e-7))/torch.mean(prob_est_list+1e-7),40)
    return result*z

def CardEst_Implement_Selective(target_rectangle,perturbation_sample=False,perturbation_int=False,perturbation_calc=False,bayes_source_attributes=None,bayes_called_attributes=None,KDE_only=False):
    torch.manual_seed(0)
    np.random.seed(0)
    attribute_exceeded_above=reg_const_one[0,:]<=target_rectangle[1,:]
    attribute_exceeded_below=reg_const_one[1,:]>=target_rectangle[0,:]
    attributes_not_covered=attribute_exceeded_above*attribute_exceeded_below*1
    target_rectangle[0:1,unit_of_variables[0,:]!=0]=(torch.ceil(target_rectangle[0:1,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]-1e-6)-0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    target_rectangle[1:2,unit_of_variables[0,:]!=0]=(torch.floor(target_rectangle[1:2,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]+1e-6)+0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    edge_length=target_rectangle[1:2,:]-target_rectangle[0:1,:]
    if torch.min(edge_length)<1e-6:
        return torch.tensor([0])
    if torch.sum(attributes_not_covered)==dimension_init:
        return torch.tensor([1])
    if torch.sum(attributes_not_covered)==dimension_init-1:
        queriedatt=torch.argmin(attributes_not_covered)
        target_rectangle=target_rectangle.numpy()
        return torch.tensor(histogram.estimate(queriedatt,target_rectangle[0,queriedatt],target_rectangle[1,queriedatt]))/size
    target_rectangle[0:1,attribute_exceeded_below]=2*reg_const_one[1:2,attribute_exceeded_below]-target_rectangle[1:2,attribute_exceeded_below]
    target_rectangle[1:2,attribute_exceeded_above]=2*reg_const_one[0:1,attribute_exceeded_above]-target_rectangle[0:1,attribute_exceeded_above]
    center=(target_rectangle[0:1,:]+target_rectangle[1:2,:])/2
    interval_of_center=torch.clamp(torch.floor((center-reg_const_one[1,:])*64/(3*reg_const_one[2,:])),max=63)
    minrad=minimum_radius[torch.arange(dimension_init),interval_of_center.to(torch.int)]/2
    target_rectangle[0,:]=torch.minimum(target_rectangle[0,:],center-minrad)
    target_rectangle[1,:]=torch.maximum(target_rectangle[1,:],center+minrad)
    modified_target_rectangle=(target_rectangle-reg_const_one[1:2,:])/reg_const_one[2:3,:]-1.5
    modified_target_rectangle=modified_target_rectangle-0.05*(modified_target_rectangle<reg_const_two[0:1,:])-0.05*(modified_target_rectangle<reg_const_two[1:2,:])+0.05*(modified_target_rectangle>reg_const_two[2:3,:])+0.05*(modified_target_rectangle>reg_const_two[3:4,:])
    clamped_target_rectangle=torch.clamp(modified_target_rectangle,min=-1.7,max=1.7)
    modified_edge_length=clamped_target_rectangle[1,:]-clamped_target_rectangle[0,:]
    values, _=torch.topk(modified_edge_length[indexes],k=3,largest=False)
    pertubation_level=torch.sum(torch.prod(values)>(10*torch.sqrt(Var_min[0:6]))**3)
    return CardEst_Implement(target_rectangle,pertubation_level,perturbation_sample,perturbation_int,perturbation_calc,bayes_source_attributes,bayes_called_attributes,attributes_not_covered,KDE_only)

def log_round(val):
    if val>20:
        return torch.round(val)
    if val<1:
        return val
    a=ma.log(ma.floor(val))
    b=ma.log(ma.ceil(val))
    c=ma.log(val)
    return ma.floor(val)*((c-a)<=(b-c))+ma.ceil(val)*((c-a)>(b-c))

def get_query_basics(target_rectangle):
    global processedqueries
    attribute_exceeded_above=reg_const_one[0,:]<=target_rectangle[1,:]
    attribute_exceeded_below=reg_const_one[1,:]>=target_rectangle[0,:]
    attribute_not_covered=attribute_exceeded_above*attribute_exceeded_below*1
    target_rectangle[0:1,unit_of_variables[0,:]!=0]=(torch.ceil(target_rectangle[0:1,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]-1e-6)-0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    target_rectangle[1:2,unit_of_variables[0,:]!=0]=(torch.floor(target_rectangle[1:2,unit_of_variables[0,:]!=0]/unit_of_variables[:,unit_of_variables[0,:]!=0]+1e-6)+0.5)*unit_of_variables[:,unit_of_variables[0,:]!=0]
    target_rectangle[0:1,attribute_exceeded_below]=2*reg_const_one[1:2,attribute_exceeded_below]-target_rectangle[1:2,attribute_exceeded_below]
    target_rectangle[1:2,attribute_exceeded_above]=2*reg_const_one[0:1,attribute_exceeded_above]-target_rectangle[0:1,attribute_exceeded_above]
    target_rectangle=(target_rectangle-reg_const_one[1:2,:])/reg_const_one[2:3,:]-1.5
    target_rectangle=target_rectangle-0.05*(target_rectangle<reg_const_two[0:1,:])-0.05*(target_rectangle<reg_const_two[1:2,:])+0.05*(target_rectangle>reg_const_two[2:3,:])+0.05*(target_rectangle>reg_const_two[3:4,:])
    target_rectangle[0:1,:]=torch.maximum(target_rectangle[0:1,:],torch.ones((1,dimension_init))*-1.7)
    target_rectangle[1:2,:]=torch.minimum(target_rectangle[1:2,:],torch.ones((1,dimension_init))*1.7)
    query_edge_length=target_rectangle[1,:]-target_rectangle[0,:]
    values, _=torch.topk(query_edge_length[indexes],k=3,largest=False)
    volume=torch.prod(values)
    minedge=torch.min(query_edge_length[indexes])
    if minedge<=0:
        print(target_rectangle)
        print(minedge)
        raise ValueError('minedge<0')
    return volume.item(),minedge,attribute_not_covered

def actual_query(target_rectangle):
    lowerleft_point=np.transpose(target_rectangle[0:1,:])
    upperright_point=np.transpose(target_rectangle[1:2,:])
    satisfy=(torch.transpose(data_init,0,1)>=lowerleft_point)*(torch.transpose(data_init,0,1)<=upperright_point)
    satisfy=torch.prod(satisfy,0)
    return torch.sum(satisfy).float()/size

def calculate_metric(q_errors):
    return np.mean(np.log(q_errors)**1.5)

def run(dname,uvar,dim,kernum,tm,workload_size,purely_time=False,bayes_source_attributes=None,bayes_called_attributes=None):
    global data_init, dataset_name, unit_of_variables, dimension, dimension_init, kernels_num, tmin, evalnet_head, evalnet_tail, kernels_matrix, timeused, j, CardEst_Compiled, kde_pointvals, diffusion_pointvals, current_query, predictions, indexes, bayesnet
    dataset_name=dname
    unit_of_variables=(torch.tensor(uvar)).float()
    dimension_init=dim
    if bayes_source_attributes is None:
        dimension=dimension_init
        indexes=[i for i in range(0,dimension_init)]
    else:
        dimension=dimension_init-len(bayes_source_attributes)
        indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    kernels_num=kernum
    tmin=tm
    starting_col=0
    data_init=pd.read_csv('original'+dataset_name+'.csv',index_col=False)
    data_init=data_init.to_numpy()
    data_init=data_init[1:,starting_col:starting_col+dimension_init]
    orisize=(torch.tensor(data_init).size())[0]
    data_init=torch.tensor(data_init)
    bayesnet=[]    
    if bayes_source_attributes is not None:
        bayesnet=[0 for i in range(0,len(bayes_source_attributes))]
        for i in range(0,len(bayes_source_attributes)):
            bayesnet[i]=BayesnetEstimator(bayes_source_attributes[i],bayes_called_attributes[i],64,45)
            (bayesnet[i]).load()
    get_reg_consts()    
    kernels_matrix=torch.tensor(np.load('KDE_params_adjusted_'+dataset_name+'.npy')).float()
    evalnet_head=torch.load(dataset_name+'_head.pkl')
    evalnet_tail=torch.load(dataset_name+'_tail.pkl')
    j=0
    min_sel=1/size
    timeused=np.zeros(workload_size+1)
    set_global(timestep,tmin,64,dimension,43,bayes_source_attributes,bayes_called_attributes)
    workloads=pd.read_csv(dataset_name+'_AreCE_trainset.csv', delimiter=' ', dtype=np.float32, header=None)
    workloads=workloads.to_numpy()
    workloads=torch.tensor(workloads)
    actual_selectivity=np.zeros((workload_size))
    error_with_diffusion=np.zeros((workload_size))
    error_without_diffusion=np.zeros((workload_size))
    error_without_diffusion_perturbed=np.zeros((workload_size))
    kde_pointvals=np.zeros((25,workload_size))
    diffusion_pointvals=np.zeros((25,workload_size))
    query_volume=np.zeros((workload_size))
    minimum_edge=np.zeros((workload_size))
    KDE_estimated_selectivity=np.zeros((workload_size))
    timestart=time.time()
    for i in range(0,workload_size):
        current_query=i
        target_rectangle=workloads[2*i:2*i+2,:]
        if bayes_called_attributes is not None:
            target_rectangle[0,bayes_called_attributes]=reg_const_one[1,bayes_called_attributes]-1.00
            target_rectangle[1,bayes_called_attributes]=reg_const_one[0,bayes_called_attributes]+1.00
            target_rectangle=target_rectangle.to(torch.float32)
        query_volume[i],minimum_edge[i],_=get_query_basics(target_rectangle)        
        estimate_selectivity=log_round(CardEst_Implement_Selective(target_rectangle,True,False,True,bayes_source_attributes,bayes_called_attributes,False)*size)/size
        a=log_round(CardEst_Implement_Selective(target_rectangle,True,False,True,bayes_source_attributes,bayes_called_attributes,True)*size)/size
        KDE_estimated_selectivity[i]=max(a,min_sel)
        e=actual_query(target_rectangle)
        actual_selectivity[i]=max(e,min_sel)
        error_with_diffusion[i]=(max(e,min_sel))/max(estimate_selectivity,min_sel)
        error_without_diffusion[i]=max(e,min_sel)/max(a,min_sel)
    print(time.time()-timestart)
    print(np.sum(timeused))
    print(np.exp(np.mean(np.log(error_with_diffusion))))
    print(np.median(error_with_diffusion))
    QError=np.maximum(error_with_diffusion,1/error_with_diffusion)
    QError_without_diffusion=np.maximum(error_without_diffusion,1/error_without_diffusion)
    z=np.sort(QError)
    print(z[ma.ceil(workload_size*0.95)])
    print(z[ma.ceil(workload_size*0.99)])
    colors=np.where(QError<=QError_without_diffusion,'red','blue').astype('<U5')
    colors[QError==QError_without_diffusion]='green'
    query_volume=np.log(query_volume)
    KDE_estimated_selectivity=np.log(KDE_estimated_selectivity)
    minimum_edge=np.log(minimum_edge)
    relative_error=np.log(error_with_diffusion)
    log_selectivity=np.log(actual_selectivity)
    plt.figure(figsize=(8, 8))  
    plt.scatter(np.exp(KDE_estimated_selectivity), np.exp(query_volume)/((3.4)**dimension), c=colors, s=1.5 , alpha=0.5, edgecolors='none') 
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
    y_labels = (q_error_corrector < q_error_predictor).astype(int)
    log_q_pred_sq = np.log(q_error_predictor)**2
    log_q_corr_sq = np.log(q_error_corrector)**2
    sample_weights = np.abs(log_q_pred_sq - log_q_corr_sq)
    X=pd.DataFrame({'vol':query_volume, 'est':KDE_estimated_selectivity})
    X_train, X_test, y_train, y_test, q_pred_train, q_pred_test, q_corr_train, q_corr_test, weights_train, weights_test = train_test_split(X, y_labels, q_error_predictor, q_error_corrector, sample_weights, test_size=0.2, random_state=0)
    print("Step 2: Training the Decision Tree with sample weights...")
    classifier = DecisionTreeClassifier(max_depth=4, random_state=0)
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
    with open('classifier_'+dataset_name+'.pkl','wb') as classifierfile:
        pickle.dump(classifier,classifierfile)
    
data_used=0
timestep=(np.load('timestep.npy')).tolist()
cutoff=43
draws=25
j=1
if j==1:
    run('forest',[[1 for i in range(0,10)]],10,1280,2/320,100000,purely_time=False)
if j==2:
    run('power',[[1e-3,1e-3,1e-2,2e-1,1,1,1]],7,1280,1/640,100000,purely_time=False,bayes_source_attributes=[3],bayes_called_attributes=[0])
if j==3:
    run('weather',[[1e-4,1e-4,1e-1,1,1,1,1]],7,1280,3/1280,100000,purely_time=False)
if j==4:
    run('advantage',[[1,1,1,1,1]],5,1280,1/320,100000,purely_time=False)

