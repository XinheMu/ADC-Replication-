import torch 
import sys
import ast
torch.set_grad_enabled(False)
import torch.nn as nn
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
import pickle
import os
import glob
import re
from typing import Tuple, Union

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
        self.filename = dataset_name+'/'+self.filename
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
            'grid_x1': self.grid_x1.to(torch.float16),
            'grid_x2': self.grid_x2.to(torch.float16),
            'cell_y_min': self.cell_y_min.to(torch.float16),
            'cell_y_max': self.cell_y_max.to(torch.float16),
            'hist_weights': self.hist_weights.to(torch.float16),
            'mcv_values': self.mcv_values.to(torch.float16),
            'mcv_probs': self.mcv_probs.to(torch.float16)
        }
        print('saving '+self.filename)
        np.save(self.filename, state)

    def load(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Model file {self.filename} not found.")
        state = np.load(self.filename, allow_pickle=True).item()
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
        prob_mcv = (mcv_hit.float() * self.mcv_probs).sum(dim=-1)
        
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
        k = torch.arange(self.bin_val, device=self.device).float() # (B)
        
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
        return total_prob

def regu_step1(starting_col=0):
    global reg_const_one, reg_const_two, reg_const_three, data_used_primitive, data_used, size, corrupted_data_size, data_init
    data_init=pd.read_csv(dataset_name+'training/'+'original'+dataset_name+'.csv',index_col=False)
    data_init=data_init.to_numpy()
    data_init=data_init[1:,starting_col:starting_col+dimension_init]
    orisize=(torch.tensor(data_init).size())[0]
    data_init=torch.tensor(data_init).to(torch.float32)
    data_init=data_init[data_init[:,0]>(nan_to+1e-6),:]
    reg_const_zero=torch.zeros((6,dimension_init))
    for i in range(0,dimension_init):
        unique_values=torch.unique(data_init[:,i],sorted=True,return_counts=False)
        gap_size=unique_values[1:]-unique_values[:-1]
        maxgap,maxgap_position=torch.topk(gap_size,2,largest=True,sorted=False)
        reg_const_zero[0,i]=unique_values[min(maxgap_position[0],maxgap_position[1])]
        reg_const_zero[1,i]=unique_values[min(maxgap_position[0],maxgap_position[1])+1]
        reg_const_zero[2,i]=reg_const_zero[1,i]-max((unique_values[min(maxgap_position[0],maxgap_position[1])+1]-unique_values[min(maxgap_position[0],maxgap_position[1])])-(torch.max(data_init[:,i])-torch.min(data_init[:,i]))/15,0)
        reg_const_zero[3,i]=unique_values[max(maxgap_position[0],maxgap_position[1])]
        reg_const_zero[4,i]=unique_values[max(maxgap_position[0],maxgap_position[1])+1]
        reg_const_zero[5,i]=reg_const_zero[4,i]-max((unique_values[max(maxgap_position[0],maxgap_position[1])+1]-unique_values[max(maxgap_position[0],maxgap_position[1])])-(torch.max(data_init[:,i])-torch.min(data_init[:,i]))/15,0)
    data_used=data_init-(reg_const_zero[1:2,:]-reg_const_zero[2:3,:])*(data_init>(reg_const_zero[2:3,:]))-(reg_const_zero[4:5,:]-reg_const_zero[5:6,:])*(data_init>(reg_const_zero[5:6,:]))
    reg_const_one=torch.zeros((3,dimension_init))
    for i in range(0,dimension_init):
        reg_const_one[:,i:i+1]=torch.tensor([[torch.max(data_used[:,i])],[torch.min(data_used[:,i])],[(1/3)*(torch.max(data_used[:,i])-torch.min(data_used[:,i]))]])
    data_used=(data_used-reg_const_one[1,:])/(reg_const_one[2,:])-1.5
    data_used_primitive=data_used.clone()
    size=(data_used.size())[0]
    corrupted_data_size=orisize-size
    reg_const_two=torch.zeros((4,dimension_init))
    for i in range(0,dimension_init):
        unique_values,counts=torch.unique(data_used[:,i],return_counts=True)
        grad_left=torch.zeros_like(unique_values)
        for j in range(1,(unique_values.size())[0]):
            if counts[j]<size/200:
                grad_left[j]=0
            else:
                grad_left[j]=counts[j]-torch.sum((data_used[:,i]<unique_values[j])*(data_used[:,i]>(unique_values[j]-0.05)))*(0.05/min(unique_values[j]+1.5,0.05))
        top_counts, top_indices=torch.topk(grad_left,2)
        reg_const_two[0:2,i]=unique_values[top_indices]*(grad_left[top_indices]>(size/200))-100*(grad_left[top_indices]<=(size/200))
        grad_right=torch.zeros_like(unique_values)
        for j in range(0,(unique_values.size())[0]-1):
            if counts[j]<size/200:
                grad_right[j]=0
            else:
                grad_right[j]=counts[j]-torch.sum((data_used[:,i]>unique_values[j])*(data_used[:,i]<(unique_values[j]+0.05)))*(0.05/min(1.5-unique_values[j],0.05))
        top_counts, top_indices=torch.topk(grad_right,2)
        reg_const_two[2:4,i]=unique_values[top_indices]*(grad_right[top_indices]>(size/200))-100*(grad_right[top_indices]<=(size/200))
        data_used[:,i]=data_used[:,i]-0.05*(data_used[:,i]<reg_const_two[0,i])-0.05*(data_used[:,i]<reg_const_two[1,i])+0.05*(data_used[:,i]>reg_const_two[2,i])+0.05*(data_used[:,i]>reg_const_two[3,i])
    reg_const_three=torch.zeros((1,dimension_init))
    for i in range(0,dimension_init):
        reg_const_three[0,i]=torch.mean(data_used[:,i])
        data_used[:,i]=data_used[:,i]-reg_const_three[0,i]
    reg_consts=np.zeros((15,dimension_init))
    reg_consts[0:3,:]=reg_const_one[0:3,:]
    reg_consts[3:7,:]=reg_const_two[0:4,:]
    reg_consts[7:8,:]=reg_const_three[0:1,:]
    reg_consts[8,0]=size
    reg_consts[8,1]=orisize-size
    reg_consts[9:15,:]=reg_const_zero[0:6,:]
    np.save(dataset_name+'/'+'reg_consts_'+dataset_name+'.npy',reg_consts)
    used_indexes=[i for i in range(0,dimension_init)]
    data_used=data_used[:,used_indexes].to(torch.float32)

def test_functional_dependency():
    shared_volume=torch.zeros((dimension,dimension))
    for i in range(0,dimension):
        for j in range(0,dimension):
            if i==j:
                shared_volume[i,j]=10
            else:
                correlation_data=torch.zeros(2,256)
                counts=torch.zeros(1,256)
                for k in range(0,256):
                    lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*k/256
                    upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(k+1)/256+0.01*(k==255)
                    mask=(data_init[:,i]<upper_bound)*(data_init[:,i]>=lower_bound)
                    counts[0,k]=torch.sum(mask)
                    if torch.sum(mask)==0:
                        correlation_data[0,k]=1000000
                        correlation_data[1,k]=1000000
                    else:
                        correlation_data[0,k]=torch.min(data_init[mask,j])
                        correlation_data[1,k]=torch.max(data_init[mask,j])
                shared_volume[i,j]=0.8*torch.sum(counts*(correlation_data[1,:]-correlation_data[0,:])/(reg_const_one[2,j]*size))+0.2*torch.mean((correlation_data[1,:]-correlation_data[0,:])/reg_const_one[2,j])
        print('fimished evaluating '+str(i+1)+' out of '+str(dimension)+' attributes')
    print(shared_volume)
    return shared_volume

def test_finegrain_functional_dependency(source_attribute,called_attribute,cuts,cutstwo):
    print('testing more finegrained functional dependency to choose assisting attribute')
    shared_volume=torch.zeros((dimension))
    for i in range(0,dimension):
        if i==source_attribute or i==called_attribute:
            shared_volume[i]=100
        else:
            correlation_data=torch.zeros((2,cuts,cutstwo))
            counts=torch.zeros((cuts,cutstwo))
            for j in range(0,cuts):
                lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*j/cuts
                upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(j+1)/cuts+0.01*(j==cuts-1)
                mask_j=(data_init[:,i]<upper_bound)*(data_init[:,i]>=lower_bound)
                for k in range(0,cutstwo):
                    lower_bound=reg_const_one[1,source_attribute]+3*reg_const_one[2,source_attribute]*k/cutstwo
                    upper_bound=reg_const_one[1,source_attribute]+3*reg_const_one[2,source_attribute]*(k+1)/cutstwo+0.01*(k==cutstwo-1)
                    mask_k=(data_init[:,source_attribute]<upper_bound)*(data_init[:,source_attribute]>=lower_bound)
                    mask=mask_k*mask_j
                    counts[j,k]=torch.sum(mask)
                    if torch.sum(mask)==0:
                        correlation_data[0,j,k]=1000000
                        correlation_data[1,j,k]=1000000
                    else:
                        correlation_data[0,j,k]=torch.min(data_init[mask,called_attribute])
                        correlation_data[1,j,k]=torch.max(data_init[mask,called_attribute])
            shared_volume[i]=0.8*torch.sum(counts*(correlation_data[1,:,:]-correlation_data[0,:,:])/(reg_const_one[2,called_attribute]*size))+0.2*torch.mean((correlation_data[1,:,:]-correlation_data[0,:,:])/reg_const_one[2,called_attribute])
    print('assisting atribute chosen to be the one who can best lower the bayes_called_attribute range')
    print(shared_volume)
    return torch.argmin(shared_volume)

class SingleAttributeHistogram:
    """
    A class to create, save, load, and query single-attribute histograms
    for a multi-dimensional database. It handles Most Common Values (MCVs)
    separately for improved accuracy.
    """

    def __init__(self, num_bins: int, num_mcvs: int):
        """
        Initializes the histogram model with hyperparameters.

        Args:
            num_bins (int): The number of bins to use for the histogram part.
            num_mcvs (int): The number of most common values to store separately.
        """
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

        Args:
            data (np.ndarray): The database table, with shape (num_rows, num_dimensions).
            hist_path (str): Path to save the histogram counts array.
            mcv_path (str): Path to save the MCV information (values and counts).
            meta_path (str): Path to save metadata (min/max values, total rows).
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
                
                # Sort by counts in descending order
                sorted_indices = torch.argsort(counts, descending=True)
                
                # Get the top N MCVs
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

        Args:
            hist_path (str): Path to the histogram counts array file.
            mcv_path (str): Path to the MCV information file.
            meta_path (str): Path to the metadata file.
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

if __name__ == "__main__":
    params=['CardEst','power','7','-100000000']
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    dimension_init=int(params[2])
    dimension=dimension_init
    nan_to=float(params[3])
    data_init=pd.read_csv(dataset_name+'training/'+'original'+dataset_name+'.csv',index_col=False)
    data_init=data_init.to_numpy()
    data_init=data_init[data_init[:,0]>(nan_to+1e-6),:]
    print("Training Single Attribute Histograms")
    hist_estimator = SingleAttributeHistogram(num_bins=200, num_mcvs=20)
    hist_estimator.train(data_init,dataset_name+'/'+dataset_name+'_histogram.npy',dataset_name+'/'+dataset_name+'_mcv.npz',dataset_name+'/'+dataset_name+'_meta.npz')
    print('Preparing data for Bayesnet test')
    regu_step1()
    use_default=True
    if use_default:
        print("Testing Functional Dependency: small number indicate larger dependency with 0.15 being the Bayesnet threshold")
        shared_volume=test_functional_dependency()
        max_correlation=torch.argmin(shared_volume)
        max_correlation=[max_correlation//dimension,max_correlation%dimension]
        if shared_volume[max_correlation]<0.2:
            bayes_source_attribute=[max_correlation[0]]
            bayes_called_attribute=[max_correlation[1]]
            bayes_assist_attribute=[test_finegrain_functional_dependency(bayes_source_attribute[0],bayes_called_attribute[0],10,100)]
            print("Near functional dependency detected")
            print("Bayes_source_attribute is:")
            print(bayes_source_attribute)
            print("Bayes_called_attribute is:")
            print(bayes_called_attribute)
            print("Bayes_assist_attribute is:")
            print(bayes_assist_attribute)
            BayesNet=BayesnetEstimator(bayes_called_attribute[0],bayes_source_attribute[0],bayes_assist_attribute[0],150,8,10,1)
            BayesNet.train(data_used)
            bayes_source_attribute=[bayes_source_attribute[0].item()]
            bayes_called_attribute=[bayes_called_attribute[0].item()]
            bayes_assist_attribute=[bayes_assist_attribute[0].item()]
        else:
            bayes_source_attribute=[]
            bayes_called_attribute=[]
            bayes_assist_attribute=[]
    bayesarray=np.array([bayes_source_attribute,bayes_called_attribute,bayes_assist_attribute])
    np.save(dataset_name+'/'+dataset_name+'_bayesarray.npy',bayesarray)
