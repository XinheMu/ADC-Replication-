import torch 
torch.set_grad_enabled(False)
import sys
import ast
import torch.nn as nn
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
from sklearn.cluster import KMeans

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
    data_init_preprocessed=data_init-(reg_const_zero[1:2,:]-reg_const_zero[2:3,:])*(data_init>(reg_const_zero[2:3,:]))-(reg_const_zero[4:5,:]-reg_const_zero[5:6,:])*(data_init>(reg_const_zero[5:6,:]))
    minimum_radius=torch.zeros((dimension_init,64))
    for i in range(0,dimension_init):
        for j in range(0,64):
            lower_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(j-1/2)/64
            upper_bound=reg_const_one[1,i]+3*reg_const_one[2,i]*(j+3/2)/64
            checked_slice=data_init_preprocessed[:,i]
            checked_slice=checked_slice[(checked_slice>lower_bound)*(checked_slice<upper_bound)]
            minimum_radius[i,j]=3*reg_const_one[2,i]/(32*max(len(torch.unique(checked_slice)),4))
    np.save(dataset_name+'/'+'minrad_'+dataset_name+'.npy',minimum_radius.numpy())
    indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    minimum_radius_mcvs=torch.zeros((50,dimension_init,2))
    for i in range(dimension_init):
        unique_vals,counts=torch.unique(data_used[:,i],return_counts=True)
        k_actual=min(50,unique_vals.shape[0])
        topk_counts,topk_indices=torch.topk(counts,k_actual)
        topk_vals = unique_vals[topk_indices]
        topk_counts=topk_counts.float()
        for j in range(0,k_actual):
            min_val=topk_vals[j]-0.075
            max_val=topk_vals[j]+0.075
            size_j=torch.sum((data_used[:,i]<max_val)*(data_used[:,i]>min_val))
            topk_counts[j]=topk_counts[j]/size_j
        minimum_radius_mcvs[:k_actual,i,0]=topk_vals
        minimum_radius_mcvs[:k_actual,i,1]=topk_counts.float()
    np.save(dataset_name+'/'+'minrad_mcvs_'+dataset_name+'.npy',minimum_radius_mcvs.numpy())
    used_indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    data_used=data_used[:,used_indexes].to(torch.float32)
    print('end of normalization step')
def sample(draws):
    global size, data_used
    pos=torch.randperm(size)
    sample=data_used[pos[0:draws],:]
    return torch.transpose(sample,0,1)
def set_datapoint(samplesize):
    global size, samples, samples_num
    samplesize=min(samplesize,size)
    z=sample(samplesize)
    samples=torch.transpose(z,0,1)
    samples_num=samplesize
def likelihood_calc(kernels_matrix,stepsize=1000,return_likelihood_list=False):
    global samples, samples_num, dimension, kernels_num
    samples_start=0
    samples_end=min(samples_start+stepsize,samples_num)
    derivatives=torch.zeros((2*dimension+1,kernels_num))
    likelihood=0
    likelihood_list=torch.zeros(samples_num)
    while samples_start<samples_num:
        kernels_matrix=kernels_matrix.reshape((2*dimension+1,kernels_num,1))
        derivatives_matrix=torch.zeros((2*dimension+1,kernels_num,samples_end-samples_start))
        samples_used=torch.transpose(samples[samples_start:samples_end,:],0,1)
        calc_matrix=torch.zeros((dimension,1,samples_end-samples_start))
        calc_matrix[:,0,:]=samples_used[:,:]
        calc_matrix=calc_matrix-kernels_matrix[:dimension,:,0:1]
        derivatives_matrix[:dimension,:,:]=calc_matrix/kernels_matrix[dimension:2*dimension,:,:]**2
        derivatives_matrix[dimension:2*dimension,:,:]=calc_matrix**2/kernels_matrix[dimension:2*dimension,:,:]**3-1/kernels_matrix[dimension:2*dimension,:,:]
        '''要不要1.5*?'''
        derivatives_matrix[2*dimension,:,:]=1.0
        calc_matrix=(1/kernels_matrix[dimension:2*dimension,:])*torch.exp(-0.5*(calc_matrix/kernels_matrix[dimension:2*dimension,:])**2)
        calc_matrix=torch.prod(calc_matrix,0)
        derivatives_matrix=derivatives_matrix*calc_matrix.unsqueeze(0)
        derivatives_matrix[:2*dimension,:,:]=derivatives_matrix[:2*dimension,:,:]*kernels_matrix[2*dimension:,:,:]
        calc_matrix=torch.sum(calc_matrix*kernels_matrix[2*dimension,:],0)+1e-10
        derivatives_matrix=derivatives_matrix/((calc_matrix.unsqueeze(0)).unsqueeze(0))
        calc_matrix=torch.log(calc_matrix)
        derivatives=derivatives+torch.sum(derivatives_matrix,2)
        likelihood=likelihood+torch.sum(calc_matrix)
        likelihood_list[samples_start:samples_end]=calc_matrix
        samples_start=samples_end
        samples_end=min(samples_start+stepsize,samples_num)
    derivatives[2*dimension:,:]=derivatives[2*dimension:,:]-samples_num/torch.sum(kernels_matrix[2*dimension,:])
    total_weight=torch.sum(kernels_matrix[2*dimension,:])
    likelihood=likelihood/samples_num-torch.log(total_weight)
    likelihood_list=likelihood_list-torch.log(total_weight)
    if return_likelihood_list:
        return likelihood,derivatives,likelihood_list
    return likelihood, derivatives
def resample(likelihood_list,samples_num):
    global kernels_matrix,dimension,kernels_num,full_kernels_num
    sequence=torch.arange(full_kernels_num)
    weight_too_small=(kernels_matrix[2*dimension,:]<1e-3)*(sequence<kernels_num)
    z=torch.nonzero(weight_too_small)
    w=(1/torch.exp(likelihood_list))/torch.sum(1/torch.exp(likelihood_list))
    new_coord=np.random.choice(min(samples_num,size),np.count_nonzero(weight_too_small),False,(1/torch.exp(likelihood_list).to(torch.float64))/torch.sum(1/torch.exp(likelihood_list).to(torch.float64)))
    kernels_matrix[:dimension,weight_too_small]=torch.transpose(samples[new_coord,:],0,1).to(torch.float32)
    kernels_matrix[dimension:2*dimension,weight_too_small]=0.25
    kernels_matrix[2*dimension:,weight_too_small]=0.002
    if torch.sum(weight_too_small*1)>0:
        clusters_mean, clusters_std=calculate_cluster_stats(samples,torch.transpose(kernels_matrix[:dimension,:],0,1),torch.nonzero(weight_too_small).squeeze())
        kernels_matrix[:dimension,weight_too_small]=torch.transpose(clusters_mean,0,1)
        kernels_matrix[dimension:2*dimension,weight_too_small]=torch.clamp(torch.transpose(clusters_std,0,1),min=1/40)
    kernels_matrix[dimension:2*dimension,:]=torch.clamp(kernels_matrix[dimension:2*dimension,:],min=1/400)
    print("Adding new kernels to low density regions and resampling kernels with weights too low. Kernels to be added or resampled are:")
    print(torch.nonzero(weight_too_small))
    return weight_too_small
def calculate_cluster_stats(sample_points,kernel_points,subset_indices):
    samples_sq = torch.sum(sample_points**2, dim=1, keepdim=True)
    kernels_sq = torch.sum(kernel_points**2, dim=1)                
    dot_product = torch.matmul(sample_points, kernel_points.T)
    dist_sq = samples_sq - 2 * dot_product + kernels_sq
    assignments = torch.argmin(dist_sq, dim=1)
    means_list = []
    stds_list = []
    d = sample_points.shape[1]
    for kernel_idx in subset_indices.flatten():
        mask = (assignments == kernel_idx)
        if torch.any(mask):
            points_in_cluster = sample_points[mask]
            mean = torch.mean(points_in_cluster, dim=0)
            std = torch.sqrt(torch.mean((points_in_cluster-mean)**2,dim=0))
            means_list.append(mean)
            stds_list.append(std)
        else:
            means_list.append(kernel_points[kernel_idx,:])
            stds_list.append(torch.ones((dimension))*0.15)            
    if not means_list: # Handle case where subset_indices is empty
        return torch.empty(0, d), torch.empty(0, d)
    return torch.stack(means_list), torch.stack(stds_list)
def call_back(xk):
    iteration_count+=1
    if iteration_count%10==0:
        print(f"Iteration {iteration_count}:Function value={likelihood_calc(xk)}")
def em_step_gmm(kernel_matrix, stepsize=40000, epsilon=1e-9):
    d = dimension
    N = kernels_num
    size=samples_num
    means = kernel_matrix[0:d,:kernels_num]  # Shape: (d, N)
    stds = kernel_matrix[d:2*d,:kernels_num]  # Shape: (d, N)
    weights = kernel_matrix[2*d:2*d+1,:kernels_num]  # Shape: (1, N)
    samples_used = torch.transpose(samples,0,1)
    points_start=0
    points_end=stepsize
    responsibilities = torch.zeros((size, N))
    while points_start<size:
        resp_calc=samples_used[:,points_start:points_end].unsqueeze(1)-means.unsqueeze(2)
        resp_calc=resp_calc**2/(2*stds.unsqueeze(2)**2)
        resp_calc=torch.exp(-1*torch.sum(resp_calc,0))
        resp_calc=resp_calc/(torch.prod(stds,0)*(2*3.14159265)**(d/2)).unsqueeze(1)
        responsibilities[points_start:points_end,:]=torch.transpose(resp_calc,0,1)*weights
        points_start=points_start+stepsize
        points_end=min(points_end+stepsize,size)
    row_sums =torch.sum(responsibilities,axis=1,keepdims=True)
    responsibilities = responsibilities / (row_sums + epsilon)
    new_kernel_matrix = torch.zeros_like(kernel_matrix[:,:kernels_num])
    for i in range(N):
        resp_sum =torch.sum(responsibilities[:, i]) + epsilon
        new_kernel_matrix[2*d:2*d+1, i] = resp_sum / size
        for dim in range(d):
            new_kernel_matrix[dim, i] = torch.sum(responsibilities[:, i]*samples_used[dim, :]) / resp_sum
        for dim in range(d):
            diff_squared = (samples_used[dim, :] - new_kernel_matrix[dim, i])**2
            new_kernel_matrix[dim+d, i] = torch.sqrt(torch.sum(responsibilities[:, i] * diff_squared) / resp_sum)
    new_kernel_matrix[d:2*d,:]=torch.clamp(new_kernel_matrix[d:2*d,:],min=1/400)
    new_kernel_matrix[2*d:2*d+1, :] = new_kernel_matrix[2*d:2*d+1, :] / torch.mean(new_kernel_matrix[2*d:2*d+1, :])
    return new_kernel_matrix

def sgd_step_gmm(kernels_matrix):
    global kernels_num
    likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
    derivatives[:dimension,:kernels_num]=(derivatives[:dimension,:kernels_num]*learnrate[0]*kernels_matrix[dimension:2*dimension,:kernels_num]**3)/kernels_matrix[2*dimension:,:kernels_num]
    derivatives[dimension:2*dimension,:kernels_num]=(derivatives[dimension:2*dimension,:kernels_num]*learnrate[1]*kernels_matrix[dimension:2*dimension,:kernels_num]**2)/kernels_matrix[2*dimension:,:kernels_num]
    derivatives[2*dimension:,:kernels_num]=derivatives[2*dimension:,:kernels_num]*learnrate[2]*torch.sqrt(kernels_matrix[2*dimension:,:kernels_num])
    derivatives=derivatives*(0.2+torch.rand(2*dimension+1,kernels_num)*0.8)
    derivatives[dimension:2*dimension,:kernels_num]=torch.maximum(torch.minimum(derivatives[dimension:2*dimension,:kernels_num],torch.ones((dimension,kernels_num))*0.05),kernels_matrix[dimension:2*dimension,:kernels_num]*(-0.2))
    derivatives[2*dimension,:]=torch.minimum(derivatives[2*dimension,:],torch.ones((kernels_num))*0.1)
    kernels_matrix[:,:kernels_num]=kernels_matrix[:,:kernels_num]+derivatives
    kernels_matrix[dimension:2*dimension,:kernels_num]=torch.clamp(kernels_matrix[dimension:2*dimension,:kernels_num],min=1/400,max=0.2)
    kernels_matrix[2*dimension:,:kernels_num]=torch.maximum(kernels_matrix[2*dimension:,:kernels_num],torch.ones((1,kernels_num))/10000)
    kernels_matrix[2*dimension:,:kernels_num]=kernels_matrix[2*dimension:,:kernels_num]/torch.mean(kernels_matrix[2*dimension:,:kernels_num])
    print(likelihood)
    return kernels_matrix 

def train_KDE_model(dim,full_kernum,learnrate,resample_and_sgd=True,loadmatrix=False,matrixloaded=0):
    global samples_num, samples, dimension, kernels_num, kernels_matrix
    dimension=dim
    kernels_num=full_kernum-128
    samples_num=0
    kernels_matrix=torch.zeros((2*dimension+1,full_kernum))
    set_datapoint(full_kernum)
    kernels_matrix[:dimension,:]=torch.transpose(samples,0,1)
    kernels_matrix[dimension:2*dimension,:]=torch.ones((dimension,full_kernum))*(0.1+torch.rand((dimension,full_kernum))*0.25)
    kernels_matrix[2*dimension,:kernels_num]=1
    if loadmatrix:
        kernels_matrix=torch.tensor(matrixloaded)
        kernels_num=full_kernum
    else:
        print('Initializing kernel location via KMeans')
        kmeans=KMeans(n_clusters=kernels_num,n_init=5,random_state=0)
        set_datapoint(4000000)
        kmeans.fit(samples)
        print('Initialization complete')
        for k in range(0,kernels_num):
            cluster_points_mask=(kmeans.labels_== k)
            cluster_points=data_used[cluster_points_mask,:]        
            if len(cluster_points) == 0:
                kernels_matrix[2*dimension,k]=1e-6 
                kernels_matrix[dimension:2*dimension,k]=torch.ones((dimension))
                kernels_matrix[:dimension,k]=torch.tensor(kmeans.cluster_centers_[k]) 
                continue
            kernels_matrix[2*dimension,k]=len(cluster_points)/size
            kernels_matrix[:dimension,k]=torch.tensor(kmeans.cluster_centers_[k])
            kernels_matrix[dimension:2*dimension,k]=torch.sqrt(torch.mean((cluster_points-kernels_matrix[:dimension,k])**2,0))
            kernels_matrix[dimension:2*dimension,:]=torch.clamp(kernels_matrix[dimension:2*dimension,:],min=(1/40))
        kernels_matrix[2*dimension:,:kernels_num]=kernels_matrix[2*dimension:,:kernels_num]/torch.mean(kernels_matrix[2*dimension:,:kernels_num])
    set_datapoint(test_sample_num)
    print("starting random exploration via SDE updates, outputting likelihood after every 10 iterations")
    for i in range(1,151):
        kernels_matrix[:,:kernels_num]=sgd_step_gmm(kernels_matrix[:,:kernels_num])
        if i%10==0:
            likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
            print(i)
            print(likelihood)
            if i%10==0:
                np.save(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy',kernels_matrix.numpy())
                kernels_num=min(full_kernum,kernels_num+16)
                weight_too_small=resample(likelihood_list,test_sample_num)
            if i%20==0:
                set_datapoint(test_sample_num)
    print("starting fcused EM updates, outputting likelihood after every 10 iterations")
    set_datapoint(test_sample_num)
    for i in range(151,241):
        kernels_matrix[:,:kernels_num]=em_step_gmm(kernels_matrix[:,:kernels_num])
        if i%10==0:
            likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
            print(i)
            print(likelihood)
            if i%10==0:
                np.save(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy',kernels_matrix.numpy())
            if i%20==0:
               set_datapoint(test_sample_num)
    set_datapoint(8*test_sample_num)
    print("finetuning model on larger trainset using EM algorithm")
    for i in range(241,251):
        kernels_matrix[:,:kernels_num]=em_step_gmm(kernels_matrix[:,:kernels_num])
        if i%5==0:
            likelihood, derivatives, likelihood_list=likelihood_calc(kernels_matrix[:,:kernels_num],calc_step_size,True)
            print(i)
            print(likelihood)
            if i%10==0:
                np.save(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy',kernels_matrix.numpy())
if __name__ == "__main__":
    params=['CardEst','power','7','-100000000']
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    dataset_name=params[1]
    dimension_init=int(params[2])
    nan_to=float(params[3])
bayes_called_attributes=(np.load(dataset_name+'/'+dataset_name+'_bayesarray.npy')[1,:])
print(bayes_called_attributes)
load=False
test_sample_num=500000
calc_step_size=8000
dimension=dimension_init-len(bayes_called_attributes)
full_kernels_num=1280
regu_step1()
learnrate=torch.tensor([600,600,300])/min(test_sample_num,size)
if load:
    train_KDE_model(dimension,full_kernels_num,learnrate,True,True,np.load(dataset_name+'/'+'KDE_params_adjusted_'+dataset_name+'.npy'))
else:
    train_KDE_model(dimension,full_kernels_num,learnrate,True)

