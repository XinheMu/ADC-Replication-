import torch 
import torch.nn as nn
import matplotlib as mtp
import pandas as pd
import numpy as np
import scipy as sp
import math as ma
import time
import sys
import ast
from fractions import Fraction

def set_global(Intervals,TimeAllowed_min,batch_size,dim,cutoff):
    global length, BatchSize, TimeMax, TimeChart, SignalDecay, NoiseVar, NoiseVar_inv, NoiseVarSquare, NoiseVarSquare_inv, Weight, Weight_sqrt, PointArray, dimension, PointArrayScore, cutoff_length, cutoff_time
    cutoff_length=cutoff
    cutoff_time=Intervals[cutoff]
    TimeMax=Intervals[len(Intervals)-1]
    dimension=dim
    BatchSize=batch_size
    length=len(Intervals)-1
    TimeChart=np.zeros((1,(length)*batch_size))
    TimeChart[0,:]=np.repeat([(Intervals[0]+Intervals[1])/2]+[(Intervals[i]+Intervals[i+1])/2 for i in range(1,length)],batch_size)
    TimeChart=TimeChart+TimeAllowed_min
    SignalDecay=np.exp(-1*TimeChart)
    NoiseVarSquare=(1-SignalDecay**2)
    NoiseVar=np.sqrt(NoiseVarSquare)
    Weight=np.zeros((1,(length)*batch_size))
    Weight[0,:]=np.repeat([(Intervals[i+1]-Intervals[i]) for i in range(0,length)],batch_size)
    Weight=torch.tensor(Weight)
    Weight_sqrt=torch.sqrt(Weight)
    SobolSampler=sp.stats.qmc.Sobol(dim,scramble=False)
    HaltonSampler=sp.stats.qmc.Halton(dim,scramble=True,seed=304516)
    UniSobol=np.transpose(SobolSampler.random(2**ma.ceil(ma.log2(batch_size*2))))
    HaltonShift=np.transpose(HaltonSampler.random(ma.ceil((length)/2)))
    UniSobol=torch.tensor(UniSobol)
    SobolBackup=UniSobol
    UniSobol=UniSobol+HaltonShift[:,0:1]
    for i in range(1,ma.ceil((length)/2)):
        UniSobol=torch.cat((UniSobol,SobolBackup+HaltonShift[:,i:i+1]),1)
    UniSobol=(UniSobol-1*(UniSobol>1)).numpy()
    NormalSobol=sp.stats.norm.ppf(UniSobol*0.99999+0.000005)
    PointArray=torch.tensor(NormalSobol[:,0:(length)*batch_size]*NoiseVar)
    PointArrayScore=-1*PointArray/NoiseVarSquare
    TimeChart=torch.tensor(TimeChart)
    SignalDecay=torch.tensor(SignalDecay)
    NoiseVar=torch.tensor(NoiseVar)
    NoiseVarSquare=torch.tensor(NoiseVarSquare)
    NoiseVar_inv=1/NoiseVar
    NoiseVarSquare_inv=1/NoiseVarSquare
    TimeChart=TimeChart.to(torch.float32)
    SignalDecay=SignalDecay.to(torch.float32)
    NoiseVar=NoiseVar.to(torch.float32)
    NoiseVarSquare=NoiseVarSquare.to(torch.float32)
    Weight=Weight.to(torch.float32)
    Weight_sqrt=Weight_sqrt.to(torch.float32)


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
    used_indexes=[(i not in bayes_called_attributes) for i in range(0,dimension_init)]
    data_used=data_used[:,used_indexes].to(torch.float32)
      
def sample(draws):
    global size, data_used
    pos=torch.randperm(size)
    sample=data_used[pos[0:draws],:]
    samplelength=size
    while samplelength<draws:
        sample=torch.cat((sample,data_used[pos[0:draws],:]),0)
        samplelength=samplelength+size
    return np.transpose(sample[0:draws,:])

def set_datapoint(samplesize):
    global size, samples
    samplesize=min(samplesize,size)
    z=sample(samplesize)
    samples=np.transpose(z)

def score_cauc(positions,samplesize,stepsize):
    global size, samples
    if (size!=samplesize[0][0]):
        set_datapoint(samplesize[0][0])
    grad_val=torch.zeros_like(positions[0:dimension,:])
    val=torch.zeros_like(positions[0:1,:])
    TimeDecay=torch.exp(-1*positions[dimension:dimension+1,:])
    processed_points=0
    j=0
    i=len(samplesize[0])-1
    samplesize[0][0]=min(size,samplesize[0][0])
    while i>=0:
        bufferpool_val=torch.zeros((1,samplesize[1][i],80))
        bufferpool_grad_val=torch.zeros((dimension,samplesize[1][i],80))
        samplesize[0][i]=min(size,samplesize[0][i])
        while processed_points<samplesize[0][i]:
            new_grad_val, new_val=batch_process(positions[:,:samplesize[1][i]],processed_points,min(processed_points+stepsize[i],samplesize[0][0]),TimeDecay)
            bufferpool_val[:,:,j%80]=new_val
            bufferpool_grad_val[:,:,j%80]=new_grad_val
            processed_points=processed_points+stepsize[i]
            j=j+1
            if j%80==0:
                print(processed_points)
                val[:,:samplesize[1][i]]=val[:,:samplesize[1][i]]+torch.sum(bufferpool_val,2)
                grad_val[:,:samplesize[1][i]]=grad_val[:,:samplesize[1][i]]+torch.sum(bufferpool_grad_val,2)
                bufferpool_val=torch.zeros((1,samplesize[1][i],80))
                bufferpool_grad_val=torch.zeros((dimension,samplesize[1][i],80))
        val[:,:samplesize[1][i]]=val[:,:samplesize[1][i]]+torch.sum(bufferpool_val,2)
        grad_val[:,:samplesize[1][i]]=grad_val[:,:samplesize[1][i]]+torch.sum(bufferpool_grad_val,2)
        i=i-1
        TimeDecay=TimeDecay[:,:samplesize[1][i]]
    return grad_val/(val+1e-7)


def batch_process(positions,startpoint,endpoint,TimeDecay):
    with torch.no_grad():
        samples_decay=(samples[startpoint:endpoint,:].unsqueeze(1))*(TimeDecay.unsqueeze(2))
        samples_decay=(samples_decay.transpose(1,2)).transpose(0,1)
        positions_used=positions[:dimension,:].unsqueeze(1)
        val=torch.zeros([1,endpoint-startpoint,(positions.shape)[1]])
        grad_val=samples_decay-positions_used
        val[0,:,:]=torch.sum(grad_val**2,0)
        grad_val=grad_val*(positions[dimension+2:dimension+3,:].unsqueeze(0))
        val[0,:,:]=val[0,:,:]*(-0.5)*positions[dimension+2:dimension+3,:]
        mask=(val>-20)
        val=mask*val
        val=torch.exp(val)*mask
        grad_val=grad_val*val
        return torch.sum(grad_val,1), torch.sum(val,1)

def decay_function(x):
    return 1/(torch.exp(x)-torch.exp(-x))

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
        y=self.structure(torch.transpose(torch.cat((x[0:dimension],torch.sqrt(x[dimension:dimension+1])),0),0,1))
        return torch.transpose(y,0,1)*decay_function(x[dimension:dimension+1])

def cycle_tail(batch_size,samplesize,explore,Time_min=1/320):
    global network, dimension
    set_global(timestep,Time_min,batch_size,dimension,43)
    explore=cutoff_length+explore
    xtrainone=torch.cat((sample(length*BatchSize)*SignalDecay+torch.normal(0,1,(dimension,length*BatchSize))*NoiseVar,TimeChart,NoiseVar_inv,NoiseVarSquare_inv),0)[:,cutoff_length*batch_size:explore*batch_size]
    xtrainone=xtrainone.to(torch.float32)
    ytrainone=score_cauc(xtrainone,samplesize,[400,240,160,80])+xtrainone[:dimension,]*NoiseVarSquare_inv[:,cutoff_length*batch_size:explore*batch_size].to(torch.float32)
    xtraintwo=torch.cat((sample(length*BatchSize)*SignalDecay+torch.normal(0,1,(dimension,length*BatchSize))*NoiseVar,TimeChart,NoiseVar_inv,NoiseVarSquare_inv),0)[:,cutoff_length*batch_size:explore*batch_size]
    xtraintwo=xtraintwo.to(torch.float32)
    ytraintwo=score_cauc(xtraintwo,samplesize,[400,240,160,80])+xtraintwo[:dimension,]*NoiseVarSquare_inv[:,cutoff_length*batch_size:explore*batch_size].to(torch.float32)
    loss=nn.MSELoss()
    optimizer=torch.optim.Adam(network.parameters(),lr=1e-5)
    for i in range(0,150*batch_size):
        if i==0:
            ypredtwo=network(xtraintwo)
            losstwo=loss(ytraintwo*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),ypredtwo*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
        ypredone=network(xtrainone)
        lossone=loss(ytrainone*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),ypredone*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
        optimizer.zero_grad()
        lossone.backward()
        optimizer.step()
        if i%100==0:
            print(i,lossone)
        if i>(25*batch_size) and i%(25*batch_size)==0:
            losstwo_backup=losstwo
            ypredtwo=network(xtraintwo)
            losstwo=loss(ytraintwo*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),ypredtwo*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
            if losstwo_backup<losstwo:
                break
            torch.save(network,dataset_name+'/'+dataset_name+'_tail.pkl')
    for i in range(0,150*batch_size):
        if i==0:
            ypredone=network(xtrainone)
            lossone=loss(ytrainone*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),ypredone*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
        ypredtwo=network(xtraintwo)
        losstwo=loss(ytraintwo*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),ypredtwo*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
        optimizer.zero_grad()
        losstwo.backward()
        optimizer.step()
        if i%100==0:
            print(i,losstwo)
        if i>(25*batch_size) and i%(25*batch_size)==0:
            lossone_backup=lossone
            ypredone=network(xtrainone)
            lossone=loss(ytrainone*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),ypredone*Weight_sqrt[:,cutoff_length*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
            if lossone_backup<lossone:
                break
            torch.save(network,dataset_name+'/'+dataset_name+'_tail.pkl')

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
        y=self.structure(torch.transpose(torch.cat((x[0:dimension]*torch.exp(x[dimension:dimension+1]),(x[dimension+1:dimension+2])*(0.1)),0),0,1))
        return (torch.transpose(y[:,0:dimension],0,1))*x[dimension+2:dimension+3]*0.04+(torch.transpose(y[:,dimension:2*dimension],0,1))*x[dimension+1:dimension+2]*0.2+(torch.transpose(y[:,2*dimension:3*dimension],0,1))

def cycle_head(batch_size,samplesize,stepsize,explore,gen_new_trainset=True,Time_min=1/320,learnrate=1e-5,begin=0):
    global network, dimension
    set_global(timestep,Time_min,batch_size,dimension,43)
    if gen_new_trainset:
        xtrainone=torch.cat((sample(length*BatchSize)*SignalDecay+torch.normal(0,1,(dimension,length*BatchSize))*NoiseVar,TimeChart,NoiseVar_inv,NoiseVarSquare_inv),0)[:,:cutoff_length*BatchSize]
        xtrainone=xtrainone.to(torch.float32)
        ytrainone=score_cauc(xtrainone,samplesize,stepsize)
        np.save(dataset_name+'training/'+dataset_name+'ytrainone.npy',ytrainone.numpy())
        np.save(dataset_name+'training/'+dataset_name+'xtrainone.npy',xtrainone.numpy())
        xtraintwo=torch.cat((sample(length*BatchSize)*SignalDecay+torch.normal(0,1,(dimension,length*BatchSize))*NoiseVar,TimeChart,NoiseVar_inv,NoiseVarSquare_inv),0)[:,:cutoff_length*BatchSize]
        xtraintwo=xtraintwo.to(torch.float32)
        ytraintwo=score_cauc(xtraintwo,samplesize,stepsize)
        np.save(dataset_name+'training/'+dataset_name+'ytraintwo.npy',ytraintwo.numpy())
        np.save(dataset_name+'training/'+dataset_name+'xtraintwo.npy',xtraintwo.numpy())
    xtrainone=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'xtrainone.npy'))
    ytrainone=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'ytrainone.npy'))
    xtraintwo=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'xtraintwo.npy')).to(torch.float32)
    ytraintwo=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'ytraintwo.npy')).to(torch.float32)
    smallcycle(xtrainone[:,:batch_size*explore],ytrainone[:,:batch_size*explore],xtraintwo[:,:batch_size*explore],ytraintwo[:,:batch_size*explore],batch_size,explore,learnrate,begin)
    if explore>40 or time.time()-timestart<24000:
        smallcycle(xtraintwo[:,:batch_size*explore],ytraintwo[:,:batch_size*explore],xtrainone[:,:batch_size*explore],ytrainone[:,:batch_size*explore],batch_size,explore,learnrate,begin)

def smallcycle(trainx,trainy,testx,testy,batch_size,explore,learnrate,begin=0):
    global network, activated_nodes, max_nodes, activation_array
    masks=[0 for i in range(0,32)]
    for i in range(0,32):
        masks[i]=[j*32+i for j in range(0,ma.floor(explore*batch_size/32))]
    slices_trainx=[trainx[:,masks[i]] for i in range(0,32)]
    slices_trainy=[trainy[:,masks[i]] for i in range(0,32)]
    slices_weightsqrt=[Weight_sqrt[:,masks[i]] for i in range(0,32)]
    unit=128
    loss=nn.MSELoss()
    optimizer=torch.optim.Adam(network.parameters(),lr=learnrate)
    randints=np.random.randint(0,32,1000*unit)
    for i in range(0,1000*unit):
        if i==0:
            predtesty=network(testx)
            losstest=loss(testy[:,begin*batch_size:explore*batch_size]*Weight_sqrt[:,begin*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),predtesty[:,begin*batch_size:explore*batch_size]*Weight_sqrt[:,begin*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
        predtrainy=network(slices_trainx[randints[i]])
        losstrain=loss(slices_trainy[randints[i]]*slices_weightsqrt[randints[i]]*100*ma.sqrt(length/TimeMax),predtrainy*slices_weightsqrt[randints[i]]*100*ma.sqrt(length/TimeMax))
        optimizer.zero_grad()
        losstrain.backward()
        optimizer.step()
        if i%1000==0:
            print(i,losstrain,randints[i])
        if i>(250*unit) and i%(250*unit)==0:
            losstest_backup=losstest
            predtesty=network(testx)
            losstest=loss(testy[:,begin*batch_size:explore*batch_size]*Weight_sqrt[:,begin*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax),predtesty[:,begin*batch_size:explore*batch_size]*Weight_sqrt[:,begin*batch_size:explore*batch_size]*100*ma.sqrt(length/TimeMax))
            torch.save(network,dataset_name+'/'+dataset_name+'_head.pkl')        
            if losstest_backup<losstest or (explore<40 and time.time()-timestart>24000):
                break

def run_head(dname, uvar, dim, Tmin, gen_new_train_dataset, begin, load, struct1=0):
    global dataset_name, unit_of_variables, dimension, network
    dataset_name=dname
    unit_of_variables=torch.tensor(uvar)
    dimension=dim
    regu_step1()
    if load:
        network=torch.load(dataset_name+'/'+dataset_name+'_head.pkl')
    else:
        network=net_head(struct1)
    stepsize=[480,240,120,60,40]
    cycle_head(batch_size,[[12000000,4000000,2000000,1000000,700000],[5*batch_size,15*batch_size,25*batch_size,35*batch_size,43*batch_size]],stepsize,43,gen_new_train_dataset,Tmin,4e-5,begin)
    cycle_head(batch_size,[[12000000,4000000,2000000,1000000,700000],[5*batch_size,15*batch_size,25*batch_size,35*batch_size,43*batch_size]],stepsize,25,False,Tmin,1e-5,begin)
    cycle_head(batch_size,[[12000000,4000000,2000000,1000000,700000],[5*batch_size,15*batch_size,25*batch_size,35*batch_size,43*batch_size]],stepsize,43,False,Tmin,1e-5,begin)
    cycle_head(batch_size,[[12000000,4000000,2000000,1000000,700000],[5*batch_size,15*batch_size,25*batch_size,35*batch_size,43*batch_size]],stepsize,43,False,Tmin,1e-5,begin)

def run_tail(dname, uvar, dim, load, struct=0):
    global dataset_name, unit_of_variables, dimension, network
    dataset_name=dname
    unit_of_variables=torch.tensor(uvar)
    dimension=dim
    regu_step1()
    if load:
        network=torch.load(dataset_name+'/'+dataset_name+'_tail.pkl')
    else:
        network=net_tail(struct)
    cycle_tail(batch_size,[[1000000],[30*batch_size]],30)
    cycle_tail(batch_size,[[1000000,330000,100000,33000],[30*batch_size,35*batch_size,40*batch_size,46*batch_size]],46)

def gen_head_trainset(dname, uvar, dim, Tmin, batch_size, dataset_num):
    global dataset_name, unit_of_variables, dimension, Time_min
    Time_min=Tmin
    dataset_name=dname
    unit_of_variables=torch.tensor(uvar)
    dimension=dim
    regu_step1()
    samplesize=[[12000000,4000000,2000000,1000000,700000],[5*batch_size,15*batch_size,25*batch_size,35*batch_size,43*batch_size]]
    stepsize=[480,240,120,60,40]
    set_global(timestep,Time_min,batch_size,dimension,43)
    xtrain=torch.cat((sample(length*BatchSize)*SignalDecay+torch.normal(0,1,(dimension,length*BatchSize))*NoiseVar,TimeChart,NoiseVar_inv,NoiseVarSquare_inv),0)[:,:cutoff_length*BatchSize]
    xtrain=xtrain.float()
    ytrain=score_cauc(xtrain,samplesize,stepsize)
    if dataset_num==1:
        np.save(dname+'training/'+dname+'ytraina.npy',ytrain.numpy())
        np.save(dname+'training/'+dname+'xtraina.npy',xtrain.numpy())
    if dataset_num==2:
        np.save(dname+'training/'+dname+'ytrainb.npy',ytrain.numpy())
        np.save(dname+'training/'+dname+'xtrainb.npy',xtrain.numpy())
    if dataset_num==3:
        np.save(dname+'training/'+dname+'ytrainc.npy',ytrain.numpy())
        np.save(dname+'training/'+dname+'xtrainc.npy',xtrain.numpy())
    if dataset_num==4:
        np.save(dname+'training/'+dname+'ytraind.npy',ytrain.numpy())
        np.save(dname+'training/'+dname+'xtraind.npy',xtrain.numpy())


if __name__ == "__main__":
    params=['CardEst','False','1','power','7','1/320','2048','False','False','True','1',-1000000000]
    for i in range(1,len(sys.argv)):
        params[i]=sys.argv[i]
    gen_trainset_only=(params[1]=='True')
    dataset_num=int(params[2])
    dataset=params[3]
    dimension_init=int(params[4])
    Time_min=float(Fraction(params[5]))
    bsize=int(params[6])
    gen_trainset_in_training_run=(params[7]=='True')
    train_tail_too=(params[8]=='True')
    load_network=(params[9]=='True')
    added_layers=int(params[10])
    nan_to=float(params[11])
    bayes_called_attributes=(np.load(dataset+'/'+dataset+'_bayesarray.npy')[1,:])
    dimension=dimension_init-len(bayes_called_attributes)
    struct_head_1=[dimension+1,60,100]+[150 for i in range(0,added_layers)]+[100,75,3*dimension]
    struct_tail_1=[dimension+1,35,60,60,35,dimension]
    uvar=[1]
    bgin=0
    timestep=(np.load('timestep.npy').tolist())
    timestart=time.time()
    if gen_trainset_only:
        batch_size=round(bsize/2)
        gen_head_trainset(dataset,[uvar],dimension,Time_min,batch_size,dataset_num)
    else:
        if train_tail_too:
            batch_size=1024
            run_tail(dataset,[uvar],dimension, load_network,struct_tail_1)
        batch_size=bsize
        run_head(dataset,[uvar],dimension,Time_min,gen_trainset_in_training_run,bgin,load_network,struct_head_1)

