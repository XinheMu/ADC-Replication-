import torch
import numpy as np
import sys

def merge():
    xtraina=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'xtraina.npy'))
    ytraina=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'ytraina.npy'))
    xtrainb=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'xtrainb.npy'))
    ytrainb=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'ytrainb.npy'))
    xtrainc=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'xtrainc.npy'))
    ytrainc=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'ytrainc.npy'))
    xtraind=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'xtraind.npy'))
    ytraind=torch.tensor(np.load(dataset_name+'training/'+dataset_name+'ytraind.npy'))
    xtrainone=torch.stack((xtraina,xtrainb),dim=2)
    xtraintwo=torch.stack((xtrainc,xtraind),dim=2)
    ytrainone=torch.stack((ytraina,ytrainb),dim=2)
    ytraintwo=torch.stack((ytrainc,ytraind),dim=2)
    xtrainone=xtrainone.reshape((xtrainone.shape)[0],-1)
    xtraintwo=xtraintwo.reshape((xtraintwo.shape)[0],-1)
    ytrainone=ytrainone.reshape((ytrainone.shape)[0],-1)
    ytraintwo=ytraintwo.reshape((ytraintwo.shape)[0],-1)
    np.save(dataset_name+'training/'+dataset_name+'xtrainone.npy',xtrainone.numpy())
    np.save(dataset_name+'training/'+dataset_name+'xtraintwo.npy',xtraintwo.numpy())
    np.save(dataset_name+'training/'+dataset_name+'ytrainone.npy',ytrainone.numpy())
    np.save(dataset_name+'training/'+dataset_name+'ytraintwo.npy',ytraintwo.numpy())
if __name__ == "__main__":
    dataset_name=sys.argv[1]
    merge()

