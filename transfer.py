import torch 
import numpy as np
from glob import glob
from matplotlib import pyplot as plt


#convert numpy array to torch tensors


for file in glob("*.npy"):
    if file == "results.npy":
        continue
        torch.save(torch.from_numpy(np.load(file).astype(float)).float(), "../footage_t/{}pt".format(file[:-3]))
    else:
        data = np.load(file)
        data1 = data[:,:,605:610,:]
        data1 = np.concatenate((data1,np.zeros((data1.shape[0],1164-874, data1.shape[2], data1.shape[3]))), 1)
        data = np.transpose(data, (0,2,1,3))[:,:,317:322,:]
        zbl = np.concatenate((data1, data), 2)
        torch.save(torch.from_numpy(zbl).float(), "../footage_t/{}pt".format(file[:-3]))




