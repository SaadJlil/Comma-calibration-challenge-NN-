import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

# model + batch_get


def batch_get(batch_size, data_size, np_fp):
    file_list = [random.randint(0,4), random.randint(0,5)]
    data_point = torch.load("./footage_t/footage{}.{}.pt".format(file_list[0], file_list[1]))
    if file_list[1] == 0:
        limit = 201-batch_size
    elif file_list[1] == 5 and file_list[0] == 4:
        limit = 195-batch_size
    elif file_list[1] == 5:
        limit = 199-batch_size
    else:
        limit = 200-batch_size
    limitation = random.randint(0,limit-np_fp)
    for final_index in range(limitation, limitation+batch_size+np_fp):
        if final_index == limitation:
            data = data_point[final_index].unsqueeze(0)
        else:
            data = torch.cat((data, data_point[final_index].unsqueeze(0)), 0)

    data_point = torch.load("./footage_t/results.pt")
    if file_list[1] == 0:
        first = 0
    elif file_list[1] == 5:
        first = 200*3+199
    else:
        first = 199 + 200 * (file_list[1]-1)
    
    first = (file_list[0])*(201+199+200*4)+first+limitation


    data_point = data_point[first:first+batch_size]

    # D,H,W,C -> N,C,D,H,W


    data = data.permute(0,3,1,2)

    for index in range(np_fp,data.shape[0]):
        if index == np_fp:
            final_data = data[index-(np_fp-1):index+1].unsqueeze(0)
        else:
            final_data = torch.cat((final_data, data[index-(np_fp-1):index+1].unsqueeze(0)), 0)
    
    return final_data, data_point

class ResBlock(nn.Module):
    def __init__(self, initial_param, final_param):
        super().__init__()
        self.conv_maxs = nn.Conv3d(initial_param, final_param, kernel_size = (3, 3, 3), padding = "same")
        self.dropout = nn.Dropout(0.25) 
        self.batch_norm = nn.BatchNorm3d(final_param)
        
        
    def forward(self, x):
        x = self.conv_maxs(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.max_pool3d(x, (1,2,1))
        return x

#3 16 8 4 3 ***** 2 1
        
class model(nn.Module):
    def __init__(self, conv_list):
        super().__init__()
        conv_liste = []
        for i in range(len(conv_list)-1):
            conv_liste.append(ResBlock(conv_list[i], conv_list[i+1]))
        self.conv_enc = nn.Sequential(*conv_liste)
        self.linear_dec = nn.Sequential(nn.Linear(6480,100), nn.ReLU(), nn.Linear(100, 2), nn.ReLU())
    def forward(self, x):
        x = self.conv_enc(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_dec(x) 
        return x

def training_loop(n_epochs, optimizer, model, loss_fn, batch_size, video_np):
    for epoch in range(n_epochs):
        loss_train = 0.0
        b = 0
        for i in range(19):
            input, target = batch_get(batch_size, 5996, video_np)
            output = model(input)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            b+=1
        print(loss_train/b)

Net = model([5,16,8,4,3])
optimizer = optim.SGD(Net.parameters(), lr = 1e-2)

training_loop(1000, optimizer, Net, nn.MSELoss(), 65, 5)
