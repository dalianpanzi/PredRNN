

from PredRNN_Model import PredRNN
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

"""
input=torch.rand(1,1,1,100,100).cuda()   # Batch_size , time_step, channels, hight/width, width/hight
target=torch.rand(1,1,1,100,100).cuda()   # Batch_size , time_step, channels, hight/width, width/hight
"""


zx=np.load(file=r'G:\cqz\pytorch\data\trans\x3tr.npy')
zy=np.load(file=r'G:\cqz\pytorch\data\trans\ytr.npy')

zx=zx.transpose(0,1,4,2,3)
zy=zy.transpose(0,1,4,2,3)
zx=zx.astype(np.float32)
zy=zy.astype(np.float32)
zx=torch.from_numpy(zx)
zy=torch.from_numpy(zy)

from torch.utils.data import TensorDataset, DataLoader
see=TensorDataset(zx,zy)
sload=DataLoader(see,batch_size=32,drop_last=False)

class PredRNN_enc(nn.Module):
    def __init__(self):
        super(PredRNN_enc, self).__init__()
        self.pred1_enc=PredRNN(input_size=(61, 41),
                input_dim=3,
                hidden_dim=[  64, 8, 1],  #128,64,8
                hidden_dim_m=[ 8, 8, 8],   #必须前后一样
                kernel_size=[(3,3), (3, 3),(3, 3) ],
                num_layers=3,
                batch_first=True,
                bias=True).cuda()
    def forward(self,enc_input):
        layer_h_c, all_time_h_m, dec_input,_ = self.pred1_enc(enc_input)
        return layer_h_c, all_time_h_m, dec_input

class PredRNN_dec(nn.Module):
    def __init__(self):
        super(PredRNN_dec, self).__init__()
        self.pred1_dec=PredRNN(input_size=(61, 41),
                input_dim=3,
                hidden_dim=[ 64, 8, 1],
                hidden_dim_m=[8, 8, 8],
                kernel_size=[(3, 3),   (3, 3), (3,3)],
                num_layers=3,
                batch_first=True,
                bias=True).cuda()
        self.relu = nn.ReLU()
    def forward(self,input,enc_hidden,enc_h_m):
        _, _, _, real_out = self.pred1_dec(input,enc_hidden,enc_h_m)
        real_out = self.relu(real_out)
        return real_out
enc=PredRNN_enc().cuda()
dec=PredRNN_dec().cuda()

import itertools
#loss_fn=nn.MSELoss()   #loss改为L1+L2  MSE就是L2
loss_fn1=nn.MSELoss()
loss_fn2=nn.L1Loss()
#position=0
optimizer=optim.Adam(itertools.chain(enc.parameters(), dec.parameters()),lr=0.0001)
"""
for epoch in range(50):
    loss_total=0
    enc_hidden, enc_h_m = enc(input)
    for i in range(input.shape[1]):
        optimizer.zero_grad()
        out, layer_h_c, last_h_m = dec(input[:,i:i+1,:,:,:], enc_hidden, enc_h_m[-1])
        loss=loss_fn1(out, targetc)+loss_fn2(out, target[:,i:i+1,:,:,:])
        loss_total+=loss #所有步长结束后的loss？
        enc_hidden = layer_h_c
        enc_h_m = last_h_m
    loss_total=loss_total/input.shape[1]
    loss_total.backward()
    optimizer.step()
    print(epoch,epoch,loss_total)


for epoch in range(50):
    loss_total=0
    for data in sload:
        input,target=data
        torch.cuda.empty_cache()
        enc_hidden,enc_h_m=enc(input)
        loss_b=0
        for i in range(input.shape[1]):
            optimizer.zero_grad()
            layer_h_c, last_h_m, real_out =dec(input, enc_hidden, enc_h_m[-1])
            loss=loss_fn1(real_out[:,i:i+1,:,:,:], target[:,i:i+1,:,:,:])+loss_fn2(real_out[:,i:i+1,:,:,:], target[:,i:i+1,:,:,:])
            loss_b+=loss
            enc_hidden=layer_h_c
            enc_h_m=last_h_m
        loss_b=loss_b/input.shape[1]
        loss_b.backward(retain_graph=True)
    loss_total+=loss_b
    loss_total=loss_total / len(sload)
    loss_total.backward()
    optimizer.step()
    print(epoch,epoch,loss_total)
"""
for epoch in range(50):
    loss_total=0
    for i , (input, target) in enumerate(sload):
        input=input.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        enc_hidden,enc_h_m, dec_input = enc(input)
        real_out = dec(input, enc_hidden, enc_h_m[-1])
        loss_b=loss_fn1(real_out, target)+loss_fn2(real_out, target)
        loss_total += loss_b.item()
        loss_b.backward()
        optimizer.step()
    loss_total=loss_total/len(sload)
    print(epoch,loss_total)

torch.save(enc.state_dict(),r'E:\cqz\encoder.pth')
torch.save(dec.state_dict(),r'E:\cqz\decoder.pth')



