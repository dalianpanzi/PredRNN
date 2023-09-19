import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import itertools
from torch.utils.data import TensorDataset, DataLoader
from predrnn_seq2seq_ccver import PredRNN_enc,PredRNN_dec
import os

zx=np.load(file=r'G:\cqz\finaltrain\zx.npy')
zy=np.load(file=r'G:\cqz\finaltrain\zy.npy')
zx=zx.transpose(0,1,4,2,3)
zy=zy.transpose(0,1,4,2,3)
zx=zx.astype(np.float32)
zy=zy.astype(np.float32)
zx=torch.from_numpy(zx)
zy=torch.from_numpy(zy)
see=TensorDataset(zx,zy)
sload=DataLoader(see,batch_size=15,drop_last=False)

enc=PredRNN_enc().cuda()
dec=PredRNN_dec().cuda()


"""
test=np.load(file=r'G:\cqz\pytorch\data\trans\x3te.npy')
test=test.astype(np.float32)
test=test.transpose(0,1,4,2,3)

test=torch.from_numpy(test)
test=TensorDataset(test)
test=DataLoader(test,batch_size=32,drop_last=False)
enc.eval()
dec.eval()
all_out=[]
with torch.no_grad():
    for batch in test:
        batch=batch[0]
        batch=batch.cuda()
        enc_hidden, enc_h_m, dec_input=enc(batch)
        predicted_output=dec(batch,enc_hidden,enc_h_m[-1])
        all_out.append(predicted_output)
all_out=torch.cat(all_out,dim=0)
no1=all_out.data.cpu().numpy()
np.save(r'E:\cqz\full.npy',no1)
os.remove(r'G:\cqz\pytorch\data\ablation\remove_wv\wv.npy')


enc.load_state_dict(torch.load(r'G:\cqz\pytorch\data\ablation\remove_wv\encoder.pth'))
dec.load_state_dict(torch.load(r'G:\cqz\pytorch\data\ablation\remove_wv\decoder.pth'))
for epoch in range(70,90):
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
torch.save(enc.state_dict(),r'G:\cqz\pytorch\mid\70-90encoder_w.pth')
torch.save(dec.state_dict(),r'G:\cqz\pytorch\mid\70-90decoder_w.pth')
"""
