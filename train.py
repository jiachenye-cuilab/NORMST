# %%
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import scanpy as sc
import numpy as np
from datasets.dataset import STDataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.sronet import SRNO
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scheduler import GradualWarmupScheduler

from utils import adata_preprocess, Averager

# %%
BATCH_SIZE=256
data_path='151675'
adata = sc.read_visium(path=f'/home/yejiachen/Workdir/ST/Data/{data_path}',count_file=f'/home/yejiachen/Workdir/ST/Data/{data_path}/raw_feature_bc_matrix.h5')
adata.var_names_make_unique()
adata=adata_preprocess(adata,n_genes=3000)
train_loader=DataLoader(STDataset(adata,repeat=30),batch_size=BATCH_SIZE,shuffle=True,num_workers=40,pin_memory=True,persistent_workers=True)
model=SRNO().cuda()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
cosine = CosineAnnealingLR(optimizer, 300)
lr_scheduler = GradualWarmupScheduler(optimizer,multiplier=50,total_epoch=50,after_scheduler=cosine)

# %%
def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.SmoothL1Loss()
    train_loss = Averager()

    data_norm = { 'inp': {'sub': [0.], 'div': [1.]},'gt': {'sub': [0.], 'div': [1.]}}

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    
    pbar = tqdm(train_loader, leave=True, desc='train')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        #psnr = metric_fn(pred, gt)
        
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None
        pbar.set_description('train loss{:.4f}'.format(train_loss.item()))
        
    return train_loss.item()

# %%
num_epochs = 350
losses=[]
for epoch in range(num_epochs):
    losses.append(train(train_loader, model, optimizer))
    lr_scheduler.step(epoch)

    if (epoch+1)%25==0:
        torch.save(model.state_dict(), f"save/{data_path}/05_12/model_{epoch+1}.pth")

np.save(f"save/{data_path}/05_12/loss.npy", losses)
# %%