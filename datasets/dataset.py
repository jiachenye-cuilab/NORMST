from torch.utils.data import Dataset

import torch
import numpy as np
from utils import make_coord, resize_matrix, pooling
import random
import scipy.sparse as sp



class STDataset(Dataset):
    def __init__(self, adata,inp_size=(26,21),scale_min=1.5,scale_max=3.0,repeat=30):

        self.data = torch.tensor(adata.X.toarray().T).unsqueeze(1)

        self.w=(adata.obs.array_col.max()+1)//2
        self.h=adata.obs.array_row.max()+1

        self.inp_size=inp_size

        self.scale_min=scale_min
        self.scale_max=scale_max

        self.repeat=repeat

    def __len__(self):
        return len(self.data)*self.repeat
    
    def __getitem__(self, idx):
        data = self.data[idx % len(self.data)].view(1, self.h, self.w)
        s = random.uniform(self.scale_min, self.scale_max)

        h_lr, w_lr = self.inp_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, self.w - w_hr)
        y0 = random.randint(0, self.h - h_hr)
        data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]

        data_lr=resize_matrix(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        id_gt = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
        #idx,_ = torch.sort(idx)
        coord = coord.view(-1, coord.shape[-1])
        coord = coord[id_gt, :]
        coord = coord.view(h_lr, w_lr, coord.shape[-1])

        gt = data_hr.contiguous().view(data_hr.shape[0], -1)
        gt = gt[:, id_gt]
        gt = gt.view(data_hr.shape[0], h_lr, w_lr)

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': gt
        }    
        



class STDataset_test(Dataset):
    def __init__(self, adata,scale=2.0):

        self.data = torch.tensor(adata.X.toarray().T).unsqueeze(1)

        self.w=(adata.obs.array_col.max()+1)//2
        self.h=adata.obs.array_row.max()+1


        self.scale=scale

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_hr = self.data[idx].view(1, self.h, self.w)

        h_hr = self.h
        w_hr = self.w
        h_lr=int(h_hr//self.scale)
        w_lr=int(w_hr//self.scale)

        data_lr=resize_matrix(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': data_hr
        }    
        

class Predict(Dataset):
    def __init__(self, adata, scale=2.0):
        self.scale = scale
        self.N = adata.X.shape[1]
        self.H = adata.obs.array_row.max() + 1
        self.W = adata.obs.array_col.max() // 2 + 1
        self.data = torch.tensor(adata.X.toarray().T).unsqueeze(1).view(-1,1, self.H, self.W)

        coord_shape = (self.H * scale, self.W * scale)
        coord = make_coord(coord_shape).unsqueeze(0).repeat(self.N, 1, 1, 1)
        cell = torch.tensor([
            2 / (self.H * scale),
            2 / (self.W * scale)
        ], dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)

        self.coord = coord
        self.cell = cell

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return {
            'inp': self.data[idx],      # [1, H, W]
            'coord': self.coord[idx],   # [2, H', W']
            'cell': self.cell[idx],     # [2]
        }


class STDataset1(Dataset):
    def __init__(self, adata,inp_size=(26,21),scale_min=1.5,scale_max=3.0,repeat=30):

        self.data = torch.tensor(adata.X.toarray().T).unsqueeze(1)

        self.w=(adata.obs.array_col.max()+1)//2
        self.h=adata.obs.array_row.max()+1

        self.inp_size=inp_size

        self.scale_min=scale_min
        self.scale_max=scale_max

        self.repeat=repeat

    def __len__(self):
        return len(self.data)*self.repeat
    
    def __getitem__(self, idx):
        data = self.data[idx % len(self.data)].view(1, self.h, self.w)
        s = random.uniform(self.scale_min, self.scale_max)

        h_lr, w_lr = self.inp_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, self.w - w_hr)
        y0 = random.randint(0, self.h - h_hr)
        data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]

        data_lr=resize_matrix(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        id_gt = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
        #idx,_ = torch.sort(idx)
        coord = coord.view(-1, coord.shape[-1])
        coord = coord[id_gt, :]
        coord = coord.view(h_lr, w_lr, coord.shape[-1])

        gt = data_hr.contiguous().view(data_hr.shape[0], -1)
        gt = gt[:, id_gt]
        gt = gt.view(data_hr.shape[0], h_lr, w_lr)

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': gt
        } 

class STDataset2(Dataset):
    def __init__(self, adata,inp_size=(26,21),scale_min=1.5,scale_max=3.0,repeat=30):

        self.data = torch.tensor(adata.X.toarray().T).unsqueeze(1)

        self.w=(adata.obs.array_col.max()+1)//2
        self.h=adata.obs.array_row.max()+1

        self.inp_size=inp_size

        self.scale_min=scale_min
        self.scale_max=scale_max

        self.repeat=repeat

    def __len__(self):
        return len(self.data)*self.repeat
    
    def __getitem__(self, idx):
        data = self.data[idx % len(self.data)].view(1, self.h, self.w)
        s = random.uniform(self.scale_min, self.scale_max)

        h_lr, w_lr = self.inp_size
        h_hr = round(h_lr * s)
        w_hr = round(w_lr * s)
        x0 = random.randint(0, self.w - w_hr)
        y0 = random.randint(0, self.h - h_hr)
        data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]

        data_lr=pooling(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        id_gt = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
        #idx,_ = torch.sort(idx)
        coord = coord.view(-1, coord.shape[-1])
        coord = coord[id_gt, :]
        coord = coord.view(h_lr, w_lr, coord.shape[-1])

        gt = data_hr.contiguous().view(data_hr.shape[0], -1)
        gt = gt[:, id_gt]
        gt = gt.view(data_hr.shape[0], h_lr, w_lr)

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': gt
        }    
    

class STDataset2_test(Dataset):
    def __init__(self, adata,scale=2.0,repeat=1):

        self.data = torch.tensor(adata.X.toarray().T).unsqueeze(1)

        self.w=(adata.obs.array_col.max()+1)//2
        self.h=adata.obs.array_row.max()+1


        self.scale=scale

        self.repeat=repeat

    def __len__(self):
        return len(self.data)*self.repeat
    
    def __getitem__(self, idx):
        data_hr = self.data[idx % len(self.data)].view(1, self.h, self.w)

        h_hr = self.h
        w_hr = self.w
        h_lr=int(h_hr//self.scale)
        w_lr=int(w_hr//self.scale)

        data_lr=pooling(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': data_hr
        }    
    


class HD_Dataset(Dataset):
    def __init__(self, adata,inp_size=(128,128),scale=2.0):

        self.data = torch.tensor(adata.X.toarray().T,dtype=torch.float32).unsqueeze(1)

        self.w=adata.obs.array_col.max()+1
        self.h=adata.obs.array_row.max()+1
        self.inp_size=inp_size

        self.scale=scale

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx].view(1, self.h, self.w)
        h_lr, w_lr = self.inp_size
        h_hr = round(h_lr * self.scale)
        w_hr = round(w_lr * self.scale)
        x0 = random.randint(0, self.w - w_hr)
        y0 = random.randint(0, self.h - h_hr)
        data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]

        data_lr=resize_matrix(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': data_hr
        }    
    
class HD_Dataset_pooling(Dataset):
    def __init__(self, adata,inp_size=(200,200),scale=2.0):

        self.data = torch.tensor(adata.X.toarray().T,dtype=torch.float32).unsqueeze(1)

        self.w=adata.obs.array_col.max()+1
        self.h=adata.obs.array_row.max()+1
        self.inp_size=inp_size

        self.scale=scale

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx].view(1, self.h, self.w)
        h_lr, w_lr = self.inp_size
        h_hr = round(h_lr * self.scale)
        w_hr = round(w_lr * self.scale)
        x0 = random.randint(0, self.w - w_hr)
        y0 = random.randint(0, self.h - h_hr)
        data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]

        data_lr=pooling(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': data_hr
        }    

class HD_Dataset_test(Dataset):
    def __init__(self, adata,gt):
        if sp.issparse(adata.X):
            adata_ = adata.X.toarray()
        else:
            adata_ = adata.X
        
        if sp.issparse(gt.X):
            gt_ = gt.X.toarray()
        else:
            gt_ = gt.X

        self.data = torch.tensor(adata_.T,dtype=torch.float32).unsqueeze(1)
        self.gt = torch.tensor(gt_.T,dtype=torch.float32).unsqueeze(1)

        self.w_hr=gt.obs.array_col.max()+1
        self.h_hr=gt.obs.array_row.max()+1
        self.w_lr=adata.obs.array_col.max()+1
        self.h_lr=adata.obs.array_row.max()+1
        # self.coord = make_coord(shape=(self.h_hr, self.w_hr))
        # self.cell = torch.tensor([2 / self.h_hr, 2 / self.w_hr], dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'inp': self.data[idx].view(1, self.h_lr, self.w_lr),
            # 'coord': self.coord,
            # 'cell': self.cell,
            'gt': self.gt[idx].view(1, self.h_hr, self.w_hr)
        }   
    

class HD_pooling(Dataset):
    def __init__(self, adata,inp_size=None,scale_min=2.0,scale_max=None,repeat=1):

        self.data = torch.tensor(adata.X.toarray().T,dtype=torch.float32).unsqueeze(1)

        self.w=adata.obs.array_col.max()+1
        self.h=adata.obs.array_row.max()+1
        self.inp_size=inp_size

        self.scale_min=scale_min
        if scale_max is None:
            scale_max=scale_min
        self.scale_max = scale_max

        self.repeat=repeat

    def __len__(self):
        return len(self.data)*self.repeat
    
    def __getitem__(self, idx):
        data = self.data[idx % len(self.data)].view(1, self.h, self.w)
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = np.floor(self.h / s + 1e-9)
            w_lr = np.floor(self.w / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            data_hr = data[:, :h_hr,:w_hr]
            data_lr=pooling(data_hr,size=(h_lr, w_lr))
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, self.w - w_hr)
            y0 = random.randint(0, self.h - h_hr)
            data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]
            data_lr=pooling(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        if self.inp_size is not None:
            id_gt = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            coord = coord.view(-1, coord.shape[-1])
            coord = coord[id_gt, :]
            coord = coord.view(h_lr, w_lr, coord.shape[-1])

            gt = data_hr.contiguous().view(data_hr.shape[0], -1)
            gt = gt[:, id_gt]
            gt = gt.view(data_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': gt
        }    
    

class HD_interpolate(Dataset):
    def __init__(self, adata,inp_size=None,scale_min=2.0,scale_max=None,repeat=1):
        if sp.issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X

        self.data = torch.tensor(data.T, dtype=torch.float32).unsqueeze(1)

        self.w=adata.obs.array_col.max()+1
        self.h=adata.obs.array_row.max()+1
        self.inp_size=inp_size

        self.scale_min=scale_min
        if scale_max is None:
            scale_max=scale_min
        self.scale_max = scale_max

        self.repeat=repeat

    def __len__(self):
        return len(self.data)*self.repeat
    
    def __getitem__(self, idx):
        data = self.data[idx % len(self.data)].view(1, self.h, self.w)
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = np.floor(self.h / s + 1e-9)
            w_lr = np.floor(self.w / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            data_hr = data[:, :h_hr,:w_hr]
            data_lr=resize_matrix(data_hr,size=(h_lr, w_lr))
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, self.w - w_hr)
            y0 = random.randint(0, self.h - h_hr)
            data_hr = data[:, y0: y0 + h_hr, x0: x0 + w_hr]
            data_lr=resize_matrix(data_hr,size=(h_lr, w_lr))

        coord = make_coord(shape=(h_hr, w_hr))

        if self.inp_size is not None:
            id_gt = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            coord = coord.view(-1, coord.shape[-1])
            coord = coord[id_gt, :]
            coord = coord.view(h_lr, w_lr, coord.shape[-1])

            gt = data_hr.contiguous().view(data_hr.shape[0], -1)
            gt = gt[:, id_gt]
            gt = gt.view(data_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / h_hr, 2 / w_hr], dtype=torch.float32)

        return {
            'inp': data_lr,
            'coord': coord,
            'cell': cell,
            'gt': gt
        }    