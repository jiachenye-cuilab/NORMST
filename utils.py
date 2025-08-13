import os
import torch
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse

def resize_matrix(matrix, size, mode='bicubic'):
    # 将二维矩阵扩展为一个 3D 张量，形状为 (batch_size, channels, height, width)
    matrix = matrix.unsqueeze(0)  # 变为 (1, 1, H, W)
    
    # 使用 interpolate 来调整大小，mode 可以是 'bilinear', 'nearest', 'bicubic' 等
    resized_matrix = F.interpolate(matrix, size=size, mode=mode, align_corners=False)
    
    # 返回调整后的矩阵，去掉 batch 维度，变回 (1, H_out, W_out)
    return resized_matrix.squeeze(0)

def pooling(matrix, size):
    
    return F.adaptive_avg_pool2d(matrix.unsqueeze(0), size).squeeze(0)



def make_coord(shape, ranges=None, flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    #ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = torch.stack(torch.meshgrid(*coord_seqs,indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v
    
def adata_preprocess(adata, min_counts=200, min_cells=100,target_sum=1e4,n_genes=0):
    """
     Filters genes and cells based on minimum counts, normalizes total counts per cell, identifies highly variable genes, and applies log transformation. 

    """
    print('===== Preprocessing Data =====')

    spatial_df = pd.DataFrame({
    "x": adata.obs.array_col,
    "y": adata.obs.array_row,
    "index": np.arange(len(adata.obs))
    })

    # 按照 y 从小到大（上到下），然后 x 从小到大的顺序排序
    sorted_indices = spatial_df.sort_values(by=["y", "x"]).index

    # 按照排序后的顺序重新排列表达矩阵和其他数据
    i_adata = adata[sorted_indices, :]
    sc.pp.filter_genes(i_adata, min_counts=min_counts)
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    if n_genes != 0:
        sc.pp.highly_variable_genes(i_adata, flavor="seurat_v3", n_top_genes=n_genes)
        i_adata = i_adata[:, i_adata.var.highly_variable]
    sc.pp.normalize_total(i_adata, target_sum=target_sum, exclude_highly_expressed=False)
    sc.pp.log1p(i_adata)


    return i_adata



def read_visium_hd(data_dir, h5_name="raw_feature_bc_matrix.h5", pos_name="spatial/tissue_positions.parquet",min_counts=0, min_cells=0,target_sum=None,n_genes=0):
    h5_path = os.path.join(data_dir, h5_name)
    adata = sc.read_10x_h5(h5_path)
    
    # 解决 var_names 重复问题
    adata.var_names_make_unique()
    adata = adata[:, ~adata.var_names.str.startswith("DEPRECATED_")]

    positions = pd.read_parquet(os.path.join(data_dir, pos_name))
    positions = positions.set_index("barcode")
    
    missing_barcodes = positions.index.difference(adata.obs_names)
    
    zero_data = ad.AnnData(
        X=scipy.sparse.csr_matrix((len(missing_barcodes), adata.n_vars)),
        obs=pd.DataFrame(index=missing_barcodes),
        var=adata.var.copy()
    )

    
    adata_full = ad.concat([adata, zero_data], merge="same")
    adata_full = adata_full[positions.index,:]
    adata_full.obs = positions.loc[adata_full.obs_names].copy()

    if min_counts>0:
        sc.pp.filter_genes(adata_full, min_counts=min_counts)
    
    if min_cells>0:
        sc.pp.filter_genes(adata_full, min_cells=min_cells)

    if n_genes > 0:
        sc.pp.highly_variable_genes(adata_full, flavor="seurat_v3", n_top_genes=n_genes)

    sc.pp.normalize_total(adata_full, target_sum=target_sum, exclude_highly_expressed=False)
    sc.pp.log1p(adata_full)

    if n_genes > 0:
        adata_full = adata_full[:, adata_full.var.highly_variable]

    # sc.pp.normalize_total(adata_full, target_sum=target_sum, exclude_highly_expressed=False)   
    
    return adata_full