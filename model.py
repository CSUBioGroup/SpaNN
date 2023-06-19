import numpy as np
import pandas as pd
import os

import torch
from torch import nn
import torch.nn.functional as F

import math
from tqdm.auto import tqdm
from collections import defaultdict

from utils import select_top_variable_genes, multi_pred_genes
from dataset import SpaDataset
from torch.utils.data import DataLoader
from logzero import logger

class MultiPred(nn.Module):
    def __init__(self, n_features, pred_dim):
        super(MultiPred, self).__init__()

        self.fc1 = nn.Linear(n_features, 1000)
        
        self.decd_fc = nn.Linear(1000, n_features)
        self.cont_fc = nn.Linear(1000, 1000) #, bias=False
        self.pred_fc = nn.Linear(1000, pred_dim)
    
    def forward(self, x):
        encode = F.relu(self.fc1(x))
        
        decode = F.relu(self.decd_fc(encode))
        pred = F.relu(self.pred_fc(encode))
        constrast = F.relu(self.cont_fc(encode))
        return decode, pred, constrast, encode

def initialize_weights_multi(self, in_dim, pred_dim):
    co_dim = in_dim+pred_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            bound = 1/math.sqrt(co_dim)
            torch.nn.init.uniform_(m.weight.data, -bound, bound)
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias.data, -bound, bound)

def loss_cosine_func(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = x.mul(y)
    
    return torch.sum(sim)

def SpaNN(spatial_df, scrna_df, device, 
          genes_to_predict,
          pos_matrix, adj_matrix,
          save_path_prefix, config, random_seed=3407):
    
    genes_to_predict = np.array(genes_to_predict)
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    shared_gene = np.intersect1d(spatial_df.columns, scrna_df.columns)
    reserved_gene = np.hstack((shared_gene,genes_to_predict))
    
    logger.info('Spatial data: {} cells * {} genes'.format(spatial_df.shape[0], spatial_df.shape[1]))
    logger.info('scRNA data: {} cells * {} genes'.format(scrna_df.shape[0], scrna_df.shape[1]))
    logger.info('{} genes need to be predicted\n'.format(genes_to_predict.shape[0]))
    logger.info('shared {} genes'.format(shared_gene.shape[0]))
    
    spatial_df = spatial_df[shared_gene]
    raw_scrna_uniq_gene = np.unique(scrna_df.columns.values[~np.isin(scrna_df.columns.values, reserved_gene)])
    scrna_df = scrna_df[np.hstack((reserved_gene, raw_scrna_uniq_gene))]
    
    spatial_df_appended = np.hstack((spatial_df.values,  np.zeros((spatial_df.shape[0], scrna_df.shape[1]-spatial_df.shape[1]))))
    spatial_df_appended = pd.DataFrame(data=spatial_df_appended, index = spatial_df.index, columns=scrna_df.columns)
    
    t_min_loss = np.array([1e7]*config['train']['t_min'])
    
    # select gene
    dedup_ind = ~scrna_df.columns.duplicated()
    spatial_df_appended = spatial_df_appended.loc[:,dedup_ind]
    scrna_df = scrna_df.loc[:,dedup_ind]
    
    other_genes = np.setdiff1d(scrna_df.columns.values, reserved_gene)
    other_genes_mtx = scrna_df[other_genes].values
    selected_ind = select_top_variable_genes(other_genes_mtx, config['train']['top_k'])
    selected_gene = other_genes[selected_ind]
    new_genes = np.hstack((shared_gene, genes_to_predict, selected_gene))
    spatial_df_appended = spatial_df_appended[new_genes]
    scrna_df = scrna_df[new_genes]

    zero_pred_res = pd.DataFrame(np.zeros((spatial_df_appended.shape[0],genes_to_predict.shape[0])), columns=genes_to_predict)

    sorted_spatial_data_label = np.ones(spatial_df_appended.shape[0])
    sorted_scRNA_data_label = np.zeros(scrna_df.shape[0])

    train_dat = torch.from_numpy(np.vstack((spatial_df_appended, scrna_df))).float()
    train_lab = torch.from_numpy(np.hstack((sorted_spatial_data_label, sorted_scRNA_data_label))).float()

    net = MultiPred(shared_gene.shape[0], train_dat.shape[1]-shared_gene.shape[0]).to(device)
    initialize_weights_multi(net, shared_gene.shape[0], train_dat.shape[1]-shared_gene.shape[0])

    optimizer = torch.optim.Adam(net.parameters(), lr = config['train']['lr'], weight_decay=0.0002) 

    logger.info("Generating Datasets")
    train_set = SpaDataset(train_dat, train_lab, pos_matrix, adj_matrix)
    train_loader = DataLoader(dataset=train_set, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=train_set, batch_size=config['train']['batch_size'], shuffle=False)
    
    for e in range(config['train']['max_epoch_num']):
        train_loss = 0.0
        train_loss_recon = 0.0
        train_loss_ref = 0.0
        train_loss_cos_cell = 0.0
        train_loss_cos_spot = 0.0
        
        for batch_idx, (train_x, train_y, pos_cell, pos_spot) in enumerate(train_loader):
            train_x = train_x.to(device)
            pos_cell = pos_cell.to(device)
            pos_spot = pos_spot.to(device)
            is_spatial = train_y==1
            
            spa_num = is_spatial.shape[0]
            rna_num = train_x.shape[0] - spa_num
            
            # decode, pred, constrast
            decode, pred, constrast, encode = net(train_x[:, :shared_gene.shape[0]]) # decoder, emd
            
            # pos cell loss
            pos_decode, pos_pred, pos_constrast, pos_encode = net(pos_cell[:, :shared_gene.shape[0]])
            loss_cosine_cell = 1 - loss_cosine_func(constrast[is_spatial, :], pos_constrast[is_spatial, :]) / spa_num

            # pos spot loss
            pos_decode, pos_pred, pos_constrast, pos_encode = net(pos_spot[:, :shared_gene.shape[0]])
            loss_cosine_spot = 1 - loss_cosine_func(constrast[is_spatial, :], pos_constrast[is_spatial, :]) / spa_num
            
            # spot recon loss
            loss_recon_target = F.mse_loss(decode[is_spatial,:], train_x[is_spatial,:shared_gene.shape[0]], reduction='mean')
            
            # cell pred loss
            train_x_new = train_x[~is_spatial]
            rna_decode, rna_pred, rna_constrast, rna_encode = net(train_x_new[:, :shared_gene.shape[0]])
            gt = train_x_new[:,shared_gene.shape[0]:]
            loss_cor_source = F.mse_loss(rna_pred, gt, reduction='mean')
            
            loss = loss_recon_target + loss_cor_source
            loss = loss + config['train']['lasso_weight']*loss_cosine_cell
            loss = loss + config['train']['spot_weight']*loss_cosine_spot

            train_loss += loss.item()
            train_loss_recon += loss_recon_target.item()
            train_loss_ref += loss_cor_source.item()
            train_loss_cos_cell += loss_cosine_cell.item()
            train_loss_cos_spot += loss_cosine_spot.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = train_loss / (batch_idx + 1)
        
        logger.info('\t[{}] loss: {:.3f}'.format(e+1, train_loss))
        
        if train_loss < max(t_min_loss):
            replace_ind = np.where(t_min_loss==max(t_min_loss))[0][0]
            t_min_loss[replace_ind] = train_loss
            torch.save({'epoch': e,'model_state_dict': net.state_dict(),'loss': train_loss, 'optimizer_state_dict': optimizer.state_dict()}, 
                       '%s-%dmin%d.pt'%(save_path_prefix,config['train']['t_min'],replace_ind))
        
        if e > 0 and train_loss < config['train']['stop_loss']: break
    
    zero_rec_res = pd.DataFrame(np.zeros((train_dat.shape[0], train_dat.shape[1])), columns=new_genes)
    t_min_loss_pred_mean = zero_pred_res.copy()
    t_min_cnt = 0
    for i_t_min in range(config['train']['t_min']):
        if os.path.exists('%s-%dmin%d.pt'%(save_path_prefix, config['train']['t_min'], i_t_min)):
            checkpoint = torch.load('%s-%dmin%d.pt'%(save_path_prefix, config['train']['t_min'], i_t_min))
            net.load_state_dict(checkpoint['model_state_dict'])
            pred = multi_pred_genes(net, val_loader, train_lab, scrna_df, genes_to_predict, config['pred']['n_neighbors'], device, shared_gene)
            t_min_loss_pred_mean += pred
            t_min_cnt += 1

    final_res = t_min_loss_pred_mean / t_min_cnt

    return final_res
