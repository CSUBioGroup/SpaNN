import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.stats as st

def save_pkl(output_file,data):
    with open(output_file,'wb') as fw:
        pkl.dump(data,fw)

def read_pkl(input_file):
    with open(input_file,'rb') as fr:
        temp_result = pkl.load(fr)
    
    return temp_result

def multi_pred_genes(net, val_loader, train_lab, scRNA_data, genes_to_predict, n_neighbors, device, shared_gene):
    net.eval()
    fm_mu = None
    for x, y, pos1, pos2 in val_loader:
        x = x.to(device)
        decode, pred, constrast, encode = net(x[:, :shared_gene.shape[0]])
        
        emd = encode.cpu().detach().numpy()
        
        if fm_mu is None:
            fm_mu = emd[:,:]
        else:
            fm_mu = np.concatenate((fm_mu, emd),axis=0)
    
    scRNA_transformed = fm_mu[train_lab!=1,:]
    spatial_transformed = fm_mu[train_lab==1,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric = 'cosine').fit(scRNA_transformed)

    pred_res = pd.DataFrame(np.zeros((spatial_transformed.shape[0],genes_to_predict.shape[0])), columns=genes_to_predict)

    distances, indices = nbrs.kneighbors(spatial_transformed)
    for j in range(0,spatial_transformed.shape[0]):
        weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
        weights = weights/(len(weights)-1)
        pred_res.iloc[j,:] = np.dot(weights,scRNA_data[genes_to_predict].iloc[indices[j,:][distances[j,:] < 1]])
    
    net.train()
    return pred_res

def select_top_variable_genes(data_mtx, top_k):
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind

def calc_corr(spatial_df, pred_res, test_gene):
    correlation = []
    for gene in test_gene:
        correlation.append(st.spearmanr(spatial_df[gene], pred_res[gene])[0])
    return correlation

def calc_cell_corr(spa_df, pred_res, test_gene):
    spa_df = spa_df[test_gene]
    pred_res = pred_res[test_gene]
    correlation = []
    for cellid in range(spa_df.shape[0]):
        correlation.append(st.spearmanr(spa_df.iloc[cellid].values, pred_res.iloc[cellid].values)[0])
    return correlation
