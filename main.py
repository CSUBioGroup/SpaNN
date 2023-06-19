import pandas as pd
import numpy as np
import click

import torch
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from logzero import logger
from ruamel.yaml import YAML
from pathlib import Path

from model import SpaNN
from utils import read_pkl, calc_corr, calc_cell_corr


@click.command()
@click.option('-c', '--config', type=click.Path(exists=True))

def main(config):
    yaml = YAML(typ='safe')
    config = yaml.load(Path(config))

    raw_spatial_df = read_pkl(config['spatial_file'])
    raw_scrna_df = read_pkl(config['scrna_file'])
    if '.csv' in config['loc_file']:
        loc_df = pd.read_csv(config['loc_file'])
    else:
        loc_df = pd.read_table(config['loc_file'], sep='\t')
    
    raw_shared_gene = np.intersect1d(raw_spatial_df.columns, raw_scrna_df.columns)
    
    locations = loc_df.values
    nbrs = NearestNeighbors(n_neighbors=11).fit(locations)
    adj_matrix = nbrs.kneighbors_graph(locations).toarray()
    adj_matrix -= np.eye(locations.shape[0])
    assert adj_matrix.shape[0]==raw_spatial_df.shape[0]

    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    kf.get_n_splits(raw_shared_gene)
    
    spatial_name = config['spatial_name']
    scrna_name = config['scrna_name']
    is_magic = config['magic']
    
    idx = 1
    for train_ind, test_ind in kf.split(raw_shared_gene):    
        logger.info("===== Fold {} =====".format(idx))
        logger.info("Number of train genes: {}, Number of test genes: {}".format(len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene  = raw_shared_gene[test_ind]

        test_spatial_df = raw_spatial_df[test_gene]
        test_rna_df = raw_scrna_df[test_gene]
        spatial_df = raw_spatial_df[train_gene]
        scrna_df   = raw_scrna_df
        
        # lasso
        lasso_w_path = config['neighbor_file'].format(spatial_name, scrna_name, is_magic, idx)
        lasso_w = read_pkl(lasso_w_path)
        
        if idx == 1:
            all_pred_res = pd.DataFrame(np.zeros((spatial_df.shape[0],raw_shared_gene.shape[0])), columns=raw_shared_gene) 
            
        save_path_prefix = './saved_model/%s-%s-fold%d'%(spatial_name, scrna_name, idx)

        SpaNN_res = SpaNN(spatial_df, scrna_df, torch.device('cuda:1'), 
                           test_gene, 
                           lasso_w, adj_matrix,
                           save_path_prefix, config)
        
        all_pred_res[SpaNN_res.columns.values] = SpaNN_res
        
        scc_gene = calc_corr(raw_spatial_df, SpaNN_res, test_gene)
        logger.info('Fold {} Gene Scc: {:.6f}'.format(idx, np.median(scc_gene)))
        
        idx += 1
    
    all_pred_res.to_csv('./result/{0}_{1}_{2}.csv'.format(spatial_name, scrna_name, is_magic), index=False)
    
    corr_res = calc_corr(raw_spatial_df, all_pred_res, raw_shared_gene)
    logger.info('Scc gene: {:.6f}'.format(np.median(corr_res)))
    scc_cell = calc_cell_corr(raw_spatial_df, all_pred_res, raw_shared_gene)
    logger.info('Scc cell: {:.6f}'.format(np.median(scc_cell)))

if __name__=='__main__':
    main()
    