import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import Lasso, MultiTaskLasso

from logzero import logger
from ruamel.yaml import YAML
from pathlib import Path

from utils import read_pkl, save_pkl


@click.command()
@click.option('-c', '--config', type=click.Path(exists=True))

def get_lasso(config):
    yaml = YAML(typ='safe')
    config = yaml.load(Path(config))
    
    all_rna_df = read_pkl(config['spatial_file'])
    all_spa_df = read_pkl(config['scrna_file'])
    
    shared_gene = np.intersect1d(all_spa_df.columns, all_rna_df.columns)
    
    idx = 1
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    kf.get_n_splits(shared_gene)
    for train_ind, test_ind in kf.split(shared_gene):    
        train_gene = shared_gene[train_ind]
        test_gene  = shared_gene[test_ind]
        
        test_spa_df = all_spa_df[test_gene]
        test_rna_df = all_rna_df[test_gene]
        
        spa_df = all_spa_df[train_gene]
        rna_df = all_rna_df[train_gene]

        X = rna_df.values.T
        Y = spa_df.values.T

        clf = Lasso(alpha=0.5, random_state=0)
        clf.fit(X,Y)
        # coef_: cell-spa * cell-rna
        # itercept: 1 * cell-spa
        w = clf.coef_.copy()
        b = clf.intercept_.copy()

        w[w<=0.0] = 0.0
        w = preprocessing.normalize(w, axis=1, norm='l1')
        save_pkl(config['neighbor_file'].format(spatial_name, scrna_name, is_magic, idx), w)

if __name__=='__main__':
    main()
    