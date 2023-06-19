from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch
from collections import defaultdict
from tqdm.auto import trange, tqdm

class SpaDataset(TensorDataset):    
    def __init__(self, data, target=None, pos_pair=None, adj_pair=None):
        if target is not None:
            assert data.size(0) == target.size(0) 
        spot_num = torch.sum(target==1).item()
        cell_num = torch.sum(target==0).item()
        if pos_pair is not None:
            assert spot_num == pos_pair.shape[0]
            assert cell_num == pos_pair.shape[1]
        
        self.data = data # (n+m) * d
        self.target = target # (n+m) * 1
        
        pos_cell = defaultdict(list)
        non_zeros_loc = np.transpose(np.nonzero(pos_pair))
        for loc in tqdm(non_zeros_loc):
            pos_cell[loc[0]].append(loc[1]+spot_num)
        for spotid in trange(spot_num):
            if len(pos_cell[spotid])==0:
                pos_cell[spotid].append(spotid)
        self.pos_cell = pos_cell # n*m
        
        pos_spot = defaultdict(list)
        non_zeros_loc = np.transpose(np.nonzero(adj_pair))
        for loc in tqdm(non_zeros_loc):
            pos_spot[loc[0]].append(loc[1])
        for spotid in trange(spot_num):
            if len(pos_spot[spotid])==0:
                pos_spot[spotid].append(spotid)
        self.pos_spot = pos_spot

    def __getitem__(self, index): 
        data = self.data[index]
        if self.target is None:
            return data
        
        target = self.target[index]
        if target == 1:
            # spot
            
            # choose pos rna
            pos_cell_sample = np.random.choice(self.pos_cell[index])
            pos_cell_data = self.data[pos_cell_sample]
            pos_cell_target = self.target[pos_cell_sample]
            # choose pos spot
            pos_spot_sample = np.random.choice(self.pos_spot[index])
            pos_spot_data = self.data[pos_spot_sample]
            pos_spot_target = self.target[pos_spot_sample]
            
            return (data, target, pos_cell_data, pos_spot_data)
        else:
            # cell
            return (data, target, data, data)

    def __len__(self):
        return self.data.size(0)
