import os
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Dataset

class EBirdDataset(Dataset):
    """[summary]
    get inspo from https://github.com/maximiliense/GLC/blob/master/data_loading/pytorch_dataset.py

    Args:
        root : string or pathlib.Path
            Root directory of dataset.
        subset: "train", "train+val", "val", "test"
    """
    # TODO
    def __init__(self, root, subset):
        #SET TARGETS IF TRAINING 
        self.root = Path(root)
        self.subset = subset
        possible_subsets = ["train", "val", "train+val", "test"]
        if subset not in possible_subsets:
            raise ValueError("Possible values for 'subset' are: {} (given {})".format(possible_subsets, subset))

        self.observation_ids =  ...

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self):
        #GET PATCH FROM RASTERS
        return()
    
def get_loader(root, subset, opts):
    batch_size = opts.batch_size
    #num_workers = 
    return DataLoader(EBirdDataset(root,subset), batch_size = batch_size, shuffle=True, pin_memory=True, drop_last=True,)
