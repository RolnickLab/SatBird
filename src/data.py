import os
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from utils import is_image_file
from imageio import imread
from PIL import Image
from src.utils import json_load, yaml_load

def patch_loader(path, source):
    """load data of a patch as tensors
    Args:
        path (str): path to data
        source ([str]): "rgb", "nir", "land"
    Returns:
        [Tensor]: 1 x C x H x W
    """
    if is_image_file(path):
        arr = imread(path).astype(np.float32)
        return torch.load(path)
    else:
        raise ValueError("Unknown data type {}".format(path))


class EBirdDataset(Dataset):
    """[summary]
    get inspo from https://github.com/maximiliense/GLC/blob/master/data_loading/pytorch_dataset.py

    Args:
        root : string or pathlib.Path
            Root directory of dataset.
        subset: "train", "train+val", "val", "test"
    """
    # TODO
    def __init__(self, config_file):

        config = yaml_load(config_file)
        self.mode = config['mode']
        self.percent = config['percent']
        self.month = config['month']
        self.s_list = config['species_list']
        
        self.dataset = json_load(Path(config['data']['base'] / config['data'][self.mode]))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """
        i: index of the item to retrieve
        """
        #GET PATCH FROM RASTERS
        item = self.dataset[i]
        #load species data for the hotspot 
        hotspot = json_load(item['s'])
        
        #to be completed from item["x"]
        patch = patch_loader()
        
        target = None
        #SET TARGETS IF TRAINING 
        if self.mode == "training":
            target = self.get_target(hotspot, self.percent, self.month, s.s_list)
    
            
        return patch, target
    
    def get_target(self, hotspot, percent=True, month=None, s_list=None):
        species = np.array(hotspot['species'])
        if not(s_list) is None:
            species = species[s_list, :]
        if percent == True:
            species = species / np.array(hotspot['checklists'])
        if not(month is None):
            species = species[:, month]
        return(species)
    
def get_loader(root, subset, opts):
    batch_size = opts.batch_size
    #num_workers = 
    return DataLoader(EBirdDataset(root,subset), batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True,)
