from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from src.dataset.geo import VisionDataset 
from src.dataset.utils import load_file, is_image_file 
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor
import numpy as np
from PIL import Image
import torch
import os
import pandas as pd
import time 

def get_path(df, index, band):
    
    return Path(df.iloc[index][band])

class Identity(Module):  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Do nothing.
        Args:
            sample: the input
        Returns:
            the unchanged input
        """
        return sample

def get_img_bands(band_npy):
    """
    band_npy: list of tuples (band_name, item_paths) item_paths being paths to npy objects
    
    Returns: 
        stacked satellite bands data as numpy array
    """
    bands = []
    for elem in band_npy:
        b, band= elem
        if b == "rgb":
            bands+= [load_file(band)]
        elif b == "nir":
            nir_band = load_file(band)
            nir_band = (nir_band/nir_band.max())*255
            nir_band = nir_band.astype(np.uint8)
            bands+= [nir_band]
    npy_data =np.vstack(bands)/255
    return (npy_data)
            
def get_subset(subset):
    if subset == "songbirds":
        return (np.load('/network/scratch/t/tengmeli/scratch/ecosystem-embedding/songbirds_idx.npy'))
    elif subset == "ducks":
        return ([37])
    else:
        return None
        
class EbirdVisionDataset(VisionDataset):
    def __init__(self,                 
                 df_paths,
                 bands,
                 env,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict [str, Any]]] = None,
                 mode : Optional[str] = "train",
                 datatype = "refl",
                 target = "probs", 
                 subset = None)-> None:
        """
        df_paths: dataframe with paths to data for each hotspot
        bands: list of bands to include, anysubset of  ["r", "g", "b", "nir"] or  "rgb" (for image dataset) 
        env: list eof env data to take into account [ped, bioclim]
        transforms:
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep 
        """
        
        super().__init__()
        self.df = df_paths
        self.total_images = len(df_paths)
        self.transform = transforms
        self.bands = bands
        self.env = env
        self.mode = mode
        self.type = datatype
        self.target = target
        self.subset = get_subset(subset)

    def __len__(self) -> int:

        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:

        meta = load_file(get_path(self.df, index, "meta"))
        band_npy = [(b,get_path(self.df, index, b)) for b in self.bands if get_path(self.df, index, b).suffix == ".npy"]
        env_npy = [(b,get_path(self.df, index, b)) for b in self.env if get_path(self.df, index, b).suffix == ".npy"]
        item_ = {}
    
        
        assert len(band_npy) > 0, "No item to fetch"
        
        if self.type == "img":
            npy_data = get_img_bands(band_npy)
        else:
            bands = [load_file(band) for (_,band) in band_npy]
            npy_data = np.stack(bands, axis = 1).astype(np.int32)
            
        for (b,band) in env_npy: 
            item_[b] = torch.from_numpy(load_file(band))
            
        
        item_["sat"] = torch.from_numpy(npy_data)           
        
        if self.transform:
            item_ = self.transform(item_)
        
        for e in self.env:
            item_["sat"] = torch.cat([item_["sat"],item_[e]], dim = 1)
         
        #add target
        species = load_file(get_path(self.df, index, "species"))
        
        if self.target == "probs":
            if not self.subset is None:
                item_["target"] = np.array(species["probs"])[self.subset]
            else: 
                item_["target"] = species["probs"]
            item_["target"] = torch.Tensor(item_["target"])
            
        elif self.target == "binary":
            if not self.subset is None:
                targ = np.array(species["probs"])[self.subset]
            else: 
                targ = species["probs"]
            item_["original_target"] = torch.Tensor(targ)
            item_["target"] = torch.Tensor([1 if targ[i]>0 else 0 for i in range(len(targ))])
            
        else:
            raise NameError("type of target not supported, should be probs or binary")
        
        
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        
        #add metadata information (hotspot info)
        meta.pop('earliest_date', None)
        item_.update(meta)
        
        
        return item_