from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from src.dataset.geo import VisionDataset 
from torchvision import transforms as trsfs

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
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_path(df, index, band):
    
    return Path(df.iloc[index][band])

def encode_loc(loc, concat_dim=1, elev = False):
    #loc is (lon, lat ) or (lon, lat, elev)
    
    feats = torch.cat((torch.sin(math.pi*loc[:,:2]), torch.cos(math.pi*loc[:,:2])), concat_dim)
    if elev:
        elev_feats = torch.unsqueeze(loc_ip[:, 2], concat_dim)
        feats = torch.cat((feats, elev_feats), concat_dim)
    return(feats)

def convert_loc_to_tensor(x, elev = False, device=None):
    # input is in lon {-180, 180}, lat {90, -90}
    xt = x
    xt[:,0] /= 180.0 # longitude
    xt[:,1] /= 90.0 # latitude
    if elev:
        xt[:,2] /= 5000.0 # elevation
    if device is not None:
        xt = xt.to(device)
    return xt

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
            bands+= [np.squeeze(load_file(band))]
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
    elif subset == "not_songbirds":
        return (np.load('/network/projects/_groups/ecosystem-embeddings/species_splits/not_songbirds_idx.npy'))
    elif subset == "ducks":
        return ([37])
    elif subset=="code1":
        return(np.load("/network/projects/_groups/ecosystem-embeddings/species_splits/code1.npy"))
    elif subset == "hawk":
        return([2])
    elif subset == "oystercatcher":
        print("using oystercatcher")#Haematopus palliatus
        return([290])
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
                 subset = None, 
                 use_loc = False, 
                 res=[],
                 loc_type = None)-> None:
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
        self.use_loc = use_loc
        self.loc_type = loc_type
        self.res=res
        self.speciesA = get_subset("songbirds") 
        
    def __len__(self):
        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:

        
        band_npy = [(b,get_path(self.df, index, b)) for b in self.bands if get_path(self.df, index, b).suffix == ".npy"]
        env_npy = [(b,get_path(self.df, index, b)) for b in self.env if get_path(self.df, index, b).suffix in ".npy"]
        
        item_ = {}
    
        
        assert len(band_npy) > 0, "No item to fetch"
        
       
        
      
        if self.type == "img":
                npy_data = get_img_bands(band_npy)
        else:
                bands = [load_file(band) for (_,band) in band_npy]
                npy_data = np.stack(bands, axis = 1).astype(np.int32)
        env_len=0
        for (b,band) in env_npy: 
                item_[b] = torch.from_numpy(load_file(band))
                env_len+=1
        sats=torch.from_numpy(npy_data)
        sats=sats.squeeze(0)
        C, _, _ = sats.shape
        item_["sat"]=sats
        if "landuse" in self.bands:
                landuse=torch.from_numpy(np.array(Image.open(get_path(self.df, index, "landuse")))/10)
                landuse = torch.unsqueeze(landuse, 0)
        if  len(self.res)>1 :
                sat_list = []
              
#                 item_['sat']=torch.unsqueeze(sats,0)
                if "landuse" in self.bands:      
                       landuselist=[]
#                        landuse=torch.unsqueeze(landuse, 0)
                       item_['landuse']=landuse                
                crops, transforms= self.transform[0],self.transform[1]
                #perform different crops and transformation on  both sat and landuse & env data
                for c in crops:
                    transforms.insert(0,c)
                    t=trsfs.Compose(transforms)    
                    item_=t(item_)
                    sat_list.append(item_['sat'])
                    if 'landuse' in self.bands:
                        landuselist.append(item_['landuse'])
            
                item_["sat"] = torch.cat(sat_list,dim=0)  
                if 'landuse' in self.bands:
                    item_["landuse"] = torch.cat(landuselist,dim=0)  
                    item_['sat'] = torch.cat((item_['sat'],item_['landuse']),dim=-3)
                    print('shape ',item_['sat'].shape)
                    assert item_['sat'].shape==(len(self.res),C+1,224,224),'shape of item_sat with land use is wrong'
                else:
                    
                     assert item_['sat'].shape==(len(self.res),C,224,224),'shape of item_sat is wrong'
                

        elif len(self.res) <= 1:
             item_['sat']=sats
             t=trsfs.Compose(self.transform)
             if "landuse" in self.bands:
                     item_['landuse']=landuse  
                
             item_=t(item_)
             if 'landuse' in self.bands:
                 item_['sat'] = torch.cat((item_['sat'],item_['landuse']),dim=-3)
        
        else:
            raise ValueError("Unknown transforms_length {}".format(len(self.transform)))

             
        #print(item_["landuse"].size())
        #print(item_["sat"].size())
        if "species" in self.df.columns: 
            #add target
            species = load_file(get_path(self.df, index, "species"))
            item_["speciesA"] = np.array(species["probs"])[self.speciesA]
            if self.target == "probs":
                if not self.subset is None:
                    item_["target"] = np.array(species["probs"])[self.subset]
                else: 
                    item_["target"] = species["probs"]
                item_["target"] = torch.Tensor(item_["target"])

            elif self.target == "binary":
                if self.subset is not None:
                    targ = np.array(species["probs"])[self.subset]
                else: 
                    targ = species["probs"]
                item_["original_target"] = torch.Tensor(targ)
                targ[targ>0] = 1 
                item_["target"] =torch.Tensor(targ)

            elif self.target == "log":
                if not self.subset is None:
                    item_["target"] = np.array(species["probs"])[self.subset]
                else: 
                    item_["target"] = species["probs"]
                
            else:
                raise NameError("type of target not supported, should be probs or binary")
        
        
            item_["num_complete_checklists"] = species["num_complete_checklists"]

        item_["state_id"] = self.df["state_id"][index]

        if "meta" in self.df.columns: 
            meta = load_file(get_path(self.df, index, "meta"))
            #add metadata information (hotspot info)
            meta.pop('earliest_date', None)
            item_.update(meta)
        else: 
            item_["hotspot_id"] = os.path.basename(get_path(self.df, index, "b")).strip(".npy")

        if self.use_loc:
            if self.loc_type == "latlon":
                
                lon, lat = torch.Tensor([item_["lon"]]), torch.Tensor([item_["lat"]])
                loc = torch.cat((lon, lat)).unsqueeze(0)
                loc = encode_loc(convert_loc_to_tensor(loc))
                item_["loc"] = loc
                
            elif self.loc_type == "state":
                item_["state_id"] = self.df["state_id"][index]
                item_["loc"] = torch.zeros([51])
                item_["loc"][item_["state_id"]] = 1 
        #print(item_.keys())
        return item_

    
class EbirdSpeciesEnvDataset(VisionDataset):
    def __init__(self,                 
                 df_paths,
                 bands,
                 env,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict [str, Any]]] = None,
                 mode : Optional[str] = "train",
                 datatype = "refl",
                 target = "probs", 
                 subset = None, use_loc = False, loc_type = None)-> None:
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
        self.use_loc = use_loc
        self.loc_type = loc_type
        self.speciesA = get_subset("songbirds") 
        
    def __len__(self):
        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:

        
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
        #item_["sat"] = item_["sat"]/ torch.amax(item_["sat"], dim=(-2,-1), keepdims=True)
    
        if self.transform:
            item_ = self.transform(item_)
        
        item_["env"] = torch.cat([item_[b] for (b,band) in env_npy], dim = 1)
        
         
        if "species" in self.df.columns: 
            #add target
            species = load_file(get_path(self.df, index, "species"))
            item_["speciesA"] = np.array(species["probs"])[self.speciesA]
            if self.target == "probs":
                if not self.subset is None:
                    item_["target"] = np.array(species["probs"])[self.subset]
                else: 
                    item_["target"] = species["probs"]
                item_["target"] = torch.Tensor(item_["target"])

            elif self.target == "binary":
                if self.subset is not None:
                    targ = np.array(species["probs"])[self.subset]
                else: 
                    targ = species["probs"]
                item_["original_target"] = torch.Tensor(targ)
                targ[targ>0] = 1 
                item_["target"] =torch.Tensor(targ)

            elif self.target == "log":
                if not self.subset is None:
                    item_["target"] = np.array(species["probs"])[self.subset]
                else: 
                    item_["target"] = species["probs"]
                
            else:
                raise NameError("type of target not supported, should be probs or binary")
        
        
            item_["num_complete_checklists"] = species["num_complete_checklists"]

        item_["state_id"] = self.df["state_id"][index]

        if "meta" in self.df.columns: 
            meta = load_file(get_path(self.df, index, "meta"))
            #add metadata information (hotspot info)
            meta.pop('earliest_date', None)
            item_.update(meta)
        else: 
            item_["hotspot_id"] = os.path.basename(get_path(self.df, index, "b")).strip(".npy")

        if self.use_loc:
            if self.loc_type == "latlon":
                
                lon, lat = torch.Tensor([item_["lon"]]), torch.Tensor([item_["lat"]])
                loc = torch.cat((lon, lat)).unsqueeze(0)
                loc = encode_loc(convert_loc_to_tensor(loc))
                item_["loc"] = loc
                
            elif self.loc_type == "state":
                item_["state_id"] = self.df["state_id"][index]
                item_["loc"] = torch.zeros([51])
                item_["loc"][item_["state_id"]] = 1 
        #print(item_.keys())
        return item_
