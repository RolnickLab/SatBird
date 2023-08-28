from pathlib import Path
from typing import Any, Callable, Dict, Optional
import os
import math

from src.dataset.geo import VisionDataset
from src.dataset.utils import load_file

import torch
from torchvision import transforms as trsfs
from torch.nn import Module
from torch import Tensor
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_path(df, index, band):
    return Path(df.iloc[index][band])


def encode_loc(loc, concat_dim=1, elev=False):
    # loc is (lon, lat ) or (lon, lat, elev)

    feats = torch.cat((torch.sin(math.pi * loc[:, :2]), torch.cos(math.pi * loc[:, :2])), concat_dim)
    if elev:
        elev_feats = torch.unsqueeze(loc_ip[:, 2], concat_dim)
        feats = torch.cat((feats, elev_feats), concat_dim)
    return (feats)


def convert_loc_to_tensor(x, elev=False, device=None):
    # input is in lon {-180, 180}, lat {90, -90}
    xt = x
    xt[:, 0] /= 180.0  # longitude
    xt[:, 1] /= 90.0  # latitude
    if elev:
        xt[:, 2] /= 5000.0  # elevation
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
        b, band = elem
        if b == "rgb":
            bands += [np.squeeze(load_file(band))]
        elif b == "nir":
            nir_band = load_file(band)
            nir_band = (nir_band / nir_band.max()) * 255
            nir_band = nir_band.astype(np.uint8)
            bands += [nir_band]
    npy_data = np.vstack(bands) / 255
    return npy_data


def get_subset(subset, num_species=684):
    """
    subset can be the filename instead
    """
    if not subset:
        return [i for i in range(num_species)]
    else:
        if os.path.isfile(subset):
            return np.load(subset).astype(int)
        else:
            raise TypeError("Only npy files are allowed")

    # if subset == "songbirds":
    #     return np.load('/network/projects/_groups/ecosystem-embeddings/species_splits/songbirds_idx.npy')
    # elif subset == "not_songbirds":
    #     return np.load('/network/projects/_groups/ecosystem-embeddings/species_splits/not_songbirds_idx.npy')
    # elif subset == "ducks":
    #     return [37]
    # elif subset == "code1":
    #     return np.load("/network/projects/_groups/ecosystem-embeddings/species_splits/code1.npy")
    # elif subset == "hawk":
    #     return [2]
    # elif subset == "oystercatcher":
    #     print("using oystercatcher")  # Haematopus palliatus
    #     return [290]
    # else:
    #     return [i for i in range(num_species)]


class EbirdVisionDataset(VisionDataset):

    def __init__(self,
                 df_paths,
                 data_base_dir,
                 bands,
                 env,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 mode: Optional[str] = "train",
                 datatype="refl",
                 target="probs",
                 subset=None,
                 use_loc=False,
                 res=[],
                 loc_type=None, 
                 num_species = 684) -> None:
        """
        df_paths: dataframe with paths to data for each hotspot
        data_base_dir: base directory for data
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
        self.data_base_dir = data_base_dir
        self.total_images = len(df_paths)
        self.transform = transforms
        self.bands = bands
        self.env = env
        self.mode = mode
        self.type = datatype
        self.target = target
        self.subset = get_subset(subset, num_species)
        self.use_loc = use_loc
        self.loc_type = loc_type
        self.res = res
        self.num_species = num_species

    def __len__(self):
        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']

        if self.type == 'img':
            img_path = os.path.join(self.data_base_dir, "images_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, "images", hotspot_id + '.tif')

        if self.type == "img":
            img = load_file(img_path)
        elif self.type == 'refl':
            img = load_file(img_path)

        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

        if len(self.env) > 0:
            env_npy = os.path.join(self.data_base_dir, "environmental_data", hotspot_id + '.npy')
            env_data = load_file(env_npy)
            item_["bioclim"] = torch.from_numpy(env_data[:19, :, :])
            #TODO: make it generic
            if self.num_species == 684:
                item_["ped"] = torch.from_numpy(env_data[19:, :, :])

        t = trsfs.Compose(self.transform)
        item_ = t(item_)

        for e in self.env:
            
            item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=-3).float()

        species = load_file(os.path.join(self.data_base_dir, "corrected_targets", hotspot_id + '.json'))

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
            targ[targ > 0] = 1
            item_["target"] = torch.Tensor(targ)

        elif self.target == "log":
            if not self.subset is None:
                item_["target"] = np.array(species["probs"])[self.subset]
            else:
                item_["target"] = species["probs"]

        else:
            raise NameError("type of target not supported, should be probs or binary")

        item_["num_complete_checklists"] = species["num_complete_checklists"]

        # item_["state_id"] = self.df["state_id"][index]

        if self.use_loc:
            if self.loc_type == "latlon":
                lon, lat = torch.Tensor([item_["lon"]]), torch.Tensor([item_["lat"]])
                loc = torch.cat((lon, lat)).unsqueeze(0)
                loc = encode_loc(convert_loc_to_tensor(loc))
                item_["loc"] = loc

        item_["hotspot_id"] = hotspot_id

        return item_


class EbirdSpeciesEnvDataset(VisionDataset):
    def __init__(self,
                 df_paths,
                 bands,
                 env,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 mode: Optional[str] = "train",
                 datatype="refl",
                 target="probs",
                 subset=None, use_loc=False, loc_type=None) -> None:
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
        self.subset = get_subset(subset, num_species)
        self.use_loc = use_loc
        self.loc_type = loc_type
        # self.speciesA = get_subset("songbirds")

    def __len__(self):
        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:

        # print(get_path(self.df, index, 'landuse').suffix == ".npy")
        band_npy = [(b, get_path(self.df, index, b)) for b in self.bands if
                    get_path(self.df, index, b).suffix == ".npy"]
        env_npy = [(b, get_path(self.df, index, b)) for b in self.env if get_path(self.df, index, b).suffix == ".npy"]
        item_ = {}

        assert len(band_npy) > 0, "No item to fetch"

        if self.type == "img":
            npy_data = get_img_bands(band_npy)
        else:
            bands = [load_file(band) for (_, band) in band_npy]
            npy_data = np.stack(bands, axis=1).astype(np.int32)

        for (b, band) in env_npy:
            item_[b] = torch.from_numpy(load_file(band))

        item_["sat"] = torch.from_numpy(npy_data)
        # item_["sat"] = item_["sat"]/ torch.amax(item_["sat"], dim=(-2,-1), keepdims=True)

        if self.transform:
            item_ = self.transform(item_)

        item_["env"] = torch.cat([item_[b] for (b, band) in env_npy], dim=-3)

        if "species" in self.df.columns:
            # add target
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
                targ[targ > 0] = 1
                item_["target"] = torch.Tensor(targ)

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
            # add metadata information (hotspot info)
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
                # print(item_.keys())
        return item_
