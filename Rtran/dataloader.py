import random
from typing import Any, Callable, Dict, Optional
import os

from src.dataset.geo import VisionDataset
from src.dataset.utils import load_file, get_subset

import torch
from torchvision import transforms as trsfs
import numpy as np


def get_unkown_mask_indices(num_labels, known_labels, max_unknown=0.5):
    # sample random number of known labels during training; in testing, everything is unknown
    # TODO: use structured masking instead of random masking
    if known_labels > 0:
        random.seed()
        num_unknown = random.randint(0, int(num_labels * max_unknown))
        unk_mask_indices = random.sample(range(num_labels), num_unknown)
    else:
        # for testing, everything is unknown
        unk_mask_indices = np.arange(0, num_labels)

    return unk_mask_indices


class SDMVisionMaskedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 targets_folder="corrected_targets", subset=None, num_species=684) -> None:
        """
        df_paths: dataframe with paths to data for each hotspot
        data_base_dir: base directory for data
        env: list eof env data to take into account [ped, bioclim]
        transforms: transforms functions
        mode : train|val|test
        datatype: "refl" (reflectance values ) or "img" (image dataset)
        target : "probs" or "binary"
        subset : None or list of indices of the indices of species to keep
        """

        super().__init__()
        self.df = df
        self.data_base_dir = data_base_dir
        self.transform = transforms
        self.env = env
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.data_type = datatype
        self.targets_folder = targets_folder
        self.subset = get_subset(subset, num_species)
        self.num_species = num_species

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item_ = {}

        hotspot_id = self.df.iloc[index]['hotspot_id']

        # loading satellite image
        if self.data_type == 'img':
            img_path = os.path.join(self.data_base_dir, "images_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, "images", hotspot_id + '.tif')

        img = load_file(img_path)
        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

        # loading environmental rasters, if any
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, "environmental_data", hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * self.env_var_sizes[i - 1]
            e_i = self.env_var_sizes[i] + s_i
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        # applying transforms
        if self.transform:
            t = trsfs.Compose(self.transform)
            item_ = t(item_)

        # concatenating env rasters, if any, with satellite image
        for e in self.env:
            item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=-3).float()

        item_["sat"] = item_["sat"].squeeze(0)
        # constructing targets
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder, hotspot_id + '.json'))

        if not self.subset is None:
            item_["target"] = np.array(species["probs"])[self.subset]
        else:
            item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        self.known_labels = 0
        if self.mode == "train":
            self.known_labels = 100
        unk_mask_indices = get_unkown_mask_indices(num_labels=self.num_species, known_labels=self.known_labels)
        mask = item_["target"].clone()
        mask[mask != 0] = 1
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)
        item_["mask"] = mask

        # meta data
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        item_["hotspot_id"] = hotspot_id
        return item_
