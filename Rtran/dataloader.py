import random
from typing import Any, Callable, Dict, Optional
import os

from src.dataset.geo import VisionDataset
from src.dataset.utils import load_file, get_subset

import torch
from torchvision import transforms as trsfs
import numpy as np


def get_unknown_mask_indices(num_labels, mode, max_unknown=0.5, absent_species=0, species_set=None, predict_family_of_species=-1):
    # sample random number of known labels during training; in testing, everything is unknown
    if mode == 'train':
        random.seed()
        if random.random() < 0.5 and absent_species == 0: # 50% of the time when butterflies are there, mask all butterflies
            unk_mask_indices = np.arange(species_set[0], species_set[0] + species_set[1])
        else:
            num_unknown = random.randint(0, int((num_labels - absent_species) * max_unknown))
            unk_mask_indices = random.sample(range(num_labels - absent_species), num_unknown)
    else:
        # for testing, everything is unknown
        unk_mask_indices = random.sample(range(num_labels - absent_species), int((num_labels- absent_species) * max_unknown))

        if predict_family_of_species != -1:
            # to predict butterflies only
            if predict_family_of_species == 1:
                unk_mask_indices = np.arange(species_set[0], species_set[0] + species_set[1])
            if predict_family_of_species == 0:
                # to predict birds only
                unk_mask_indices = np.arange(0, species_set[0])

    return unk_mask_indices


class SDMVisionMaskedDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 targets_folder="corrected_targets", targets_folder_2="butterfly_targets_2", env_data_folder="environmental",
                 maximum_unknown_labels_ratio=0.5, subset=None, num_species=670, species_set=None, predict_family=-1) -> None:
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
        self.env_data_folder = env_data_folder
        self.subset = get_subset(subset, num_species)
        self.num_species = num_species
        self.maximum_unknown_labels_ratio = maximum_unknown_labels_ratio

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
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder, hotspot_id + '.npy')
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

        if self.subset:
            item_["target"] = np.array(species["probs"])[self.subset]
        else:
            item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode, max_unknown=self.maximum_unknown_labels_ratio)
        mask = item_["target"].clone()
        mask[mask != 0] = 1
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)
        item_["mask"] = mask
        # meta data
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        item_["hotspot_id"] = hotspot_id
        return item_


class SDMVJointDataset(VisionDataset):
    def __init__(self, df, data_base_dir, env, env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, mode="train", datatype="refl",
                 targets_folder="corrected_targets", targets_folder_2="butterfly_targets_2", env_data_folder="environmental",
                maximum_unknown_labels_ratio=0.5, subset=None, num_species=670, species_set=None, predict_family=-1) -> None:
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
        self.targets_folder_2 = targets_folder_2
        self.env_data_folder = env_data_folder
        self.subset = get_subset(subset, num_species)
        self.num_species = num_species
        self.species_set = species_set
        self.maximum_unknown_labels_ratio = maximum_unknown_labels_ratio
        self.predict_family_of_species = predict_family

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
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder, hotspot_id + '.npy')
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
        species_2_to_exclude = 0
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder, hotspot_id + '.json'))
        if os.path.exists(os.path.join(self.data_base_dir, self.targets_folder_2, hotspot_id + '.json')):
            species_2 = load_file(os.path.join(self.data_base_dir, self.targets_folder_2, hotspot_id + '.json'))
        else:
            species_2 = {}
            species_2["probs"] = [-2] * self.species_set[1]
            species_2_to_exclude = self.species_set[1]

        species["probs"] = species["probs"] + species_2["probs"]

        if self.subset:
            item_["target"] = np.array(species["probs"])[self.subset]
        else:
            item_["target"] = species["probs"]
        item_["target"] = torch.Tensor(item_["target"])

        # constructing mask for R-tran
        unk_mask_indices = get_unknown_mask_indices(num_labels=self.num_species, mode=self.mode, max_unknown=self.maximum_unknown_labels_ratio,
                                                    absent_species=species_2_to_exclude, species_set=self.species_set, predict_family_of_species=self.predict_family_of_species)
        mask = item_["target"].clone()
        mask[mask > 0] = 1
        mask.scatter_(dim=0, index=torch.Tensor(unk_mask_indices).long(), value=-1.0)
        item_["mask"] = mask
        # print(item_["mask"].unique())
        # print(len(unk_mask_indices))
        # meta data
        item_["num_complete_checklists"] = species["num_complete_checklists"]
        item_["hotspot_id"] = hotspot_id
        return item_
