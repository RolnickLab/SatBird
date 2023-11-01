# main data-loader
import os
from typing import Any, Callable, Dict, Optional
import numpy as np
import torch
from torchvision import transforms as trsfs

from src.dataset.geo import VisionDataset
from src.dataset.utils import get_subset, load_file, encode_loc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EbirdVisionDataset(VisionDataset):

    def __init__(self,
                 df_paths,
                 data_base_dir,
                 bands,
                 env,
                 env_var_sizes,
                 transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 mode: Optional[str] = "train",
                 datatype="refl",
                 target="probs",
                 targets_folder="targets",
                 env_data_folder="environmental_bounded",
                 images_folder="images",
                 subset=None,
                 use_loc=False,
                 res=[],
                 loc_type=None,
                 num_species=684) -> None:
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
        self.env_var_sizes = env_var_sizes
        self.mode = mode
        self.type = datatype
        self.target = target
        self.targets_folder = targets_folder
        self.env_data_folder = env_data_folder
        self.images_folder = images_folder
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

        # satellite image
        if self.type == 'img':
            img_path = os.path.join(self.data_base_dir, self.images_folder + "_visual", hotspot_id + '_visual.tif')
        else:
            img_path = os.path.join(self.data_base_dir, self.images_folder, hotspot_id + '.tif')

        img = load_file(img_path)
        sats = torch.from_numpy(img).float()
        item_["sat"] = sats

        assert len(self.env) == len(self.env_var_sizes), "env variables sizes must be equal to the size of env vars specified`"
        # env rasters
        for i, env_var in enumerate(self.env):
            env_npy = os.path.join(self.data_base_dir, self.env_data_folder, hotspot_id + '.npy')
            env_data = load_file(env_npy)
            s_i = i * self.env_var_sizes[i - 1]
            e_i = self.env_var_sizes[i] + s_i
            item_[env_var] = torch.from_numpy(env_data[s_i:e_i, :, :])

        t = trsfs.Compose(self.transform)
        item_ = t(item_)

        for e in self.env:
            item_["sat"] = torch.cat([item_["sat"], item_[e]], dim=-3).float()

        # target labels
        species = load_file(os.path.join(self.data_base_dir, self.targets_folder, hotspot_id + '.json'))
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

        if self.use_loc:
            if self.loc_type == "latlon":
                lon, lat = torch.Tensor([item_["lon"]]), torch.Tensor([item_["lat"]])
                loc = torch.cat((lon, lat)).unsqueeze(0)
                loc = encode_loc(loc)
                item_["loc"] = loc

        item_["hotspot_id"] = hotspot_id

        return item_
