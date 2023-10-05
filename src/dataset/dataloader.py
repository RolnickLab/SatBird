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
                 targets_folder="corrected_targets",
                 env_data_folder="environmental",
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
