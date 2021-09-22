from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from dataset.geo import RasterDataset, VisionDataset
from dataset.sampler import RandomGeoSampler

from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor
import numpy as np
from PIL import Image


from rasterio.crs import CRS
import os

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

class EbirdRaster(RasterDataset):
    filename_glob = "*.npy*"
    separate_files = True
    filename_regex = r"""
        ^.*npy*$
    """

    # Plotting
    all_bands = ["r", "g", "b", "nir", "rgb"]
    rgb_bands = ["r", "g", "b"]


class EbirdVisionDataset(VisionDataset):
    def __init__(self) -> None:
        super().__init__()
        self.datadir = datadir
        self.transform = transform
        all_images = os.listdir(self.datadir)
        self.total_images = len(all_images)

    def __len__(self) -> int:
        return self.total_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_loc = os.path.join(self.datadir, self.total_images[index])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return {"image": tensor_image}


if __name__ == "__main__":
    ebird = EbirdVisionDataset("../toydata", transform=Identity())
    dataset = ebird
    sampler = RandomGeoSampler(ebird.index, size=1000, length=10)
    dataloader = DataLoader(dataset, sampler=sampler)

