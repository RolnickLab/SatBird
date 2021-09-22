import abc
import functools
import glob
import math
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.merge
import torch
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset

from .utils import BoundingBox, disambiguate_timestamp

Dataset.__module__ = "torch.utils.data"


class GeoDataset(Dataset[Dict[str, Any]], abc.ABC):
    crs:CRS 
    res:float


    def __init__(self, transforms: Optional[Callable[[Dict[str, Any]], Dict [str, Any]]] = None,) -> None:
        self.transforms = transforms
        self.index = Index(interleaved=False, properties=Property(dimension=3))

    
    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> List[Dict[str, Any]]:
        pass

    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: GeoDataset
    bbox: {self.bounds}"""

    @property

    def bounds(self) -> BoundingBox:
        """Bounds of the index.
        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)


class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files
    """
    filename_glob = "*"
    filename_regex = ".*"
    date_format = "%Y%m%d"
    is_image=True
    separate_files = False
    all_bands: List[str]=[]
    rgb_bands: List[str]=[]
    stretch = False
    cmap: Dict[int, Tuple[int, int, int]] = {}

    def __init__(
        self,
        root:str,
        crs: Optional[CRS] = None,
        res: Optional[float]=None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict [str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new dataset"""
        super().__init__(transforms)

        self.root = root
        self.cache = cache

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            print(i)
            match = re.match(filename_regex, os.path.basename(filepath))
            
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # print(src)
                        # See if file has a color map
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        self.crs = cast(CRS, crs)
        self.res = cast(float, res)


    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.
        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            sample of image/mask and metadata at that index
        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]
        

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: List[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in getattr(self, "bands", self.all_bands):
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "date" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                        if "resolution" in match.groupdict():
                            start = match.start("resolution")
                            end = match.end("resolution")
                            filename = filename[:start] + "*" + filename[end:]
                    filepath = glob.glob(os.path.join(directory, filename))[0]
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.stack(data_list)
        else:
            data = self._merge_files(filepaths, query)

        key = "image" if self.is_image else "mask"
        sample = {
            key: data,
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load and merge one or more files.
        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        dest, _ = rasterio.merge.merge(vrt_fhs, bounds, self.res)
        dest = dest.astype(np.int32)

        tensor: Tensor = torch.tensor(dest)  # type: ignore[attr-defined]
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.
        Args:
            filepath: file to load and warp
        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.
        Args:
            filepath: file to load and warp
        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src

    def plot(self, data: Tensor) -> None:
        """Plot a data sample.
        Args:
            data: the data to plot
        Raises:
            AssertionError: if ``is_image`` is True and ``data`` has a different number
                of channels than expected
        """
        array = data.squeeze().numpy()

        if self.is_image:
            bands = getattr(self, "bands", self.all_bands)
            assert array.shape[0] == len(bands)

            # Only plot RGB bands
            if bands and self.rgb_bands:
                indices = np.array([bands.index(band) for band in self.rgb_bands])
                array = array[indices]

            # Convert from CxHxW to HxWxC
            array = np.rollaxis(array, 0, 3)

        if self.cmap:
            # Convert from class labels to RGBA values
            cmap = np.array([self.cmap[i] for i in range(len(self.cmap))])
            array = cmap[array]

        if self.stretch:
            # Stretch to the range of 2nd to 98th percentile
            per02 = np.percentile(array, 2)  # type: ignore[no-untyped-call]
            per98 = np.percentile(array, 98)  # type: ignore[no-untyped-call]
            array = (array - per02) / (per98 - per02)
            array = np.clip(array, 0, 1)

        # Plot the data
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()


class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips.
    """
    

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and labels at that index
        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.
        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""
