import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
import yaml
from PIL import Image
from addict import Dict
from typing import Optional, Union
from typing import Tuple

comet_kwargs = {
    "auto_metric_logging": False,
    "parse_args": True,
    "log_env_gpu": True,
    "log_env_cpu": True,
    "display_summary_level": 0,
}

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)


def json_load(file_path):
    """
    loads a json file given path
    """
    with open(file_path, "r") as f:
        return json.load(f)


def yaml_load(file_path):
    """
    loads a yaml file given path
    """
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def is_image_file(filename):
    """Check that a file's name points to a known image format"""
    if isinstance(filename, Path):
        return filename.suffix in IMG_EXTENSIONS
    return Path(filename).suffix in IMG_EXTENSIONS


def load_geotiff_visual(file):
    img = tiff.imread(file).astype(np.float32)

    img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

    return img


def load_geotiff(file):
    img = tiff.imread(file)
    new_band_order = [2, 1, 0, 3]  # r, g, b, nir
    img = img[:, :, new_band_order].astype(float)
    img = np.reshape(img, (img.shape[2], img.shape[0], img.shape[1]))

    return img


def load_file(file_path):
    if is_image_file(file_path):
        return Image.open(file_path)
    elif file_path.split('.')[-1] == "yaml":
        return yaml_load(file_path)
    elif file_path.split('.')[-1] == "json":
        return json_load(file_path)
    elif file_path.split('.')[-1] == "npy":
        return np.load(file_path)
    elif file_path.split('.')[-1] == "tif":
        if 'visual' in str(file_path):
            return load_geotiff_visual(file_path)
        else:
            return load_geotiff(file_path)


def get_path(df, index, band):
    return Path(df.iloc[index][band])


def encode_loc(loc, concat_dim=1, elev=False):
    # loc is (lon, lat ) or (lon, lat, elev)
    loc = convert_loc_to_tensor(loc)
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
        return np.arange(0, num_species)
    else:
        if os.path.isfile(subset):
            return np.load(subset).astype(int)
        else:
            raise TypeError("Only npy files are allowed")


def copy_run_files(opts: Dict) -> None:
    """
    Copy the opts's sbatch_file to output_path
    Args:
        opts (addict.Dict): options
    """
    if opts.sbatch_file:
        p = Path(opts.sbatch_file)
        if p.exists():
            o = Path(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)
    if opts.exp_file:
        p = Path(opts.exp_file)
        if p.exists():
            o = Path(opts.output_path)
            if o.exists():
                shutil.copyfile(p, o / p.name)


def merge(
        source: Union[dict, Dict], destination: Union[dict, Dict]
) -> Union[dict, Dict]:
    """
    run me with nosetests --with-doctest file.py
    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == {
        'first' : {
            'all_rows' : { '
                pass' : 'dog',
                'fail' : 'cat',
                'number' : '5'
            }
        }
    }
    True
    """
    for key, value in source.items():
        try:
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                merge(value, node)
            else:
                if isinstance(destination, dict):
                    destination[key] = value
                else:
                    destination = {key: value}
        except TypeError as e:
            print(traceback.format_exc())
            print(">>>", source)
            print(">>>", destination)
            print(">>>", key)
            print(">>>", value)
            raise Exception(e)

    return destination


def load_opts(
        path: Optional[Union[str, Path]] = None,
        default: Optional[Union[str, Path, dict, Dict]] = None,
        commandline_opts: Optional[Union[Dict, dict]] = None,
) -> Dict:
    """Loadsize a configuration Dict from 2 files:
    1. default files with shared values across runs and users
    2. an overriding file with run- and user-specific values
    Args:
        path (pathlib.Path): where to find the overriding configuration
            default (pathlib.Path, optional): Where to find the default opts.
            Defaults to None. In which case it is assumed to be a default config
            which needs processing such as setting default values for lambdas and gen
            fields
    Returns:
        addict.Dict: options dictionnary, with overwritten default values
    """

    if path is None and default is None:
        path = Path(__file__).parent.parent / "configs" / "defaults.yaml"

    if path:
        path = Path(path).resolve()

    if default is None:
        default_opts = {}
    else:
        if isinstance(default, (str, Path)):
            with open(default, "r") as f:
                default_opts = yaml.safe_load(f)
        else:
            default_opts = dict(default)

    if path is None:
        overriding_opts = {}
    else:
        with open(path, "r") as f:
            overriding_opts = yaml.safe_load(f) or {}

    opts = Dict(merge(overriding_opts, default_opts))

    if commandline_opts is not None and isinstance(commandline_opts, dict):
        opts = Dict(merge(commandline_opts, opts))

    return set_data_paths(opts)


def set_data_paths(opts: Dict) -> Dict:
    """Update the data files paths in data.files.train and data.files.val
    from data.files.base
    Args:
        opts (addict.Dict): options
    Returns:
        addict.Dict: updated options
    """

    for mode in ["train", "val", "test"]:
        if opts.data.files.base:
            opts.data.files[mode] = str(
                Path(opts.data.files.base) / opts.data.files[mode]
            )
            assert Path(
                opts.data.files[mode]
            ).exists(), "Cannot find {}".format(str(opts.data.files[mode]))

    return opts


class BoundingBox(Tuple[float, float, float, float, float, float]):
    """Data class for indexing spatiotemporal data.
    Attributes:
        minx (float): western boundary
        maxx (float): eastern boundary
        miny (float): southern boundary
        maxy (float): northern boundary
        mint (float): earliest boundary
        maxt (float): latest boundary
    """

    def __new__(
            cls,
            minx: float,
            maxx: float,
            miny: float,
            maxy: float,
            mint: float,
            maxt: float,
    ) -> "BoundingBox":
        """Create a new instance of BoundingBox.
        Args:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary
        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)
        """
        if minx > maxx:
            raise ValueError(f"Bounding box is invalid: 'minx={minx}' > 'maxx={maxx}'")
        if miny > maxy:
            raise ValueError(f"Bounding box is invalid: 'miny={miny}' > 'maxy={maxy}'")
        if mint > maxt:
            raise ValueError(f"Bounding box is invalid: 'mint={mint}' > 'maxt={maxt}'")

        # Using super() doesn't work with mypy, see:
        # https://stackoverflow.com/q/60611012/5828163
        return tuple.__new__(cls, [minx, maxx, miny, maxy, mint, maxt])

    def __init__(
            self,
            minx: float,
            maxx: float,
            miny: float,
            maxy: float,
            mint: float,
            maxt: float,
    ) -> None:
        """Initialize a new instance of BoundingBox.
        Args:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary
        """
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.mint = mint
        self.maxt = maxt

    def __getnewargs__(self) -> Tuple[float, float, float, float, float, float]:
        """Values passed to the ``__new__()`` method upon unpickling.
        Returns:
            tuple of bounds
        """
        return self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt

    def __repr__(self) -> str:
        """Return the formal string representation of the object.
        Returns:
            formal string representation
        """
        return (
            f"{self.__class__.__name__}(minx={self.minx}, maxx={self.maxx}, "
            f"miny={self.miny}, maxy={self.maxy}, mint={self.mint}, maxt={self.maxt})"
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether or not two bounding boxes intersect.
        Args:
            other: another bounding box
        Returns:
            True if bounding boxes intersect, else False
        """
        return (
                self.minx <= other.maxx
                and self.maxx >= other.minx
                and self.miny <= other.maxy
                and self.maxy >= other.miny
                and self.mint <= other.maxt
                and self.maxt >= other.mint
        )


def disambiguate_timestamp(date_str: str, format: str) -> Tuple[float, float]:
    """Disambiguate partial timestamps.
    TorchGeo stores the timestamp of each file in a spatiotemporal R-tree. If the full
    timestamp isn't known, a file could represent a range of time. For example, in the
    CDL dataset, each mask spans an entire year. This method returns the maximum
    possible range of timestamps that ``date_str`` could belong to. It does this by
    parsing ``format`` to determine the level of precision of ``date_str``.
    Args:
        date_str: string representing date and time of a data point
        format: format codes accepted by :meth:`datetime.datetime.strptime`
    Returns:
        (mint, maxt) tuple for indexing
    """
    mint = datetime.strptime(date_str, format)

    # TODO: This doesn't correctly handle literal `%%` characters in format
    # TODO: May have issues with time zones, UTC vs. local time, and DST
    # TODO: This is really tedious, is there a better way to do this?

    if not any([f"%{c}" in format for c in "yYcxG"]):
        # No temporal info
        return 0, sys.maxsize
    elif not any([f"%{c}" in format for c in "bBmjUWcxV"]):
        # Year resolution
        maxt = datetime(mint.year + 1, 1, 1)
    elif not any([f"%{c}" in format for c in "aAwdjcxV"]):
        # Month resolution
        if mint.month == 12:
            maxt = datetime(mint.year + 1, 1, 1)
        else:
            maxt = datetime(mint.year, mint.month + 1, 1)
    elif not any([f"%{c}" in format for c in "HIcX"]):
        # Day resolution
        maxt = mint + timedelta(days=1)
    elif not any([f"%{c}" in format for c in "McX"]):
        # Hour resolution
        maxt = mint + timedelta(hours=1)
    elif not any([f"%{c}" in format for c in "ScX"]):
        # Minute resolution
        maxt = mint + timedelta(minutes=1)
    elif not any([f"%{c}" in format for c in "f"]):
        # Second resolution
        maxt = mint + timedelta(seconds=1)
    else:
        # Microsecond resolution
        maxt = mint + timedelta(microseconds=1)

    mint -= timedelta(microseconds=1)
    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()
