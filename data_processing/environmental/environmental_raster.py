#code adapted from https://github.com/maximiliense/GLC/

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
"""
MEANS = {"bio_1":11.994303908599168, "bio_2":12.162265840542254, "bio_3":36.942481755599964, "bio_4":805.720459451272, "bio_5":29.44890890443498, "bio_6":-4.561721325088685, "bio_7":34.01063026068749, "bio_8":15.816412685879353, "bio_9":7.808452188833704,
    "bio_10":21.77499490648628, "bio_11":1.9399000365976449, "bio_12":902.9704985980738, "bio_13":114.61111788370108, "bio_14":42.027672802633184, "bio_15":37.11493780524879, "bio_16":315.34206997439964, "bio_17":145.09703766914544,
    "bio_18":231.19724491039864, "bio_19":220.06619529440448, "bdticm": 2230.563616964524,
 "bldfie": 1374.6855161376368,
 "cecsol": 20.45478794345159,
 "clyppt":19.0492131234996,
 "orcdrc": 31.119631901840492,
 "phihox": 61.24246465724193,
 "sltppt": 36.68711656441718,
  "sndppt" : 44.25620165377434 }
"""

"""
winter
"""
MEANS = {'bio_1': 12.740144948743144,
 'bio_2': 12.230345398403946,
 'bio_3': 38.25656206187823,
 'bio_4': 770.5181806119864,
 'bio_5': 29.762035675902258,
 'bio_6': -3.3439501335222666,
 'bio_7': 33.10598581851715,
 'bio_8': 15.160649527422294,
 'bio_9': 10.074888284022924,
 'bio_10': 22.122542917204193,
 'bio_11': 3.1582102336961704,
 'bio_12': 920.5306976347024,
 'bio_13': 117.66142942532662,
 'bio_14': 43.12688924942362,
 'bio_15': 38.869635061973,
 'bio_16': 323.9086329092306,
 'bio_17': 147.8199982922039,
 'bio_18': 216.59713090257023,
 'bio_19': 244.02390914524807,
 'bdticm': 2121.0403494551024,
 'bldfie': 1393.9937854633883,
 'cecsol': 19.29955867783482,
 'clyppt': 18.963073043321625,
 'orcdrc': 27.870665585877692,
 'phihox': 61.041069981086196,
 'sltppt': 36.61019544267315,
 'sndppt': 44.42637125101324}

bioclimatic_raster_names = [
    "bio_1", "bio_2", "bio_3", "bio_4", "bio_5", "bio_6", "bio_7", "bio_8", "bio_9",
    "bio_10", "bio_11", "bio_12", "bio_13", "bio_14", "bio_15", "bio_16", "bio_17",
    "bio_18", "bio_19"
]
pedologic_raster_names = [
    "bdticm", "bldfie", "cecsol", "clyppt", "orcdrc", "phihox", "sltppt", "sndppt"
]
raster_names =bioclimatic_raster_names + pedologic_raster_names


class Raster(object):
    """
    Handles the loading and the patch extraction for a single raster
    """

    def __init__(self, path, country, size=256, nan=np.nan, out_of_bounds="error"):
        """Loads a GeoTIFF file containing an environmental raster

        Parameters
        ----------
        path : string / pathlib.Path
            Path to the folder containing all the rasters.
        country : string, either "FR" or "USA"
            Which country to load raster from.
        size : integer
            Size in pixels (size x size) of the patch to extract around each location.
        nan : float or None
            Value to use to replace missing data in original rasters, if None, leaves default values.
        out_of_bounds : string, either "error", "warn" or "ignore"
            If "error", raises an exception if the location requested is out of bounds of the rasters. Set to "warn" to only produces a warning and to "ignore" to silently ignore it and return a patch filled with missing data.
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(
                "path should be the path to a raster, given non-existant path: {}".format(
                    path
                )
            )

        self.path = path
        self.name = path.name
        self.size = size
        self.out_of_bounds = out_of_bounds
        self.nan = nan

        # Loading the raster
        filename = path / "{}_{}.tif".format(self.name, country)
        with rasterio.open(filename) as dataset:
            self.dataset = dataset
            self.raster = dataset.read(1).astype(np.float32)
            mask = self.dataset.read_masks(1).astype(bool)

        # Changes nodata to user specified value
        if nan:
            self.raster[np.isnan(self.raster)] = nan
            self.raster[~mask] = nan

        # setting the shape of the raster
        self.shape = self.raster.shape

    def _extract_patch(self, coordinates):
        """Extracts the patch around the given GPS coordinates.
        Avoid using this method directly.

        Parameter
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)

        Returns
        -------
        patch : 2d array of floats, [size, size], or single float if size == 1
            Extracted patch around the given coordinates.
        """
        row, col = self.dataset.index(coordinates[1], coordinates[0])

        if self.size == 1:
            # Environmental vector
            patch = self.raster[row, col]
        else:
            # FIXME: can it happen that part of the patch is outside the raster? (and how about the mask of the dataset?)
            half_size = int(self.size / 2)
            # FIXME: only way to trigger an exception? (slices don't)
            self.raster[row, col]
            patch = self.raster[
                row - half_size:row + half_size,
                col - half_size:col + half_size
            ]

        patch = patch[np.newaxis]
        return patch

    def __len__(self):
        """Number of bands in the raster (should always be equal to 1).

        Returns
        -------
        n_bands : integer
            Number of bands in the raster
        """
        return self.dataset.count

    def __getitem__(self, coordinates):
        """Extracts the patch around the given GPS coordinates.

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)

        Returns
        -------
        patch : 2d array of floats, [size, size], or single float if size == 1
            Extracted patch around the given coordinates.
        """
        try:
            return self._extract_patch(coordinates)
        except IndexError as e:
            if self.out_of_bounds == "error":
                raise e
            else:
                if self.out_of_bounds == "warn":
                    warnings.warn("GPS coordinates ({}, {}) out of bounds".format(*coordinates))

                if self.size == 1:
                    return np.array([self.nan], dtype=np.float32)
                else:
                    patch = np.empty((1, self.size, self.size), dtype=np.float32)
                    patch.fill(self.nan)
                    return patch

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "name: " + self.name + "\n"


class PatchExtractor(object):
    """
    Handles the loading and extraction of an environmental tensor from multiple rasters given GPS coordinates.
    """

    def __init__(self, root_path, country, size=256):
        """Constructor

        Parameters
        ----------
        root_path : string or pathlib.Path
            Path to the folder containing all the rasters.
        size : integer
            Size in pixels (size x size) of the patches to extract around each location.
        """
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise ValueError(
                "root_path should be the directory containing the rasters, given a non-existant path: {}".format(
                    root_path
                )
            )

        self.size = size

        #self.rasters_fr = []
        self.rasters_us = []
        self.country = "USA"

    def add_all_rasters(self, **kwargs):
        """Add all variables (rasters) available

        Parameters
        ----------
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        for raster_name in raster_names:
            self.append(raster_name, **kwargs)

    def add_all_bioclimatic_rasters(self, **kwargs):
        """Add all bioclimatic variables (rasters) available

        Parameters
        ----------
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        for raster_name in bioclimatic_raster_names:
            self.append(raster_name, **kwargs)

    def add_all_pedologic_rasters(self, **kwargs):
        """Add all pedologic variables (rasters) available

        Parameters
        ----------
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        for raster_name in pedologic_raster_names:
            self.append(raster_name, **kwargs)

    def append(self, raster_name, **kwargs):
        """Loads and appends a single raster to the rasters already loaded.

        Can be useful to load only a subset of rasters or to pass configurations specific to each raster.

        Parameters
        ----------
        raster_name : string
            Name of the raster to load, should be a subfolder of root_path.
        kwargs : dict
            Updates the default arguments passed to Raster (nan, out_of_bounds, etc.)
        """
        
        r_us = Raster(self.root_path / raster_name, self.country, size=self.size,  **kwargs)
        #nan = MEANS[raster_name]
      
        self.rasters_us.append(r_us)
       
    def clean(self):
        """Remove all rasters from the extractor.
        """
    
        self.rasters_us = []

    def _get_rasters_list(self, coordinates):
        """Returns the list of rasters from the appropriate country

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)

        Returns
        -------
        rasters : list of Raster objects
            All previously loaded rasters.
        """
        #if coordinates[1] > -10.0:
        #    return self.rasters_fr
        return(self.rasters_us)

    def __repr__(self):
        return str(self)

    def __str__(self):
        result = ""

        for rasters in [self.rasters_us]:
            for raster in rasters:
                result += "-" * 50 + "\n"
                result += str(raster)

        return result

    def __getitem__(self, coordinates):
        """Extracts the patches around the given GPS coordinates for all the previously loaded rasters.

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)

        Returns
        -------
        patch : 3d array of floats, [n_rasters, size, size], or 1d array of floats, [n_rasters,], if size == 1
            Extracted patches around the given coordinates.
        """
        rasters = self._get_rasters_list(coordinates)
        return np.concatenate([r[coordinates] for r in rasters])

    def __len__(self):
        """Number of variables/rasters loaded.

        Returns
        -------
        n_rasters : integer
            Number of loaded rasters
        """
        return len(self.rasters_us)

    def plot(self, coordinates, return_fig=False, n_cols=5, fig=None, resolution=1.0):
        """Plot an environmental tensor (only works if size > 1)

        Parameters
        ----------
        coordinates : tuple containing two floats
            GPS coordinates (latitude, longitude)
        return_fig : boolean
            If True, returns the created plt.Figure object
        n_cols : integer
            Number of columns to use
        fig : plt.Figure or None
            If not None, use the given plt.Figure object instead of creating a new one
        resolution : float
            Resolution of the created figure

        Returns
        -------
        fig : plt.Figure
            If return_fig is True, the used plt.Figure object
        """
        if self.size <= 1:
            raise ValueError("Plot works only for tensors: size must be > 1")

        rasters = self._get_rasters_list(coordinates)

        # Metadata are the name of the variables and the bounding boxes in latitude-longitude coordinates
        metadata = [
            (
                raster.name,
                [
                    coordinates[1] - (self.size // 2) * raster.dataset.res[0],
                    coordinates[1] + (self.size // 2) * raster.dataset.res[0],
                    coordinates[0] - (self.size // 2) * raster.dataset.res[1],
                    coordinates[0] + (self.size // 2) * raster.dataset.res[1],
                ],
            )
            for raster in rasters
        ]

        # Extracts the patch
        patch = self[coordinates]

        # Computing number of rows and columns
        n_rows = (patch.shape[0] + (n_cols - 1)) // n_cols

        if fig is None:
            fig = plt.figure(figsize=(n_cols * 6.4 * resolution, n_rows * 4.8 * resolution))

        axes = fig.subplots(n_rows, n_cols)
        axes = axes.ravel()

        for i, (ax, k) in enumerate(zip(axes, metadata)):
            p = np.squeeze(patch[i])
            im = ax.imshow(p, extent=k[1], aspect="equal", interpolation="none")

            ax.set_title(k[0], fontsize=20)
            fig.colorbar(im, ax=ax)

        for ax in axes[len(metadata):]:
            ax.axis("off")

        fig.tight_layout()

        if return_fig:
            return fig
