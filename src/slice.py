# slicing code 
## Necessary libs 
from pathlib import Path
import argparse
import os 
import numpy as np 
from geopandas.geodataframe import GeoDataFrame 
from skimage.util.shape import view_as_windows
import pandas as pd 
from shapely import geometry
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt 
import shapely


#squash function to return mean and std of a list
def squash(x):
    return (x - x.min()) / x.ptp()

## Create RGB composite
def create_rgb_composite(input_paths):
    image_data = []
    composite_downsample_factor = 1
    composite_norm_value = 15000

    for fn in input_paths:
        with rasterio.open(fn,'r') as raster:
            h = int(raster.height/composite_downsample_factor)
            w = int(raster.width/composite_downsample_factor)
            band_array = raster.read(1, out_shape=(1, h, w))
            raster.close()
            band_array = band_array / composite_norm_value
            image_data.append(band_array)

    rgb = np.dstack((image_data[0],image_data[1],image_data[2]))
    return rgb

def slice_tile(img, size=(512,512), overlap=6):
    """
    Slice a tile from an image
    """
    size_ = (size[0], size[1], img.shape[2])
    patches = view_as_windows(img, size_,step=size[0] - overlap)
    
    result = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
        result.append(patches[i,j,0])
    return result

def slices_metadata(imgf, img_path, size=(512, 512), overlap=6):
    """
    Write geometry and source information to metadata
    """
    meta = slice_polys(imgf, size, overlap)
    # print(img_path)
    # meta["img_source"] = img_path
    return meta


def slice_polys(imgf, size=(512, 512), overlap=6):
    """
    Get Polygons Corresponding to Slices
    """
    ix_row = np.arange(0, imgf.meta["height"], size[0] - overlap)
    ix_col = np.arange(0, imgf.meta["width"], size[1] - overlap)
    lats = np.linspace(imgf.bounds.bottom, imgf.bounds.top, imgf.meta["height"])
    longs = np.linspace(imgf.bounds.left, imgf.bounds.right, imgf.meta["width"])

    polys = []
    for i in range(len(ix_row) - 1):
        for j in range(len(ix_col) - 1):
            box = shapely.geometry.box(
                longs[ix_col[j]],
                lats[ix_row[i]],
                longs[ix_col[j + 1]],
                lats[ix_row[i + 1]],
            )
            polys.append(box)

    return GeoDataFrame(geometry=polys, crs=imgf.meta["crs"].to_string())


def write_pair_slices(img_path, out_dir, out_base="slice",**kwargs):
    
    """ Write sliced images and masks to numpy arrays
    Args:
        img_path(String): the path to the raw image tiff
        mask_path(String): the paths to the mask array
        border_path(String): teh path to the border array
        output_base(String): The basenames for all the output numpy files
        out_dir(String): The directory to which all the results will be stored
    Returns:
        Writes a csv to metadata path
    """

    imgf = rasterio.open(img_path[0])
    print(imgf)
    img = create_rgb_composite(img_path)

    img_slices = slice_tile(img, **kwargs)
   
    metadata = slices_metadata(imgf, img_path, **kwargs)

    # loop over slices for individual tile
    slice_stats = []
    for k in tqdm(range(len(img_slices))):
        img_slice_path = Path(out_dir, f"{out_base}_img_{k:03}.npy")
        np.save(img_slice_path, img_slices[k])

        # update metadata
        stats = {"img_slice": str(img_slice_path)}
        img_slice_mean = np.nan_to_num(img_slices[k]).mean()
        stats.update({"img_mean": img_slice_mean})
        slice_stats.append(stats)

    slice_stats = pd.DataFrame(slice_stats)
    return pd.concat([metadata, slice_stats], axis=1)


def plot_slices(slice_dir, processed=False, n_cols=3, div=3000, n_examples=5):
    """Helper to plot slices in a directory
    """
    files = list(Path(slice_dir).glob("*img*npy"))
    _, ax = plt.subplots(n_examples, n_cols, figsize=(15,15))
    for i in range(n_examples):
        index = np.random.randint(0, len(files))
        img = np.load(files[index])
        # mask = np.load(str(files[index]).replace("img", "mask"))

        if not processed:
            ax[i, 0].imshow(img)
        else:
            ax[i, 0].imshow(squash(img))

    return ax

# main
if __name__ == "__main__":
    ## Helper code
    paths = {}
    data_dir = "data"
    input_folder = data_dir
    input_paths = list(Path(input_folder).glob("*.tif*"))
    img_path = input_paths[0]
    out_dir="output"
    metadata_ = write_pair_slices(input_paths, out_dir)
    out_path = Path(out_dir, "patches.geojson")
    metadata_.to_file(out_path, index=False, driver="GeoJSON")