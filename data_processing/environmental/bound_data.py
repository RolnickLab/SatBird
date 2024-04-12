"""
post-processing to environmental data after extraction
This assumes the rasters have been originally extracted to a folder named "environmental"
and filled with interpolation using fill_env_nans.py, and saved to a folder named "environmental_filled".
The remaining NaN values in the rasters are filled with the mean of each environmental variable over the training set and the filled rasters are saved to "environmental_temp"
The min and max ranges of the original (not filled)rasters are computed and used to bound the data, and saved in a folder names "environmental_temp_2"

We subsequently renamed "environmental_temp_2" to "environmental" and this is what is in the released dataset
"""
import argparse
import functools
import itertools
import json
import multiprocessing as mp
import os
import os.path
from pathlib import Path
import shutil
import csv
import numpy as np
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import glob
import shutil


def bound_env_data(root_dir, mini, maxi):
    """
    bound env data after the interpolation
    """

    rasters = glob.glob(root_dir + "/environmental_temp/*.npy")  # '/network/projects/_groups/ecosystem-

    for raster_file in tqdm(rasters):
        file_name = os.path.basename(raster_file)
        try:
            arr = np.load(raster_file, allow_pickle=True)
            for i, elem in enumerate(arr):
                elem = elem.clip(mini[i], maxi[i])
                arr[i] = elem
            np.save(os.path.join(root_dir + "/environmental_bounded_2",
                file_name), arr)

        except:
            print(raster_file)


def fill_nan_values(root_dir, dataframe_name="all_summer_hotspots_withnan.csv"):
    """
    fill values that still have nans after interpolation with mean point values
    """
    rasters = glob.glob(os.path.join(root_dir, "environmental_filled", "*.npy"))
    dst = os.path.join(root_dir, "environmental_temp")

    train_df = pd.read_csv(os.path.join(root_dir, dataframe_name))

    bioclim_env_column_names = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5',
                                'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12',
                                'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
    ped_env_column_names = ['bdticm', 'bldfie', 'cecsol', 'clyppt', 'orcdrc', 'phihox', 'sltppt', 'sndppt']

    env_column_names = bioclim_env_column_names + ped_env_column_names
    env_means = [train_df[env_column_name].mean(skipna=True) for env_column_name in env_column_names]
    count = 0
    print(len(rasters))

    for raster_file in tqdm(rasters):
        file_name = os.path.basename(raster_file)
        try:
            arr = np.load(raster_file)
            for i, elem in enumerate(arr):
                nans = np.isnan(elem)
                if nans.all() or nans.any():
                    elem[nans] = env_means[i]
                    arr[i] = elem
                    count += 1
            np.save(dst + "/" + file_name , arr)
        except:
            print(raster_file)

    print("Number of rasters that have been filles: ", count) #43


def compute_min_max_ranges(root_dir):
    """
    computes minimum and maximum of env data
    """
    rasters = glob.glob(os.path.join(root_dir, "environmental", "*.npy"))

    nan_count = 0

    mins = np.ones(27) * 1e9
    maxs = np.ones(27) * -1e9

    for raster_file in tqdm(rasters):
        try:
            arr = np.load(raster_file)
            for i, elem in enumerate(arr):
                mins[i] = min(np.nanmin(arr[i]), mins[i])
                maxs[i] = max(np.nanmax(arr[i]), maxs[i])
                nans = np.isnan(elem)
                if nans.all() or nans.any():
                    nan_count += 1
        except:
            print(raster_file)

    print("Number of nans: ", nan_count) #43
    print("Minimum values: ", mins)
    print("Maximum values: ",maxs)

    return mins, maxs


def move_missing_file(root_dir):
    """
    a utility that have been used once to move files around
    """
    rasters_origin = glob.glob(os.path.join(root_dir, "environmental", "*.npy"))

    names = [os.path.basename(x) for x in glob.glob(os.path.join(root_dir, "environmental_bounded", "*.npy"))]
    for raster in rasters_origin:
        file_name = os.path.basename(raster)
        if file_name not in names:
            shutil.copyfile(raster, os.path.join(root_dir, "environmental_temp", file_name))


def remove_files(root_dir):
    rasters_origin = glob.glob(os.path.join(root_dir, "environmental_bounded", "*.npy"))

    names = [os.path.basename(x) for x in glob.glob(os.path.join(root_dir, "environmental_temp", "*.npy"))]
    for raster in rasters_origin:
        file_name = os.path.basename(raster)
        if file_name in names:
            os.remove(raster)


if __name__ == '__main__':
    
    root_dir = "/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer"
    fill_nan_values(root_dir=root_dir)
    # move_missing_file(root_dir=root_dir)
    # remove_files(root_dir=root_dir)
    mini, maxi = compute_min_max_ranges(root_dir=root_dir)
    bound_env_data(root_dir=root_dir, mini=mini, maxi=maxi)
    
    
