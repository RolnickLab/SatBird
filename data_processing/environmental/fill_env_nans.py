from __future__ import annotations

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
from scipy.interpolate import interp2d
import glob

from tqdm import tqdm
N_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))



H, W = 50,50
# Example data with NaN values
x = np.linspace(0, H-1, H)
y = np.linspace(0, W-1, W)
xi, yi = np.meshgrid(x, y)

def bilinear_interpolation(elem, nans):
    # Perform bilinear nterpolation
    f = interp2d(xi[~nans], yi[~nans], elem[~nans], kind='linear', bounds_error=False )
    return f(x, y)


def process_raster(file):
    r = np.load(file)
    u = r.copy()
    for i,elem in enumerate(r): 
        nans = np.isnan(elem)
        nonnan = ~np.isnan(elem)
        if nans.all():
                #all values are nan, need to fill with something
                #TODO
            print(file)
            continue #for now
        elif nans.any() and nonnan.any():
                # Define the coordinates of NaN values
                # Create a function for bicubic interpolation with extrapolation
            try:
                arr = bilinear_interpolation(elem, nans)
                arr[~nans] = elem[~nans]
                u[i] = arr
            except:
                #for example when only 2 points are available, interpolation is not possible with linear
                u[i]=r[i]
                print(os.path.basename(file))
    return(u, file)


        
if __name__=="__main__":
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader_chunk_size",
        type=int,
        default=100,
        help="Read the CSV file this number of lines at a time.",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=N_CPUS,
        help="Number of processes to use for parallel preprocessing.",
    )
    parser.add_argument(
        "--worker_chunk_size",
        type=int,
        default=512,
        help="Number of tasks to send to each worker process at a time.",
    )
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--range", type=int, default=10000)
    
    args = parser.parse_args()
    # Might get even better results by tuning these parameters here, but this works fine for now.
    n_processes = args.n_processes
    rasters = glob.glob("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_winter/environmental/*") # '/network/projects/_groups/ecosystem-
    reader_chunk_size = args.reader_chunk_size

    worker_chunk_size = args.worker_chunk_size


    print(f"Using {n_processes=}, {reader_chunk_size=} and {worker_chunk_size=}")

    index = args.index - 1
    range = args.range

    starting_index = index * range
    end_index = (index+1) * range
    print(starting_index, end_index)
    all_data =rasters[starting_index:end_index]

                
    files_iterator = tqdm(
        all_data,
        desc="iterating over rasters",
        unit="Files",
        )
    with mp.Pool(processes=n_processes) as pool:
        for data, file in pool.map(functools.partial(process_raster),
            files_iterator,):
            np.save(os.path.join("/network/projects/_groups/ecosystem-embeddings/SatBird_data_v2/USA_winter/environmental_filled/", os.path.basename(file)), data)
                
    print("Done")
