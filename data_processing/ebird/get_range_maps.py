from __future__ import annotations
import pickle
import pandas as pd
import os
from collections import defaultdict
import geopandas as gdp
from shapely.geometry import Point, shape
from tqdm.auto import tqdm
from functools import partial

import argparse
import multiprocessing as mp
import os.path
from pathlib import Path

SCRATCH = Path(os.environ["SCRATCH"])
SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
N_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 12))


#CHANGE
output_dir = "/home/mila/h/hager.radi/scratch/range_maps"

# DON'T CHANGE
save_dir = "/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/range_maps"
all_data_path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_clean_splits.csv"
species_list_path = "/network/projects/ecosystem-embeddings/ebird_dataset/species_data.csv"
shape_file_path = "/network/projects/_groups/ecosystem-embeddings/ebird_dataset/bigdata.shp"
def process_loc(row, shape_file, species):
    # No need to do that, since we're using the `parse_dates` arg of the `read_csv` function.
    # chunk["OBSERVATION DATE"] = pd.to_datetime(chunk["OBSERVATION DATE"])
    # sub[["lon", "lat", "hotspot_id"]].values
    mapp = defaultdict(partial(defaultdict, int))
    lon, lat, hotspot_id = row.lon, row.lat, row.hotspot_id
    loc = Point(lon, lat)
    Id = hotspot_id
    mapp["loc"] = Id
    for s in (species['scientific_name']):

        feature = shape_file.loc[shape_file['scntfc_'] == s][['geometry']]
        # print(type(feature))
        for idx, f in feature.iterrows():
            # print(f)
            poly = shape(f['geometry']) if f['geometry'] else None
            if poly:
                if loc.within(poly):
                    # or loc.touches(poly):

                    mapp[(lon, lat)][s] = True
                    break
                else:
                    mapp[(lon, lat)][s] = False
    return mapp


def group_by_locality_and_write(processed_chunk,
                                output_dir):
    # saving seperate file for each location
    loc = processed_chunk["loc"]
    file_name = os.path.join(output_dir, loc + '.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(processed_chunk, f)
    print(f'done saving {loc}')


def main():
    # Some very large file csv file (302 Gb)

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
    parser.add_argument("--range", type=int, default=5000)

    args = parser.parse_args()
    # Might get even better results by tuning these parameters here, but this works fine for now.
    n_processes = args.n_processes
    all_data = pd.read_csv(all_data_path)  # '/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/hotspots_june_filtered.csv')
    # species names that are in the R package
    species = pd.read_csv(species_list_path)
    # get shape file (scientific names x geometry (from R package))
    shape_file = gdp.read_file(shape_file_path)
    # mapp=defaultdict(lambda : defaultdict(lambda : 0))
    # sub_locs=locs.loc[50000:,:]

    reader_chunk_size = args.reader_chunk_size
    # See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap for
    # more info on this parameter.
    worker_chunk_size = args.worker_chunk_size
    # NOTE: Asking for more CPUs *should* make this faster until the filesystem becomes the
    # bottleneck (if it isn't already). Current runtime is ~8 hours though with 4 CPUs, so
    # depending on your context it might be worth it to ask for more CPUs to make that shorter,
    # up to you.

    print(f"Using {n_processes=}, {reader_chunk_size=} and {worker_chunk_size=}")

    index = args.index - 1
    range = args.range

    starting_index = index * range
    end_index = (index+1) * range
    print(starting_index, end_index)

    all_data = all_data.loc[starting_index:end_index, :]

    existing_files = os.listdir(save_dir)
    # Remove the extension (.tif) to get the IDs of the downloaded images
    existing_ids = [os.path.splitext(file)[0] for file in existing_files]
    # Filter the DataFrame to only include rows where 'hotspot_id' is not in 'existing_ids'
    all_data = all_data[~all_data["hotspot_id"].isin(existing_ids)]

    process = partial(process_loc, shape_file=shape_file, species=species)

    with mp.Pool(processes=n_processes, maxtasksperchild=100) as pool:
        rows = (f for _, f in all_data.iterrows())
        with tqdm(total=range) as pbar:
            for processed_chunk in pool.imap(process, rows):
                group_by_locality_and_write(processed_chunk, output_dir)
                pbar.update()

    print("Done")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()