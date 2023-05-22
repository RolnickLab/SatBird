from __future__ import annotations

import multiprocessing as mp
import os
import os.path
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRATCH = Path(os.environ["SCRATCH"])
SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
TOTAL_LINES = 102602192

# NOTE: Unused, but might be useful for debugging to see the column names and data types.
nan = float("nan")


# -----------------
# Optimized version
# -----------------


def process_chunk(chunk: pd.DataFrame):
    # No need to do that, since it's already part of the `read_csv` function.
    # chunk["OBSERVATION DATE"] = pd.to_datetime(chunk["OBSERVATION DATE"])
    chunk = chunk[chunk["country"]== "United States"]
    chunk = chunk[chunk["ALL SPECIES REPORTED"]==1]
    chunk = chunk[~chunk["STATE"].isin(["Alaska", "Hawaii"])]
    try:
        chunk = chunk[chunk["OBSERVATION DATE"].dt.year >= 2010]
        summer = chunk[chunk["OBSERVATION DATE"].dt.month.isin([6,7])]
        winter = chunk[chunk["OBSERVATION DATE"].dt.month.isin([12,1])]
        return summer, winter
    except:
        print(chunk["OBSERVATION DATE"])
        chunk["OBSERVATION DATE"] = pd.to_datetime(chunk["OBSERVATION DATE"])
        chunk = chunk[chunk["OBSERVATION DATE"].dt.year >= 2010]
        summer = chunk[chunk["OBSERVATION DATE"].dt.month.isin([6,7])]
        winter = chunk[chunk["OBSERVATION DATE"].dt.month.isin([12,1])]
        return summer, winter
    


def group_by_locality_and_write(processed_chunk: pd.DataFrame, output_dir: Path):
    for grp, dfg in processed_chunk.groupby("LOCALITY ID"):
        if not os.path.isfile(output_dir / f"{grp}.csv"):
            dfg.to_csv(output_dir / f"{grp}.csv", index=False)
        else:  # else it exists so append without writing the header
            dfg.to_csv(output_dir / f"{grp}.csv", mode="a", index=False, header=False)


def main():
    print("start task")

    # EBIRD DATA FILE LOCATION
    data_path = "/network/projects/_groups/ecosystem-embeddings/ebird_new/ebd_sampling_relMar-2023.txt"
    output_dir = Path("/network/projects/_groups/ecosystem-embeddings/ebird_new/checklists_USA")

    # Might get even better results by tuning this parameter here, but this works fine for now.
    reader_chunk_size = 10000

    # NOTE: Asking for more CPUs *will* make this faster, close to linearly, until the filesystem
    # becomes the bottleneck (if it isn't already). Current runtime is ~8 hours though with 4 CPUs,
    # so depending on your context it might be worth it to ask for more CPUs to make that shorter,
    # up to you.
    n_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))

    # Do NOT allow the output directory to exist, since we want "clean" preprocessed dataset.
    output_dir.mkdir(parents=True, exist_ok=False)

    reader = pd.read_csv(
        data_path,
        sep="\t",
        chunksize=reader_chunk_size,
        parse_dates=["OBSERVATION DATE", "LAST EDITED DATE"],
    )

    chunks_iterator = tqdm(
        reader,
        unit="Lines",
        unit_scale=reader_chunk_size,
        total=int(TOTAL_LINES / reader_chunk_size),  # number of chunks.
    )

    with mp.Pool(processes=n_cpus) as pool:
        for processed_chunk in pool.imap_unordered(process_chunk, chunks_iterator):
            summer , winter = processed_chunk
            if not os.path.isfile(output_dir / "summer.csv"):
                summer.to_csv(output_dir / "summer.csv", index=False)
            else:
                summer.to_csv(output_dir / "summer.csv", mode="a", index=False, header=False)
            if not os.path.isfile(output_dir / "winter.csv"):
                winter.to_csv(output_dir / "winter.csv", index=False)
            else:  # else it exists so append without writing the header
                winter.to_csv(output_dir /"winter.csv", mode="a", index=False, header=False)
    print("Done")
    
def process2(chunk):
    chunk = chunk[['STATE','COUNTY', 'COUNTY CODE', 'STATE CODE', "LOCALITY", "LOCALITY ID", "OBSERVATION DATE",  'LATITUDE',
           'LONGITUDE', 'SAMPLING EVENT IDENTIFIER']]
    return(chunk)
"""
def main2():
    
    
    
    LINES = 8077930 #summer 6302533 #winter 8077930
    data_path = Path("/network/projects/_groups/ecosystem-embeddings/ebird_new/checklists_USA/winter.csv")
    output_dir =  Path("/network/projects/_groups/ecosystem-embeddings/ebird_new/checklists_USA/winter/")
    
    # Might get even better results by tuning this parameter here, but this works fine for now.
    reader_chunk_size = 10000

    # NOTE: Asking for more CPUs *will* make this faster, close to linearly, until the filesystem
    # becomes the bottleneck (if it isn't already). Current runtime is ~8 hours though with 4 CPUs,
    # so depending on your context it might be worth it to ask for more CPUs to make that shorter,
    # up to you.
    n_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))

    # Do NOT allow the output directory to exist, since we want "clean" preprocessed dataset.
    output_dir.mkdir(parents=True, exist_ok=False)

    reader = pd.read_csv(
        data_path,
        sep=",",
        chunksize=reader_chunk_size,
        parse_dates=["OBSERVATION DATE", "LAST EDITED DATE"],
    )
    chunks_iterator = tqdm(
        reader,
        unit="Lines",
        unit_scale=reader_chunk_size,
        total=int(LINES / reader_chunk_size),  # number of chunks.
    )


    with mp.Pool(processes=n_cpus) as pool:
        for processed_chunk in pool.imap_unordered(process2, chunks_iterator):
            group_by_locality_and_write(processed_chunk, output_dir=output_dir)

    print("Done")
"""    
if __name__ == "__main__":
    # main_unoptimized()
    main()
