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
import warnings

import pandas as pd
from tqdm import tqdm
import rich.logging
import logging
from logging import getLogger as get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)
try:
    import rich.logging

    # from tqdm.rich import tqdm
    logger.addHandler(rich.logging.RichHandler(rich_tracebacks=True))
except ImportError:
    pass

SCRATCH = Path(os.environ["SCRATCH"])
SLURM_TMPDIR = Path(os.environ["SLURM_TMPDIR"])
N_CPUS = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))

# -----------------
# Optimized version
# -----------------


def filter_lines(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk[chunk["OBSERVATION DATE"].dt.year >= 2010]
    chunk = chunk[chunk["ALL SPECIES REPORTED"] == 1]
    chunk = chunk[chunk["OBSERVATION DATE"].dt.month.isin([6, 7, 12, 1])]
    return chunk


def preprocess_chunk(chunk: pd.DataFrame):
    # No need to do that, since we're using the `parse_dates` arg of the `read_csv` function.
    # chunk["OBSERVATION DATE"] = pd.to_datetime(chunk["OBSERVATION DATE"])
    chunk = filter_lines(chunk)
    return chunk


def group_by_locality_and_write(processed_chunk: pd.DataFrame, hotspots_output_dir: Path):
    for grp, dfg in processed_chunk.groupby("LOCALITY ID"):
        if not os.path.isfile(hotspots_output_dir / f"{grp}.csv"):
            dfg.to_csv(hotspots_output_dir / f"{grp}.csv", index=False)
        else:  # else it exists so append without writing the header
            dfg.to_csv(hotspots_output_dir / f"{grp}.csv", mode="a", index=False, header=False)


def process_hotspot(
    hotspot: Path,
    filler: pd.DataFrame, season = "summer"
) -> tuple[dict[str, tuple[dict, pd.DataFrame]], dict[str, tuple[dict, pd.DataFrame]]]:
    if not os.path.exists(os.path.join("/network/scratch/t/tengmeli/newebd/output/split-" + hotspot[-3:], hotspot+".csv")):
        return {"hotspotnotexist":(0,0)}
    hotspot_df = pd.read_csv(os.path.join("/network/scratch/t/tengmeli/newebd/output/split-" + hotspot[-3:], hotspot+".csv"), sep ="\t", parse_dates=["OBSERVATION DATE"])
    #hotspot_name = hotspot_file.stem
    # Dicts mapping from
    targets_and_entries: dict[str, tuple[dict, pd.DataFrame]] = {}
    #winter_targets_and_entries: dict[str, tuple[dict, pd.DataFrame]] = {}

    if hotspot_df["STATE"].isin(["Hawaii", "Alaska"]).any():
        return {}, {}
    if season=="summer":
        months = [6,7]
    if season=="winter":
        months = [12,1]
    hotspot_df = hotspot_df[hotspot_df["OBSERVATION DATE"].dt.month.isin(months)]
    for df, destination in zip(
        [hotspot_df], [targets_and_entries]
    ):
        if df["SAMPLING EVENT IDENTIFIER"].nunique() <= 5:
            continue
        target, entry = target_and_entry(df, filler=filler, hotspot_name=hotspot)
        assert hotspot not in destination
        destination[hotspot] = (target, entry)
    return targets_and_entries


def target_and_entry(
    filtered_hotspot_df: pd.DataFrame, filler: pd.DataFrame, hotspot_name: str
) -> tuple[dict, pd.DataFrame]:
    df = filtered_hotspot_df

    #sometimes species might be reported twice in the complete checklists because observers will indicate subspecies. In our application, we do not consider subspecies and thus only count one of the observations of the same species in the same checklist. 
    targets = (
        df[["SCIENTIFIC NAME", "SAMPLING EVENT IDENTIFIER"]].drop_duplicates(keep="first").groupby("SCIENTIFIC NAME").agg("count")
    )
    num_checklists = df["SAMPLING EVENT IDENTIFIER"].nunique()
    targets = targets/ num_checklists
    
    
    targ = targets.reset_index().merge(filler, how="right")
    targ = list(targ.fillna(0)["SAMPLING EVENT IDENTIFIER"].values)
    target = {
        "hotspot_id": hotspot_name,
        "probs": targ,
        "num_complete_checklists": num_checklists,
    }

    def _first_non_nan_value(series: pd.Series) -> pd.Series:
        unique_values = series.unique()
        assert unique_values.size == 1
        return pd.Series(unique_values[0])

    # get dataframes of hotspot info
    hotspot_entry = pd.DataFrame(
        {
            "hotspot_id": _first_non_nan_value(df["LOCALITY ID"]),
            "lon": _first_non_nan_value(df["LONGITUDE"]),
            "lat": _first_non_nan_value(df["LATITUDE"]),
            "county": _first_non_nan_value(df["COUNTY"]),
            "county_code": _first_non_nan_value(df["COUNTY CODE"]),
            "state": _first_non_nan_value(df["STATE"]),
            "state_code": _first_non_nan_value(df["STATE CODE"]),
            "num_complete_checklists": [num_checklists],
            "num_different_species": [df["SCIENTIFIC NAME"].nunique()],
        },
    )
    return target, hotspot_entry


def get_filler(data_dir: Path) -> pd.DataFrame:
    species_file = data_dir / "species_list.txt"
    species = species_file.read_text().splitlines()
    # when reading, it adds an extra ""
    if species[-1] == "":
        species = species[:-1]

    assert len(species) == 684
    return pd.DataFrame({"SCIENTIFIC NAME": species})


def _is_empty(directory: Path) -> bool:
    return len(list(itertools.islice(directory.iterdir(), 1))) == 0


def _remove_dir_if_it_exists(directory: Path, completed_file: Path) -> None:
    if not directory.exists():
        return
    if _is_empty(directory):
        directory.rmdir()
        return
    warnings.warn(
        RuntimeWarning(
            f"Erasing everything in {directory} because {completed_file} doesn't exist."
        )
    )
    answer = input("Continue? [y/N] ")
    if answer != "y":
        raise RuntimeError("Aborting")
    shutil.rmtree(directory)


def main():
    #choose summer or winter 
    season = "summer"
    data_dir = Path("/network/scratch/t/tengmeli/newebd/output/") 
    # Some very large file csv file (302 Gb)
    summer_path = Path("/network/projects/ecosystem-embeddings/ebird_new/checklists_USA/summer_list.txt")
    winter_path = Path("/network/projects/ecosystem-embeddings/ebird_new/checklists_USA/winter_list.txt")
    #or csv file with hotspot ids
    
    csv_file =  "/network/projects/_groups/ecosystem-embeddings/ebird_dataset_v2/USA_summer/all_summer_hotspots.csv"

    #"/network/projects/_groups/ecosystem-embeddings/SatBird_data/extra_summer.csv"
    
    #total_lines = 843_311_790

    # Directory where the output files should be created.
    output_dir = Path("/network/projects/ecosystem-embeddings/SatBird_data_v2/")

    # Do NOT allow the output directory to exist, since we want a "clean" preprocessed dataset.
    #output_dir.mkdir(parents=True, exist_ok=True)  # FIXME: Set `exist_ok` to False before posting

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader_chunk_size",
        type=int,
        default=10_000,
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
        default=1,
        help="Number of tasks to send to each worker process at a time.",
    )
    args = parser.parse_args()
    # Might get even better results by tuning these parameters here, but this works fine for now.
    n_processes = args.n_processes
    reader_chunk_size = args.reader_chunk_size
    # See https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap for
    # more info on this parameter.
    worker_chunk_size = args.worker_chunk_size


    
    
    print("Creating the datasets for the summer and winter seasons...")
    targets_dir = output_dir / f"{season}_targets"
    _remove_dir_if_it_exists(targets_dir, targets_dir)
    targets_dir.mkdir(parents=True, exist_ok=False)


    #with open(summer_path, "r") as f:
    #    data_summer = [line.rstrip().split(",")[0] for line in f]
    #with open(winter_path, "r") as f:
    #    data_winter = [line.rstrip().split(",")[0] for line in f]
  
    data  = pd.read_csv(csv_file).hotspot_id.to_list()
    print(len(data))
    files_iterator = tqdm(
        data,
        desc="Second phase (Reading split files + creating datasets + targets)",
        unit="Files",
        )

    filler = get_filler(Path("/network/projects/_groups/ecosystem-embeddings/species_splits/"))

    with mp.Pool(processes=n_processes) as pool:
            # No multiprocessing:
            # for summer_and_winter_data in map(functools.partial(process_hotspot, filler=filler),
            #     files_iterator,
            # ):
        for data in map(functools.partial(process_hotspot, filler=filler),
                files_iterator,):
            _targets_and_entries = data
            for target_dir, targets_and_entries in [
                    ( targets_dir, _targets_and_entries),
                ]:
                for hotspot_name, (target, entry_df) in targets_and_entries.items():
                        # Write the target
                    if hotspot_name == "hotspotnotexist":
                        continue
                    logger.debug(
                            f"Writing target for  hotspot {hotspot_name}"
                        )
                    with open(target_dir / (hotspot_name + ".json"), "x") as f:
                        json.dump(target, f)

                        # Write the entry
                    out_file = output_dir / f"all_{season}_hotspots.csv"
                    entry_df.to_csv(
                            out_file,
                            index=False,
                            mode="a" if out_file.exists() else "w",
                            header=not out_file.exists(),
                        )
    print("Done")


if __name__ == "__main__":
    main()
