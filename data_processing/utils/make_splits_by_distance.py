from sklearn.cluster import DBSCAN
from collections import defaultdict
from copy import deepcopy
import random
import pandas as pd
import os
import json
import random
import numpy as np
import glob
from math import radians


RADIUS_EARTH = 6356.7523  # in km, polar radius of Earth


def get_lat_for_distance(d):
    # https://github.com/sustainlab-group/africa_poverty
    '''
    Helper function

    Calculates the degrees latitude for some North-South distance.

    Makes (incorrect) assumption that Earth is a perfect sphere.
    Uses the smaller polar radius (instead of equatorial radius), so
        actual degrees latitude <= returned value

    Args
    - d: numeric, distance in km

    Returns
    - lat: float, approximate degrees latitude
    '''
    lat = d / RADIUS_EARTH  # in radians
    lat = lat * 180.0 / np.pi  # convert to degrees
    return lat


def get_lon_for_distance(lat, d):
    # https://github.com/sustainlab-group/africa_poverty
    '''
    Helper Function
    Calculates the degrees longitude for some East-West distance at a given latitude.

    Makes (incorrect) assumption that Earth is a perfect sphere.
    Uses the smaller polar radius (instead of equatorial radius), so
        actual degrees longitude <= returned value

    Args
    - lat: numeric, latitude in degrees
    - d: numeric, distance in km

    Returns
    - lon: float, approximate degrees longitude
    '''
    lat = np.abs(lat) * np.pi / 180.0  # convert to radians
    r = RADIUS_EARTH * np.cos(lat)  # radius at the given lat
    lon = d / r
    lon = lon * 180.0 / np.pi  # convert to degrees
    return lon


from sklearn.cluster import DBSCAN
def cluster_based_on_dist(X, dist):
    '''
    label the cluster each input belongs to , where the cluster is a group of points in that are not far away from other by  dist km
    each group of points contains at least 2 samples , if you used more than 2 samples=> less number of clusters
    the distance in km is converted to lat, lon using some formulas , then that total distance is approximated as a straight line between lat & lon
    more on #https://jonisalonen.com/2014/computing-distance-between-coordinates-can-be-simple-and-fast/


    Args
    X: array of shape(data_size ,2) contains [lat, lon]
    dist: numeric, maximum distance  in km between points in a cluster

    Returns
    - cluster_labels :array of shape(data_size,) labeling each [lat,lon]
    - clusters_dict  :dict(cluster_label:[indices]) map each label to its list of indices in X
    '''
    clustering = DBSCAN( eps=dist, min_samples=2, metric='haversine').fit(X)
    cluster_labels = clustering.labels_
    # map each label to its indices, each outlier (label of -1) is treated as its own cluster
    loc_indices = defaultdict(list)
    for i, loc in enumerate(X):
        loc_indices[tuple(loc)].append(i)
    clusters_dict = defaultdict(list)
    neg = -1
    for loc, label in zip(X, cluster_labels):
        indices = loc_indices[tuple(loc)]
        if label < 0:
            label = neg
            neg -= 1
        clusters_dict[label].extend(indices)
    return clusters_dict, cluster_labels


def make_splits(splits_names, sizes, clusters_dict):
    '''
    maps each split to its indices with the size of the split , making sure that each cluster is all in one split and not scattered between different splits
    Args
    - splits: list of string, naming splits
    - sizes: list of floats , size of each split
    - clusters_dict : dict(cluster_label:[indices]) map each label to its indices

    Returns
    - splits: dict(split:[indices]) maps split to list of indices
    '''
    splits = defaultdict(list)

    cop_cluster = deepcopy(clusters_dict)
    for s, size in zip(splits_names, sizes):
        while cop_cluster:
            c = random.choice(list(cop_cluster))
            # print(c)
            splits[s].extend(clusters_dict[c])
            cop_cluster.pop(c)

            if len(splits[s]) >= size:
                # print(c)
                break
    return splits


def write_array_text(arr, file):
    with open(file, "w") as txt_file:
        for elem in arr:
            txt_file.write(elem + "\n")


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    # Reading Hotspot data

    locs = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_clean.csv")
    print(len(locs))
    #pd.read_csv('/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/hotspots_june_filtered.csv')
    locs = locs.sort_values("county_code").reset_index()
    locs = locs.loc[:, ~locs.columns.str.contains('^Unnamed')]
    u = locs[['lat', 'lon']]
    u['lat']= np.radians(u["lat"])
    u['lon']= np.radians(u["lon"])
    # splitting ---
    X = u.values
   
    clusters_dict, cluster_labels = cluster_based_on_dist(u.values, dist=5/RADIUS_EARTH )
    train = 0.70
    valid = 0.15
    splits_names = ['valid','test','train']

    train_size = int(train * len(X))
    val_size = int(valid * len(X))
    test_size = len(locs) - (train_size + val_size)

    print((train_size , val_size, test_size))
    sizes = [val_size, test_size,  train_size]

    splits = make_splits(splits_names, sizes, clusters_dict)
    print([(name, len(splits[name])) for name in splits_names])
    # Write to text files
    for name in splits_names:
        lats_lons = locs.loc[splits[name], 'hotspot_id']
        write_array_text(lats_lons,
                         f"/network/scratch/t/tengmeli/scratch/ecosystem-embedding/training/new_{name}_clustered_summer_5.txt")

    # Write to csv files:
    df = locs
    df["split"] = ""
    for name in splits_names:
        df.loc[splits[name], "split"] = name #.reset_index(drop=True)
    df.to_csv(f'/network/scratch/t/tengmeli/scratch/ecosystem-embedding/training/satbird_clustered_summer.csv')

    # print(df.head())
