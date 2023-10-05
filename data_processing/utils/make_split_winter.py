import pandas as pd
from data_processing.utils.make_splits_by_distance import cluster_based_on_dist, make_splits
import numpy as np

if __name__=="__main__":
    #for the winter dataset, we want hotspots that are also present in the summer dataset to be assigned to the same split
    #we assign the remaining hotspots following the same clustering strategy
    
    csv = pd.read_csv("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_winter/all_winter_hotspots.csv")
    csv_summer = pd.read_csv("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_summer/all_summer_hotspots_final.csv")
    csv = csv.merge(csv_summer[["hotspot_id", "split"]], how = "left", left_on = "hotspot_id", right_on = "hotspot_id")
    
    nans = csv[csv["split"].isna()]
    nans = nans.reset_index()


    RADIUS_EARTH = 6356.7523 

    csv["split"].value_counts() / csv["split"].value_counts().sum()

    u = nans[['lat', 'lon']]
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
    test_size = len(u) - (train_size + val_size)

    print((train_size , val_size, test_size))
    sizes = [val_size, test_size,  train_size]

    splits = make_splits(splits_names, sizes, clusters_dict)
    print([(name, len(splits[name])) for name in splits_names])
    # Write to text files


    nans["split"] = ""
    for name in splits_names:
        nans.loc[splits[name], "split"] = name 

    for i, row in nans.iterrows():
        csv.loc[row['index'], "split"] = row["split"]

    csv.to_csv("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_winter/all_winter_hotspots_splits.csv)