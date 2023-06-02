import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

def merge_duplicate_lat_lon(path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan.csv",
                           out_path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan_merged_vf.csv",
                           duplicate_list_path = "/network/projects/ecosystem-embeddings/ebird_new/to_merge_3.txt"):
    """
    Some hotspots have the same laitude and longitude and address, however, they have different hotspot IDs. we suspect they haven't been properly merged in the eBird dataset and thus, 
    "merge" those hotspots and only keeo 1 hotspot ID. 
    We keep track of the list of hotspots with same lat-lon for 
    We apply this for summer and winter US datasets
    winter paths : 
        path: "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan.csv"
        out_path: "/network/projects/ecosystem-embeddings/ebird_new/winter_hotspots_with_bioclim_withnan_merged_vf.csv"
        duplicate_list_path: "/network/projects/ecosystem-embeddings/ebird_new/to_merge_winter.txt"
    """
    #clean dataset from duplicate lat-lon values, keep track of hotspots associated to the same lat-lon for future merging
    s = pd.read_csv(path)
    s = s.drop_duplicates(["hotspot_id"], keep= "first")

    #dups = s[s[['lat', 'lon']].duplicated()]
    grouped = s.groupby(['lon', 'lat'])
    new_df= pd.DataFrame(columns = s.columns)

    hotspots_merge= []

    for i, (name, group) in enumerate(grouped):
        if i%10000==0:
            print(i)
        if len(group)==1:
            new_df = pd.concat([new_df, group])
        else:
            group["num_complete_checklists"] = group["num_complete_checklists"].sum()
            group["num_different_species"] = group["num_different_species"].sum()
            hotspots_merge.append([list(group["hotspot_id"].values)])
            group = group.drop_duplicates(["lon"], keep = "first")
            new_df = pd.concat([new_df, group])


    liste = [h[0] for h in hotspots_merge]

    #list of targets to merge 
    with open(duplicate_list_path, "w") as f:
        for s in liste:
            f.write((' ').join(s) +"\n")

    new_df.to_csv(out_path)


def add_split(path= "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan_merged_vf.csv",
             out_path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf.csv"):
    """
    apply to outpath of merge_duplicate_lat_lon to add a column with the splits + also fill nan values of bioclim variables with training set stats
    """
    #add split column in dataset
    summer = pd.read_csv(out_path)

    summer["split"]=""
    
    #TODO later: pass path to as arg.
    for name in ["train", "valid","test"]:
        with open(f"/network/scratch/t/tengmeli/scratch/ecosystem-embedding/training/new_{name}_clustered_summer_2.txt", "r") as f:
            data = [line.strip("\n") for line in f.readlines()]
        print(len(data))
        summer.loc[list(summer[summer["hotspot_id"].isin(data)].index),"split"]=name

    summer = summer.drop(columns=["Unnamed: 0"])

    dict_means = {}

    #get stats on training set of env. variables
    cols = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8',
           'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15',
           'bio_16', 'bio_17', 'bio_18', 'bio_19', 'bdticm', 'bldfie', 'cecsol',
           'clyppt', 'orcdrc', 'phihox', 'sltppt', 'sndppt']
    values = summer[summer["split"]=="train"][cols].mean().values
    for i,elem in enumerate(cols):
        dict_means[elem] = values[i]
    #Dict to be used in Rasters for xtraction of environmental rasters. 
    print(dict_means)

    #fill nans 
    summer = summer.fillna(value=dict_means)
    summer.to_csv(out_path, index=False)
    

def get_merge_dict():
    """
    merge_dict and clean dataset
    get dictionary to be used to  merge targets from hotspots that have duplicate lat lons: {hotspot_in_the_hotspot_csv: [duplicate_lat_lon_hotspots]}
    since winter locations are included in summer locations, we only generate one merging dictionary 
    """
    with open("/network/projects/ecosystem-embeddings/ebird_new/to_merge_3.txt", "r") as f:
    data = [line.strip("\n") for line in f.readlines()]

    #path to raw data
    root = Path("/network/scratch/t/tengmeli/newebd/output/")
    to_drop = []
    for i in range(len(data)):
        hotspots = data[i].split(" ")

        paths = []
        for h in hotspots:
            ending = h[-3:]
            paths += [root / f"split-{ending}" / f"{h}.csv"]
        observers = [] 
        dates = []
        #clean data from hotspots with the same lat-lon, that onl have 1 observer and 1 date of observation (probably an error)
        for path in paths:
            df= pd.read_csv(path, sep = "\t")
            observers += [tuple(df["OBSERVER ID"].unique())]
            dates += list(df["OBSERVATION DATE"].unique())
        if len(set(dates))==1 and len(set(observers))==1 :
            to_drop += hotspots
    print(to_drop)
    summer = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf.csv")
    
    #drop problematic hotspots
    ind = summer[summer["hotspot_id"].isin(to_drop)].index
    summer = summer.drop(ind)
    
    summer.drop(columns="Unnamed: 0.1").to_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf_clean.csv")

    dict_merges = {}
    for i in range(len(data)):
        hotspots = data[i].split(" ")
        value = []
        key = ""
        for h in hotspots:
            if h in summer["hotspot_id"].values:
                key = h
            else:
                value+= [h]
        if key != "":
            dict_merges[key] = value

    merge_path = Path("/network/projects/ecosystem-embeddings/ebird_new/to_merge_dict.json")
    json_str = json.dumps(dict_merges, indent=4) + '\n'
    merge_path.write_text(json_str, encoding='utf-8')
    
if name == "__main__":
    #TODO have cleaner path handling
    path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan.csv",
    out_path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan_merged_vf.csv",
    duplicate_list_path = "/network/projects/ecosystem-embeddings/ebird_new/to_merge_3.txt"
    merge_duplicate_lat_lon(path, out_path, duplicate_list_path)
    add_split(path= out_path,
             out_path = "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf.csv")
    get_merge_dict()
    #final output is "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf_clean.csv"
    
    
    #for winter we notice the winter locations are included in the summer locations
    #we keep the same split as for summer and additionnally when cleaning the duplicate lat lon keep those we kept for summer. 
    merge_duplicate_lat_lon(path= "/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_withnan.csv",
        out_path= "/network/projects/ecosystem-embeddings/ebird_new/winter_hotspots_with_bioclim_withnan_merged_vf.csv",
        duplicate_list_path= "/network/projects/ecosystem-embeddings/ebird_new/to_merge_winter.txt")
    
    new_df = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/winter_hotspots_with_bioclim_withnan_merged_vf.csv")
    summer = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf.csv")
    aa = new_df.merge(summer[["hotspot_id", "split"]], how = "left")
    to_keep = aa[aa["split"].isin(["train","valid","test"])]
    extras = aa[~aa["split"].isin(["train","valid","test"])]

    merged = extras.merge(summer[["hotspot_id", "lat","lon"]], how = "inner", left_on=["lat","lon"], right_on= ["lat", "lon"])
    merged["hotspot_id"] = merged["hotspot_id_y"]
    merged = merged.drop(columns= ["hotspot_id_y","hotspot_id_x", "Unnamed: 0"])

    final = pd.concat([to_keep, merged]).drop(columns = ["split", "Unnamed: 0"])

    final = final.merge(summer[["hotspot_id", "split"]], how = "left")
    print(final.split.value_counts())
    final.to_csv("/network/projects/ecosystem-embeddings/ebird_new/winter_hotspots_with_bioclim_splits_vf.csv")
    
    """
    dict_means = {}

    #get stats on training set of env. variables
    cols = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5', 'bio_6', 'bio_7', 'bio_8',
           'bio_9', 'bio_10', 'bio_11', 'bio_12', 'bio_13', 'bio_14', 'bio_15',
           'bio_16', 'bio_17', 'bio_18', 'bio_19', 'bdticm', 'bldfie', 'cecsol',
           'clyppt', 'orcdrc', 'phihox', 'sltppt', 'sndppt']
    values = final[final["split"]=="train"][cols].mean().values
    for i,elem in enumerate(cols):
        dict_means[elem] = values[i]
    print(dict_means)

    dict_means
    """