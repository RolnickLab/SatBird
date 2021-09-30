import glob
import os
import json 
import pandas as pd

datapath = "/network/scratch/a/akeraben/akera/ecosystem-embedding/data/sentinel2/"
species_data = "/network/scratch/t/tengmeli/ecosystem-embedding/ebird_data_june/"
#txt file with csv names
hotspots = "/network/scratch/t/tengmeli/ecosystem-embedding/training/train_june.txt"
#path of csv where we want to save info 
save_path = "/network/scratch/t/tengmeli/ecosystem-embedding/training/train_june.csv"

if __name__=="__main__":
    with open(hotspots, "r") as f:
        hs_list = [line.rstrip() for line in f]


    hs_list = list(dict.fromkeys(hs_list))


    rgb_paths = [os.path.join(datapath, f"{hs}_rgb.npy") for hs in hs_list]
    json_paths = [os.path.join(datapath, f"{hs}.json") for hs in hs_list]
    nir_paths = [os.path.join(datapath, f"{hs}_ni.npy") for hs in hs_list]
    r_paths = [os.path.join(datapath, f"{hs}_r.npy") for hs in hs_list]
    g_paths = [os.path.join(datapath, f"{hs}_g.npy") for hs in hs_list]
    b_paths = [os.path.join(datapath, f"{hs}_b.npy") for hs in hs_list]

    species = [os.path.join(species_data, f"{hs}.json") for hs in hs_list]


    dataset = pd.DataFrame(list(zip(hs_list, r_paths, g_paths, b_paths,nir_paths, json_paths, rgb_paths,
                                    species,
                                    )),

                  columns=["hotspot", "r", "g", "b","nir", "meta", "rgb","species"])


    dataset.to_csv(save_path)