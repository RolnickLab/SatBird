import glob
import os
import json 
import pandas as pd

datapath = "/network/scratch/t/tengmeli/scratch/ecosystem-embedding/satellite_data/"
species_data = "/network/scratch/t/tengmeli/scratch/ecosystem-embedding/ebird_data_june/"
#txt file with csv names
hotspots = "/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/cali_val.txt"
#""/network/scratch/t/tengmeli/scratch/ecosystem-embedding/training/train_june.txt"
#path of csv where we want to save info 
save_path = "/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/cali_val.csv"
#"/network/scratch/t/tengmeli/ecosystem-embedding/training/train_june.csv"
#/network/scratch/t/tengmeli/scratch/ecosystem-embedding/training/train_june_vf.csv


if __name__=="__main__":
    with open(hotspots, "r") as f:
        hs_list = [line.rstrip() for line in f]

    keys = [os.path.basename(a).strip(".json") for a in glob.glob(species_data + "/*")]   

    hs_list = list(dict.fromkeys(hs_list))
    hs_list = [hs for hs in hs_list if hs in keys]


    rgb_paths = [os.path.join(datapath, f"{hs}_rgb.npy") for hs in hs_list]
    json_paths = [os.path.join(datapath, f"{hs}.json") for hs in hs_list]
    nir_paths = [os.path.join(datapath, f"{hs}_ni.npy") for hs in hs_list]
    r_paths = [os.path.join(datapath, f"{hs}_r.npy") for hs in hs_list]
    g_paths = [os.path.join(datapath, f"{hs}_g.npy") for hs in hs_list]
    b_paths = [os.path.join(datapath, f"{hs}_b.npy") for hs in hs_list]
    species = [os.path.join(species_data, f"{hs}.json") for hs in hs_list]

    env = [os.path.join("/network/scratch/t/tengmeli/scratch/ecosystem-embedding/hotspot_env_var", f"{hs}.npy") for hs in hs_list]
    ped = [os.path.join("/network/scratch/t/tengmeli/scratch/ecosystem-embedding/hotspot_ped_var", f"{hs}.npy") for hs in hs_list]
    bioclim = [os.path.join("/network/scratch/t/tengmeli/scratch/ecosystem-embedding/hotspot_bio_var", f"{hs}.npy") for hs in hs_list]
    


    dataset = pd.DataFrame(list(zip(hs_list, r_paths, g_paths, b_paths,nir_paths, json_paths, rgb_paths,
                                    species,env, ped, bioclim
                                    )),

                  columns=["hotspot", "r", "g", "b","nir", "meta", "rgb","species", "env","ped","bioclim"])


    dataset.to_csv(save_path)
    print(len(dataset))