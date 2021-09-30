import pandas as pd
import os
import glob

datapath = "/network/scratch/a/akeraben/akera/ecosystem-embedding/data"
species_data = "/network/scratch/t/tengmeli/ecosystem-embedding/ebird_data_june/"


all_paths = glob.glob(datapath+"/sentinel2/*")
rgb_paths = glob.glob(datapath+"/sentinel2/*rgb*")
json_paths = glob.glob(datapath+"/sentinel2/*.json")
nir_paths = glob.glob(datapath+"/sentinel2/*_ni.npy")
r_paths = glob.glob(datapath+"/sentinel2/*_r.npy")
g_paths = glob.glob(datapath+"/sentinel2/*_g.npy")
b_paths = glob.glob(datapath+"/sentinel2/*_b.npy")

hotspot_ids = [i.split("sentinel2/")[1].split("_")[0] for i in rgb_paths]



species_paths = glob.glob(species_data+"*")
species_ids = [i.split("ebird_data_june/")[1].split(".")[0] for i in species_paths]

#ch
ids = [i.split("sentinel2/")[1].split(".")[0] if "json" in i else i.split("sentinel2/")[1].split("_")[0] for i in json_paths]

dataset = pd.DataFrame(list(zip(species_ids,
                                species_paths,
                                )),

              columns=["species_ids","species_paths"])

dataset.to_csv("json_species_paths.csv")
