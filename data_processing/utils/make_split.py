import pandas as pd
import os 
import json
import random
import numpy as np 
import glob

path = "../../hotspots_june_filtered.csv"


if __name__ == "__main__":
    train=0.70
    val=0.15
    test=0.15

    df = pd.read_csv(path)


    df = df.drop_duplicates("hotspot_id")

    ids = os.listdir("/network/scratch/t/tengmeli/ecosystem-embedding/ebird_data_june/")
    ids = [i.strip(".json") for i in ids]
    
    ids2 = os.listdir("/network/scratch/a/akeraben/akera/ecosystem-embedding/data/sentinel2")
    ids2 = [i.split(".")[0] for i in ids2 if i.endswith("json")]
    
    a = pd.DataFrame({"hotspot_id":ids})
    b =  pd.DataFrame({"hotspot_id":ids2})
    a = a.merge(b,on="hotspot_id", how = "inner")
    df = df.merge(a, on="hotspot_id", how = "inner")
    
    counties = list(df["county_code"].unique())
    l = len(counties)

    shuffled = random.sample(counties, l)

    train_c = counties[:int(train*l)]
    val_c = counties[int(train*l):int(train*l)+int(val*l)]
    test_c = counties[int(train*l)+int(val*l):]

    df[df["county_code"].isin(train_c)]["hotspot_id"].values

    train_hs = df[df["county_code"].isin(train_c)]["hotspot_id"].values
    val_hs = df[df["county_code"].isin(val_c)]["hotspot_id"].values
    test_hs = df[df["county_code"].isin(test_c)]["hotspot_id"].values

    print(len(train_hs))
    print(len(val_hs))
    print(len(test_hs))

    def write_array_text(arr, file):
        with open(file, "w") as txt_file:    
            for elem in arr:
                txt_file.write(elem + "\n") 

    write_array_text(train_hs,"/network/scratch/t/tengmeli/ecosystem-embedding/training/train_june.txt")
    write_array_text(val_hs,"/network/scratch/t/tengmeli/ecosystem-embedding/training/val_june.txt")
    write_array_text(test_hs,"/network/scratch/t/tengmeli/ecosystem-embedding/training/test_june.txt")