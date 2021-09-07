import pandas as pd
import random 

if __name__=="__main__":
    path = "hotspots_data.csv"

    df = pd.read_csv(path)

    train = 0.75
    val = 0.15
    test = 0.15

    l = len(counties)

    counties = list(df["county_code"].unique())

    shuffled = random.sample(counties, l)

    train_c = counties[:int(train*l)]
    val_c = counties[int(train*l):int(train*l)+int(val*l)]
    test_c = counties[int(train*l)+int(val*l):]

    df[df["county_code"].isin(train_c)]["hotspot_id"].values

    train_hs = df[df["county_code"].isin(train_c)]["hotspot_id"].values
    val_hs = df[df["county_code"].isin(val_c)]["hotspot_id"].values
    test_hs = df[df["county_code"].isin(test_c)]["hotspot_id"].values

    def write_array_text(arr, file):
        with open(file, "w") as txt_file:    
            for elem in arr:
                txt_file.write(elem + "\n") 


    write_array_text(train_hs,"data/train.txt")
    write_array_text(val_hs,"data/val.txt")
    write_array_text(test_hs,"data/test.txt")