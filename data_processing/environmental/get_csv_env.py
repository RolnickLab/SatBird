
import os
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
sys.path.append(str(Path().resolve().parent.parent))

import numpy as np
import pandas as pd
from data_processing.environmental.environmental_raster import PatchExtractor

import matplotlib.pyplot as plt

if __name__ == "__main__":
    DATA_PATH = "/network/scratch/t/tengmeli/geolifeclef-2022/rasters/"

    extractor = PatchExtractor(DATA_PATH, country="USA", size = 1)


    extractor.add_all_rasters()


    #extractor.add_all_rasters()
    print("Number of rasters: {}".format(len(extractor)))

    df_ref = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/winter_hotspots.csv") #ADD DF TO WHICH WE WANT TO ADD ENV VARIABLES
    df_ref = df_ref.drop_duplicates(["hotspot_id"], keep="first")
    print("length of dataframe: {}".format(len(df_ref)))
    var_names = [extractor.rasters_us[i].name for i in range(len(extractor))]

    var_to_add = [elem for elem in var_names if elem not in df_ref.columns]

    variables = df_ref.apply(lambda row: extractor[row.lat, row.lon], axis=1)

    values = [variables.apply(lambda row: row[:][i]).values.tolist() for i in range(len(var_to_add))]

    for i in range(len(var_to_add)):
        df_ref[var_to_add[i]] = values[i]

    df_ref.to_csv("/network/projects/ecosystem-embeddings/ebird_new/winter_hotspots_with_bioclim_withnan.csv")