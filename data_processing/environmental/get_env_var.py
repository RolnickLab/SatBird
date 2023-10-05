import os
import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))
sys.path.append(str(Path().resolve().parent.parent))
import numpy as np
import pandas as pd

from data_processing.environmental.environmental_raster import PatchExtractor

DATA_PATH = Path("/network/scratch/t/tengmeli/geolifeclef-2022/rasters")

extractor = PatchExtractor(DATA_PATH, country="USA", size = 50)
extractor.add_all_rasters()
print("Number of rasters: {}".format(len(extractor)))

if __name__=="__main__":
    df = pd.read_csv("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_winter/winter_hotspots.csv") #"hotspots_data_with_bioclim.csv")
    for index, row in df.iterrows():
        i = 0
        if index % 100 == 0:
            print(index)
        try : 
            val = extractor[row.lat, row.lon]
            np.save("/network/projects/ecosystem-embeddings/SatBird_data_v2/USA_winter/environmental/" + row.hotspot_id + ".npy", val)
        except :
            i+= 1
        #try:
        #    val = extrator2[row.lat, row.lon]
         #   np.save("/network/scratch/t/tengmeli/scratch/ecosystem-embedding/hotspot_env_var/" + row.hotspot_id + ".npy", val)
        #except : 
            print(row.hotspot_id, row.index) #, row['Unnamed: 0'])
            pass
