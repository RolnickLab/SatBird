import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path 
import os, numpy, json, tifffile

def filter_by_sat():
    summer = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_vf_clean.csv")
    hotspots = [r.strip(".tif") for r in os.listdir("/network/projects/_groups/ecosystem-embeddings/ebird_new/rasters_new/summer_rasters/")]
    uu = summer[summer["hotspot_id"].isin(hotspots)]
    uu.drop(columns = ["Unnamed: 0"]).to_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_with_bioclim_splits_final.csv")

def filter_by_geography():
    root = Path("/network/projects/ecosystem-embeddings/ebird_new/")
    summer = pd.read_csv(root / "summer_hotspots_with_bioclim_splits_final.csv")
    gdf = gpd.read_file(root / "cb_2018_us_nation_5m.shp")
    indices = []
    for i, elem in enumerate(summer[["lon", "lat"]].values):
        if not gdf.geometry[0].contains(Point(elem[0], elem[1])):
            indices += [i]

    summer_clean = summer.drop(indices)
    summer_clean.to_csv(root / "summer_hotspots_final.csv")

def filter_by_size():
    df = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_final.csv")
    indices = []
    for i, elem in enumerate(df.hotspot_id.values):
        w,h,b = tifffile.imread(f"/network/projects/_groups/ecosystem-embeddings/ebird_new/rasters_new/summer_rasters/{elem}.tif").shape
        if w<128 or h<128:
            indices += [i]
    df = df.drop(columns = ["Unnamed: 0.1",'Unnamed: 0'])
    df = df.drop(indices)
    print(len(indices))
    print(len(df))
    df.to_csv("/network/projects/ecosystem-embeddings/ebird_new/summer_hotspots_clean.csv")
    
if __name__=="__main__":
    #filter_by_sat()
    #filter_by_geography()
    filter_by_size()