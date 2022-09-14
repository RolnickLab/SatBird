import pickle
import pandas as pd
import os
from collections import defaultdict
import pyreadr
import geopandas as gdp
from shapely.geometry import Point,shape
from shapely.geometry.polygon import Polygon
from tqdm.auto import tqdm
from functools import partial

#read the hotspots
all_data=pd.read_csv('/network/projects/_groups/ecosystem-embeddings/hotspot_split_june/hotspots_june_filtered.csv')
locs= all_data.loc[:, ~all_data.columns.str.contains('^Unnamed')][['lon','lat']]
#species names that are in the R package
species=pd.read_csv('species_data.csv')
#get shape file (scientific names x geometry (from R package))
shape_file=gdp.read_file('bigdata.shp')
#mapp=defaultdict(lambda : defaultdict(lambda : 0))  

#map from shape file to hotspot locs      #map of(locations x scientific names)
mapp=defaultdict(partial(defaultdict, int))

for index, row in tqdm(locs.iterrows()):
    #print(loc)
    loc=Point(row['lon'],row['lat'])
    Id=all_data['hotspot_id']
    
    for s in (species['scientific_name']):
    
        print(s)
        feature=shape_file.loc[shape_file['scntfc_']==s][['geometry']]
        #print(type(feature))
        for idx,f in feature.iterrows():
            #print(f)
            poly=shape(f['geometry']) if f['geometry'] else None
            if poly:
                if loc.within(poly) or loc.touches(poly):
                    
                   mapp[(row['lon'],row['lat'])][s]=True
                   break
    # saving seperate file for each location
    file_name=os.path.join('range_maps',str(Id)+'.pkl')
    with open(file_name,'wb') as f :
        pickle.dump(mapp[(row['lon'],row['lat'])],f)
    print(f'done saving {loc}')
                

with open('ALL_rangemapp.pkl','wb') as f:
    pickle.dump(mapp,f)
