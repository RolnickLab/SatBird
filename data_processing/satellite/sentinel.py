import os
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc
import numpy as np
from shapely.geometry import Polygon
import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp
import pandas as pd
from PIL import Image
import json


import os
from azure.storage.blob import BlobServiceClient


#set subscription key to planetary computer
pc.settings.set_subscription_key("")

def get_corner(lon, lat, bearingInDegrees = 45, d=10):
    R = 6371 
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    brng = np.deg2rad(bearingInDegrees)
    lat2 = np.arcsin( np.sin( lat) * np.cos( d / R ) +np.cos( lat) * np.sin( d / R ) * np.cos( brng ) )
    lon2 = lon + np.arctan2( np.sin( brng ) * np.sin( d / R ) *np.cos( lat),np.cos( d / R ) - np.sin( lat) * np.sin( lat2 ) )
    return([np.rad2deg(lon2), np.rad2deg(lat2)])


def get_square(lon, lat, d = 1):

    tr = get_corner(lon, lat, 45, d)
    tl = get_corner(lon, lat,  115, d)
    bl = get_corner(lon, lat,205, d)
    br = get_corner(lon, lat, 295, d)

    area_of_interest = Polygon([tr, tl, bl, br, tr])
    return(area_of_interest)

def get_least_cloudy(items):
    least_cloudy_item = sorted(items, key=lambda item: eo.ext(item).cloud_cover)[0]

    print(
        f"Choosing {least_cloudy_item.id} from {least_cloudy_item.datetime.date()}"
        f" with {eo.ext(least_cloudy_item).cloud_cover}% cloud cover"
    )
    return (least_cloudy_item)


def get_cropped(signed_href, area_of_interest):
    with rasterio.open(signed_href) as ds:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        band_data = ds.read(window=aoi_window)
    return(band_data)

def get_data(area_of_interest):
    years = [y for y in range(2021, 2016, -1)]
    found = False
    for y in years:
        if not found:
            time_of_interest = f"{str(y)}-06-01/{str(y)}-06-30"
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=area_of_interest,
                datetime=time_of_interest,
                query={"eo:cloud_cover": {"lt": 10}},
            )

            # Check how many items were returned
            items = list(search.get_items())
            if len(items) > 0:
                print(f"Returned {len(items)} Items for year {str(y)}")
                found = True
    if not found:
        print("Nothing found for hotspot")
        return(None)


    least_cloudy_item = get_least_cloudy(items)
    
    asset_href = least_cloudy_item.assets["visual"].href
    ni_href = least_cloudy_item.assets["B08"].href
    signed_href = pc.sign(asset_href)
    signed_nihref = pc.sign(ni_href)
    
    band_data = get_cropped(signed_href, area_of_interest)
    ni_data = get_cropped(signed_nihref, area_of_interest)
    return(band_data, ni_data, least_cloudy_item.datetime.date())
    
def get_metadata(lon, lat, hotspot_id, area_of_interest, date):
    obj = {
        "hotspot_id" : hotspot_id,
        "lon" : lon,
        "lat" : lat, 
        "area_geom" :  area_of_interest.wkt,
        "date": str(date)
    }
    return(obj)


def get_patch(lon,lat, hotspot_id, save_path="."):
    area_of_interest = get_square(lon, lat, d=5)
    data = get_data(area_of_interest)
    if data is None:
        return(None)
    else :
        band_data, ni_data, date = data
        np.save(os.path.join(save_path, str(hotspot_id)+"_rgb.npy"), band_data)
        np.save(os.path.join(save_path, str(hotspot_id)+"_ni.npy"), ni_data)
        metadata = get_metadata(lon, lat, hotspot_id, area_of_interest, date)
        with open(os.path.join(save_path, str(hotspot_id)+".json"), 'w') as f:
            json.dump(metadata, f)
        f.close()
        print(f"Saved patch for hotspot {hotspot_id}")
        return(band_data, ni_data, metadata)


    
def display_image(data, mode = "rgb"):
    """
    mode : 'rgb' or "ni'
    """
    if mode == "rgb":
        img = Image.fromarray(np.transpose(data, axes=[1, 2, 0]))
    elif mode == "ni":
        ni_data = (255 * (data / np.max(data))).astype(np.uint8)
        img = Image.fromarray(ni_data[0], mode="L")
        
    else:
        print("Invalid mode")
        return()
    return(img)

if __name__ == "__main__":
    connection_string = "" #connection string to Azure storage account
    service_client = BlobServiceClient.from_connection_string(connection_string)

    df = pd.read_csv() #CSV file with hotspots lat long data "~/PlanetaryComputerExamples/datasets/sentinel-2-l2a/hotspots_latlon.csv")
    
    container_name = 'sentinel'

    for index, row in df.iterrows():

        hotspot_id = row['hotspot_id']
        lon = row['lon']
        lat = row['lat']
        a = get_patch(lon,lat, hotspot_id, save_path=".")
        if a is None:
            print("nothing found for " + hotspot_id) 
        else:
            band_data, ni_data, metadata = a
            print("\nUploading to Azure Storage as blob:\n\t" + str(hotspot_id))

            with open(str(hotspot_id)+"_rgb.npy","rb") as band_data:
                blob_client = service_client.get_blob_client(container=container_name', blob= str(hotspot_id)+"_rgb.npy")
                blob_client.upload_blob(band_data)
            with open(str(hotspot_id)+"_ni.npy", "rb") as ni_data:
                blob_client = service_client.get_blob_client(container=container_name, blob= str(hotspot_id)+"_ni.npy")
                blob_client.upload_blob(ni_data)
            with open(str(hotspot_id)+".json", "rb") as metadata:
                blob_client = service_client.get_blob_client(container=container_name, blob= str(hotspot_id)+".json")
                blob_client.upload_blob(metadata)

            os.remove(str(hotspot_id)+"_rgb.npy")
            os.remove(str(hotspot_id)+"_ni.npy")
            os.remove(str(hotspot_id)+".json")
            print("removed files, moving to the next")
