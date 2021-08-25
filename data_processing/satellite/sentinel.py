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

#from https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/eo/eo.py

def metre_per_degree(lat: float):
    # https://gis.stackexchange.com/questions/75528/understanding-terms-in
    # -length-of-degree-formula
    # see the link above to explain the magic numbers
    m_per_degree_lat = (
        111132.954
        + (-559.822 * cos(radians(2.0 * lat)))
        + (1.175 * cos(radians(4.0 * lat)))
        + (-0.0023 * cos(radians(6 * lat)))
    )
    m_per_degree_lon = (
        (111412.84 * cos(radians(lat)))
        + (-93.5 * cos(radians(3 * lat)))
        + (0.118 * cos(radians(5 * lat)))
    )

    return m_per_degree_lat, m_per_degree_lon

def bounding_box_from_centre(
    mid_lat, mid_lon, surrounding_metres):

    m_per_deg_lat, m_per_deg_lon = metre_per_degree(mid_lat)

    if isinstance(surrounding_metres, int):
        surrounding_metres = (surrounding_metres, surrounding_metres)

    surrounding_lat, surrounding_lon = surrounding_metres

    deg_lat = surrounding_lat / m_per_deg_lat
    deg_lon = surrounding_lon / m_per_deg_lon

    max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
    max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

    return Polygon([[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat]]
    )

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
    asset_rref = least_cloudy_item.assets["B04"].href
    asset_gref = least_cloudy_item.assets["B03"].href
    asset_bref = least_cloudy_item.assets["B02"].href
    ni_href = least_cloudy_item.assets["B08"].href
    signed_href = pc.sign(asset_href)
    signed_rref = pc.sign(asset_rref)
    signed_gref = pc.sign(asset_gref)
    signed_bref = pc.sign(asset_bref)
    signed_nihref = pc.sign(ni_href)
    
    band_data = get_cropped(signed_href, area_of_interest)
    r_data = get_cropped(signed_rref, area_of_interest)
    g_data = get_cropped(signed_gref, area_of_interest)
    b_data = get_cropped(signed_bref, area_of_interest)
    ni_data = get_cropped(signed_nihref, area_of_interest)
    
    return(band_data, r_data, g_data, b_data, ni_data, least_cloudy_item.datetime.date())
    
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
    area_of_interest = bounding_box_from_centre(lat, lon, 3000)
    data = get_data(area_of_interest)
    if data is None:
        return(None)
    else :
        band_data, r_data, g_data, b_data, ni_data, date = data
        np.save(os.path.join(save_path, str(hotspot_id)+"_rgb.npy"), band_data)
        np.save(os.path.join(save_path, str(hotspot_id)+"_r.npy"),r_data)
        np.save(os.path.join(save_path, str(hotspot_id)+"_g.npy"), g_data)
        np.save(os.path.join(save_path, str(hotspot_id)+"_b.npy"), b_data)
        np.save(os.path.join(save_path, str(hotspot_id)+"_ni.npy"), ni_data)
        metadata = get_metadata(lon, lat, hotspot_id, area_of_interest, date)
        with open(os.path.join(save_path, str(hotspot_id)+".json"), 'w') as f:
            json.dump(metadata, f)
        f.close()
        print(f"Saved patch for hotspot {hotspot_id}")
        return(band_data, r_data, g_data, b_data, ni_data, metadata)

    
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
    
    container = 'sentinel2'


    for index, row in df.iterrows():

        hotspot_id = row['hotspot_id']
        lon = row['lon']
        lat = row['lat']
        a = get_patch(lon,lat, hotspot_id, save_path=".")
        if a is None:
            print("nothing found for " + hotspot_id) 
        else:
            band_data, r_data, g_data, b_data,ni_data, metadata = a
            print("\nUploading to Azure Storage as blob:\n\t" + str(hotspot_id))

            with open(str(hotspot_id)+"_rgb.npy","rb") as band_data:
                blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_rgb.npy")
                blob_client.upload_blob(band_data)
            with open(str(hotspot_id)+"_r.npy","rb") as band_data:
                blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_r.npy")
                blob_client.upload_blob(band_data)
            with open(str(hotspot_id)+"_g.npy","rb") as band_data:
                blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_g.npy")
                blob_client.upload_blob(band_data)
            with open(str(hotspot_id)+"_b.npy","rb") as band_data:
                blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_b.npy")
                blob_client.upload_blob(band_data)
            with open(str(hotspot_id)+"_ni.npy", "rb") as ni_data:
                blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_ni.npy")
                blob_client.upload_blob(ni_data)
            with open(str(hotspot_id)+".json", "rb") as metadata:
                blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+".json")
                blob_client.upload_blob(metadata)

            os.remove(str(hotspot_id)+"_rgb.npy")
            os.remove(str(hotspot_id)+"_ni.npy")
            os.remove(str(hotspot_id)+"_r.npy")
            os.remove(str(hotspot_id)+"_g.npy")
            os.remove(str(hotspot_id)+"_b.npy")
            os.remove(str(hotspot_id)+".json")
            print("removed files, moving to the next")
