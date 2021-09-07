from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer as pc
import numpy as np
from PIL import Image
from rasterio import features
from shapely.geometry import Polygon, shape
import pandas as pd
import json
import pandas as pd
import os
import stackstac
from sentinel_data import bounding_box_from_centre, get_metadata
from azure.storage.blob import BlobServiceClient
import xrspatial.multispectral as ms


# Set the environment variable PC_SDK_SUBSCRIPTION_KEY, or set it here.
# The Hub sets PC_SDK_SUBSCRIPTION_KEY automatically.



def get_items(area_of_interest, cc):
    years = [y for y in range(2021, 2016, -1)]
    found = False
    items_ = []
    for y in years:
        time_of_interest = f"{str(y)}-06-01/{str(y)}-06-30"
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=area_of_interest,
            datetime=time_of_interest,
            query={"eo:cloud_cover": {"lt": cc}},
        )

        # Check how many items were returned
        items_ += list(search.get_items())
    print(len(items_))
    return(items_)

def save_local(save_path, hotspot_id, lon, lat, band_data, r_data, g_data, b_data, ni_data, metadata):
    np.save(os.path.join(save_path, str(hotspot_id)+"_rgb.npy"), band_data)
    np.save(os.path.join(save_path, str(hotspot_id)+"_r.npy"),r_data)
    np.save(os.path.join(save_path, str(hotspot_id)+"_g.npy"), g_data)
    np.save(os.path.join(save_path, str(hotspot_id)+"_b.npy"), b_data)
    np.save(os.path.join(save_path, str(hotspot_id)+"_ni.npy"), ni_data)
   
    with open(os.path.join(save_path, str(hotspot_id)+".json"), 'w') as f:
        json.dump(metadata, f)
    f.close()
    print(f"Saved patch for hotspot {hotspot_id}")
    
if __name__=="__main__":
    pc.settings.set_subscription_key("")

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    connection_string = ""
    service_client = BlobServiceClient.from_connection_string(connection_string)

    with open('incomplete.txt', 'r') as dat:   #second step "incomplete_flagged.txt"
        lines = dat.readlines()

    lines = [l.strip('\n') for l in lines]

    df = pd.read_csv('./data/hotspots_latlon.csv')

    inc = df[df['hotspot_id'].isin(lines)].copy()
    inc = inc.reset_index()

    flagged_txt = "./data/incomplete_flagged.txt" #second step "incomplete_flagged_2.txt"
        save_path = "./completed/"

    container = 'sentinel2complete'

    for i, row in inc.iterrows():

        hotspot_id = row["hotspot_id"]
        print(hotspot_id)

        if not service_client.get_blob_client(container, blob= str(hotspot_id)+"_rgb.npy").exists():
            lat = row["lat"]
            lon = row["lon"]
            poly = bounding_box_from_centre(lat, lon, 3000)
            items_ = get_items_step1(poly, 10) #20 in step 2


            signed_items = []
            for item in items_:
                item.clear_links()
                signed_items.append(pc.sign(item).to_dict())

            try:
                rgbni = (
                    stackstac.stack(
                        signed_items,
                        assets=["B04", "B03", "B02", "B08"],  # red, green, blue
                        chunksize=4096,
                        resolution=10,
                        bounds_latlon=features.bounds(poly)
                    )
                    .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
                    .assign_coords(band=lambda x: x.common_name.rename("band"))  # use common names
                )

                mosaic = rgbni.median(dim="time").compute()

                if np.sum(mosaic.isnull()).values > 0: 
                    with open(flagged_txt, "a") as f:
                        f.write(hotspot_id)
                        f.write('\n')
                    f.close()
                else: 
                    colored = ms.true_color(*mosaic[:3,:,:]) 
                    colored = colored[:,:,:3].astype(np.uint8) #only keep rgb (drop A)
                    colored = np.transpose(np.array(colored), (2,0,1))

                    r,g,b, ni = mosaic.data.astype(np.uint16)

                    metadata = get_metadata(lon, lat, hotspot_id, poly, rgbni.coords["time"][-1].values, earliest = rgbni.coords["time"][0].values)

                    save_local(save_path, hotspot_id, lon, lat, colored, r, g, b, ni,metadata)


               # if not service_client.get_blob_client(container, blob= str(hotspot_id)+"_rgb.npy").exists():
                    print("\nUploading to Azure Storage as blob:\n\t" + str(hotspot_id))
                    with open(os.path.join(save_path, str(hotspot_id)+"_rgb.npy"),"rb") as banddata:
                        blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_rgb.npy")
                        blob_client.upload_blob(banddata)
                    banddata.close()
                    with open(os.path.join(save_path, str(hotspot_id)+"_r.npy"),"rb") as rdata:
                        blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_r.npy")
                        blob_client.upload_blob(rdata)
                    rdata.close()
                    with open(os.path.join(save_path, str(hotspot_id)+"_g.npy"),"rb") as gdata:
                        blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_g.npy")
                        blob_client.upload_blob(gdata)
                    gdata.close()
                    with open(os.path.join(save_path, str(hotspot_id)+"_b.npy"),"rb") as bdata:
                        blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_b.npy")
                        blob_client.upload_blob(bdata)
                    bdata.close()
                    with open(os.path.join(save_path, str(hotspot_id)+"_ni.npy"), "rb") as ni_data:
                        blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+"_ni.npy")
                        blob_client.upload_blob(ni_data)
                    ni_data.close()
                    with open(os.path.join(save_path, str(hotspot_id)+".json"), "rb") as metadata:
                        blob_client = service_client.get_blob_client(container, blob= str(hotspot_id)+".json")
                        blob_client.upload_blob(metadata)
                    metadata.close()

                    os.remove(os.path.join(save_path, str(hotspot_id)+"_rgb.npy"))
                    os.remove(os.path.join(save_path, str(hotspot_id)+"_ni.npy"))
                    os.remove(os.path.join(save_path, str(hotspot_id)+"_r.npy"))
                    os.remove(os.path.join(save_path, str(hotspot_id)+"_g.npy"))
                    os.remove(os.path.join(save_path, str(hotspot_id)+"_b.npy"))
                    os.remove(os.path.join(save_path, str(hotspot_id)+".json"))
                    print("removed files, moving to the next")
            except:
                with open("./data/problem.txt", "a") as f:
                    f.write(hotspot_id)
                    f.write('\n')
                f.close()
                continue
