import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import windows, features, warp
from shapely import wkt
from shapely.geometry import mapping
from retrying import retry
from tqdm import tqdm
import pystac_client
import planetary_computer
import argparse

planetary_computer.settings.set_subscription_key("api-key")
# This should be a secret!! ask me for mine

# Incase Planetary computer sleeps off,
# Define the number of retries and the wait interval between retries
NUM_RETRIES = 5
WAIT_INTERVAL = 1  # seconds


# Open client and set subscription key
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    # modifier=planetary_computer.sign_inplace, --> this is depricated ??
)

# Define the bands we are interested in --> r,g,b,nir and true color image or "visual"
BANDS = ["B02", "B03", "B04", "B08"]

time_of_interest = "2022-06-01/2022-07-31" #this is for summer, if winter use "2022-12-01/2023-01-31"


def convert_polygon(polygon_str):
    """
    Convert a POLYGON string to the desired dictionary format.

    Parameters:
        polygon_str (str): The POLYGON string representing the geometry.

    Returns:
        dict: The geometry in the desired dictionary format.
    """
    # Convert string to shapely Polygon object
    polygon_obj = wkt.loads(polygon_str)
    # Convert the Polygon object to dictionary format
    polygon_dict = mapping(polygon_obj)
    # Convert tuples to lists
    polygon_dict["coordinates"] = [
        [list(point) for point in polygon] for polygon in polygon_dict["coordinates"]
    ]

    return polygon_dict


@retry(stop_max_attempt_number=NUM_RETRIES, wait_fixed=WAIT_INTERVAL * 1000)
def process_row(row, save_dir):
    area_of_interest = row["geometry"]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 10}},
    )

    items = search.get_all_items()

    items = planetary_computer.sign(items)

    least_cloudy_item = min(items, key=lambda item: item.properties["eo:cloud_cover"])

    # Create an empty list to hold band data
    concatenated_band_data = []

    for band in BANDS:
        asset_href = least_cloudy_item.assets[band].href
        with rasterio.open(asset_href) as ds:
            aoi_bounds = features.bounds(area_of_interest)
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            band_data = ds.read(window=aoi_window)
            concatenated_band_data.append(band_data)

    # Stack arrays along new third axis
    concatenated_band_data = np.stack(concatenated_band_data, axis=0)
    concatenated_band_data = np.squeeze(concatenated_band_data)

    # Save the image with a name based on hotspot_id
    image_filename = os.path.join(save_dir, f"{row['hotspot_id']}.tif")

    with rasterio.open(
        image_filename,
        "w",
        driver="GTiff",
        height=concatenated_band_data.shape[1],
        width=concatenated_band_data.shape[2],
        count=concatenated_band_data.shape[0],
        dtype=concatenated_band_data.dtype,
        crs=ds.crs,
        transform=ds.transform,
    ) as dest:
        dest.write(concatenated_band_data)

        
@retry(stop_max_attempt_number=NUM_RETRIES, wait_fixed=WAIT_INTERVAL * 1000)
def process_row_mosaic(row, save_dir,  cc=20, no_item_found_txt="no_item_found.txt"):
    """
    Use this after trying to extract images with process_row first
    """
    area_of_interest = row["geometry"]

    search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=area_of_interest,
            datetime=time_of_interest,
            query={"eo:cloud_cover": {"lt": cc}},
        )

    items_ = list(search.get_items())
    items = []
    for item in items_:
        item.clear_links()
        items.append(planetary_computer.sign(item).to_dict())
    
    if len(items) == 0:
        print(f"no item found for hotspot {row['hotspot_id']}")
        with open( no_item_found_txt, "a") as f:
            f.write(row['hotspot_id'])
            f.write('\n')
    else:
            # Create an empty list to hold band data

        try:    
            concatenated_band_data = (
                stackstac.stack(
                    items,
                    assets=BANDS, 
                    chunksize=4096,
                    resolution=10,
                    bounds_latlon=features.bounds(area_of_interest)
                )
                .where(lambda x: x > 0, other=np.nan)  # sentinel-2 uses 0 as nodata
                .assign_coords(band=lambda x: x.common_name.rename("band"))  # use common names
            )

            concatenated_band_data = concatenated_band_data.median(dim="time").compute()

            if np.sum(concatenated_band_data.isnull()).values > 0: 
                #this means the extracted tile is incomplete 
                with open(no_item_found_txt, "a") as f:
                    f.write(row["hotspot_id"])
                    f.write('\n')
                f.close()


            # Save the image with a name based on hotspot_id
            image_filename = os.path.join(save_dir, f"{row['hotspot_id']}.tif")

            with rasterio.open(
                image_filename,
                "w",
                driver="GTiff",
                height=concatenated_band_data.shape[1],
                width=concatenated_band_data.shape[2],
                count=concatenated_band_data.shape[0],
                dtype=concatenated_band_data.dtype,
                #crs=ds.crs,
                #transform=ds.transform,
            ) as dest:
                dest.write(concatenated_band_data)
        except Exception as e:
            print(f"Error processing row: {e}")
            with open(no_item_found_txt, "a") as f:
                    f.write(row["hotspot_id"])
                    f.write('\n')
                    

def main():
    
    # Specify the directory to save the rasters
    WINTER_SAVE_DIRECTORY = "/network/projects/ecosystem-embeddings/ebird_new/rasters_new/winter_rasters/"
    SUMMER_SAVE_DIRECTORY = "/network/projects/_groups/ecosystem-embeddings/ebird_new/rasters_new/summer_rasters"

    winter_polygons = "/network/projects/ecosystem-embeddings/ebird_new/polygons_winter.csv"
    summer_polygons = "/network/projects/ecosystem-embeddings/ebird_new/polygons_summer.csv"

    arg_parser = argparse.ArgumentParser(
        prog='DownloadData',
        description='download rasters from planetary compute')

    arg_parser.add_argument('-i', '--index', default=1, type=int)
    arg_parser.add_argument('-r', '--range', default=20000, type=int)
    arg_parser.add_argument('-s', '--season', default="summer", type=str)
    args = arg_parser.parse_args()

    index = int(args.index) - 1
    range = int(args.range)
    season_of_interest = args.season

    # Create the save directory if it doesn't exist
    os.makedirs(WINTER_SAVE_DIRECTORY, exist_ok=True)
    os.makedirs(SUMMER_SAVE_DIRECTORY, exist_ok=True)

    if season_of_interest == "summer":
        save_dir = SUMMER_SAVE_DIRECTORY
        polygons_file = summer_polygons
    else:
        save_dir = WINTER_SAVE_DIRECTORY
        polygons_file = winter_polygons

    # swap data input from here: Winter Polygons if winter, summer if summer
    data = pd.read_csv(polygons_file)

    data = data.iloc[index*range: (index+1)*range]
    data_df = gpd.GeoDataFrame(data)

    # data_df = data_df[:50]
    # Loop over the rows and process each row with retry logic
    data_df["geometry"] = data_df["geometry"].apply(convert_polygon)

    for _, row in tqdm(
        data_df.iterrows(), total=data_df.shape[0], desc="Processing images"
    ):
        try:
            #process_row or process_row_mosaic
            process_row(row, save_dir)
        except Exception as e:
            print(f"Error processing row: {e}")
            # Handle the error or raise an exception if desired
            # Retry mechanism will automatically retry the row processing

    print("Image processing completed.")


# that's all, folks :)

if __name__ == "__main__":
    main()
