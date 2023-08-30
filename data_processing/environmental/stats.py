import tifffile
import pandas as pd
import numpy as np


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]


def main():
    stats_df = pd.DataFrame(columns=["hotspot_id", "r","g","b","nir"])
    df = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/summer_hotspots_train.csv")
    
    for i, row in df.iterrows():
        hs = row["hotspot_id"]
        arr = tifffile.imread(f"/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/images/{hs}.tif")
        cropped = crop_center(arr, 64, 64)
        means = np.mean(np.mean(cropped, axis = 0),  axis = 0)
        new_row = {'hotspot_id':hs, 'r':means[2], 'g':means[1], 'b':means[0], 'nir':means[3]}
        stats_df= stats_df.append(new_row, ignore_index=True)
    stats_df.to_csv('/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/means_summer_images.csv')

    stats_df = pd.read_csv('/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/means_summer_images.csv')
    mean_r = stats_df["r"].mean()
    mean_g = stats_df["g"].mean()
    mean_b =stats_df["b"].mean()
    mean_nir =stats_df["nir"].mean()
    means = np.array([mean_r, mean_g, mean_b, mean_nir])
    print(means)
    stats_df_2 = pd.DataFrame(columns=["hotspot_id", "r_std","g_std","b_std","nir_std"])
    for i, row in df.iterrows():
        hs = row["hotspot_id"]
        arr = tifffile.imread(f"/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/images/{hs}.tif")
        cropped = crop_center(arr, 64,64)
        std = ((cropped-means)**2 /(64*64)).sum(axis = 0).sum(axis=0)
        new_row = {'hotspot_id':hs, 'r_std':std[2], 'g_std':std[1], 'b_std':std[0], 'nir_std':std[3]}
        stats_df_2= stats_df_2.append(new_row, ignore_index=True)
    stats_df_2.to_csv('/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/std_summer_images.csv')
    std_r = stats_df_2["r"].mean()
    std_g = stats_df_2["g"].mean()
    std_b = stats_df_2["b"].mean()
    std_nir = stats_df_2["nir"].mean()
    stds = np.array([np.sqrt(std_r), np.sqrt(std_g), np.sqrt(std_b), np.sqrt(std_nir)])
    print(stds)



if __name__ == "__main__":
    main()
