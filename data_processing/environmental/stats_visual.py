"""
[78.20186183 83.78967871 58.99852628]
[64.32016159 49.08529428 46.45643505]
"""
import tifffile
import pandas as pd
import numpy as np

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[starty:starty+cropy, startx:startx+cropx, :]

def main():
    stats_df = pd.DataFrame(columns=["hotspot_id", "r","g","b"])
    df = pd.read_csv("/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/summer_hotspots_train.csv")

    for i, row in df.iterrows():
        hs = row["hotspot_id"]
        arr = tifffile.imread(f"/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/images_visual/{hs}_visual.tif")
        cropped = crop_center(arr, 64, 64)
        means = np.mean(np.mean(cropped, axis = 0),  axis = 0)
        new_row = {'hotspot_id':hs, 'r':means[0], 'g':means[1], 'b': means[2]}
        stats_df= stats_df.append(new_row, ignore_index=True)
    stats_df.to_csv('/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/means_summer_images_visual.csv')

    
    stats_df = pd.read_csv('/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/means_summer_images_visual.csv')
    mean_r = stats_df["r"].mean()
    mean_g = stats_df["g"].mean()
    mean_b = stats_df["b"].mean()
    means = np.array([mean_r, mean_g, mean_b])
    print(means)
    stats_df_2 = pd.DataFrame(columns=["hotspot_id", "r_std","g_std","b_std","nir_std"])
    for i, row in df.iterrows():
        hs = row["hotspot_id"]
        arr = tifffile.imread(f"/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/images_visual/{hs}_visual.tif")
        cropped = crop_center(arr, 64,64)
        std = ((cropped-means)**2 /(64*64)).sum(axis = 0).sum(axis=0)
        new_row = {'hotspot_id':hs, 'r_std':std[0], 'g_std':std[1], 'b_std':std[2]}
        stats_df_2= stats_df_2.append(new_row, ignore_index=True)
    stats_df_2.to_csv('/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer/std_summer_images_visual.csv')

    std_r = np.sqrt(stats_df_2["r_std"].mean())
    std_g = np.sqrt(stats_df_2["g_std"].mean())
    std_b = np.sqrt(stats_df_2["b_std"].mean())
    stds = np.array([std_r, std_g, std_b])

    print(stds)

if __name__=="__main__":
    main()
