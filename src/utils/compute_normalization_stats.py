"""
utility file for computing normalization means and stds for training
"""
import os
import tifffile
import pandas as pd
import numpy as np
from pandas import DataFrame
from tqdm import tqdm


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def compute_means_stds_images(root_dir, train_csv, output_file_means="stats/means_summer_rgbnir.npy",
                              output_file_std="stats/stds_summer_rgbnir.npy"):
    """
    computes normalization statistics (means, stds) on training data, for RGBNIR refl channels
    """

    df = pd.read_csv(os.path.join(root_dir, train_csv))
    stats_df = pd.DataFrame(columns=["hotspot_id", "r", "g", "b", "nir"])
    output_file_means_path = os.path.join(root_dir, output_file_means)
    if os.path.exists(output_file_means_path):
        means = np.load(output_file_means_path)
    else:
        for i, row in tqdm(df.iterrows()):
            hs = row["hotspot_id"]
            arr = tifffile.imread(os.path.join(root_dir, f"images/{hs}.tif"))
            cropped = crop_center(arr, 64, 64)
            means = np.mean(np.mean(cropped, axis=0), axis=0)
            new_row = {'hotspot_id': hs, 'r': means[2], 'g': means[1], 'b': means[0], 'nir': means[3]}
            stats_df = stats_df.append(new_row, ignore_index=True)

        mean_r = stats_df["r"].mean()
        mean_g = stats_df["g"].mean()
        mean_b = stats_df["b"].mean()
        mean_nir = stats_df["nir"].mean()
        means = np.array([mean_b, mean_g, mean_r, mean_nir])
        means_to_save = np.array([mean_r, mean_g, mean_b, mean_nir])
        np.save(output_file_means_path, means_to_save)

    print("Images RGBNIR means: ", means)

    output_file_stds_path = os.path.join(root_dir, output_file_std)
    if os.path.exists(output_file_stds_path):
        stds = np.load(output_file_stds_path, allow_pickle=True, fix_imports=True, encoding='latin1')
    else:
        stats_df_2 = pd.DataFrame(columns=["hotspot_id", "r_std", "g_std", "b_std", "nir_std"])
        for i, row in tqdm(df.iterrows()):
            hs = row["hotspot_id"]
            arr = tifffile.imread(os.path.join(root_dir, f"images/{hs}.tif"))
            cropped = crop_center(arr, 64, 64)
            std = ((cropped - means) ** 2 / (64 * 64)).sum(axis=0).sum(axis=0)
            new_row = {'hotspot_id': hs, 'r_std': std[2], 'g_std': std[1], 'b_std': std[0], 'nir_std': std[3]}
            stats_df_2 = stats_df_2.append(new_row, ignore_index=True)
            std_r = np.sqrt(stats_df_2["r_std"].mean())
            std_g = np.sqrt(stats_df_2["g_std"].mean())
            std_b = np.sqrt(stats_df_2["b_std"].mean())
            std_nir = np.sqrt(stats_df_2["nir_std"].mean())
            stds = np.array([std_r, std_g, std_b, std_nir])
            np.save(output_file_stds_path, stds)

    print("Images RGBNIR stds: ", stds)

    return means.tolist(), stds.tolist()


def compute_means_stds_images_visual(root_dir, train_csv, output_file_means="stats/means_summer_images_visual.npy",
                                     output_file_std="stats/stds_summer_images_visual.npy"):
    """
    computes normalization statistics (means, stds) on training data, for RGB visual channels
    """
    stats_df: DataFrame = pd.DataFrame(columns=["hotspot_id", "r", "g", "b", "nir"])
    df = pd.read_csv(os.path.join(root_dir, train_csv))

    output_file_means_path = os.path.join(root_dir, output_file_means)
    if os.path.exists(output_file_means_path):
        means = np.load(output_file_means_path)
    else:
        for i, row in tqdm(df.iterrows()):
            hs = row["hotspot_id"]
            arr = tifffile.imread(os.path.join(root_dir, f"images_visual/{hs}.tif"))
            cropped = crop_center(arr, 64, 64)
            means = np.mean(np.mean(cropped, axis=0), axis=0)
            new_row = {'hotspot_id': hs, 'r': means[2], 'g': means[1], 'b': means[0], 'nir': means[3]}
            stats_df = stats_df.append(new_row, ignore_index=True)

        mean_r = stats_df["r"].mean()
        mean_g = stats_df["g"].mean()
        mean_b = stats_df["b"].mean()
        mean_nir = stats_df["nir"].mean()
        means = np.array([mean_b, mean_g, mean_r, mean_nir])
        means_to_save = np.array([mean_r, mean_g, mean_b, mean_nir])
        np.save(output_file_means_path, means_to_save)
    print("Images-visual RGB means: ", means_to_save)

    output_file_stds_path = os.path.join(root_dir, output_file_std)
    if os.path.exists(output_file_stds_path):
        stds = np.load(output_file_stds_path)
    else:
        stats_df_2: DataFrame = pd.DataFrame(columns=["hotspot_id", "r_std", "g_std", "b_std", "nir_std"])
        for i, row in tqdm(df.iterrows()):
            hs = row["hotspot_id"]
            arr = tifffile.imread(os.path.join(root_dir, f"images_visual/{hs}.tif"))
            cropped = crop_center(arr, 64, 64)
            std = ((cropped - means) ** 2 / (64 * 64)).sum(axis=0).sum(axis=0)
            new_row = {'hotspot_id': hs, 'r_std': std[2], 'g_std': std[1], 'b_std': std[0], 'nir_std': std[3]}
            stats_df_2 = stats_df_2.append(new_row, ignore_index=True)
            std_r = np.sqrt(stats_df_2["r_std"].mean())
            std_g = np.sqrt(stats_df_2["g_std"].mean())
            std_b = np.sqrt(stats_df_2["b_std"].mean())
            std_nir = np.sqrt(stats_df_2["nir_std"].mean())
            stds = np.array([std_r, std_g, std_b, std_nir])
            np.save(output_file_stds_path, stds)

    print("Images-visual RGB stds: ", stds)
    return means.tolist(), stds.tolist()


def compute_means_stds_env_vars(root_dir, train_csv):
    """
    computes normalization statistics (means, stds) on training data, for environmental variables
    """
    bioclim_env_column_names = ['bio_1', 'bio_2', 'bio_3', 'bio_4', 'bio_5',
                                'bio_6', 'bio_7', 'bio_8', 'bio_9', 'bio_10', 'bio_11', 'bio_12',
                                'bio_13', 'bio_14', 'bio_15', 'bio_16', 'bio_17', 'bio_18', 'bio_19']
    ped_env_column_names = ['bdticm', 'bldfie', 'cecsol', 'clyppt', 'orcdrc', 'phihox', 'sltppt', 'sndppt']

    df = pd.read_csv(os.path.join(root_dir, train_csv))

    bioclim_means = np.nanmean(df[bioclim_env_column_names].values.tolist(), axis=0)
    bioclim_stds = np.nanstd(df[bioclim_env_column_names].values.tolist(), axis=0)

    ped_means = np.nanmean(df[ped_env_column_names].values.tolist(), axis=0)
    ped_stds = np.nanstd(df[ped_env_column_names].values.tolist(), axis=0)

    return bioclim_means.tolist(), bioclim_stds.tolist(), ped_means.tolist(), ped_stds.tolist()


if __name__ == "__main__":
    bioclim_means, bioclim_stds, ped_means, ped_stds = compute_means_stds_env_vars(root_dir= "/network/projects/ecosystem-embeddings/ebird_dataset/USA_summer",
                train_csv="summer_hotspots_train_with_nan_envvars.csv")

    print(bioclim_means, bioclim_stds)
    print(ped_means, ped_stds)

    # image_means, image_stds = compute_means_stds_images(
    #     root_dir="/network/projects/ecosystem-embeddings/ebird_dataset_v2/USA_summer",
    #     train_csv="butterfly_hotspots_train.csv")

    # image_visual_means, image_visual_stds = compute_means_stds_images_visual(root_dir="/network/projects/ecosystem-embeddings/ebird_dataset_v2/USA_summer",
    #                                                     train_csv="butterfly_hotspots_train.csv")
