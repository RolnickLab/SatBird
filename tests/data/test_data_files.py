import numpy as np
import pandas as pd
import os
import tifffile as tiff
import pytest
from tqdm import tqdm

from src.utils.config_utils import load_opts


@pytest.fixture
def config():
    base_dir = '.'
    default_config = os.path.join(base_dir, "configs/defaults.yaml")
    base_config = os.path.join(base_dir, "configs/base.yaml")

    config = load_opts(base_config, default=default_config, commandline_opts=None)

    return config


@pytest.fixture
def hotspots_list(config):
    df_train = pd.read_csv(os.path.join(config.data.files.base, config.data.files.train))
    train_hotspots = df_train["hotspot_id"].values.tolist()

    df_val = pd.read_csv(os.path.join(config.data.files.base, config.data.files.val))
    val_hotspots = df_val["hotspot_id"].values.tolist()

    df_test = pd.read_csv(os.path.join(config.data.files.base, config.data.files.test))
    test_hotspots = df_test["hotspot_id"].values.tolist()

    return train_hotspots + val_hotspots + test_hotspots


@pytest.mark.fast
def test_corresponding_files_from_csv(config, hotspots_list) -> None:
    """
    given hotspots, verify that their corresponding data files (in env data, targets, images and images visual) are there
    """
    # assert if environmental data doesn't contain all hotspot files
    environmental_data = os.listdir(os.path.join(config.data.files.base, "environmental"))
    assert len(environmental_data) >= len(hotspots_list)
    hotspots_from_env_files = [file.split('.')[0] for file in environmental_data]
    assert set(hotspots_list).issubset(hotspots_from_env_files)

    # assert if targets doesn't contain all hotspot files
    targets = os.listdir(os.path.join(config.data.files.base, config.data.files.targets_folder))
    assert len(targets) >= len(hotspots_list)
    hotspots_from_target_files = [file.split('.')[0] for file in targets]
    assert set(hotspots_list).issubset(hotspots_from_target_files)

    # assert if refl images doesn't contain all hotspot files
    sat_images = os.listdir(os.path.join(config.data.files.base, "images"))
    assert len(sat_images) >= len(hotspots_list)
    hotspots_refl_image_files = [file.split('.')[0] for file in sat_images]
    assert set(hotspots_list).issubset(hotspots_refl_image_files)

    # assert if visual images doesn't contain all hotspot files
    sat_images = os.listdir(os.path.join(config.data.files.base, "images_visual"))
    assert len(sat_images) >= len(hotspots_list)
    hotspots_from_visual_image_files = [file.split('_')[0] for file in sat_images]
    assert set(hotspots_list).issubset(hotspots_from_visual_image_files)


@pytest.mark.slow
def test_nan_refl_image_values(hotspots_list, config) -> None:
    """
    test that images have non Nan values
    """
    hotspots_with_nans = []
    sat_images = os.listdir(os.path.join(config.data.files.base, "images"))
    hotspots_from_sat_files = [file.split('.')[0] for file in sat_images]
    hotspots_from_sat_files = list(set(hotspots_from_sat_files).intersection(hotspots_list))

    for tif_file in tqdm(hotspots_from_sat_files):
        image_path = os.path.join(config.data.files.base, "images", tif_file + '.tif')
        img = tiff.imread(image_path)
        # Check if there are any NaN values in the image
        has_nan = np.isnan(img).any()
        if np.sum(has_nan):
            hotspots_with_nans.append(tif_file)
        # assert not has_nan
    np.save(os.path.join(config.data.files.base, "hotspots_with_nan_refl_images.npy"), np.array(hotspots_with_nans))
    assert len(hotspots_with_nans) == 0


@pytest.mark.slow
def test_nan_visual_image_values(hotspots_list, config) -> None:
    """
    test that images have non Nan values
    """
    hotspots_with_nans = []
    sat_images = os.listdir(os.path.join(config.data.files.base, "images_visual"))
    hotspots_from_sat_files = [file.split('_')[0] for file in sat_images]
    hotspots_from_sat_files = list(set(hotspots_from_sat_files).intersection(hotspots_list))

    for tif_file in tqdm(hotspots_from_sat_files):
        image_path = os.path.join(config.data.files.base, "images_visual", tif_file + '_visual.tif')
        img = tiff.imread(image_path)
        # Check if there are any NaN values in the image
        has_nan = np.isnan(img).any()
        if np.sum(has_nan):
            hotspots_with_nans.append(tif_file)
            # assert not has_nan
    np.save(os.path.join(config.data.files.base, "hotspots_with_nan_visual_images.npy"), np.array(hotspots_with_nans))
    assert len(hotspots_with_nans) == 0
