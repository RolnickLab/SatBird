# unit-tests for dataloader (size of image, label, mask)
import os
import pandas as pd
import torch
import pytest

from Rtran.dataloader import SDMVisionMaskedDataset
from src.utils.config_utils import load_opts


@pytest.fixture
def config():
    base_dir = '.'
    default_config = os.path.join(base_dir, "configs/defaults.yaml")
    base_config = os.path.join(base_dir, "configs/base.yaml")

    config = load_opts(base_config, default=default_config, commandline_opts=None)

    return config


@pytest.fixture
def dataframe(config):
    df = pd.read_csv(os.path.join(config.data.files.base, config.data.files.train))
    return df


@pytest.mark.fast
def test_data_loader(config, dataframe) -> None:
    """
    test that data loader has correct shapes
    """

    dataset = SDMVisionMaskedDataset(df=dataframe[0:100], data_base_dir=config.data.files.base, env=config.data.env,
        env_var_sizes=config.data.env_var_sizes, num_species=config.data.total_species)

    image_input_channels = len(config.data.bands)
    image_input_channels += sum(config.data.env_var_sizes) if len(config.data.env) > 0 else 0

    minibatch = dataset.__getitem__(0)
    assert minibatch["sat"].size()[0] == image_input_channels
    assert list(minibatch["target"].size()) == [config.data.total_species]


@pytest.mark.fast
def test_masked_data_loader(config, dataframe):
    """
    test the creation of masks
    """

    dataset = SDMVisionMaskedDataset(df=dataframe[0:100], data_base_dir=config.data.files.base, env=config.data.env,
        env_var_sizes=config.data.env_var_sizes, num_species=config.data.total_species)

    # for training, mask can include 3 values (unknown -1, negative 0, known 1)
    minibatch = dataset.__getitem__(0)
    assert len(torch.unique(minibatch["mask"])) == 3
    assert list(minibatch["mask"].size()) == [config.data.total_species]

    for mode in ["val", "test"]:
        dataset = SDMVisionMaskedDataset(df=dataframe[0:100], data_base_dir=config.data.files.base, env=config.data.env,
            mode=mode, env_var_sizes=config.data.env_var_sizes, num_species=config.data.total_species)

        # for validation/testing, mask should include (unknown)
        minibatch = dataset.__getitem__(0)
        assert len(torch.unique(minibatch["mask"])) == 1
        assert list(minibatch["mask"].size()) == [config.data.total_species]
