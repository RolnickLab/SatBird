from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from os.path import expandvars
from src.dataset.utils import set_data_paths
from typing import cast


def resolve(path):
    """
    fully resolve a path:
    resolve env vars ($HOME etc.) -> expand user (~) -> make absolute
    Returns:
        pathlib.Path: resolved absolute path
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def load_opts(path, default, commandline_opts):
    """
        Args:
        path (pathlib.Path): where to find the overriding configuration
            default (pathlib.Path, optional): Where to find the default opts.
            Defaults to None. In which case it is assumed to be a default config
            which needs processing such as setting default values for lambdas and gen
            fields
     """

    if path is None and default is None:
        path = (resolve(Path(__file__)).parent.parent / "configs" / "defaults.yaml")
        print(path)
    else:
        print("using config ", path)

    if default is None:
        default_opts = {}
    else:
        print(default)
        if isinstance(default, (str, Path)):
            default_opts = OmegaConf.load(default)
        else:
            default_opts = dict(default)

    if path is None:
        overriding_opts = {}
    else:
        print("using config ", path)
        overriding_opts = OmegaConf.load(path)

    opts = OmegaConf.merge(default_opts, overriding_opts)

    if commandline_opts is not None and isinstance(commandline_opts, dict):
        opts = OmegaConf.merge(opts, commandline_opts)

    # conf = set_data_paths(opts)
    conf = cast(DictConfig, opts)
    return conf
