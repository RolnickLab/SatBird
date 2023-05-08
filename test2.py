import comet_ml
import os
from pathlib import Path
from os.path import expandvars
import hydra
from hydra.utils import get_original_cwd
from addict import Dict
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, cast

import src.trainer.trainer as general_trainer
import src.trainer.geo_trainer as geo_trainer
import src.trainer.state_trainer as state_trainer
import src.trainer.multires_trainer as multires_trainer
from src.dataset.utils import set_data_paths

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger

hydra_config_path = Path(__file__).resolve().parent / "configs/hydra.yaml"


def resolve(path):
    """
    fully resolve a path:
    resolve env vars ($HOME etc.) -> expand user (~) -> make absolute
    Returns:
        pathlib.Path: resolved absolute path
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def set_up_omegaconf() -> DictConfig:
    """Helps with loading config files"""

    conf = OmegaConf.load("./configs/defaults.yaml")
    command_line_conf = OmegaConf.from_cli()

    if "config_file" in command_line_conf:

        config_fn = command_line_conf.config_file

        if os.path.isfile(config_fn):
            user_conf = OmegaConf.load(config_fn)
            conf = OmegaConf.merge(conf, user_conf)
        else:
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

    conf = OmegaConf.merge(
        conf, command_line_conf
    )
    conf = set_data_paths(conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright

    # if commandline_opts is not None and isinstance(commandline_opts, dict):
    #    opts = Dict(merge(commandline_opts, opts))
    return conf


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
        path = (
                resolve(Path(__file__)).parent.parent
                / "config"
                / "defaults.yaml"
        )
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
        print("Commandline opts", commandline_opts)

    conf = set_data_paths(opts)
    conf = cast(DictConfig, opts)
    return conf


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):
    hydra_opts = dict(OmegaConf.to_container(opts))
    print("hydra_opts", hydra_opts)
    args = hydra_opts.pop("args", None)

    base_dir = args['base_dir']
    if not base_dir:
        base_dir = get_original_cwd()
    print(base_dir)
    config_path = os.path.join(base_dir, args['config'])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    conf = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    conf.base_dir = base_dir
    print(conf.base_dir)
    conf.save_path = os.path.join(base_dir, conf.save_path, os.environ["SLURM_JOB_ID"])
    pl.seed_everything(conf.program.seed)

    print('multires len:', len(conf.data.multiscale))
    if not conf.loc.use and len(conf.data.multiscale) > 1:
        print('using multiscale net')
        task = multires_trainer.EbirdTask(conf)
        datamodule = EbirdDataModule(conf)
    elif "speciesAtoB" in conf.keys() and conf.speciesAtoB:
        print("species A to B")
        task = EbirdSpeciesTask(conf)
        datamodule = EbirdDataModule(conf)
    elif not conf.loc.use:
        task = general_trainer.EbirdTask(conf)
        datamodule = general_trainer.EbirdDataModule(conf)
    elif conf.loc.loc_type == "latlon":
        print("Using geo information")
        task = geo_trainer.EbirdTask(conf)
        datamodule = geo_trainer.EbirdDataModule(conf)
    elif conf.loc.loc_type == "state":
        print("Using geo information")
        task = state_trainer.EbirdTask(conf)
        datamodule = state_trainer.EbirdDataModule(conf)

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))

    if conf.comet.experiment_key:
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            # save_dir=".",
            project_name=conf.comet.project_name,
            experiment_name=conf.comet.experiment_name,
            experiment_key=conf.comet.experiment_key
        )

        trainer_args["logger"] = comet_logger

    # "/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527294/last.ckpt"
    # above with landuse : /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527309
    # sat landuse env 512  /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527306
    # Sat landuse  env 224 /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527294
    print("Checkpoint: ", os.path.join(base_dir, conf.load_ckpt_path))
    if conf.load_ckpt_path:
        print("Loading existing checkpoint")
        try:
            task = task.load_from_checkpoint(os.path.join(base_dir, conf.load_ckpt_path),
                                         save_preds_path=conf.save_preds_path)
        # to prevent older models from failing, because there are new keys in conf
        except:
            task.load_state_dict(torch.load(os.path.join(base_dir, conf.load_ckpt_path))['state_dict'])
    else:
        print("No checkpoint provided...Evaluating a random model")

    trainer = pl.Trainer(**trainer_args)
    trainer.validate(model=task, datamodule=datamodule)
    trainer.test(model=task,
                 dataloaders=datamodule.test_dataloader(),
                 verbose=True)


if __name__ == "__main__":
    main()
