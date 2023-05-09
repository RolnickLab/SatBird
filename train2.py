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
from src.dataset.utils import set_data_paths

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, BackboneFinetuning


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
                / "configs"
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
    args = hydra_opts.pop("args", None)

    base_dir = args['base_dir']
    if not base_dir:
        base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args['config'])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    conf = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    conf.save_path = os.path.join(base_dir, conf.save_path, os.environ["SLURM_JOB_ID"])
    pl.seed_everything(conf.program.seed)
    conf.base_dir = base_dir

    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)
    with open(os.path.join(conf.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(config=conf, f=fp)
    fp.close()

    if "speciesAtoB" in conf.keys() and conf.speciesAtoB:
        print("species A to B")
        task = EbirdSpeciesTask(conf)
        datamodule = general_trainer.EbirdDataModule(conf)
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

    if conf.log_comet:

        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            # save_dir=".",  # Optional
            project_name=conf.comet.project_name,  # Optional
            experiment_name=conf.comet.experiment_name,
        )
        comet_logger.experiment.add_tags(list(conf.comet.tags))
        print(conf.comet.tags)
        trainer_args["logger"] = comet_logger
    else:
        wandb_logger = WandbLogger(project='test-project')
        print('in wandb logger')
        trainer_args["logger"] = wandb_logger

    checkpoint_callback = ModelCheckpoint(
        monitor="val_topk_epoch",
        dirpath=conf.save_path,
        save_top_k=2,
        mode="max",
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_topk",
        min_delta=0.00,
        patience=4,
        mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer_args["callbacks"] = [checkpoint_callback]
    trainer_args["overfit_batches"] = conf.overfit_batches  # 0 if not overfitting
    trainer_args['max_epochs'] = conf.max_epochs

    if not conf.loc.use:

        trainer = pl.Trainer(**trainer_args)
        if conf.log_comet:
            trainer.logger.experiment.add_tags(list(conf.comet.tags))
        if conf.auto_lr_find:
            lr_finder = trainer.tuner.lr_find(task, datamodule=datamodule)

            # Results can be found in
            """   #lr_finder.results

            # Plot with
            fig = lr_finder.plot(suggest=True)
            fig.show()
            fig.savefig("learningrate.jpg")
            """
            # Pick point based on plot, or get suggestion
            new_lr = lr_finder.suggestion()

            # update hparams of the model
            task.hparams.learning_rate = new_lr
            task.hparams.lr = new_lr
            trainer.tune(model=task, datamodule=datamodule)
    else:

        trainer = pl.Trainer(**trainer_args)
        if conf.log_comet:
            trainer.logger.experiment.add_tags(list(conf.comet.tags))

    ## Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
