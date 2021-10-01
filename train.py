# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Tuple, Type, cast
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.trainer.trainer import EbirdTask, EbirdDataModule
from omegaconf import DictConfig, OmegaConf

print(EbirdTask)

TASK_TO_MODULES_MAPPING: Dict[
    str, Tuple[Type[pl.LightningModule], Type[pl.LightningDataModule]]
] = {
    "ebird_classifier": (EbirdTask, EbirdDataModule),
    
}




def set_up_omegaconf()-> DictConfig:
    """Helps with loading config files"""
    
    conf = OmegaConf.load("config/defaults.yaml")
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

    task_name = conf.experiment.task
    task_config_fn = os.path.join("config", "task_defaults", f"{task_name}.yaml")
    if task_name == "test":
        task_conf = OmegaConf.create()
    elif os.path.exists(task_config_fn):
        task_conf = cast(DictConfig, OmegaConf.load(task_config_fn))
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    conf = OmegaConf.merge(task_conf, conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright

    return conf

def main(conf: DictConfig) -> None:
    """Main Training loop"""
    experiment_name = conf.experiment.name
    task_name = conf.experiment.task

    if os.path.isfile(conf.program.output_dir):
        raise NotAdirectoryError("program.output_dir must be a directory")

    os.makedirs(conf.program.output_dir, exist_ok=True)
    
    experiment_dir = os.path.join(conf.program.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok = True)

    if len(os.listdir(experiment_dir)) > 0:
        if conf.program.overwrite:
            print(
                f"WARNING! The experiment directory, {experiment_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            raise FileExistsError(
                f"The experiment directory, {experiment_dir}, already exists and isn't "
                + "empty. We don't want to overwrite any existing results, exiting..."
            )

    with open(os.path.join(experiment_dir, "experiment_config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    task_args = cast(Dict[str, Any], OmegaConf.to_object(conf.experiment.module))
    datamodule_args = cast(
        Dict[str, Any], OmegaConf.to_object(conf.experiment.datamodule)
    )

    datamodule: pl.LigtningDataModuleModule
    task: pl.LightningModule

    if task_name in TASK_TO_MODULES_MAPPING:
        task_class, datamodule_class = TASK_TO_MODULES_MAPPING[task_name]
        task = task_class(**task_args)
        datamodule = datamodule_class(**datamodule_args)
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    # Setup trainer

    tb_logger = pl_loggers.TensorBoardLogger(conf.program.log_dir, name=experiment_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_dir,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
    )

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))
    trainer_args["callbacks"] = [checkpoint_callback, early_stopping_callback]
    trainer_args["logger"] = tb_logger
    trainer = pl.Trainer(**trainer_args)

    ## Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)

if __name__ == "__main__":
    conf = set_up_omegaconf()
    pl.seed_everything(conf.program.seed)
    
    # Main training procedure
    main(conf)