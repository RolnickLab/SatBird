"""
main training script
To run: python train.py args.config=$CONFIG_FILE_PATH
"""
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, cast
import comet_ml

import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.config_utils import load_opts
import src.trainer.trainer as general_trainer
import src.trainer.geo_trainer as geo_trainer
from src.utils.compute_normalization_stats import *


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):
    hydra_opts = dict(OmegaConf.to_container(opts))
    args = hydra_opts.pop("args", None)

    base_dir = args['base_dir']
    run_id = args["run_id"]
    if not base_dir:
        base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args['config'])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    conf = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    global_seed = (run_id * (conf.program.seed + (run_id - 1))) % (2 ** 31 - 1)

    # naming experiment folders with seed information
    conf.save_path = os.path.join(base_dir, conf.save_path, str(global_seed))
    conf.comet.experiment_name = conf.comet.experiment_name + '_seed_' + str(global_seed)
    conf.base_dir = base_dir

    # compute means and stds for normalization
    conf.variables.bioclim_means, conf.variables.bioclim_stds, conf.variables.ped_means,\
        conf.variables.ped_stds = compute_means_stds_env_vars(root_dir=conf.data.files.base, train_csv=conf.data.files.train)

    if conf.data.datatype == "refl":
        conf.variables.rgbnir_means, conf.variables.rgbnir_std = compute_means_stds_images(root_dir=conf.data.files.base,
                                                                                           train_csv=conf.data.files.train,
                                                                                           output_file_means=conf.data.files.rgbnir_means,
                                                                                           output_file_std=conf.data.files.rgbnir_stds)

    if conf.data.datatype == "img":
        conf.variables.visual_means, conf.variables.visual_stds = compute_means_stds_images_visual(root_dir=conf.data.files.base,
                                                                                                   train_csv=conf.data.files.train,
                                                                                                   output_file_means=conf.data.files.rgb_means,
                                                                                                   output_file_std=conf.data.files.rgb_stds)

    pl.seed_everything(global_seed)

    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)
    with open(os.path.join(conf.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(config=conf, f=fp)
    fp.close()

    # using general trainer without location information
    if not conf.loc.use:
        print("Using general trainer..")
        task = general_trainer.EbirdTask(conf)
        datamodule = general_trainer.EbirdDataModule(conf)
    # using geo-trainer (location encoder)
    elif conf.loc.use and conf.loc.loc_type == "latlon":
        print("Using geo-trainer with lat/lon info..")
        task = geo_trainer.EbirdTask(conf)
        datamodule = geo_trainer.EbirdDataModule(conf)
    else:
        print("cannot specify trainers based on config..")
        exit(0)

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
        save_top_k=1,
        mode="max",
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=True
    )

    trainer_args["callbacks"] = [checkpoint_callback]
    trainer_args["overfit_batches"] = conf.overfit_batches  # 0 if not overfitting
    trainer_args['max_epochs'] = conf.max_epochs

    if not conf.loc.use:
        trainer = pl.Trainer(**trainer_args)
        if conf.log_comet:
            trainer.logger.experiment.add_tags(list(conf.comet.tags))
        if conf.auto_lr_find:
            lr_finder = trainer.tuner.lr_find(task, datamodule=datamodule)

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
    # Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)

    # logging the best checkpoint to comet ML
    if conf.log_comet:
        print(checkpoint_callback.best_model_path)
        trainer.logger.experiment.log_asset(checkpoint_callback.best_model_path, file_name='best_checkpoint.ckpt')


if __name__ == "__main__":
    main()