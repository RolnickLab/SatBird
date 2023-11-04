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

    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    global_seed = (run_id * (config.program.seed + (run_id - 1))) % (2 ** 31 - 1)

    # naming experiment folders with seed information
    config.save_path = os.path.join(base_dir, config.save_path, str(global_seed))
    config.comet.experiment_name = config.comet.experiment_name + '_seed_' + str(global_seed)
    config.base_dir = base_dir

    # compute means and stds for normalization
    if len(config.data.env) > 0:
        config.variables.bioclim_means, config.variables.bioclim_std, config.variables.ped_means, config.variables.ped_std = compute_means_stds_env_vars(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            env=config.data.env,
            env_data_folder=config.data.files.env_data_folder,
            output_file_means=config.data.files.env_means,
            output_file_std=config.data.files.env_stds
        )

    if config.data.datatype == "refl":
        config.variables.rgbnir_means, config.variables.rgbnir_std = compute_means_stds_images(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            output_file_means=config.data.files.rgbnir_means,
            output_file_std=config.data.files.rgbnir_stds)
    elif config.data.datatype == "img" and not config.data.transforms[4].normalize_by_255:
        config.variables.visual_means, config.variables.visual_stds = compute_means_stds_images_visual(
            root_dir=config.data.files.base,
            train_csv=config.data.files.train,
            output_file_means=config.data.files.rgb_means,
            output_file_std=config.data.files.rgb_stds)

    # set global seed
    pl.seed_everything(global_seed)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    with open(os.path.join(config.save_path, "config.yaml"), "w") as fp:
        OmegaConf.save(config=config, f=fp)
    fp.close()

    # using general trainer without location information
    print("Using general trainer..")
    task = general_trainer.EbirdTask(config)
    datamodule = general_trainer.EbirdDataModule(config)

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(config.trainer))

    if config.log_comet:
        if os.environ.get("COMET_API_KEY"):
            comet_logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                # save_dir=".",  # Optional
                project_name=config.comet.project_name,  # Optional
                experiment_name=config.comet.experiment_name,
            )
            comet_logger.experiment.add_tags(list(config.comet.tags))
            trainer_args["logger"] = comet_logger
        else:
            print("no COMET API Key found..continuing without logging..")
            return

    checkpoint_callback = ModelCheckpoint(
        monitor="val_topk",
        dirpath=config.save_path,
        save_top_k=1,
        mode="max",
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=True
    )

    trainer_args["callbacks"] = [checkpoint_callback]
    trainer_args["overfit_batches"] = config.overfit_batches  # 0 if not overfitting
    trainer_args['max_epochs'] = config.max_epochs

    trainer = pl.Trainer(**trainer_args)
    if config.log_comet:
        trainer.logger.experiment.add_tags(list(config.comet.tags))
    if config.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(task, datamodule=datamodule)

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        task.hparams.learning_rate = new_lr
        task.hparams.lr = new_lr
        trainer.tune(model=task, datamodule=datamodule)

    # Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)

    # logging the best checkpoint to comet ML
    if config.log_comet:
        print(checkpoint_callback.best_model_path)
        trainer.logger.experiment.log_asset(checkpoint_callback.best_model_path, file_name='best_checkpoint.ckpt')


if __name__ == "__main__":
    main()
