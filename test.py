"""
main testing script
To run: python test.py args.config=CONFIG_FILE_PATH
"""
from pathlib import Path
from typing import Any, Dict, cast
import csv

import hydra
from hydra.utils import get_original_cwd

import comet_ml
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import CometLogger

from src.utils.config_utils import load_opts
from src.utils.compute_normalization_stats import *
import src.trainer.trainer as general_trainer

hydra_config_path = Path(__file__).resolve().parent / "configs/hydra.yaml"


def load_existing_checkpoint(task, base_dir, checkpint_path, save_preds_path):
    print("Loading existing checkpoint")
    try:
        task = task.load_from_checkpoint(os.path.join(base_dir, checkpint_path),
                                         save_preds_path=save_preds_path)

    # to prevent older models from failing, because there are new keys in conf
    except:
        task.load_state_dict(torch.load(os.path.join(base_dir, checkpint_path))['state_dict'])

    return task


def save_test_results_to_csv(results, root_dir, file_name='test_results.csv'):
    output_file = os.path.join(root_dir, file_name)

    with open(output_file, 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        csvfile.seek(0)
        if not csvfile.read():
            writer.writeheader()  # Write the header row based on the dictionary keys

        csvfile.seek(0, os.SEEK_END)
        writer.writerow(results)  # Write the values row by row

    print(f"CSV file '{output_file}' has been saved.")


def get_seed(run_id, fixed_seed):
    return (run_id * (fixed_seed + (run_id - 1))) % (2 ** 31 - 1)


@hydra.main(config_path="configs", config_name="hydra")
def main(opts):
    hydra_opts = dict(OmegaConf.to_container(opts))
    print("hydra_opts", hydra_opts)
    args = hydra_opts.pop("args", None)

    base_dir = args['base_dir']
    if not base_dir:
        base_dir = get_original_cwd()

    config_path = os.path.join(base_dir, args['config'])
    default_config = os.path.join(base_dir, "configs/defaults.yaml")

    config = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
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

    run_id = args["run_id"]
    global_seed = get_seed(run_id, config.program.seed)

    config.save_path = os.path.join(base_dir, config.save_path, str(global_seed))
    pl.seed_everything(config.program.seed)

    # using general trainer without location information
    print("Using general trainer..")
    task = general_trainer.EbirdTask(config)
    datamodule = general_trainer.EbirdDataModule(config)

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(config.trainer))

    if config.comet.experiment_key:
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=config.comet.project_name,
            experiment_name=config.comet.experiment_name,
            experiment_key=config.comet.experiment_key
        )

        trainer_args["logger"] = comet_logger

    def test_task(task):
        trainer = pl.Trainer(**trainer_args)
        trainer.validate(model=task, datamodule=datamodule)
        test_results = trainer.test(model=task,
                                    dataloaders=datamodule.test_dataloader(),
                                    verbose=True)

        print("Final test results: ", test_results)
        return test_results

    # if a single checkpoint is given
    if config.load_ckpt_path:
        if config.load_ckpt_path.endswith('.ckpt'):
            task = load_existing_checkpoint(task=task, base_dir=config.base_dir,
                                            checkpint_path=config.load_ckpt_path,
                                            save_preds_path=config.save_preds_path)

            test_results = test_task(task)
            save_test_results_to_csv(results=test_results[0],
                                     root_dir=os.path.join(config.base_dir, os.path.dirname(config.load_ckpt_path)))

        else:
            # get the number of experiments based on folders given
            n_runs = len(os.listdir(os.path.join(config.base_dir, config.load_ckpt_path)))
            # loop over all seeds
            for run_id in range(1, n_runs + 1):
                # get path of a single experiment
                run_id_path = os.path.join(config.load_ckpt_path, str(get_seed(run_id, config.program.seed)))
                # get path of the best checkpoint (not last)
                files = os.listdir(os.path.join(config.base_dir, run_id_path))
                best_checkpoint_file_name = [file for file in files if 'last' not in file and file.endswith('.ckpt')][0]
                checkpoint_path_per_run_id = os.path.join(run_id_path, best_checkpoint_file_name)
                # load the best checkpoint for the given run
                task = load_existing_checkpoint(task=task, base_dir=config.base_dir,
                                                checkpint_path=checkpoint_path_per_run_id,
                                                save_preds_path=config.save_preds_path)
                test_results = test_task(task)
                save_test_results_to_csv(results=test_results[0],
                                         root_dir=os.path.join(config.base_dir, config.load_ckpt_path))

    else:
        print("No checkpoint provided...Evaluating a random model")
        _ = test_task(task)


if __name__ == "__main__":
    main()
