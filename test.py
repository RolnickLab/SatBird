"""
main testing script
To run: python test.py args.config=CONFIG_FILE_PATH
"""
import os
from pathlib import Path
from typing import Any, Dict, cast
import csv
import math

import hydra
from hydra.utils import get_original_cwd

import comet_ml
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import CometLogger

from src.utils.config_utils import load_opts
import src.trainer.trainer as general_trainer
import src.trainer.geo_trainer as geo_trainer

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

    conf = load_opts(config_path, default=default_config, commandline_opts=hydra_opts)
    conf.base_dir = base_dir

    run_id = args["run_id"]
    global_seed = get_seed(run_id, conf.program.seed)

    conf.save_path = os.path.join(base_dir, conf.save_path, str(global_seed))
    pl.seed_everything(conf.program.seed)

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

    if conf.comet.experiment_key:
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=conf.comet.project_name,
            experiment_name=conf.comet.experiment_name,
            experiment_key=conf.comet.experiment_key
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
    if conf.load_ckpt_path:
        if conf.load_ckpt_path.endswith('.ckpt'):
            task = load_existing_checkpoint(task=task, base_dir=conf.base_dir,
                                            checkpint_path=conf.load_ckpt_path,
                                            save_preds_path=conf.save_preds_path)

            test_results = test_task(task)
            save_test_results_to_csv(results=test_results[0],
                                     root_dir=os.path.join(conf.base_dir, os.path.dirname(conf.load_ckpt_path)))

        else:
            # get the number of experiments based on folders given
            n_runs = len(os.listdir(os.path.join(conf.base_dir, conf.load_ckpt_path)))
            # loop over all seeds
            for run_id in range(1, n_runs + 1):
                # get path of a single experiment
                run_id_path = os.path.join(conf.load_ckpt_path, str(get_seed(run_id, conf.program.seed)))
                # get path of the best checkpoint (not last)
                files = os.listdir(os.path.join(conf.base_dir, run_id_path))
                best_checkpoint_file_name = [file for file in files if 'last' not in file and file.endswith('.ckpt')][0]
                checkpoint_path_per_run_id = os.path.join(run_id_path, best_checkpoint_file_name)
                # load the best checkpoint for the given run
                task = load_existing_checkpoint(task=task, base_dir=conf.base_dir,
                                                checkpint_path=checkpoint_path_per_run_id,
                                                save_preds_path=conf.save_preds_path)

                test_results = test_task(task)
                save_test_results_to_csv(results=test_results[0],
                                         root_dir=os.path.join(conf.base_dir, conf.load_ckpt_path))

    else:
        print("No checkpoint provided...Evaluating a random model")
        _ = test_task(task)


if __name__ == "__main__":
    main()
