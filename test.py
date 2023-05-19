import os
from os.path import expandvars
from pathlib import Path
from typing import Any, Dict, cast
import csv

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import CometLogger

import src.trainer.geo_trainer as geo_trainer
import src.trainer.multires_trainer as multires_trainer
import src.trainer.state_trainer as state_trainer
import src.trainer.trainer as general_trainer
from src.dataset.utils import set_data_paths

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
            project_name=conf.comet.project_name,
            experiment_name=conf.comet.experiment_name,
            experiment_key=conf.comet.experiment_key
        )

        trainer_args["logger"] = comet_logger

    # "/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527294/last.ckpt"
    # above with landuse : /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527309
    # sat landuse env 512  /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527306
    # Sat landuse  env 224 /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527294

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
                run_id_path = os.path.join(conf.load_ckpt_path, str(run_id*conf.program.seed))
                # get path of the best checkpoint (not last)
                files = os.listdir(os.path.join(conf.base_dir, run_id_path))
                best_checkpoint_file_name = [file for file in files if 'last' not in file and file.endswith('.ckpt')][0]
                checkpoint_path_per_run_id = os.path.join(run_id_path, best_checkpoint_file_name)
                # load the best checkpoint for the given run
                task = load_existing_checkpoint(task=task, base_dir=conf.base_dir, checkpint_path=checkpoint_path_per_run_id,
                                                save_preds_path=conf.save_preds_path)

                test_results = test_task(task)
                save_test_results_to_csv(results=test_results[0], root_dir=os.path.join(conf.base_dir, conf.load_ckpt_path))

    else:
        print("No checkpoint provided...Evaluating a random model")
        _ = test_task(task)


if __name__ == "__main__":
    main()
