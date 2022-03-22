import comet_ml
import os
import sys
from pathlib import Path
from os.path import expandvars
#sys.path.append(str(Path().resolve().parent))
#sys.path.append(str(Path().resolve().parent.parent))
import hydra
from addict import Dict
from omegaconf import OmegaConf, DictConfig
from src.trainer.trainer import EbirdTask, EbirdDataModule
import src.trainer.geo_trainer as geo_trainer
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from typing import Any, Dict, Tuple, Type, cast
from src.dataset.utils import set_data_paths
import pdb

hydra_config_path = Path(__file__).resolve().parent / "configs/hydra.yaml"

def resolve(path):
    """
    fully resolve a path:
    resolve env vars ($HOME etc.) -> expand user (~) -> make absolute
    Returns:
        pathlib.Path: resolved absolute path
    """
    return Path(expandvars(str(path))).expanduser().resolve()


def set_up_omegaconf()-> DictConfig:
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
    
    #if commandline_opts is not None and isinstance(commandline_opts, dict):
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
        opts =  OmegaConf.merge(opts, commandline_opts)
        print("Commandline opts", commandline_opts)

    conf = set_data_paths(opts)
    conf = cast(DictConfig, opts)
    return conf


    
@hydra.main(config_path="configs", config_name = "hydra") #hydra_config_path)
def main(opts):

    hydra_opts = dict(OmegaConf.to_container(opts))
    print("hydra_opts", hydra_opts)
    args = hydra_opts.pop("args", None)

    config_path = args['config']
    default = "/home/mila/t/tengmeli/ecosystem-embedding/configs/defaults.yaml" #args['default']
    #default = Path(__file__).parent / "configs/defaults.yaml"
    conf = load_opts(config_path, default=default, commandline_opts=hydra_opts)
    conf.save_path = conf.save_path+os.environ["SLURM_JOB_ID"]
    pl.seed_everything(conf.program.seed)
    
    if not os.path.exists(conf.save_path):
        os.makedirs(conf.save_path)
    with open(os.path.join(conf.save_path, "config.yaml"),"w") as fp:
        OmegaConf.save(config = conf, f = fp)
    fp.close()
    
    print(conf.log_comet)
    
    print(conf) 
    
    if not conf.loc.use :
        task = EbirdTask(conf)
        datamodule = EbirdDataModule(conf)
    else:
        print("Using geo information")
        task = geo_trainer.EbirdTask(conf)
        datamodule = geo_trainer.EbirdDataModule(conf)
        
    
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))
        
    if conf.log_comet:

        comet_logger= CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
           # save_dir=".",  # Optional
            project_name=conf.comet.project_name,  # Optional
        )
        comet_logger.experiment.add_tags(list(conf.comet.tags))
        print(conf.comet.tags)
        trainer_args["logger"] = comet_logger
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=conf.save_path,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
         monitor="val_loss",
        min_delta=0.00,
        patience=4,
        mode = "min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    
    trainer_args["callbacks"] = [lr_monitor] #, #early_stopping_callback]checkpoint_callback, 
    #trainer_args["max_epochs"] = 1000
    

    trainer_args["profiler"]="simple"
    trainer_args["overfit_batches"] = conf.overfit_batches #0 if not overfitting
    
   # trainer_args["track_grad_norm"]=2
    
    if not conf.loc.use :
    
        trainer_args["auto_lr_find"]=conf.auto_lr_find
        
        trainer = pl.Trainer(**trainer_args)
        if conf.log_comet:
            trainer.logger.experiment.add_tags(list(conf.comet.tags))
        
        #lr_finder = trainer.tuner.lr_find(task,  datamodule=datamodule)

        # Results can be found in
        #lr_finder.results

        # Plot with
        #fig = lr_finder.plot(suggest=True)
        #fig.show()
        #fig.savefig("test.jpg")
        
        # Pick point based on plot, or get suggestion
        #new_lr = lr_finder.suggestion()

        # update hparams of the model
        #task.hparams.learning_rate = new_lr
        #task.hparams.lr = new_lr
        trainer.tune(model = task, datamodule=datamodule)
    else : 
        
        trainer = pl.Trainer(**trainer_args)     
        if conf.log_comet:
            trainer.logger.experiment.add_tags(list(conf.comet.tags))
      
    ## Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)
    
    
if __name__ == "__main__":
    main()

    
