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
#from src.trainer.trainer_species import EbirdSpeciesTask
import src.trainer.geo_trainer as geo_trainer
import src.trainer.state_trainer as state_trainer
import src.trainer.multires_trainer as multires_trainer

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

    config_path = "/network/scratch/a/amna.elmustafa/final/ecosystem-embedding/configs/custom_amna.yaml"
    #args['config'] #"/home/mila/t/tengmeli/ecosystem-embedding/configs/custom_meli_2.yaml" 
    default = "/network/scratch/a/amna.elmustafa/final/ecosystem-embedding/configs/defaults.yaml" #args['default']
    #default = Path(__file__).parent / "configs/defaults.yaml"
    conf = load_opts(config_path, default=default, commandline_opts=hydra_opts)
    conf.save_path = conf.save_path+os.environ["SLURM_JOB_ID"]
    pl.seed_everything(conf.program.seed)

    #if not os.path.exists(conf.save_preds_path):
     #   os.makedirs(conf.save_preds_path)
    #with open(os.path.join(conf.save_preds_path, "config.yaml"),"w") as fp:
     #   OmegaConf.save(config = conf, f = fp)
    #fp.close()
    
    print(conf.log_comet)
    
    print(conf)
    print('multires len:', len(conf.data.multiscale))
    if not conf.loc.use and len(conf.data.multiscale)>1:
         print('using multiscale net')
         task = multires_trainer.EbirdTask(conf)
         datamodule = EbirdDataModule(conf)
    elif "speciesAtoB" in conf.keys() and conf.speciesAtoB:
            print("species A to B")
            task = EbirdSpeciesTask(conf)
            datamodule = EbirdDataModule(conf)
    elif not conf.loc.use :
        task = EbirdTask(conf)
        datamodule = EbirdDataModule(conf)
   
        
    elif conf.loc.loc_type == "latlon":
        print("Using geo information")
        task = geo_trainer.EbirdTask(conf)
        datamodule = geo_trainer.EbirdDataModule(conf)
    elif conf.loc.loc_type == "state":
        print("Using geo information")
        task = state_trainer.EbirdTask(conf)
        datamodule = state_trainer.EbirdDataModule(conf)
        
    
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))
    if conf.load_ckpt_path != "":
        print("Loading existing checkpoint")
    conf.load_ckpt_path = "/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527294/last.ckpt"
    #"/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2477166/last.ckpt"
    #/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2477165  512_224
    #/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2477127  64
    #/network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2477123  512
    # /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2477120  224
    # /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2485167 satenv224
    #land use only /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2484606
    # /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2485253 satenvlanduse224
    #"/network/scratch/t/tengmeli/ecosystem-embeddings/checkpoint_base_2034177/epoch=53-step=2537.ckpt" 
    #"/network/scratch/t/tengmeli/ecosystem-embeddings/checkpoint_loc_2034124/epoch=3-step=187.ckpt" #"/network/scratch/t/tengmeli/ecosystem-embeddings/checkpoint_loc2033871/last-copy.ckpt"
    #sat 224 512 rangemaps: /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527315
    #above with landuse : /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527309
    #sat landuse env 512  /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527306
    #Sat landuse  env 224 /network/scratch/a/amna.elmustafa/ecosystem-embeddings/ckpts2527294
    
    task = task.load_from_checkpoint(conf.load_ckpt_path, save_preds_path = conf.save_preds_path)    
        
    trainer = pl.Trainer(**trainer_args)
    trainer.validate(model=task, datamodule=datamodule)
    trainer.test(model=task, 
                       dataloaders=datamodule.test_dataloader(),
               
                       #ckpt_path='best',
                       verbose=True)
    trainer.test(model=task, dataloaders=datamodule.train_dataloader(),verbose=True)
    
    
if __name__ == "__main__":
    main()

    
