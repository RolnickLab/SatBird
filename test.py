import comet_ml
import os
import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))
sys.path.append(str(Path().resolve().parent.parent))

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
import torch 
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_up_omegaconf()-> DictConfig:
    """Helps with loading config files"""
    
    conf = OmegaConf.load("./configs/defaults.yaml")
    command_line_conf = OmegaConf.from_cli()

    if "config_file" in command_line_conf:
        
        config_fn = command_line_conf.config_file

        if os.path.isfile(config_fn):
            user_conf = OmegaConf.load(config_fn)

    if "test_config_file" in command_line_conf:
        
        config_fn_test = command_line_conf.test_config_file

        if os.path.isfile(config_fn_test):
            user_conf_test = OmegaConf.load(config_fn_test)
    conf = OmegaConf.merge(conf, user_conf, user_conf_test)
 
    conf = set_data_paths(conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright
    
    return conf


if __name__ == "__main__":
    
    conf = set_up_omegaconf()

    pl.seed_everything(conf.program.seed)
    if not os.path.exists(conf.save_preds_path):
        os.makedirs(conf.save_preds_path)
    
    if not conf.loc.use :        
        task = EbirdTask(conf)
        datamodule = EbirdDataModule(conf)
    else:       

        task = geo_trainer.EbirdTask(conf)
        datamodule = geo_trainer.EbirdDataModule(conf)
    
    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))
        
    if conf.log_comet:

        comet_logger= CometLogger(
            api_key= os.environ.get("COMET_API_KEY"),
            project_name=conf.comet.project_name,  # Optional

        )
        print(conf.comet.tags)
        trainer_args["logger"] = comet_logger
        #trainer_args["logger"].experiment.add_tags(conf.comet.tags)


    trainer = pl.Trainer(**trainer_args)
    if conf.log_comet:
        trainer.logger.experiment.add_tags(list(conf.comet.tags))

    trainer = Trainer()
    if conf.load_ckpt_path != "":
        print("Loading existing checkpoint")
        task = task.load_from_checkpoint(conf.load_ckpt_path, save_preds_path = conf.save_preds_path)

    result = trainer.test(model=task, datamodule=datamodule)
    print(result)
    print("finished testing")
