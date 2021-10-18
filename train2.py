import comet_ml
import os
import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))
sys.path.append(str(Path().resolve().parent.parent))
from train import set_up_omegaconf, OmegaConf
from src.trainer.trainer import EbirdTask, EbirdDataModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Any, Dict, Tuple, Type, cast

if __name__ == "__main__":

    conf = OmegaConf.load("./configs/defaults.yaml")
    task = EbirdTask("./configs/defaults.yaml")
    datamodule = EbirdDataModule("./configs/defaults.yaml")
    comet_logger= CometLogger(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace= "", 
        project_name="ebird",  
        experiment_name="default",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./ckpt",
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
    trainer_args["logger"] = comet_logger
    trainer = pl.Trainer(**trainer_args)

    ## Run experiment
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)