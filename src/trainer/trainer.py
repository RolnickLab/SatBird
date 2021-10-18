import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import models

from src.dataset.utils import load_opts
from src.transforms.transforms import get_transforms
from torchvision import transforms as trsfs
import pandas as pd
import torch.nn.functional as F
from src.losses.losses import CustomCrossEntropyLoss
import torchmetrics

from typing import Any, Dict, Optional
from src.dataset.dataloader import EbirdVisionDataset

criterion = CustomCrossEntropyLoss()#BCEWithLogitsLoss()
m = nn.Sigmoid()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EbirdTask(pl.LightningModule):
    def __init__(self, opts = 'configs/defaults.yaml',**kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.save_hyperparameters()
        self.config_task(opts, **kwargs)
        self.opts = load_opts(opts)
        
        
    def config_task(self, opts, **kwargs: Any) -> None:
        self.opts = load_opts(opts)
        
        if self.opts.experiment.module.model == "resnet18":
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, 684) 
            self.model.to(device)
            self.loss = CustomCrossEntropyLoss() #BCEWithLogitsLoss()
            self.m = nn.Sigmoid()
            self.criterion = CustomCrossEntropyLoss()
            self.mse = torchmetrics.MeanSquaredError()
            self.mae = torchmetrics.MeanAbsoluteError()

        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")


    def forward(self, x:Tensor) -> Any:
        return self.model(x)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int )-> Tensor:
        
        """Training step"""
        
        x = batch['sat'].squeeze(1).to(device)
        y = batch['target'].to(device)
    
        y_hat = self.forward(x)
        loss = self.loss(m(y_hat), y)
        mse = self.mse(m(y_hat), y)
        mae = self.mae(m(y_hat),y)
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_mse", mse )
        
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int )->None:

        """Validation step """

        
        x = batch['sat'].squeeze(1).to(device)
        y = batch['target'].to(device)
        
        y_hat = self.forward(x)
        loss = self.loss(m(y_hat), y)
        mse = self.mse(m(y_hat), y)
        mae = self.mae(m(y_hat),y)
        self.log("val_loss", loss)
        self.log("val_mae", mae)
        self.log("val_mse", mse )
    

    def test_step(
        self, batch: Dict[str, Any], batch_idx:int
    )-> None:
        """Test step """

        x = batch['sat'].squeeze(1).to(device)
        y = batch['target'].to(device)
        y_hat = self.forward(x)
        loss = self.loss(m(y_hat), y)
        mse = self.mse(m(y_hat), y)
        mae = self.mae(m(y_hat),y)
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        self.log("test_mse", mse )

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.opts.experiment.module.lr, #CHECK IN CONFIG
        )
        return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience = self.opts.experiment.module.lr_schedule_patience   #CHECK IN CONFIG
            ),
            "monitor":"val_loss",
            }
        }

class EbirdDataModule(pl.LightningDataModule):
    def __init__(self, opts) -> None:
        super().__init__() 
        self.opts = load_opts(opts)
        
        self.seed = self.opts.program.seed
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers

        self.df_train = pd.read_csv(self.opts.data.files.train)
        self.df_val = pd.read_csv(self.opts.data.files.val)
        self.df_test = pd.read_csv(self.opts.data.files.test)
        self.bands = self.opts.data.bands    

    def prepare_data(self) -> None:
        """_ = EbirdVisionDataset(
            # pd.Dataframe("/network/scratch/a/akeraben/akera/ecosystem-embedding/data/train_june.csv"), 
            df_paths = self.df_paths,
            bands = self.bands,
            split = "train",
            transforms = trsfs.Compose(get_transforms(self.opts, "train"))
        )"""
        print("prepare data")
        
        

    def setup(self, stage: Optional[str]=None)->None:
        """create the train/test/val splits"""
        self.all_train_dataset = EbirdVisionDataset(
            self.df_train,
            bands=self.bands,
            split = "train",
            transforms = trsfs.Compose(get_transforms(self.opts, "train"))

        )

        self.all_test_dataset = EbirdVisionDataset(                
            self.df_test, 
            bands = self.bands,
            split = "test",
            transforms = trsfs.Compose(get_transforms(self.opts, "val")),
        )

        self.all_val_dataset = EbirdVisionDataset(
            self.df_val,
            bands=self.bands,
            split = "val",
            transforms = trsfs.Compose(get_transforms(self.opts, "val")),
        )

        #TODO: Create subsets of the data
        
        self.train_dataset = self.all_train_dataset
           
        self.test_dataset = self.all_test_dataset
           
        self.val_dataset = self.all_val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
        )
