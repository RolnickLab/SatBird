import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import models

from omegaconf import OmegaConf
#from src.dataset.utils import load_opts
from src.transforms.transforms import get_transforms
from torchvision import transforms as trsfs
import pandas as pd
import torch.nn.functional as F
from src.losses.losses import CustomCrossEntropyLoss, TopKAccuracy, get_metrics
import torchmetrics
from torch.nn import BCELoss
from typing import Any, Dict, Optional
from src.dataset.dataloader import EbirdVisionDataset
from src.dataset.dataloader import get_subset
import time 

criterion = CustomCrossEntropyLoss()#BCEWithLogitsLoss()
m = nn.Sigmoid()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_nb_bands(bands):
    """
    Get number of channels in the satellite input branch (stack bands of satellite + environmental variables)
    """
    n = 0
    for b in bands:
        if b in ["r","g","b","nir"]:
            n+=1
        elif b == "ped":
            n+=8
        elif b == "bioclim":
            n+= 19
    return(n)

    
def get_scheduler(optimizer, opts):
    if opts.scheduler.name == "ReduceLROnPlateau":
        return (ReduceLROnPlateau(optimizer, factor = opts.scheduler.reduce_lr_plateau.factor,
                  patience = opts.scheduler.reduce_lr_plateau.lr_schedule_patience))
    elif opts.scheduler.name == "StepLR":
        return (StepLR(optimizer, opts.scheduler.step_lr.step_size, opts.scheduler.step_lr.gamma))
    elif opts.scheduler.name == "WarmUp":     
        return(LinearWarmupCosineAnnealingLR(optimizer, opts.scheduler.warmup.warmup_epochs,
        opts.scheduler.warmup.max_epochs))
    elif opts.scheduler.name == "Cyclical":
        return(CosineAnnealingWarmRestarts(optimizer, opts.scheduler.cyclical.warmup_epochs))
    else:
        raise ValueError(f"Scheduler'{self.opts.scheduler.name}' is not valid")
        

class EbirdTask(pl.LightningModule):
    def __init__(self, opts,**kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""
        
        super().__init__()
        self.save_hyperparameters(opts)
        self.config_task(opts, **kwargs)
        self.opts = opts
        #define self.learning_rate to enable learning rate finder
        self.learning_rate = self.opts.experiment.module.lr
        
    def config_task(self, opts, **kwargs: Any) -> None:
        self.opts = opts
        
        #get target vector size (number of species we consider)
        subset = get_subset(self.opts.data.target.subset)
        self.target_size= len(subset) if subset is not None else self.opts.data.total_species
        self.target_type = self.opts.data.target.type
        
        if self.target_type == "binary":
            #ground truth is 0-1. if bird is reported at a hotspot, target = 1
            self.criterion = BCELoss()
            print("Training with BCE Loss")
        else:
            #target is num checklists reporting species i / total number of checklists at a hotspot
            self.criterion = CustomCrossEntropyLoss(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs) 
            print("Training with Custom CE Loss")
            
        if self.opts.experiment.module.model == "resnet18":
            
            self.model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands)!=3 or len(self.opts.data.env) > 0:
                bands = self.opts.data.bands + self.opts.data.env
                self.model.conv1 = nn.Conv2d(get_nb_bands(bands), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(512, self.target_size) 
            self.model.to(device)
            self.m = nn.Sigmoid()
            
            
        elif self.opts.experiment.module.model == "resnet50":
            self.model = models.resnet50(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands)!=3 or len(self.opts.data.env) > 0:
                bands = self.opts.data.bands + self.opts.data.env
                self.model.conv1 = nn.Conv2d(get_nb_bands(bands), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.fc = nn.Linear(2048, self.target_size) 
            self.model.to(device)
            self.m = nn.Sigmoid()

            
        elif self.opts.experiment.module.model == "inceptionv3":
            self.model = models.inception_v3(pretrained=self.opts.experiment.module.pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size) 
            self.model.to(device)
            self.m = nn.Sigmoid()

        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")
        
        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics


    def forward(self, x:Tensor) -> Any:
        return self.model(x)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int )-> Tensor:
        
        """Training step"""
        
        x = batch['sat'].squeeze(1).to(device)
        y = batch['target'].to(device)      
        
        #check weights are moving
        #for p in self.model.fc.parameters(): 
        #    print(p.data)
        if self.opts.experiment.module.model == "inceptionv3":
            y_hat, aux_outputs = self.forward(x)
            pred = m(y_hat)
            aux_pred = m(aux_outputs)
            loss1 = self.criterion(pred, y)
            loss2 = self.criterion(aux_pred, y)
            loss = loss1 + loss2
            
        else:
            y_hat = self.forward(x)
            pred = m(y_hat)
            loss = self.criterion(pred, y)

        pred_ = pred.clone()
        
        if self.opts.data.target.type == "binary":
            pred_[pred_>0.5] = 1
            pred_[pred_<0.5] = 0
        
        for (name, _, scale) in self.metrics:
            nname = "train_" + name
            metric = getattr(self,name)(pred_, y)
            if name == "topk":
                self.log(nname, metric[0] * scale)
            else:
                self.log(nname, metric * scale)
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int )->None:

        """Validation step """

        
        x = batch['sat'].squeeze(1).to(device)
        y = batch['target'].to(device)

        y_hat = self.forward(x)
      
        pred = m(y_hat)
        loss = self.criterion(pred, y)

        pred_ = pred.clone()
        if self.opts.data.target.type == "binary":
            pred_[pred_>0.5] = 1
            pred_[pred_<0.5] = 0
        
        for (name, _, scale) in self.metrics:
            nname = "val_" + name
            metric = getattr(self,name)(pred_, y)
            if name == "topk":
                self.log(nname, metric[0] * scale)
            else:
                self.log(nname, metric * scale)
        self.log("val_loss", loss)

    

    def test_step(
        self, batch: Dict[str, Any], batch_idx:int
    )-> None:
        """Test step """

        x = batch['sat'].squeeze(1).to(device)
        y = batch['target'].to(device)
        y_hat = self.forward(x)
        pred = m(y_hat)
        pred_ = pred.clone()
        for (name, _, scale) in self.metrics:
            nname = "test_" + name
            metric = getattr(self,name)(pred_, y)
            if name == "topk":
                self.log(nname, metric[0] * scale)
            else:
                self.log(nname, metric * scale)

    def get_optimizer(self, model, opts):
        
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(   #
                model.parameters(),
                lr=self.learning_rate # self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate #self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate  
                )
        else :
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return(optimizer)
    
    def configure_optimizers(self) -> Dict[str, Any]:
 
        optimizer = self.get_optimizer(self.model, self.opts)       
        scheduler = get_scheduler(optimizer, self.opts)
        
        return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            "monitor":"val_loss",
            }
        }


class EbirdDataModule(pl.LightningDataModule):
    def __init__(self, opts) -> None:
        super().__init__() 
        self.opts = opts
        
        self.seed = self.opts.program.seed
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
        self.df_train = pd.read_csv(self.opts.data.files.train)
        self.df_val = pd.read_csv(self.opts.data.files.val)
        self.df_test = pd.read_csv(self.opts.data.files.test)
        self.bands = self.opts.data.bands 
        self.env = self.opts.data.env
        self.datatype = self.opts.data.datatype
        self.target = self.opts.data.target.type
        self.subset = self.opts.data.target.subset
        self.use_loc = self.opts.loc.use
        
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
            df_paths = self.df_train,
            bands = self.bands,
            env = self.env,
            transforms = trsfs.Compose(get_transforms(self.opts, "train")),
            mode = "train",
            datatype = self.datatype,
            target = self.target, 
            subset = self.subset,
            use_loc = self.use_loc
        )

        self.all_test_dataset = EbirdVisionDataset(                
            self.df_test, 
            bands = self.bands,
            env = self.env,
            transforms = trsfs.Compose(get_transforms(self.opts, "val")),
            mode = "test",
            datatype = self.datatype,
            target = self.target, 
            subset = self.subset,
            use_loc = self.use_loc
            )

        self.all_val_dataset = EbirdVisionDataset(
            self.df_val,
            bands=self.bands,
            env = self.env,
            transforms = trsfs.Compose(get_transforms(self.opts, "val")),
            mode = "val",
            datatype = self.datatype,
            target = self.target, 
            subset = self.subset,
            use_loc = self.use_loc
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
