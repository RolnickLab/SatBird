import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import models
import numpy as np
from omegaconf import OmegaConf
#from src.dataset.utils import load_opts
from src.transforms.transforms import get_transforms
from torchvision import transforms as trsfs
import pandas as pd
import torch.nn.functional as F
from src.losses.losses import CustomCrossEntropyLoss,CustomCrossEntropy, get_metrics
import torchmetrics
from torch.nn import BCELoss, BCEWithLogitsLoss
from typing import Any, Dict, Optional
from src.dataset.dataloader import EbirdVisionDataset
from src.dataset.dataloader import get_subset
import src.models.resnet_tabular as resnet_tabular
import time 
import os 
import json
from torch.nn.functional import l1_loss
#criterion = CustomCrossEntropyLoss()#BCEWithLogitsLoss()
mse=nn.MSELoss()
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
        elif b == "rgb":
            n+=3
    return(n)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
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
        return(CosineAnnealingWarmRestarts(optimizer, opts.scheduler.cyclical.t0, opts.scheduler.cyclical.tmult))
    elif opts.scheduler.name == "":
        return(None)
    else:
        raise ValueError(f"Scheduler'{self.opts.scheduler.name}' is not valid")
        

class EbirdSpeciesTask(pl.LightningModule):
    def __init__(self, opts,**kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""
        
        super().__init__()
        print(opts)
        self.save_hyperparameters(opts)
        self.config_task(opts, **kwargs)
        self.opts = opts
        print(self.opts.save_preds_path)
        #define self.learning_rate to enable learning rate finder
        self.learning_rate = self.opts.experiment.module.lr
        #for (name, _, scale) in self.metrics:
    
    def config_task(self, opts, **kwargs: Any) -> None:
        self.opts = opts
        self.means = None
        #get target vector size (number of species we consider)
        subset = get_subset(self.opts.data.target.subset)
        
        self.target_size = len(subset) if subset is not None else self.opts.data.total_species
        print("Predicting ", self.target_size, "species")
        self.target_type = self.opts.data.target.type
        
        if self.target_type == "binary":
            #ground truth is 0-1. if bird is reported at a hotspot, target = 1
            self.criterion = BCEWithLogitsLoss()
            print("Training with BCE Loss")
        elif self.target_type == "log":
            self.criterion = nn.MSELoss()
            print("Training with MSE Loss")
        else:
            #target is num checklists reporting species i / total number of checklists at a hotspot
            self.criterion =CustomCrossEntropyLoss()
            #CustomCrossEntropy(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs)
            #mse 
            #CustomCrossEntropy(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_absCustomCrossEntropy(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs) 
            print("Training with Custom CE Loss")
        if self.opts.experiment.module.model == "train_linear":
            self.feature_extractor = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands)!=3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                self.feature_extractor.conv1 = nn.Conv2d(get_nb_bands(self.bands), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            if self.opts.experiment.module.fc == "linear":
                self.feature_extractor.fc = nn.Linear(512, self.target_size)
            ckpt = torch.load(self.opts.experiment.module.resume)
            for key in list(ckpt["state_dict"].keys()):
                ckpt["state_dict"][key.replace('model.', '')] = ckpt["state_dict"].pop(key)
            self.feature_extractor.load_state_dict(ckpt["state_dict"])
            print("initialized network, freezing weights")
            self.feature_extractor.fc = nn.Sequential()
            #self.feature_extractor.freeze()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(512, self.target_size)
            #self.means = np.load(self.opts.experiment.module.means_path)[0,subset]
            #means = torch.Tensor(self.means)

            #means = torch.logit(means, eps=1e-10)
            #self.model.bias.data =  means

        elif self.opts.experiment.module.model == "resnet18":
            
            self.sat_model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
        
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.sat_model.conv1.in_channels
                weights = self.sat_model.conv1.weight.data.clone()
                self.sat_model.conv1 = nn.Conv2d(
                        get_nb_bands(self.bands),
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False,
                )
                #assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    self.sat_model.conv1.weight.data[:, :orig_channels, :, :] = weights
            self.sat_model.fc = Identity()
            #self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.bands_feature_dim = 512
            
        elif self.opts.experiment.module.model == "resnet50":
            self.sat_model = models.resnet50(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.sat_model.conv1.in_channels
                weights = self.sat_model.conv1.weight.data.clone()
                self.sat_model.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                #assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    self.sat_model.conv1.weight.data[:, :orig_channels, :, :] = weights
           
            self.sat_model.fc = Identity() # = nn.Sequential(*list(model.children())[:-1])
            self.bands_feature_dim = 2048
        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")
        
        if self.opts.experiment.module.init_bias=="means":
            print("initializing biases with mean predictor")
            self.means = np.load(self.opts.experiment.module.means_path)[0,subset]
            means = torch.Tensor(self.means)
            
            means = torch.logit(means, eps=1e-10)
            if self.opts.experiment.module.model != "linear":
                if self.opts.experiment.module.fc == "linear_net":
                    self.model.fc[2].bias.data = means
                else:
                    self.model.fc.bias.data =  means
            
        else:
            print("no initialization of biases")
        d_in = 305  # number of env vars
        
        self.species_dim = 256  # number of last fc layer

        #self.species_model = #nn.Sequential(nn.Linear(d_in, 128),nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, self.species_dim),nn.ReLU())
        self.species_model = resnet_tabular.ResNet.make_baseline(
            d_in=d_in,
            d_main=128,
            d_hidden=256,
            dropout_first=0.2,
            dropout_second=0.0,
            n_blocks=2,
            d_out=self.species_dim,
        )
    
        
        self.last_layer = nn.Sequential(nn.Flatten(), torch.nn.Linear(
            in_features=(self.species_dim + self.bands_feature_dim),
            out_features=self.target_size,
        ))
        #self.last_layer = nn.Sequential(nn.Flatten(), self.fc)
        self.dropout = torch.nn.Dropout(0.2)
        #self.model #.to(device)
      
        
        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics

    def forward(self, x:Tensor, species:Tensor) -> Any:
        band_out = self.sat_model(x)
        species_out = self.species_model(species.type_as(x)) #.unsqueeze(-1).unsqueeze(-1)
        print(band_out.shape)
        print(species_out.shape)
        #species_out = species.type_as(x).unsqueeze(-1).unsqueeze(-1)
        combined = torch.cat([band_out, species_out], dim=1)
        output = self.last_layer(combined) #torch.nn.functional.relu(self.last_layer(combined))
        return output
    
      

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int )-> Tensor:
       # from pdb import set_trace; set_trace()
        """Training step"""
        m = nn.Sigmoid()
        x = batch['sat'].squeeze(1)
        y = batch['target']

        b, no_species = y.shape
        species = batch["speciesA"]
        
        #print("Model is on cuda", next(self.model.parameters()).is_cuda)
        
        y_hat = self.forward(x, species)
        pred = m(y_hat).type_as(y)
                
        pred_ = pred.clone().type_as(y)
          
            #print('maximum ytrue in trainstep',y.max())
        loss = self.criterion(y, pred)
        self.log("train_loss", loss) 
       
        for (name, _, scale) in self.metrics:
            nname = "train_" + name
            if name == "accuracy":
                getattr(self,name)(pred_, y.type(torch.uint8))
          
                print(nname,getattr(self,name)(pred_,  y.type(torch.uint8)))
                
            else:
               
                getattr(self,name)(y, pred_)
                print(nname,getattr(self,name)(y, pred_) )
               
            self.log(nname, getattr(self,name))
        
        return loss
 
    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int )->None:

        """Validation step """

        m = nn.Sigmoid()
        x = batch['sat'].squeeze(1)
        y = batch['target']

        b, no_species = y.shape
        species = batch["speciesA"]
        
        #print("Model is on cuda", next(self.model.parameters()).is_cuda)
        
        y_hat = self.forward(x, species)
        pred = m(y_hat).type_as(y)
                
        pred_ = pred.clone().type_as(y)
          
            #print('maximum ytrue in trainstep',y.max())
        loss = self.criterion(y, pred)
        self.log("val_loss", loss, on_step = True, on_epoch = True)

        for (name, _, scale) in self.metrics:
            nname = "val_" + name
            if name == "accuracy":
                getattr(self,name)(pred_, y.type(torch.uint8))
                print(nname,getattr(self,name)(pred_,  y.type(torch.uint8)))
          
            else:
                getattr(self,name)(y, pred_)
         
            self.log(nname, getattr(self, name), on_step=False, on_epoch=True) 
        

    

    def test_step(
        self, batch: Dict[str, Any], batch_idx:int
    )-> None:
        """Test step """
        
        m = nn.Sigmoid()
        x = batch['sat'].squeeze(1)
        y = batch['target']

        b, no_species = y.shape
        species = batch["speciesA"]
        
        #print("Model is on cuda", next(self.model.parameters()).is_cuda)
        
        y_hat = self.forward(x, species)
        pred = m(y_hat).type_as(y)
                
        pred_ = pred.clone().type_as(y)
        
        if self.opts.save_preds_path != "":       
            for i, elem in enumerate(pred):
                np.save(os.path.join(self.opts.save_preds_path, batch["hotspot_id"][i] + ".npy"), elem.cpu().detach().numpy())
        print("saved elems")


    def get_optimizer(self, model, opts):
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(   #
                model.parameters(),
                lr=self.learning_rate, # self.opts.experiment.module.lr,  
                weight_decay=0.00001
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
    
    def get_optimizer_from_params(self,param, opts):
        
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(   #
                param,
                lr=self.learning_rate#self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                param,
                lr=self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                param,
                lr=self.learning_rate#self.opts.experiment.module.lr,  
                )
        else :
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return(optimizer)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        parameters = (
            list(self.sat_model.parameters())
            + list(self.species_model.parameters())
            + list(self.last_layer.parameters())
                )
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        print(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable components out of {len(parameters)}."
            )

        optimizer = self.get_optimizer_from_params(trainable_parameters, self.opts)
        scheduler = get_scheduler(optimizer, self.opts)

        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                            },
                    }

    """ 
        optimizer = self.get_optimizer(self.model, self.opts)       
        scheduler = get_scheduler(optimizer, self.opts)
        print("scheduler", scheduler)
        if scheduler is None:
            return optimizer
        else:
            return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            "monitor":"val_loss",
            "frequency":1
            }
        }

    """
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
