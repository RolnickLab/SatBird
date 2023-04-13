import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torch.autograd import Variable

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
import src.models.geomodels as geomodels

criterion = CustomCrossEntropyLoss()#BCEWithLogitsLoss()
m = nn.Sigmoid()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_nb_bands(bands):
    n = 0
    for b in bands:
        if b in ["r","g","b","nir"]:
            n+=1
        elif b == "ped":
            n+=8
        elif b == "bioclim":
            n+= 19
    return(n)

def get_target_size(opts):
    subset = get_subset(opts.data.target.subset)
    target_size= len(subset) if subset is not None else opts.data.total_species
    return(target_size)


class LocEncoder(torch.nn.Module):
    def __init__(self, opts,**kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""
        
        super().__init__()
        self.opts = opts
        self.target_size = get_target_size(self.opts)
        
        num_inputs = 4
        if self.opts.loc.elev:
            num_inputs = 5
        self.model = geomodels.FCNet(num_inputs, num_classes=self.target_size,num_filts=256).to(device)     
        
    def forward(self, loc):
        return(self.model(loc))
    
    def __str__(self):
        return("Location encoder")


def create_loc_encoder(opts, verbose=0):
    encoder = LocEncoder(opts)

    if verbose > 0:
        print(f"  - Add {encoder.__class__.__name__}")
    return encoder



class EbirdTask(pl.LightningModule):
    def __init__(self, opts,**kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""
        
        super().__init__()
        self.save_hyperparameters(opts)
        
        self.automatic_optimization = False
        self.config_task(opts, **kwargs)
        self.opts = opts
        
    def get_sat_model(self):
        
        if self.opts.experiment.module.model == "resnet18":
            
            self.sat_model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands)!=3 or len(self.opts.data.env) > 0:
                bands = self.opts.data.bands + self.opts.data.env
                self.sat_model.conv1 = nn.Conv2d(get_nb_bands(bands), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.sat_model.fc = nn.Linear(512, self.target_size) 
            self.sat_model.to(device)
            self.m = nn.Sigmoid()
            
            
        elif self.opts.experiment.module.model == "resnet50":
            self.sat_model = models.resnet50(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands)!=3 or len(self.opts.data.env) > 0:
                bands = self.opts.data.bands + self.opts.data.env
                self.sat_model.conv1 = nn.Conv2d(get_nb_bands(bands), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.sat_model.fc = nn.Linear(2048, self.target_size) 
            self.sat_model.to(device)
            self.m = nn.Sigmoid()

            
        elif self.opts.experiment.module.model == "inceptionv3":
            self.sat_model = models.inception_v3(pretrained=self.opts.experiment.module.pretrained)
            self.sat_model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.sat_model.fc = nn.Linear(2048, self.target_size) 
            self.sat_model.to(device)
            #self.loss = CustomCrossEntropyLoss(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs) 
     
            self.m = nn.Sigmoid()

        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")
        return(self.sat_model)
        
    def config_task(self, opts, **kwargs: Any) -> None:
        self.opts = opts
        self.target_size= get_target_size(self.opts)
        self.target_type = self.opts.data.target.type
        
        if self.target_type == "binary":
            #self.target_type = "binary"
            self.criterion = BCELoss()
            print("Training with BCE Loss")
        else:
            self.criterion = CustomCrossEntropyLoss(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs) 
            print("Training with Custom CE Loss")
        
        self.encoders = {}
        if self.opts.loc.use:
            self.encoders["loc"] =  LocEncoder(self.opts)
        self.encoders["sat"] = self.get_sat_model()   
        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics


    def forward(self, x:Tensor, loc_tensor = None) -> Any:
        # need to fix use of inceptionv3 to be able to use location too 
        if self.opts.experiment.module.model == "inceptionv3":
            out = self.encoders["sat"](x)
        else:
            out_sat = m(self.encoders["sat"](x))
            print(self.encoders["sat"](x))
            if self.opts.loc.use:            
                out_loc = self.encoders["loc"](loc_tensor).squeeze(1)
                #print(self.encoders["loc"].model.feats[2].w1.weight[0])
                #out = torch.multiply(out_sat, out_loc) 
                return out_sat, out_loc
            return out_sat
    
    def training_step(
        self, batch: Dict[str, Any], batch_idx: int )-> Tensor:
        
        """Training step"""
        
        x = batch['sat'].squeeze(1).to(device)
        
        if self.opts.loc.use:
            loc_tensor= batch["loc"]
        sat_opt, loc_opt = self.optimizers()
        print(sat_opt)
        y = batch['target'].to(device)      

        print(self.encoders["sat"].parameters())
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
            if self.opts.loc.use:
            #    pred = self.forward(x, loc_tensor) 
            #else:
                out_sat, out_loc = self.forward(x, loc_tensor)
                y_hat = torch.multiply(m(out_sat), out_loc)#self.forward(x)
                
                pred = y_hat         
        
        #loss = Variable(loss.data, requires_grad=True)
        loss = self.criterion(pred, y)
   
        #for opt in [optims]:
        sat_opt.zero_grad()
        self.manual_backward(loss,retain_graph=True)
        sat_opt.step() 
        
        loc_opt.zero_grad()
        self.manual_backward(loss)
        loc_opt.step()
            

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
        
        if self.opts.loc.use:
            loc_tensor= batch["loc"]
            
        sat_opt, loc_opt = self.optimizers()
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
            if self.opts.loc.use:
           #     pred = self.forward(x, loc_tensor) 
           # else:
                out_sat, out_loc = self.forward(x, loc_tensor)
                y_hat = torch.multiply(m(out_sat), out_loc)#self.forward(x)
                pred = y_hat         
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
                lr=self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.opts.experiment.module.lr,  
                )
        else :
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return(optimizer)
    
    def configure_optimizers(self) -> Dict[str, Any]:
 
        sat_opt = self.get_optimizer( self.encoders["sat"], self.opts)       
        sat_scheduler = get_scheduler(sat_opt, self.opts)
        optimizers = [{
            "optimizer": sat_opt,
            "lr_scheduler": {
                "scheduler": sat_scheduler,
            "monitor":"val_loss",
            }}]
        if self.opts.loc.use:
            loc_opt = self.get_optimizer(self.encoders["loc"].model, self.opts)       
            loc_scheduler = StepLR(loc_opt, self.opts.scheduler.step_lr.step_size, self.opts.scheduler.step_lr.gamma)#get_scheduler(loc_opt, self.opts)
            optimizers += [{
                "optimizer": loc_opt,
                "lr_scheduler": {
                    "scheduler": loc_scheduler,
                "monitor":"val_loss",
                }}]
        
        return optimizers
        

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
            use_loc =self.use_loc
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

    
def configure_optimizers(self):
                    
    parameters = dict.fromkeys(self.model.modules.keys())

    self.optims = []
    self.scheds = []

    for key in self.encoders["sat"].modules.keys():
        parameters[key] = {'params':filter(lambda p: p.requires_grad, self.encoders["sat"].modules[key].parameters()),
                               'lr':self.hparams['bb_model']['optimizers']['modules'][key]['lr'],
                               'name':key
                              
            
        # If we want to update all optimizers at every step
        if self.hparams['bb_model']['optimizers']['same_interval'] is True:

            optimizer = torch.optim.Adam([parameters[key] for key in parameters])

            self.optims.append(optimizer)

            '''
            Configure Scheduler
            '''


            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optims[0],mode='min'
                                                             ,factor=0.2
                                                             , patience=100, threshold=0.01
                                                             , threshold_mode='rel', cooldown=20
                                                             , min_lr=1e-6, eps=1e-08, verbose=True)
            self.scheds.append(lr_scheduler)

            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"
                                   ,"frequency": 1, "monitor": "loss_train"}

            return self.optims, [lr_scheduler_config]


        # If we want to create different optimizers for different parameter groups so that we can update them
        # at different intervals
        else:

            opt_sched_dict = {}

            for key in parameters.keys():

                # parameters[key]={'params': ..., 'lr': ...}. In order to pass it to torch.optim.Adam,
                # you need to put it inside [] (list)
                optimizer = torch.optim.Adam([parameters[key]])
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min'
                                                             ,factor=0.2
                                                             , patience=100, threshold=0.01
                                                             , threshold_mode='rel', cooldown=20
                                                             , min_lr=1e-6, eps=1e-08, verbose=True)

                lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"
                                   ,"frequency": 1, "monitor": "loss_train"}


                opt_sched_dict[key] = {"optimizer": optimizer, "lr_scheduler":lr_scheduler_config}


            # The return value for this function could be any of the 6 in the docs, and when defined in this way
            # it should be a tuple of dictionaries. (Those dicts should each be {"optimizer": optimzer_obj,
            # "lr_scheduler": lr_sched_obj}
            return tuple(opt_sched_dict.values())

       