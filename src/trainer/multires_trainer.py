import os
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
import numpy as np
from omegaconf import OmegaConf
#from src.dataset.utils import load_opts
from src.transforms.transforms import get_transforms
from torchvision import transforms as trsfs
import pandas as pd
import torch.nn.functional as F
from src.losses.losses import CustomCrossEntropyLoss,get_metrics
import torchmetrics
from torch.nn import BCELoss
from typing import Any, Dict, Optional
from src.dataset.dataloader import EbirdVisionDataset
from src.dataset.dataloader import get_subset
import time 
import pickle 
import copy 

criterion = CustomCrossEntropyLoss()#BCEWithLogitsLoss()
m = nn.Sigmoid()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_nb_bands(bands):
    n = 0
    for b in bands:
        if b in ["r","g","b","nir", "landuse"]:
            n+=1
        elif b == "ped":
            n+=8
        elif b == "bioclim":
            n+= 19
        elif b == "rgb":
            n+=3
    return(n)

def get_target_size(opts):
    subset = get_subset(opts.data.target.subset)
    target_size= len(subset) if subset is not None else opts.data.total_species
    return(target_size)


class LocEncoder(torch.nn.Module):
    def __init__(self, opt,**kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""
        
        super().__init__()
        self.opts_ = opt
        self.target_size = get_target_size(self.opts_)
        
        num_inputs = 4
        if self.opts_.loc.elev:
            num_inputs = 5
        self.model = geomodels.FCNet(num_inputs, num_classes=self.target_size,num_filts=256)   
      
        
    def forward(self, loc):
        
        return(self.model(loc, class_of_interest=None, return_feats=self.opts_.loc.concat))
    
    def __str__(self):
        return("Location encoder")



def create_loc_encoder(opt, verbose=0):
    print("using location")
    encoder = LocEncoder(opt)

    if verbose > 0:
        print(f"  - Add {encoder.__class__.__name__}")
    return encoder

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class EbirdTask(pl.LightningModule):
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
           

        elif self.opts.experiment.module.model == "resnet18":
            
            self.model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
        
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.model.conv1.in_channels
                weights = self.model.conv1.weight.data.clone()
                self.model.conv1 = nn.Conv2d(
                        get_nb_bands(self.bands),
                        64,
                        kernel_size=(7, 7),
                        stride=(2, 2),
                        padding=(3, 3),
                        bias=False,
                )
                #assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    self.model.conv1.weight.data[:, :orig_channels, :, :] = weights
            #loading seco model
            if self.opts.experiment.module.resume:
                
                ckpt = torch.load(self.opts.experiment.module.resume)
                self.model.fc = nn.Sequential()
                loaded_dict=ckpt['state_dict']
                model_dict=self.model.state_dict()
                del loaded_dict["queue"] 
                del loaded_dict["queue_ptr"]
                #load state dict keys
                for key_model, key_seco in zip(model_dict.keys(),loaded_dict.keys()):
                    #ignore first layer weights(use imagenet ones)
                    if key_model=='conv1.weight':
                        continue
                    model_dict[key_model]=loaded_dict[key_seco]
                    
                    

           
                self.model.load_state_dict(model_dict)
            if self.opts.experiment.module.fc == "linear":
                self.model.fc = nn.Linear(512, self.target_size)
            elif self.opts.experiment.module.fc == "linear_net":
                self.model.fc = nn.Sequential(nn.Linear(512, 512),
                          nn.ReLU(),
                          nn.Linear(512, self.target_size))
            else : 
                self.model.fc = nn.Linear(512, self.target_size)
            
        elif self.opts.experiment.module.model == "resnet50":
            self.model = models.resnet50(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.model.conv1.in_channels
                weights = self.model.conv1.weight.data.clone()
                self.model.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                #assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    self.model.conv1.weight.data[:, :orig_channels, :, :] = weights
           
            if self.opts.experiment.module.fc == "linear":
                self.model.fc = nn.Linear(2048, self.target_size)
            elif self.opts.experiment.module.fc == "linear_net":
                self.model.fc = nn.Sequential(nn.Linear(2048, 2048),
                          nn.ReLU(),
                          nn.Linear(2048, self.target_size))
            else :
                self.model.fc = nn.Linear(2048, self.target_size)

            
        elif self.opts.experiment.module.model == "inceptionv3":
            self.model = models.inception_v3(pretrained=self.opts.experiment.module.pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size) 
            
        elif self.opts.experiment.module.model == "linear":
            nb_bands = get_nb_bands(self.opts.data.bands + self.opts.data.env)
            self.model = nn.Linear(nb_bands*64*64, self.target_size)  

        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")
        
        #multiscale experiment assuming all scales will have the same model
        
        if len(self.opts.data.multiscale)>1:
#             if self.opts.data.multiscale_agg=='vectors' :
                models_scales=[]
                in_features=self.model.fc.in_features
                self.model.fc =nn.Sequential()
                if self.opts.data.multiscale_agg=='vectors' :
                    linear_layer = nn.Linear(len(self.opts.data.multiscale)*in_features,self.target_size)
                elif self.opts.data.multiscale_agg=='features' :
                     linear_layer = nn.Linear(in_features,self.target_size)
                     layer3=copy.deepcopy(self.model.layer3)
                     layer4=copy.deepcopy(self.model.layer4)
                     avgpool=copy.deepcopy(self.model.avgpool)

                for i in self.opts.data.multiscale:
                    models_scales.append(copy.deepcopy(self.model))
                self.model= nn.ModuleDict({ f'{key}_scale':model for key,model in zip(self.opts.data.multiscale,models_scales)})
                self.model.update({'fc':linear_layer})
                if self.opts.data.multiscale_agg=='features' :
                    self.model.update({'layer3':layer3 ,'layer4':layer4 , 'avgpool':avgpool})
                
            #elif self.opts.data.multiscale_agg=='features' :
                
            
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
            
        #self.model #.to(device)
        self.m = nn.Sigmoid()
        
        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, "val_" + name, value)
        for (name, value, _) in metrics:

            setattr(self, "train_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

        with open(self.opts.data.files.correction_thresh,'rb') as f:
            self.correction_data=pickle.load(f)

        self.correction=  self.correction_data.iloc[:,subset]

        assert self.correction.shape[1]==len(subset)
        
        
        #watch model gradients
        #wandb.watch(self.model, log='all', log_freq=5)


    def forward(self, x:Tensor) -> Any:
            assert len(self.opts.data.multiscale)>1, "the input is not multiscale, use normal trainer instead!"
            
            out_sat=[]
            if self.opts.data.multiscale_agg=='vectors' :
                for i,res in enumerate(self.opts.data.multiscale):

                        out_sat.append(self.model[f"{res}_scale"](x[:,i,:,:,:]))

                out_sat=torch.cat(out_sat,dim=-1)
    #             print(out_sat.requires_grad)
    #             print('out_sat shape',out_sat.shape)
                assert out_sat.shape[-1]== ((self.model['fc'].in_features)), 'shape of output after concat is wrong'
                out_sat=out_sat/(2**0.5)
                out = self.model['fc'](out_sat)
            elif self.opts.data.multiscale_agg=='features' :
                 scales=self.opts.data.multiscale
                 scales.sort(reverse=True)
                 for i,res in enumerate(scales):
                        out = self.model[f"{res}_scale"].conv1(x[:,i,:,:,:])
                        out = self.model[f"{res}_scale"].bn1(out)
                        out = self.model[f"{res}_scale"].relu(out)
                        out = self.model[f"{res}_scale"].maxpool(out)
                        # x,_=self.attn(x)
                        out = self.model[f"{res}_scale"].layer1(out)
                        out = self.model[f"{res}_scale"].layer2(out)
                        out_sat.append(out)
                 #shape of the largest image(lowest res)
                 print('largest res shape: ',out_sat[0].shape)
                 b,c,h,w=out_sat[0].shape
                 #pad other outs to match the largest image feature map
                 for i,s in enumerate(scales[1:]):
                        w_pad=(w-out_sat[i].shape[-1])//2
                        h_pad=(h-out_sat[i].shape[-2])//2
                        out_sat[i]=F.pad(out_sat[i], (w_pad,w_pad,h_pad,h_pad), "constant", 0) 
                 out_sat=torch.stack(out_sat,dim=1)
                 out_sat=torch.sum(out_sat,dim=1)
                 print('out sat shape after aggregation: ',out_sat.shape) 
                 x = self.model[f"{scales[0]}_scale"].layer3(out_sat)
                 x = self.model[f"{scales[0]}_scale"].layer4(x)
                 x = self.model[f"{scales[0]}_scale"].avgpool(x)
                 x = torch.flatten(x, 1)
               
                 out = self.model['fc'](x)
                 print('out shape: ',out.shape)

            
            return out

            
        

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int )-> Tensor:
       # from pdb import set_trace; set_trace()
        """Training step"""
        m = nn.Sigmoid()
        x = batch['sat']
        print('input shape:', x.shape)
        y = batch['target']

        b, no_species = y.shape        
        hotspot_id=batch['hotspot_id']
        correction= self.correction[self.correction_data['hotspot_id'].isin(list(hotspot_id))]
        correction=torch.tensor(correction.to_numpy(),device=y.device)
        assert correction.shape==(b,no_species) ,'shape of correction factor is not as expected'
        
       
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        #check weights are moving
        #for p in self.model.fc.parameters(): 
        #    print(p.data)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "inceptionv3":
            y_hat, aux_outputs = self.forward(x)
            if self.opts.data.correction_factor.use=='before':
                print('before correction ',y_hat[:10])
                y_hat*=correction
                print('after correction ', y_hat[:10])
    
               
            if self.target_type == "log":
                pred = y_hat.type_as(y)
                aux_pred = aux_outputs.type_as(y)
            else:
                pred = m(y_hat).type_as(y)
                aux_pred = m(aux_outputs).type_as(y)
                if self.opts.data.correction_factor.use=='after':
                        print('preds before, ', preds[:10])
                        preds=pred*correction
                        aux_preds=aux_pred*corretcion
                        cloned_pred=preds.clone().type_as(preds)
                        aux_clone=aux_preds.clone().type_as(aux_preds)
                        
                        #apply sigmoid after to normalize again ! not sure if this is mathematically correct
                        #pred=m(cloned_pred)
                        #aux_pred=m(aux_clone)
                        pred=torch.clip(cloned_pred, min=0, max=0.98)
                        aux_pred=torch.clip(aux_clone,min=0,max=0.98)
                elif self.opts.data.correction_factor.thresh:
                    mask=correction
                    cloned_pred=pred.clone().type_as(pred)
                    print('predictons before: ',cloned_pred)
                    cloned_pred*=mask.int()
                    y*=mask.int()
                    pred=cloned_pred

            #pred = m(y_hat)
            #aux_pred = m(aux_outputs)
            loss1 = self.criterion(y, pred)
            loss2 = self.criterion(y, aux_pred)
            loss = loss1 + loss2
        if self.opts.experiment.module.model == "train_linear":
            inter= self.feature_extractor(x)
            y_hat = self.forward(inter)
            if self.opts.data.correction_factor.use=='before':
                
                y_hat*=correction
           
            pred = m(y_hat).type_as(y)
            
            if self.opts.data.correction_factor.use=='after':
                        print('preds before, ', preds[:10])
                        preds=pred*correction
                        cloned_pred=preds.clone().type_as(preds)

                        #pred=m(cloned_pred)
                        pred=torch.clip(cloned_pred, min=0, max=0.98)
            elif self.opts.data.correction_factor.thresh:
                
                #mask=torch.le(pred, correction)
                mask=correction
                
                cloned_pred=pred.clone().type_as(pred)
                cloned_pred[~mask]=0
                
                pred=cloned_pred
                
            pred_ = pred.clone().type_as(y)

            
            loss = self.criterion(y, pred)
        else:
            y_hat = self.forward(x)
            if self.opts.data.correction_factor.use=='before':
               
                y_hat*=correction
            elif self.opts.data.correction_factor.thresh=='before':
                 y_hat*=correction
              
            if self.target_type == "log" or self.target_type == "binary":
                pred = y_hat.type_as(y)
                #pred_ = m(pred).clone().type_as(y)
            else :

                pred = m(y_hat).type_as(y)
               
              
                
            if self.opts.data.correction_factor.use=='after':
                        print('preds before correction Validation',preds[:10])
                        preds=pred*correction
                        cloned_pred=preds.clone().type_as(preds)
                        #pred=m(cloned_pred)
                        pred=torch.clip(cloned_pred, min=0, max=0.98)
                        
                        
            elif self.opts.data.correction_factor.thresh=='after':
          
                mask=correction
                
                cloned_pred=pred.clone().type_as(pred)
                print('predictons before: ',cloned_pred)
                
                
               
                cloned_pred*=mask.int()
                y*=mask.int()
                
                    
                pred=cloned_pred
                print('predictions after: ',pred)
            
            pred_ = pred.clone().type_as(y)

                
            if self.target_type == "binary":
                loss =  self.criterion(pred, y)
            elif self.target_type == "log":
                loss =  self.criterion(pred, torch.log(y + 1e-10))
            else:
                #print('maximum ytrue in trainstep',y.max())
                loss = self.criterion(y, pred)
                print('train_loss',loss)
        
        if self.target_type == "log":
            pred_ = torch.exp(pred_)
       # if self.current_epoch in [0,1]:
        #print("target", y) 
        #print("pred", pred_)
        if self.opts.data.target.type == "binary":
            
            pred_[pred_>=0.5] = 1
            pred_[pred_<0.5] = 0
        
        for (name, _, scale) in self.metrics:
            nname = "train_" + name
            if name == "accuracy":
                getattr(self,nname)(pred_, y.type(torch.uint8))
                #if getattr(self,name)(pred_,  y.type(torch.uint8)) != 1:
                #    print("pred_train", pred_)
                #    print("y", y)
                    #print(batch["hotspot_id"])
                #print(nname,getattr(self,nname)(pred_,  y.type(torch.uint8)))
            #elif name=='r2':
                #print('in r2')
                #getattr(self,name)(pred_, y.type(torch.uint8))
                #if getattr(self,name)(pred_,  y.type(torch.uint8)) != 1:
                #    print("pred_train", pred_)
                #    print("y", y)
                    #print(batch["hotspot_id"])
                
               # print(nname,getattr(self,name)(pred_,  y.type(torch.uint8)))
                
            else:
               
                getattr(self,nname)(y, pred_)
               # print(getattr(self,nname)(y, pred_))
                #print(nname,getattr(self,name))
               
            self.log(nname, getattr(self,nname), on_step = True, on_epoch = True)
        self.log("train_loss", loss, on_step = True, on_epoch= True)
        return loss
   

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int )->None:

        """Validation step """

        #import pdb; pdb.set_trace()
        print('heere in valid')

        x = batch['sat'].squeeze(1)#.to(device)
         
        y = batch['target']
        b, no_species = y.shape
        state_id = batch['state_id']
        hotspot_id=batch['hotspot_id']
        correction= self.correction[self.correction_data['hotspot_id'].isin(list(hotspot_id))]

        correction=torch.tensor(correction.to_numpy(),device=y.device)
        
        #correction=self.correction

       #correction = self.correction_data[state_id]
        #print('shapes of correction and outpu in valdiation ',correction.shape, y.shape)
        assert correction.shape == (b,no_species), 'shape of correction factor is not as expected'
        #correction.unsqueeze_(-1)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)

        if self.opts.experiment.module.model == "train_linear":
            inter= self.feature_extractor(x)
            y_hat = self.forward(inter)

        else:

            y_hat = self.forward(x)
        if self.opts.data.correction_factor.use=='before':
                print('in validation y hat before correction ',y_hat[:10])
                y_hat*=correction
                print('after correction ', y_hat[:10])
        elif self.opts.data.correction_factor.thresh=='before':
                 y_hat*=correction

        if self.target_type == "log" or self.target_type == "binary":
            pred = y_hat.type_as(y)
            #pred_ = m(pred).clone().type_as(y)
        else:
            pred = m(y_hat).type_as(y)
        
        if self.opts.data.correction_factor.use=='after':
                  
                        print('preds before correction Validation',preds[:10])
                        preds=pred*correction
                        
                        cloned_pred=preds.clone().type_as(preds)
                        pred=torch.clip(cloned_pred, min=0, max=0.98)
                        #pred=m(cloned_pred)
                        
        elif self.opts.data.correction_factor.thresh=='after':
                mask=correction

                cloned_pred=pred.clone().type_as(pred)
                #print(cloned_pred.shape,mask.shape)
                cloned_pred*=mask.int()
                y*=mask.int()
                pred=cloned_pred
                
        pred_ = pred.clone().type_as(y)

        if self.target_type == "binary":
            loss = self.criterion(pred, y)
        elif self.target_type == "log":
                loss =  self.criterion(pred, torch.log(y + 1e-10))
        else:
            loss = self.criterion(y, pred)


        if self.target_type == "log":
            pred_ = torch.exp(pred_)

        if self.opts.data.target.type == "binary":
            pred_[pred_>=0.5] = 1
            pred_[pred_<0.5] = 0
            
        for (name, _, scale) in self.metrics:
            nname = "val_" + name
            if name == "accuracy":
                getattr(self,name)(pred_, y.type(torch.uint8))
                print(nname,getattr(self,name))
          
            else:
                getattr(self,nname)(y, pred_)
                print(nname,getattr(self,nname)(y, pred_))
                
            self.log(nname, getattr(self, nname), on_step=True, on_epoch=True) 
        self.log("val_loss", loss, on_step = True, on_epoch = True)

    

    def test_step(
        self, batch: Dict[str, Any], batch_idx:int
    )-> None:
        """Test step """
        
        x = batch['sat'].squeeze(1)#.to(device)
        #self.model.to(device)
        y = batch['target']
        b, no_species = y.shape
        state_id = batch['state_id']
        hotspot_id=batch['hotspot_id']
        correction= self.correction[self.correction_data['hotspot_id'].isin(list(hotspot_id))]

        correction=torch.tensor(correction.to_numpy(),device=y.device)
        
        #correction = self.correction_data[state_id]
        #print('shapes of correction and output in validation ',correction.shape, y.shape)
        #assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'
        #print('shapes of correction and output in test ',correction.shape, y.shape)
        #assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'
        #correction.unsqueeze_(-1)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        y_hat = self.forward(x)
        if self.opts.data.correction_factor.use=='before':
                y_hat*=correction
                
        elif self.opts.data.correction_factor.thresh=='before':
                 y_hat*=correction
                
                
        if self.target_type == "log" or self.target_type == "binary":
            pred = y_hat.type_as(y)
            #pred_ = m(pred).clone()
        else:
            pred = m(y_hat).type_as(y)
            
            if self.opts.data.correction_factor.use=='after':
                    preds=pred*correction
                    cloned_pred=preds.clone().type_as(preds)
                    pred=torch.clip(cloned_pred, min=0, max=1)
                        #pred=m(cloned_pred)
            elif self.opts.data.correction_factor.thresh=='after':
                mask=correction
                cloned_pred=pred.clone().type_as(pred)
              
                cloned_pred*=mask.int()
                y*=mask.int()
                pred=cloned_pred
        loss = self.criterion(y, pred)

        pred_ = pred.clone().type_as(y)
      
        for (name, _, scale) in self.metrics:
            nname = "test_" + name
            if name == "accuracy":
                getattr(self,name)(pred_, y.type(torch.uint8))
                print(nname,getattr(self,name))
          
            else:
                getattr(self,nname)(y, pred_)
                print(nname,getattr(self,nname)(y, pred_))
                
            self.log(nname, getattr(self, nname), on_step=True, on_epoch=True) 
        self.log("test_loss", loss, on_step = True, on_epoch = True)

        
#         for i, elem in enumerate(pred_):
#             np.save(os.path.join(self.opts.save_preds_path, batch["hotspot_id"][i] + ".npy"), elem.cpu().detach().numpy())
#         print("saved elems")

    def get_optimizer(self, model, opts):
        
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(   #
                model.parameters(),
                lr=self.learning_rate#self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.opts.experiment.module.lr,  
                )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate#self.opts.experiment.module.lr,  
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
    elif opts.scheduler.name == "":
        return(None)
    else:
        raise ValueError(f"Scheduler'{opts.scheduler.name}' is not valid")

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
        self.res= self.opts.data.multiscale
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
        """create the train/test/val splits and prepare the transforms for the multires"""
        self.all_train_dataset = EbirdVisionDataset(
            df_paths = self.df_train,
            bands = self.bands,
            env = self.env,
            transforms =get_transforms(self.opts, "train"),
            mode = "train",
            datatype = self.datatype,
            target = self.target, 
            subset = self.subset,
            res=self.res,
            use_loc = self.use_loc
        )

        self.all_test_dataset = EbirdVisionDataset(                
            self.df_test, 
            bands = self.bands,
            env = self.env,
            transforms = get_transforms(self.opts, "val"),
            mode = "test",
            datatype = self.datatype,
            target = self.target, 
            subset = self.subset,
            res=self.res,
            use_loc = self.use_loc
            )

        self.all_val_dataset = EbirdVisionDataset(
            self.df_val,
            bands=self.bands,
            env = self.env,
            transforms =get_transforms(self.opts, "val"),
            mode = "val",
            datatype = self.datatype,
            target = self.target, 
            subset = self.subset,
            res=self.res,
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