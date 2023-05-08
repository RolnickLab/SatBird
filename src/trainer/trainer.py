import pickle
import numpy as np
import pandas as pd
import time
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
from torch.nn import BCELoss, BCEWithLogitsLoss

from src.transforms.transforms import get_transforms
from src.losses.losses import CustomCrossEntropyLoss, CustomCrossEntropy, get_metrics

from typing import Any, Dict, Optional
from src.dataset.dataloader import EbirdVisionDataset
from src.dataset.dataloader import get_subset


mse = nn.MSELoss()
m = nn.Sigmoid()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_target_size(opts, subset=None):
    if subset is None:
        subset = get_subset(opts.data.target.subset)
    target_size = len(subset) if subset is not None else opts.data.total_species
    return target_size


def get_nb_bands(bands):
    """
    Get number of channels in the satellite input branch (stack bands of satellite + environmental variables)
    """
    n = 0
    for b in bands:
        if b in ["r", "g", "b", "nir", "landuse"]:
            n += 1
        elif b == "ped":
            n += 8
        elif b == "bioclim":
            n += 19
        elif b == "rgb":
            n += 3
    return (n)


def get_scheduler(optimizer, opts):
    if opts.scheduler.name == "ReduceLROnPlateau":
        return (ReduceLROnPlateau(optimizer, factor=opts.scheduler.reduce_lr_plateau.factor,
                                  patience=opts.scheduler.reduce_lr_plateau.lr_schedule_patience))
    elif opts.scheduler.name == "StepLR":
        return (StepLR(optimizer, opts.scheduler.step_lr.step_size, opts.scheduler.step_lr.gamma))
    elif opts.scheduler.name == "WarmUp":
        return (LinearWarmupCosineAnnealingLR(optimizer, opts.scheduler.warmup.warmup_epochs,
                                              opts.scheduler.warmup.max_epochs))
    elif opts.scheduler.name == "Cyclical":
        return (CosineAnnealingWarmRestarts(optimizer, opts.scheduler.cyclical.t0, opts.scheduler.cyclical.tmult))
    elif opts.scheduler.name == "":
        return (None)
    else:
        raise ValueError(f"Scheduler'{opts.scheduler.name}' is not valid")


class EbirdTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.save_hyperparameters(opts)
        self.config_task(opts, **kwargs)
        self.opts = opts
        # define self.learning_rate to enable learning rate finder
        self.learning_rate = self.opts.experiment.module.lr

    def config_task(self, opts, **kwargs: Any) -> None:
        self.opts = opts
        self.means = None
        self.is_transfer_learning = True if self.opts.experiment.module.resume else False

        # get target vector size (number of species we consider)
        self.subset = get_subset(self.opts.data.target.subset)
        self.target_size = get_target_size(opts, self.subset)
        print("Predicting ", self.target_size, "species")
        self.target_type = self.opts.data.target.type

        if self.target_type == "binary":
            # ground truth is 0-1. if bird is reported at a hotspot, target = 1
            self.criterion = BCEWithLogitsLoss()
            print("Training with BCE Loss")
        elif self.target_type == "log":
            self.criterion = nn.MSELoss()
            print("Training with MSE Loss")
        else:
            # target is num checklists reporting species i / total number of checklists at a hotspot
            self.criterion = CustomCrossEntropyLoss()
            # CustomCrossEntropy(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs)
            # mse
            # CustomCrossEntropy(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_absCustomCrossEntropy(self.opts.losses.ce.lambd_pres,self.opts.losses.ce.lambd_abs)
            print("Training with Custom CE Loss")

        if self.is_transfer_learning:
            self.model = self.get_sat_model_AtoB()
        else:
            self.model = self.get_sat_model()

    def get_sat_model_AtoB(self):
        """
        #TODO: merge with get_sat_model()
        transfers weights between species A to species B
        """
        self.model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
        if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
            self.bands = self.opts.data.bands + self.opts.data.env
            self.model.conv1 = nn.Conv2d(
                get_nb_bands(self.bands),
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            # assume first three channels are rgb

        # loading seco mode
        pretrained_model_path = os.path.join(self.opts.base_dir, self.opts.experiment.module.resume)
        print('loading a pretrained model..', pretrained_model_path)
        loaded_dict = torch.load(pretrained_model_path)['state_dict']
        self.model.fc = nn.Sequential()

        # load state dict keys
        for name, param in loaded_dict.items():
            if name not in self.state_dict():
                continue
            print("loaded..", name)
            self.state_dict()[name].copy_(param)
        self.model.fc = nn.Linear(512, self.target_size)

        with open(self.opts.data.files.correction_thresh, 'rb') as f:
            self.correction_t_data = pickle.load(f)

        if self.opts.data.correction_factor.use:
            with open(self.opts.data.files.correction, 'rb') as f:
                self.correction_data = pickle.load(f)
                if self.subset:
                    self.correction = self.correction_data[:, self.subset]

        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, "val_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "train_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

        return self.model

    def get_sat_model(self):
        if self.opts.experiment.module.model == "train_linear":
            self.feature_extractor = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                self.feature_extractor.conv1 = nn.Conv2d(get_nb_bands(self.bands), 64, kernel_size=(7, 7),
                                                         stride=(2, 2), padding=(3, 3), bias=False)
            if self.opts.experiment.module.fc == "linear":
                self.feature_extractor.fc = nn.Linear(512, self.target_size)
            ckpt = torch.load(self.opts.experiment.module.resume)
            for key in list(ckpt["state_dict"].keys()):
                ckpt["state_dict"][key.replace('model.', '')] = ckpt["state_dict"].pop(key)
            self.feature_extractor.load_state_dict(ckpt["state_dict"])
            print("initialized network, freezing weights")
            self.feature_extractor.fc = nn.Sequential()
            # self.feature_extractor.freeze()
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
                # assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    # self.model.conv1.weight.data[:, :orig_channels, :, :] = weights
                    self.model.conv1.weight.data = init_first_layer_weights(get_nb_bands(self.bands), weights)
            # loading seco mode
            if self.opts.experiment.module.resume:
                print('loading a pretrained model')
                ckpt = torch.load(self.opts.experiment.module.resume)
                self.model.fc = nn.Sequential()
                loaded_dict = ckpt['state_dict']
                model_dict = self.model.state_dict()
                del loaded_dict["queue"]
                del loaded_dict["queue_ptr"]
                # load state dict keys
                for key_model, key_seco in zip(model_dict.keys(), loaded_dict.keys()):
                    # ignore first layer weights(use imagenet ones)
                    if key_model == 'conv1.weight':
                        continue
                    model_dict[key_model] = loaded_dict[key_seco]

                self.model.load_state_dict(model_dict)
            if self.opts.experiment.module.fc == "linear":
                self.model.fc = nn.Linear(512, self.target_size)
            elif self.opts.experiment.module.fc == "linear_net":
                self.model.fc = nn.Sequential(nn.Linear(512, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, self.target_size))
            else:
                self.model.fc = nn.Linear(512, self.target_size)

        elif self.opts.experiment.module.model == "resnet50":
            model = models.resnet50(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = model.conv1.in_channels
                weights = model.conv1.weight.data.clone()
                model.conv1 = nn.Conv2d(
                    get_nb_bands(self.bands),
                    64,
                    kernel_size=(7, 7),
                    stride=(2, 2),
                    padding=(3, 3),
                    bias=False,
                )
                # assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    model.conv1.weight.data[:, :orig_channels, :, :] = weights
            # loading geossl model
            if self.opts.experiment.module.resume:
                path = self.opts.experiment.module.resume
                checkpoint = torch.load(path)

                loaded_dict = checkpoint['state_dict']

                model_dict = model.state_dict()
                del loaded_dict["module.queue"]
                del loaded_dict["module.queue_ptr"]
                #                 print('loaded dict keys',loaded_dict.keys(),'model_keys',model_dict.keys())
                model.conv1.weight.data[:, :orig_channels, :, :] = loaded_dict[list(loaded_dict.keys())[0]]
                # load state dict keys

                for key_model, key_seco in zip(model_dict.keys(), loaded_dict.keys()):
                    if 'fc' in key_model:
                        print('here in fc continue')
                        # ignore fc weight
                        continue

                    if key_seco == 'module.encoder_q.conv1.weight':
                        print('here in conv1')
                        continue
                    #                     print(key_model,key_seco)
                    model_dict[key_model] = loaded_dict[key_seco]
                #                 for key in list(self.model.state_dict().keys()):
                #                     model_dict['model.'+key]=model_dict.pop(key)
                msg = model.load_state_dict(model_dict, strict=False)
                print(msg)
                self.model = model

            if self.opts.experiment.module.fc == "linear":
                self.model.fc = nn.Linear(2048, self.target_size)
            elif self.opts.experiment.module.fc == "linear_net":
                self.model.fc = nn.Sequential(nn.Linear(2048, 2048),
                                              nn.ReLU(),
                                              nn.Linear(2048, self.target_size))
            else:
                self.model.fc = nn.Linear(2048, self.target_size)


        elif self.opts.experiment.module.model == "inceptionv3":
            self.model = models.inception_v3(pretrained=self.opts.experiment.module.pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, self.target_size)
            self.model.fc = nn.Linear(2048, self.target_size)

        elif self.opts.experiment.module.model == "linear":
            nb_bands = get_nb_bands(self.opts.data.bands + self.opts.data.env)
            self.model = nn.Linear(nb_bands * 64 * 64, self.target_size)

        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")

        if self.opts.experiment.module.init_bias == "means":
            print("initializing biases with mean predictor")
            self.means = np.load(self.opts.experiment.module.means_path)[0, self.subset]
            means = torch.Tensor(self.means)

            means = torch.logit(means, eps=1e-10)
            if self.opts.experiment.module.model != "linear":
                if self.opts.experiment.module.fc == "linear_net":
                    self.model.fc[2].bias.data = means
                else:
                    self.model.fc.bias.data = means
        else:
            print("no initialization of biases")

        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, "val_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "train_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

        #         if   self.opts.data.correction_factor.thresh:
        with open(self.opts.data.files.correction_thresh, 'rb') as f:

            self.correction_t_data = pickle.load(f)

        if self.opts.data.correction_factor.use:
            with open(self.opts.data.files.correction, 'rb') as f:
                self.correction_data = pickle.load(f)
                if self.subset:
                    self.correction = self.correction_data[:, self.subset]

        # assert self.correction.shape[1]==len(subset)

        # watch model gradients
        # wandb.watch(self.model, log='all', log_freq=5)

        return self.model
    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def training_step(
            self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        # from pdb import set_trace; set_trace()
        """Training step"""
        m = nn.Sigmoid()
        x = batch['sat'].squeeze(1)
        print('input shape:', x.shape)
        y = batch['target']

        b, no_species = y.shape
        hotspot_id = batch['hotspot_id']
        state_id = batch['state_id']
        #         if   self.opts.data.correction_factor.thresh:
        correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
            columns=["index"]).iloc[:, self.subset].values
        correction_t = torch.tensor(correction_t, device=y.device)

        if self.opts.data.correction_factor.use:
            self.correction_data = torch.tensor(self.correction, device=y.device)
            correction = self.correction[state_id]

        #         self.correction_data=torch.tensor(self.correction,device=y.device)
        #         correction=self.correction[state_id]
        #         print(correction.shape)
        #         assert correction.shape==(b,no_species) ,'shape of correction factor is not as expected'

        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        # check weights are moving
        # for p in self.model.fc.parameters():
        #    print(p.data)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "inceptionv3":
            y_hat, aux_outputs = self.forward(x)
            if self.opts.data.correction_factor.use == 'before':
                print('before correction ', y_hat[:10])
                y_hat *= correction
                print('after correction ', y_hat[:10])

            if self.target_type == "log":
                pred = y_hat.type_as(y)
                aux_pred = aux_outputs.type_as(y)
            else:
                pred = m(y_hat).type_as(y)
                aux_pred = m(aux_outputs).type_as(y)
                if self.opts.data.correction_factor.use == 'after':
                    print('preds before, ', preds[:10])
                    # preds=pred*correction
                    y = y * correction
                    # aux_preds=aux_pred*corretcion
                    cloned_pred = preds.clone().type_as(preds)
                    aux_clone = aux_preds.clone().type_as(aux_preds)


                elif self.opts.data.correction_factor.thresh:
                    mask = correction_t
                    cloned_pred = pred.clone().type_as(pred)
                    print('predictons before: ', cloned_pred)
                    cloned_pred *= mask.int()
                    y *= mask.int()
                    pred = cloned_pred

            # pred = m(y_hat)
            # aux_pred = m(aux_outputs)
            loss1 = self.criterion(y, pred)
            loss2 = self.criterion(y, aux_pred)
            loss = loss1 + loss2
        elif self.opts.experiment.module.model == "train_linear":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)
            if self.opts.data.correction_factor.use == 'before':
                y_hat *= correction

            pred = m(y_hat).type_as(y)

            if self.opts.data.correction_factor.use == 'after':
                preds = pred * correction
                # y= y * correction

                cloned_pred = preds.clone().type_as(preds)

                # pred=m(cloned_pred)
                pred = torch.clip(cloned_pred, min=0, max=0.99)
            elif self.opts.data.correction_factor.thresh == 'after':

                # mask=torch.le(pred, correction)
                mask = correction_t

                cloned_pred = pred.clone().type_as(pred)
                cloned_pred[~mask] = 0

                pred = cloned_pred

            pred_ = pred.clone().type_as(y)

            loss = self.criterion(y, pred)
        else:
            y_hat = self.forward(x)
            if self.opts.data.correction_factor.use == 'before':
                y_hat *= correction

            if self.target_type == "log" or self.target_type == "binary":
                pred = y_hat.type_as(y)
                # pred_ = m(pred).clone().type_as(y)
            else:

                pred = m(y_hat).type_as(y)

            if self.opts.data.correction_factor.use == 'after':
                preds = pred * correction
                # preds=pred
                # y= y * correction
                cloned_pred = preds.clone().type_as(preds)
                # pred=m(cloned_pred)
                pred = torch.clip(cloned_pred, min=0, max=0.98)


            elif self.opts.data.correction_factor.thresh == 'after':

                mask = correction_t

                cloned_pred = pred.clone().type_as(pred)
                print('predictons before: ', cloned_pred)

                cloned_pred *= mask.int()
                y *= mask.int()

                pred = cloned_pred
                print('predictions after: ', pred)
            else:

                y = y * correction_t
            pred_ = pred.clone().type_as(y)

            if self.target_type == "binary":
                loss = self.criterion(pred, y)
            elif self.target_type == "log":
                loss = self.criterion(pred, torch.log(y + 1e-10))
            else:
                # print('maximum ytrue in trainstep',y.max())
                loss = self.criterion(y, pred)
                print('train_loss', loss)

        if self.target_type == "log":
            pred_ = torch.exp(pred_)
        # if self.current_epoch in [0,1]:
        # print("target", y)
        # print("pred", pred_)
        if self.opts.data.target.type == "binary":
            pred_[pred_ >= 0.5] = 1
            pred_[pred_ < 0.5] = 0

        for (name, _, scale) in self.metrics:
            nname = "train_" + name
            if name == "accuracy":
                value = getattr(self, nname)(pred_, y.type(torch.uint8))
            elif name == 'r2':
                value = torch.mean(getattr(self, nname)(y, pred_))


            else:

                value = getattr(self, nname)(y, pred_)
            # print(getattr(self,nname)(y, pred_))
            # print(nname,getattr(self,name))

            self.log(nname, value, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int) -> None:

        """Validation step """

        # import pdb; pdb.set_trace()
        x = batch['sat'].squeeze(1)  # .to(device)

        y = batch['target']
        b, no_species = y.shape
        state_id = batch['state_id']
        hotspot_id = batch['hotspot_id']
        #         if   self.opts.data.correction_factor.thresh:
        correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
            columns=["index"]).iloc[:, self.subset].values
        correction_t = torch.tensor(correction_t, device=y.device)
        #
        if self.opts.data.correction_factor.use:
            self.correction = torch.tensor(self.correction, device=y.device)
            correction = self.correction[state_id]

        #         assert correction.shape == (b,no_species), 'shape of correction factor is not as expected'
        # correction.unsqueeze_(-1)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)

        if self.opts.experiment.module.model == "train_linear":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)

        else:

            y_hat = self.forward(x)
        if self.opts.data.correction_factor.use == 'before':
            print('in validation y hat before correction ', y_hat[:10])
            y_hat *= correction
            print('after correction ', y_hat[:10])

        if self.target_type == "log" or self.target_type == "binary":
            pred = y_hat.type_as(y)
            # pred_ = m(pred).clone().type_as(y)
        else:
            pred = m(y_hat).type_as(y)

        if self.opts.data.correction_factor.use == 'after':

            # print('preds before correction Validation',pred[:10])
            # preds=pred
            preds = pred * correction
            #                         y= y * correction
            cloned_pred = preds.clone().type_as(preds)
            pred = torch.clip(cloned_pred, min=0, max=0.99)
        # pred=m(cloned_pred)

        elif self.opts.data.correction_factor.thresh == 'after':
            mask = correction_t

            cloned_pred = pred.clone().type_as(pred)
            # print(cloned_pred.shape,mask.shape)
            cloned_pred *= mask.int()
            y *= mask.int()
            pred = cloned_pred
        else:

            y = y * correction_t
        pred_ = pred.clone().type_as(y)

        if self.target_type == "binary":
            loss = self.criterion(pred, y)
        elif self.target_type == "log":
            loss = self.criterion(pred, torch.log(y + 1e-10))
        else:
            loss = self.criterion(y, pred)

        if self.target_type == "log":
            pred_ = torch.exp(pred_)

        if self.opts.data.target.type == "binary":
            pred_[pred_ >= 0.5] = 1
            pred_[pred_ < 0.5] = 0

        for (name, _, scale) in self.metrics:
            nname = "val_" + name
            if name == "accuracy":
                value = getattr(self, name)(pred_, y.type(torch.uint8))
                print(nname, getattr(self, name))
            elif name == 'r2':
                value = torch.mean(getattr(self, nname)(y, pred_))
            else:
                value = getattr(self, nname)(y, pred_)
                print(nname, getattr(self, nname)(y, pred_))

            self.log(nname, value, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step """

        x = batch['sat'].squeeze(1)  # .to(device)
        # self.model.to(device)
        y = batch['target']
        b, no_species = y.shape
        state_id = batch['state_id']
        hotspot_id = batch['hotspot_id']
        #         if   self.opts.data.correction_factor.thresh:
        correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
            columns=["index"]).iloc[:, self.subset].values
        correction_t = torch.tensor(correction_t, device=y.device)
        #             correction =  (self.correction_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(columns = ["index"]).iloc[:,self.subset].values
        if self.opts.data.correction_factor.use:
            self.correction = torch.tensor(self.correction, device=y.device)
            correction = self.correction[state_id]

        # (self.correction_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)].reset_index().set_index('index')).iloc[:,self.subset]
        #
        #         correction=self.correction[state_id]

        # correction = self.correction_data[state_id]
        # print('shapes of correction and output in validation ',correction.shape, y.shape)
        # assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'
        # print('shapes of correction and output in test ',correction.shape, y.shape)
        # assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'
        # correction.unsqueeze_(-1)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        y_hat = self.forward(x)
        if self.opts.data.correction_factor.use == 'before':
            y_hat *= correction

        if self.target_type == "log" or self.target_type == "binary":
            pred = y_hat.type_as(y)
            # pred_ = m(pred).clone()
        else:
            pred = m(y_hat).type_as(y)

            if self.opts.data.correction_factor.use == 'after':
                #                     preds=pred
                #                     y= y * correction
                preds = pred * correction
                cloned_pred = preds.clone().type_as(preds)
                pred = torch.clip(cloned_pred, min=0, max=0.99)
            # pred=m(cloned_pred)
            elif self.opts.data.correction_factor.thresh == 'after':
                print('NNNNNNNNNNNNNNNNNNNNNNN in correction')
                mask = correction_t
                cloned_pred = pred.clone().type_as(pred)

                cloned_pred *= mask

                y *= mask
                pred = cloned_pred
            else:

                y = y * correction_t
        loss = self.criterion(y, pred)

        pred_ = pred.clone().type_as(y)

        for (name, _, scale) in self.metrics:
            nname = "test_" + name
            if name == "accuracy":
                value = getattr(self, name)(pred_, y.type(torch.uint8))
                print(nname, getattr(self, name))
            elif name == 'r2':
                value = torch.mean(getattr(self, nname)(y, pred_))
            else:
                value = getattr(self, nname)(y, pred_)
                print(nname, getattr(self, nname)(y, pred_))

            self.log(nname, value, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)

        if self.opts.save_preds_path != "":
            for i, elem in enumerate(pred):
                np.save(os.path.join(self.opts.save_preds_path, batch["hotspot_id"][i] + ".npy"),
                        elem.cpu().detach().numpy())
        print("saved elems")

    def get_optimizer(self, model, opts):
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(  #
                model.parameters(),
                lr=self.learning_rate,  # self.opts.experiment.module.lr,
                weight_decay=0.00001
            )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate  # self.opts.experiment.module.lr,
            )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate
            )
        else:
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return (optimizer)

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = self.get_optimizer(self.model, self.opts)
        scheduler = get_scheduler(optimizer, self.opts)
        print("scheduler", scheduler)
        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
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
        self.res = self.opts.data.multiscale
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

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits and prepare the transforms for the multires"""
        self.all_train_dataset = EbirdVisionDataset(
            df_paths=self.df_train,
            bands=self.bands,
            env=self.env,
            transforms=get_transforms(self.opts, "train"),
            mode="train",
            datatype=self.datatype,
            target=self.target,
            subset=self.subset,
            res=self.res,
            use_loc=self.use_loc
        )

        self.all_test_dataset = EbirdVisionDataset(
            self.df_test,
            bands=self.bands,
            env=self.env,
            transforms=get_transforms(self.opts, "val"),
            mode="test",
            datatype=self.datatype,
            target=self.target,
            subset=self.subset,
            res=self.res,
            use_loc=self.use_loc
        )

        self.all_val_dataset = EbirdVisionDataset(
            self.df_val,
            bands=self.bands,
            env=self.env,
            transforms=get_transforms(self.opts, "val"),
            mode="val",
            datatype=self.datatype,
            target=self.target,
            subset=self.subset,
            res=self.res,
            use_loc=self.use_loc
        )

        # TODO: Create subsets of the data

        self.train_dataset = self.all_train_dataset

        self.test_dataset = self.all_test_dataset

        self.val_dataset = self.all_val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def init_first_layer_weights(in_channels: int, rgb_weights,
                             hs_weight_init: str = 'random'):
    '''Initializes the weights for filters in the first conv layer.
      If we are using RGB-only, then just initializes var to rgb_weights. Otherwise, uses
      hs_weight_init to determine how to initialize the weights for non-RGB bands.
      Args
      - int: in_channesl, input channels
          - in_channesl is  either 3 (RGB), 7 (lxv3), or 9 (Landsat7) or 2 (NL)
      - rgb_weights: ndarray of np.float32, shape [64, 3, F, F]
      - hs_weight_init: str, one of ['random', 'same', 'samescaled']
      Returs
      -torch tensor : final_weights
      '''

    out_channels, rgb_channels, H, W = rgb_weights.shape
    print('rgb weight shape ', rgb_weights.shape)
    rgb_weights = torch.tensor(rgb_weights, device='cuda')
    ms_channels = in_channels - rgb_channels
    if in_channels == 3:
        final_weights = rgb_weights

    elif in_channels < 3:
        with torch.no_grad():
            mean = rgb_weights.mean()
            std = rgb_weights.std()
            final_weights = torch.empty((out_channels, in_channels, H, W), device='cuda')
            final_weights = torch.nn.init.trunc_normal_(final_weights, mean, std)
    elif in_channels > 3:
        # spectral images

        if hs_weight_init == 'same':

            with torch.no_grad():
                mean = rgb_weights.mean(dim=1, keepdim=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = mean

        elif hs_weight_init == 'random':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean()
                std = rgb_weights.std()
                ms_weights = torch.empty((out_channels, ms_channels, H, W), device='cuda')
                ms_weights = torch.nn.init.trunc_normal_(ms_weights, mean, std)
            print(f'random: {time.time() - start}')

        elif hs_weight_init == 'samescaled':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean(dim=1, keepdim=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = (mean * 3) / (3 + ms_channels)
                # scale both rgb_weights and ms_weights
                rgb_weights = (rgb_weights * 3) / (3 + ms_channels)


        else:

            raise ValueError(f'Unknown hs_weight_init type: {hs_weight_init}')

        final_weights = torch.cat([rgb_weights, ms_weights], dim=1)
    print('init__layer_weight shape ', final_weights.shape)
    return final_weights
