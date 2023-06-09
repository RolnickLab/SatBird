# main trainer
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset.dataloader import EbirdVisionDataset
from src.dataset.dataloader import get_subset
from src.losses.losses import CustomCrossEntropyLoss, WeightedCustomCrossEntropyLoss
from src.losses.metrics import get_metrics
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler, init_first_layer_weights, load_from_checkpoint
from src.transforms.transforms import get_transforms
from src.models.vit import ViTFinetune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EbirdTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.save_hyperparameters(opts)
        self.opts = opts
        self.means = None
        self.is_transfer_learning = True if self.opts.experiment.module.resume else False
        self.freeze_backbone = self.opts.experiment.module.freeze
        # get target vector size (number of species we consider)
        self.subset = get_subset(self.opts.data.target.subset)
        self.target_size = get_target_size(opts, self.subset)
        print("Predicting ", self.target_size, "species")

        self.target_type = self.opts.data.target.type

        # define self.learning_rate to enable learning rate finder
        self.learning_rate = self.opts.experiment.module.lr
        self.sigmoid_activation = nn.Sigmoid()

        self.config_task(opts, **kwargs)

    def config_task(self, opts, **kwargs: Any) -> None:

        if self.target_type == "binary":
            # ground truth is 0-1. if bird is reported at a hotspot, target = 1
            self.criterion = BCEWithLogitsLoss()
            print("Training with BCE Loss")
        elif self.target_type == "log":
            self.criterion = nn.MSELoss()
            print("Training with MSE Loss")
        else:
            # target is num checklists reporting species i / total number of checklists at a hotspot
            if self.opts.experiment.module.use_weighted_loss:
                self.criterion = WeightedCustomCrossEntropyLoss()
                print("Training with Weighted CE Loss")
            else:
                self.criterion = CustomCrossEntropyLoss()
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
            # self.state_dict()[name].copy_(param)
            if name == "model.conv1.weight":
                self.state_dict()[name].copy_(param[:, 0:23, :, :])
            else:
                self.state_dict()[name].copy_(param)
            if self.freeze_backbone:
                self.state_dict()[name].requires_grad = False
        self.model.fc = nn.Linear(512, self.target_size)

        if self.opts.data.correction_factor.thresh:
            with open(os.path.join(self.opts.data.files.base, self.opts.data.files.correction_thresh), 'rb') as f:
                self.correction_t_data = pickle.load(f)

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

        elif self.opts.experiment.module.model == "satlas":
            #first lets assume freezing the pretrained model 
            self.feature_extractor = torchvision.models.swin_transformer.swin_v2_b()
            #TODO: remove hardcoded path
            full_state_dict = torch.load('/network/scratch/a/amna.elmustafa/hager/ecosystem-embedding/src/trainer/satlas-model-v1-lowres.pth')
            swin_prefix = 'backbone.backbone.'
            swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}

            self.feature_extractor.load_state_dict(swin_state_dict)
            self.feature_extractor.to('cuda:0')
            print("initialized network, freezing weights")

            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(1000, self.target_size)
        elif self.opts.experiment.module.model =="satmae":
            satmae=ViTFinetune(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=self.target_size,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                drop_rate=0.1,
            )
            #TODO: remove the hardcoded path
            satmae=load_from_checkpoint('/network/scratch/a/amna.elmustafa/hager/ecosystem-embedding/src/trainer/fmow_pretrain.pth',satmae )
            satmae.to('cuda')
            in_feat=satmae.fc.in_features
            satmae.fc=nn.Sequential()
            print("initialized network, freezing weights")
            self.feature_extractor=satmae
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(in_feat, self.target_size)

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

        # range maps
        if self.opts.data.correction_factor.thresh:
            with open(os.path.join(self.opts.data.files.base, self.opts.data.files.correction_thresh), 'rb') as f:
                self.correction_t_data = pickle.load(f)

        return self.model

    def forward(self, x: Tensor) -> Any:
        return self.model(x)

    def training_step(
            self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        # from pdb import set_trace; set_trace()
        """Training step"""
        x = batch['sat'].squeeze(1)
        print('input shape:', x.shape)
        y = batch['target']
        hotspot_id = batch['hotspot_id']
            

        if self.opts.experiment.module.loss_weight == "sqrt":
            new_weights = torch.sqrt(batch["num_complete_checklists"])
        elif self.opts.experiment.module.loss_weight == "log":
            new_weights = torch.log(batch["num_complete_checklists"])
        else:
            new_weights = batch["num_complete_checklists"]

        new_weights = torch.ones(y.shape, device=torch.device("cuda")) * new_weights.view(
            -1, 1
        )

        if self.opts.data.correction_factor.thresh:
            correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
                columns=["index"]).iloc[:, self.subset].values
            correction_t = torch.tensor(correction_t, device=y.device)

        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        # check weights are moving
        print("Model is on cuda", next(self.model.parameters()).is_cuda)
       
        if self.opts.experiment.module.model == "inceptionv3":
            y_hat, aux_outputs = self.forward(x)

            if self.target_type == "log":
                pred = y_hat.type_as(y)
                aux_pred = aux_outputs.type_as(y)
            else:
                pred = self.sigmoid_activation(y_hat).type_as(y)
                aux_pred = self.sigmoid_activation(aux_outputs).type_as(y)

                if self.opts.data.correction_factor.thresh:
                    mask = correction_t
                    cloned_pred = pred.clone().type_as(pred)
                    # print('predictons before: ', cloned_pred)
                    cloned_pred *= mask.int()
                    y *= mask.int()
                    pred = cloned_pred

            loss1 = self.criterion(y, pred)
            loss2 = self.criterion(y, aux_pred)
            loss = loss1 + loss2

        elif self.opts.experiment.module.model == "train_linear":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)

            pred = self.sigmoid_activation(y_hat).type_as(y)

            if self.opts.data.correction_factor.thresh == 'after':
                mask = correction_t

                cloned_pred = pred.clone().type_as(pred)
                cloned_pred[~mask] = 0

                pred = cloned_pred

            pred_ = pred.clone().type_as(y)

            loss = self.criterion(y, pred)

        elif self.opts.experiment.module.model=="satlas" or self.opts.experiment.module.model=="satmae" :
                inter = self.feature_extractor(x)
                print('features shape ',inter.shape)
                y_hat = self.forward(inter)

                if self.target_type == "log" or self.target_type == "binary":
                    pred = y_hat.type_as(y)
                    # pred_ = m(pred).clone().type_as(y)
                else:

                    pred = self.sigmoid_activation(y_hat).type_as(y)

                if self.opts.data.correction_factor.thresh == 'after':
                    mask = correction_t

                    cloned_pred = pred.clone().type_as(pred)
                    # print('predictons before: ', cloned_pred)

                    cloned_pred *= mask.int()
                    y *= mask.int()

                    pred = cloned_pred
                    # print('predictions after: ', pred)

                pred_ = pred.clone().type_as(y)

                if self.target_type == "binary":
                    loss = self.criterion(pred, y)
                elif self.target_type == "log":
                    loss = self.criterion(pred, torch.log(y + 1e-10))
                else:
                    # print('maximum ytrue in trainstep',y.max())
                    loss = self.criterion(y, pred)
                    # print('train_loss', loss)
        else:
            y_hat = self.forward(x)

            if self.target_type == "log" or self.target_type == "binary":
                pred = y_hat.type_as(y)
            else:

                pred = self.sigmoid_activation(y_hat).type_as(y)

            if self.opts.data.correction_factor.thresh == 'after':
                mask = correction_t
                cloned_pred = pred.clone().type_as(pred)
                # print('predictons before: ', cloned_pred)

                cloned_pred *= mask.int()
                y *= mask.int()

                pred = cloned_pred
                # print('predictions after: ', pred)

            pred_ = pred.clone().type_as(y)

            if self.target_type == "binary":
                loss = self.criterion(pred, y)
            elif self.target_type == "log":
                loss = self.criterion(pred, torch.log(y + 1e-10))
            else:
                # print('maximum ytrue in trainstep',y.max())
                if self.opts.experiment.module.use_weighted_loss:
                    print("Using Weighted CrossEntropy Loss")
                    loss = self.criterion(y, pred, new_weights)
                else:
                    loss = self.criterion(y, pred)
                # print('train_loss', loss)

        if self.target_type == "log":
            pred_ = torch.exp(pred_)

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

            self.log(nname, value, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    

    def training_step_refactored(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Training step"""
        x = batch['sat'].squeeze(1)
        y = batch['target']
        hotspot_id = batch['hotspot_id']

        new_weights = self._get_new_weights(batch)
        correction_t = self._get_correction_t(batch, hotspot_id, y.device) if self.opts.data.correction_factor.thresh else None
        
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        print("Model is on cuda", next(self.model.parameters()).is_cuda)

        loss, pred = self._compute_loss_and_pred(x, y, new_weights, correction_t)

        self._compute_and_log_metrics(loss, pred, y)
        return loss


    def _get_new_weights(self, batch):
        """Helper function to calculate new_weights"""
        weight_type = self.opts.experiment.module.loss_weight
        if weight_type == "sqrt":
            return torch.sqrt(batch["num_complete_checklists"])
        elif weight_type == "log":
            return torch.log(batch["num_complete_checklists"])
        else:
            return batch["num_complete_checklists"]


    def _get_correction_t(self, batch, hotspot_id, device):
        """Helper function to get correction_t"""
        correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
            columns=["index"]).iloc[:, self.subset].values
        return torch.tensor(correction_t, device=device)


    def _compute_loss_and_pred(self, x, y, new_weights, correction_t):
        """Helper function to compute loss and predictions"""
        model_type = self.opts.experiment.module.model
        target_type = self.target_type
        thresh_type = self.opts.data.correction_factor.thresh
        use_weighted_loss = self.opts.experiment.module.use_weighted_loss

        if model_type in ["inceptionv3", "train_linear", "satlas", "satmae"]:
            # The below function computes loss and predictions for model types "inceptionv3", "train_linear", "satlas", "satmae".
            # It is a more complicated case that has been moved to a separate function for better readability.
            return self._compute_complex_loss_and_pred(x, y, new_weights, correction_t)
        else:
            y_hat = self.forward(x)
            pred = y_hat if target_type in ["log", "binary"] else self.sigmoid_activation(y_hat).type_as(y)
            pred = self._apply_correction_factor(pred, y, correction_t, thresh_type)

            if target_type == "binary":
                loss = self.criterion(pred, y)
            elif target_type == "log":
                loss = self.criterion(pred, torch.log(y + 1e-10))
            else:
                loss = self.criterion(y, pred, new_weights) if use_weighted_loss else self.criterion(y, pred)
            return loss, pred


    def _compute_complex_loss_and_pred(self, x, y, new_weights, correction_t):
        """Helper function to compute loss and predictions for complex model types"""
        model_type = self.opts.experiment.module.model
        target_type = self.target_type
        thresh_type = self.opts.data.correction_factor.thresh

        inter = self.feature_extractor(x)
        y_hat = self.forward(inter) if model_type != "inceptionv3" else self.forward(x)

        pred = y_hat if target_type in ["log", "binary"] else self.sigmoid_activation(y_hat).type_as(y)
        pred = self._apply_correction_factor(pred, y, correction_t, thresh_type)

        if model_type == "inceptionv3":
            aux_outputs = y_hat[1]
            aux_pred = aux_outputs if target_type == "log" else self.sigmoid_activation(aux_outputs).type_as(y)
            loss1 = self.criterion(y, pred)
            loss2 = self.criterion(y, aux_pred)
            loss = loss1 + loss2
        else:
            if target_type == "binary":
                loss = self.criterion(pred, y)
            elif target_type == "log":
                loss = self.criterion(pred, torch.log(y + 1e-10))
            else:
                loss = self.criterion(y, pred)
        return loss, pred


    def _apply_correction_factor(self, pred, y, correction_t, thresh_type):
        """Helper function to apply correction factor if necessary"""
        if thresh_type == 'after' and correction_t is not None:
            mask = correction_t
            cloned_pred = pred.clone().type_as(pred)
            cloned_pred[~mask] = 0
            pred = cloned_pred
            y *= mask.int()
        return pred


    def _compute_and_log_metrics(self, loss, pred, y):
        """Helper function to compute and log metrics"""
        pred_ = pred.clone().type_as(y)

        if self.target_type == "log":
            pred_ = torch.exp(pred_)

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

            self.log(nname, value, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)





    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int) -> None:

        """Validation step """
        x = batch['sat'].squeeze(1)  # .to(device)
        y = batch['target']

        hotspot_id = batch['hotspot_id']

        if self.opts.data.correction_factor.thresh:
            correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
                columns=["index"]).iloc[:, self.subset].values
            correction_t = torch.tensor(correction_t, device=y.device)
            self.correction = correction_t

        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)

        if self.opts.experiment.module.model == "train_linear":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)

        elif self.opts.experiment.module.model=="satlas" or self.opts.experiment.module.model=="satmae":
                        inter = self.feature_extractor(x)
                        print('inter shape ',inter.shape)
                        y_hat = self.forward(inter)
        else:
            y_hat = self.forward(x)

        if self.target_type == "log" or self.target_type == "binary":
            pred = y_hat.type_as(y)
        else:
            pred = self.sigmoid_activation(y_hat).type_as(y)

        if self.opts.data.correction_factor.thresh == 'after':
            mask = correction_t

            cloned_pred = pred.clone().type_as(pred)
            # print(cloned_pred.shape,mask.shape)
            cloned_pred *= mask.int()
            y *= mask.int()
            pred = cloned_pred

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

        x = batch['sat'].squeeze(1)
        y = batch['target']

        hotspot_id = batch['hotspot_id']
        if self.opts.data.correction_factor.thresh:
            correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
                columns=["index"]).iloc[:, self.subset].values
            correction_t = torch.tensor(correction_t, device=y.device)

        print("Model is on cuda", next(self.model.parameters()).is_cuda)
        if self.opts.experiment.module.model == "linear":
            x = torch.flatten(x, start_dim=1)
        elif self.opts.experiment.module.model=="satlas" or self.opts.experiment.module.model=="satmae":
                        inter = self.feature_extractor(x)
                        print('inter shape ',inter.shape)
                        y_hat = self.forward(inter)
        else:
            y_hat = self.forward(x)


        if self.target_type == "log" or self.target_type == "binary":
            pred = y_hat.type_as(y)
        else:
            pred = self.sigmoid_activation(y_hat).type_as(y)

            if self.opts.data.correction_factor.thresh == 'after':
                # print('Adding (after) correction factor')
                mask = correction_t
                cloned_pred = pred.clone().type_as(pred)

                cloned_pred *= mask

                y *= mask
                pred = cloned_pred

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
        self.data_base_dir = self.opts.data.files.base
        self.df_train = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.train))
        self.df_val = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.val))
        self.df_test = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.test))
        self.bands = self.opts.data.bands
        self.env = self.opts.data.env
        self.datatype = self.opts.data.datatype
        self.target = self.opts.data.target.type
        self.subset = self.opts.data.target.subset
        self.res = self.opts.data.multiscale
        self.use_loc = self.opts.loc.use

    def prepare_data(self) -> None:
        """
        """
        print("prepare data")

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits and prepare the transforms for the multires"""
        self.all_train_dataset = EbirdVisionDataset(
            df_paths=self.df_train,
            data_base_dir=self.data_base_dir,
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
            df_paths=self.df_test,
            data_base_dir=self.data_base_dir,
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
            df_paths=self.df_val,
            data_base_dir=self.data_base_dir,
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