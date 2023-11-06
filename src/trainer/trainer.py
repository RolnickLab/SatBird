"""
general trainer: supports training Resnet18, Satlas, and SATMAE
"""
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
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset.dataloader import EbirdVisionDataset, get_subset
from src.losses.losses import CustomCrossEntropyLoss, WeightedCustomCrossEntropyLoss, RMSLELoss, CustomFocalLoss
from src.losses.metrics import get_metrics
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler, init_first_layer_weights, \
    load_from_checkpoint
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

        self.freeze_backbone = self.opts.experiment.module.freeze
        # get target vector size (number of species we consider)
        self.subset = get_subset(self.opts.data.target.subset, self.opts.data.total_species)
        self.target_size = get_target_size(opts, self.subset)
        print("Predicting ", self.target_size, "species")

        self.target_type = self.opts.data.target.type

        # define self.learning_rate to enable learning rate finder
        self.learning_rate = self.opts.experiment.module.lr
        self.sigmoid_activation = nn.Sigmoid()

        self.config_task(opts, **kwargs)

    def config_task(self, opts, **kwargs: Any) -> None:

        if self.opts.losses.criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif self.opts.losses.criterion == "MAE":
            self.criterion = nn.L1Loss()
        elif self.opts.losses.criterion == "RMSLE":
            self.criterion = RMSLELoss()
        elif self.opts.losses.criterion == "Focal":
            self.criterion = CustomFocalLoss()
        else:
            # target is num checklists reporting species i / total number of checklists at a hotspot
            if self.opts.experiment.module.use_weighted_loss:
                self.criterion = WeightedCustomCrossEntropyLoss()
                print("Training with Weighted CE Loss")
            else:
                self.criterion = CustomCrossEntropyLoss()
                print("Training with Custom CE Loss")

        self.model = self.get_sat_model()

    def get_sat_model(self):
        if self.opts.experiment.module.model == "satlas":
            print('using Satlas model')
            # first lets assume freezing the pretrained model
            self.feature_extractor = torchvision.models.swin_transformer.swin_v2_b()
            # self.opts.experiment.module.resume should be path to satlas model (we used satlas-model-v1-lowres.pth)
            full_state_dict = torch.load(self.opts.experiment.module.resume, map_location=torch.device('cpu'))
            swin_prefix = 'backbone.backbone.'
            swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}

            self.feature_extractor.load_state_dict(swin_state_dict)
            self.feature_extractor.to('cuda:0')
            print("initialized network, freezing weights")

            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(1000, self.target_size)

        elif self.opts.experiment.module.model == "satmae":
            print('using SatMAE model')
            satmae = ViTFinetune(img_size=224, patch_size=16, in_chans=3, num_classes=self.target_size, embed_dim=1024,
                                 depth=24, num_heads=16, mlp_ratio=4, drop_rate=0.1, )
            # self.opts.experiment.module.resume should be set to fMoW pretrained SatMAE model
            satmae = load_from_checkpoint(self.opts.experiment.module.resume, satmae)
            satmae.to('cuda')
            in_feat = satmae.fc.in_features
            satmae.fc = nn.Sequential()
            print("initialized network, freezing weights")
            self.feature_extractor = satmae
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.model = nn.Linear(in_feat, self.target_size)

        elif self.opts.experiment.module.model == "resnet18":

            self.model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)

            if self.opts.experiment.module.transfer_weights == "SECO":
                # this works for https://zenodo.org/record/4728033/files/seco_resnet18_1m.ckpt?download=1
                # Seco ResNet-18-1M model - from which the state dict corresponding only to the ResNet18 part encoder was extracted.
                # because of some package version compatibility issues, we saved the encoder weights of the Seco ResNet-18-1M model as pkl 
                # and that is what is supported in this code. 
                print("Initializing with SeCo weights")
                with open(self.opts.experiment.module.resume, "rb") as file:
                    enc = pickle.load(file)
                pretrained = list(enc.items())
                # match the weights
                model_dict = dict(self.model.state_dict())
                count = 0
                for key, value in model_dict.items():
                    if not key.startswith("fc"):
                        if not key.startswith("conv1") and not key.startswith("bn1"):
                            layer_name, weights = pretrained[count]
                            model_dict[key] = weights
                        count += 1

            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                self.bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.model.conv1.in_channels
                weights = self.model.conv1.weight.data.clone()
                self.model.conv1 = nn.Conv2d(get_nb_bands(self.bands), 64, kernel_size=(7, 7), stride=(2, 2),
                                             padding=(3, 3), bias=False, )
                # assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    # self.model.conv1.weight.data[:, :orig_channels, :, :] = weights
                    self.model.conv1.weight.data = init_first_layer_weights(get_nb_bands(self.bands), weights)

            if self.opts.experiment.module.transfer_weights == "USA":

                # this is used for transferring USA weights to Kenya
                print("Transferring USA weights")

                ckpt = torch.load(self.opts.experiment.module.resume)
                self.model.fc = nn.Sequential()
                loaded_dict = ckpt['state_dict']
                model_dict = self.model.state_dict()

                # load state dict keys
                for key_model, key_pretrained in zip(model_dict.keys(), loaded_dict.keys()):
                    # ignore first layer weights(use imagenet ones)
                    if key_model == 'conv1.weight':
                        continue
                    model_dict[key_model] = loaded_dict[key_pretrained]

                self.model.load_state_dict(model_dict)

                if self.freeze_backbone:
                    print("initialized network, freezing weights")
                    for param in self.model.parameters():
                        param.requires_grad = False

            self.model.fc = nn.Linear(512, self.target_size)


        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")

        if self.opts.experiment.module.init_bias == "means":
            print("initializing biases with mean predictor")
            self.means = np.load(self.opts.experiment.module.means_path)[0, self.subset]
            means = torch.Tensor(self.means)

            means = torch.logit(means, eps=1e-10)
            self.model.fc.bias.data = means

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

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        # from pdb import set_trace; set_trace()
        """Training step"""
        x = batch['sat'].squeeze(1)
        y = batch['target']

        hotspot_id = batch['hotspot_id']

        weighted_loss_operations = {"sqrt": torch.sqrt, "log": torch.log, "nchklists": lambda x: x}

        weight_type = self.opts.experiment.module.loss_weight
        new_weights = weighted_loss_operations[weight_type](batch["num_complete_checklists"])
        new_weights = torch.ones(y.shape, device=torch.device("cuda")) * new_weights.view(-1, 1)

        if self.opts.experiment.module.model == "satlas" or self.opts.experiment.module.model == "satmae":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)

            pred = self.sigmoid_activation(y_hat).type_as(y)

        else:
            y_hat = self.forward(x)

            pred = self.sigmoid_activation(y_hat).type_as(y)

        if self.opts.data.correction_factor.thresh == 'after':
            # this means we are training with Range Maps
            correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').drop(columns=["index"]).loc[
                list(hotspot_id)]).iloc[:, self.subset].values
            correction_t = torch.tensor(correction_t, device=y.device)
            mask = correction_t
            cloned_pred = pred.clone().type_as(pred)

            cloned_pred *= mask.int()
            y *= mask.int()

            pred = cloned_pred

        pred_ = pred.clone().type_as(y)

        if self.opts.experiment.module.use_weighted_loss:
            print("Using Weighted CrossEntropy Loss")
            loss = self.criterion(pred, y, new_weights)
        else:
            loss = self.criterion(pred, y)

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

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:

        """Validation step """
        x = batch['sat'].squeeze(1)  # .to(device)
        y = batch['target']

        hotspot_id = batch['hotspot_id']

        if self.opts.data.correction_factor.thresh:
            correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').drop(columns=["index"]).loc[
                list(hotspot_id)]).iloc[:, self.subset].values
            correction_t = torch.tensor(correction_t, device=y.device)
            self.correction = correction_t

        if self.opts.experiment.module.model == "satlas" or self.opts.experiment.module.model == "satmae":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)
        else:
            y_hat = self.forward(x)

        pred = self.sigmoid_activation(y_hat).type_as(y)

        if self.opts.data.correction_factor.thresh == 'after':
            mask = correction_t

            cloned_pred = pred.clone().type_as(pred)
            # print(cloned_pred.shape,mask.shape)
            cloned_pred *= mask.int()
            y *= mask.int()
            pred = cloned_pred

        pred_ = pred.clone().type_as(y)

        loss = self.criterion(pred, y)

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

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step """

        x = batch['sat'].squeeze(1)
        y = batch['target']

        hotspot_id = batch['hotspot_id']
        if self.opts.data.correction_factor.thresh:
            correction_t = (self.correction_t_data.reset_index().set_index('hotspot_id').drop(columns=["index"]).loc[
                list(hotspot_id)]).iloc[:, self.subset].values
            correction_t = torch.tensor(correction_t, device=y.device)

        # print("Model is on cuda", next(self.model.parameters()).is_cuda)

        if self.opts.experiment.module.model == "satlas" or self.opts.experiment.module.model == "satmae":
            inter = self.feature_extractor(x)
            y_hat = self.forward(inter)
        else:
            y_hat = self.forward(x)

        pred = self.sigmoid_activation(y_hat).type_as(y)

        if self.opts.data.correction_factor.thresh == 'after':
            mask = correction_t
            cloned_pred = pred.clone().type_as(pred)

            cloned_pred *= mask

            y *= mask
            pred = cloned_pred

        loss = self.criterion(pred, y)

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
            preds_path = os.path.join(self.opts.base_dir, self.opts.save_preds_path)
            if os.path.exists(preds_path):
                os.mkdir(preds_path)
            for i, elem in enumerate(pred):
                np.save(os.path.join(preds_path, batch["hotspot_id"][i] + ".npy"), elem.cpu().detach().numpy())
        print("saved elems")

    def get_optimizer(self, model, opts):
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(  #
                model.parameters(), lr=self.learning_rate,  # self.opts.experiment.module.lr,
                weight_decay=0.00001)
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate  # self.opts.experiment.module.lr,
                                          )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
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
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1}}


class EbirdDataModule(pl.LightningDataModule):
    def __init__(self, opts) -> None:
        super().__init__()
        self.opts = opts

        self.seed = self.opts.program.seed
        self.batch_size = self.opts.data.loaders.batch_size
        self.num_workers = self.opts.data.loaders.num_workers
        self.data_base_dir = self.opts.data.files.base
        self.targets_folder = self.opts.data.files.targets_folder
        self.env_data_folder = self.opts.data.files.env_data_folder
        self.images_folder = self.opts.data.files.images_folder
        self.df_train = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.train))
        self.df_val = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.val))
        self.df_test = pd.read_csv(os.path.join(self.data_base_dir, self.opts.data.files.test))
        self.bands = self.opts.data.bands
        self.env = self.opts.data.env
        self.env_var_sizes = self.opts.data.env_var_sizes
        self.datatype = self.opts.data.datatype
        self.target = self.opts.data.target.type
        self.subset = self.opts.data.target.subset
        self.res = self.opts.data.multiscale
        self.use_loc = self.opts.loc.use
        self.num_species = self.opts.data.total_species

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits and prepare the transforms for the multires"""
        self.all_train_dataset = EbirdVisionDataset(df_paths=self.df_train, data_base_dir=self.data_base_dir,
                                                    bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
                                                    transforms=get_transforms(self.opts, "train"), mode="train",
                                                    datatype=self.datatype, target=self.target,
                                                    targets_folder=self.targets_folder,
                                                    env_data_folder=self.env_data_folder,
                                                    images_folder=self.images_folder, subset=self.subset, res=self.res,
                                                    use_loc=self.use_loc, num_species=self.num_species)

        self.all_test_dataset = EbirdVisionDataset(df_paths=self.df_test, data_base_dir=self.data_base_dir,
                                                   bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
                                                   transforms=get_transforms(self.opts, "val"), mode="test",
                                                   datatype=self.datatype, target=self.target,
                                                   targets_folder=self.targets_folder,
                                                   env_data_folder=self.env_data_folder,
                                                   images_folder=self.images_folder, subset=self.subset, res=self.res,
                                                   use_loc=self.use_loc, num_species=self.num_species)

        self.all_val_dataset = EbirdVisionDataset(df_paths=self.df_val, data_base_dir=self.data_base_dir,
                                                  bands=self.bands, env=self.env, env_var_sizes=self.env_var_sizes,
                                                  transforms=get_transforms(self.opts, "val"), mode="val",
                                                  datatype=self.datatype, target=self.target,
                                                  targets_folder=self.targets_folder,
                                                  env_data_folder=self.env_data_folder,
                                                  images_folder=self.images_folder, subset=self.subset, res=self.res,
                                                  use_loc=self.use_loc, num_species=self.num_species)

        # TODO: Create subsets of the data

        self.train_dataset = self.all_train_dataset

        self.test_dataset = self.all_test_dataset

        self.val_dataset = self.all_val_dataset

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, )
