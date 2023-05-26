# geo-trainer (uses location information into a seperate encoder)
import copy
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision import models

from src.dataset.dataloader import EbirdVisionDataset
from src.dataset.dataloader import get_subset
from src.losses.losses import CustomCrossEntropyLoss
from src.losses.metrics import get_metrics
from src.models.geomodels import Identity, LocEncoder
from src.trainer.utils import get_target_size, get_nb_bands, get_scheduler
from src.transforms.transforms import get_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EbirdTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()

        self.save_hyperparameters(opts)
        print(self.hparams.keys())
        # self.automatic_optimization = False
        self.opts = opts

        self.concat = self.opts.loc.concat
        self.config_task(opts, **kwargs)
        self.learning_rate = self.opts.experiment.module.lr

        self.m = nn.Sigmoid()

    def config_task(self, opts, **kwargs: Any) -> None:
        self.opts = opts
        self.target_size = get_target_size(self.opts)
        self.target_type = self.opts.data.target.type
        self.subset = get_subset(self.opts.data.target.subset)

        if self.target_type == "binary":
            # self.target_type = "binary"
            self.criterion = BCELoss()
            print("Training with BCE Loss")
        else:
            self.criterion = CustomCrossEntropyLoss()
            print("Training with Custom CE Loss")

        self.encoders = {}
        self.encoders["loc"] = self.get_loc_model()
        self.encoders["sat"] = self.get_sat_model()
        metrics = get_metrics(self.opts)
        for (name, value, _) in metrics:
            setattr(self, name, value)
        self.metrics = metrics

        # range maps
        with open(self.opts.data.files.correction_thresh, 'rb') as f:
            self.correction_data = pickle.load(f)

    def get_loc_model(self):
        self.loc_model = LocEncoder(self.opts)
        return self.loc_model

    def get_sat_model(self):
        """
        Satellite model if we multiply output with location model output
        """
        if self.opts.experiment.module.model == "resnet18":
            self.sat_model = models.resnet18(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.sat_model.conv1.in_channels
                weights = self.sat_model.conv1.weight.data.clone()
                self.sat_model.conv1 = nn.Conv2d(get_nb_bands(bands), 64, kernel_size=(7, 7), stride=(2, 2),
                                                 padding=(3, 3), bias=False)
                # assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    self.sat_model.conv1.weight.data[:, :orig_channels, :, :] = weights
            if self.concat:
                self.sat_model.fc = Identity()
                if self.opts.experiment.module.fc == "linear":
                    self.linear_layer = nn.Linear(256 + 512, self.target_size)
                if self.opts.experiment.module.fc == "linear_net":
                    self.linear_layer = nn.Sequential(nn.Linear(256 + 512, 512), nn.ReLU(),
                                                      nn.Linear(512, self.target_size))
            else:
                self.sat_model.fc = nn.Linear(512, self.target_size)

        elif self.opts.experiment.module.model == "resnet50":
            self.sat_model = models.resnet50(pretrained=self.opts.experiment.module.pretrained)
            if len(self.opts.data.bands) != 3 or len(self.opts.data.env) > 0:
                bands = self.opts.data.bands + self.opts.data.env
                orig_channels = self.sat_model.conv1.in_channels
                weights = self.sat_model.conv1.weight.data.clone()
                self.sat_model.conv1 = nn.Conv2d(get_nb_bands(bands), 64, kernel_size=(7, 7), stride=(2, 2),
                                                 padding=(3, 3), bias=False)
                # assume first three channels are rgb
                if self.opts.experiment.module.pretrained:
                    self.sat_model.conv1.weight.data[:, :orig_channels, :, :] = weights

            if self.concat:
                self.sat_model.fc = Identity()
                self.linear_layer = nn.Linear(256 + 2048, self.target_size)
            else:
                self.sat_model.fc = nn.Linear(2048, self.target_size)

        elif self.opts.experiment.module.model == "inceptionv3":
            self.sat_model = models.inception_v3(pretrained=self.opts.experiment.module.pretrained)
            self.sat_model.AuxLogits.fc = nn.Linear(768, self.target_size)
            if self.concat:
                self.sat_model.fc = Identity()
            else:
                self.sat_model.fc = nn.Linear(2048, self.target_size)
        else:
            raise ValueError(f"Model type '{self.opts.experiment.module.model}' is not valid")

        if self.opts.experiment.module.init_bias == "means":
            print("initializing biases with mean predictor")
            self.means = np.load(self.opts.experiment.module.means_path)[0, self.subset]
            means = torch.Tensor(self.means)

            means = torch.logit(means, eps=1e-10)
            self.sat_model.fc.bias.data = means
        else:
            print("no initialization of biases")
        # self.sat_model.to(device)
        return self.sat_model

    def forward(self, x: Tensor, loc_tensor=None) -> Any:
        # need to fix use of inceptionv3 to be able to use location too 

        if self.opts.experiment.module.model == "inceptionv3":
            out_sat, aux_outputs = self.encoders["sat"](x)
            out_loc = self.encoders["loc"](loc_tensor).squeeze(1)
            return out_sat, aux_outputs, out_loc
        else:
            if not self.concat:
                out_sat = self.encoders["sat"](x)
                out_loc = self.encoders["loc"](loc_tensor).squeeze(1)
                return out_sat, out_loc
            else:
                out_sat = self.encoders["sat"](x)
                out_loc = self.encoders["loc"](loc_tensor).squeeze(1)
                concat = torch.cat((out_sat, out_loc), 1)
                out = self.linear_layer(concat)
                return out

    def training_step(
            self, batch: Dict[str, Any], batch_idx: int) -> Tensor:

        """Training step"""

        x = batch['sat'].squeeze(1)
        print('input shape', x.shape)
        loc_tensor = batch["loc"]
        y = batch['target']
        b, no_species = y.shape
        hotspot_id = batch['hotspot_id']

        correction = (self.correction_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
            columns=["index"]).iloc[:, self.subset].values
        correction = torch.tensor(correction, device=y.device)

        assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'

        if self.opts.experiment.module.model == "inceptionv3":
            out_sat, aux_outputs, out_loc = self.forward(x, loc_tensor)
            y_hat = self.m(out_sat)
            aux_y_hat = self.m(aux_outputs)
            pred = torch.multiply(y_hat, out_loc)
            aux_pred = torch.multiply(aux_y_hat, out_loc)
            y *= correction
            loss1 = self.criterion(y, pred)
            loss2 = self.criterion(y, aux_pred)
            loss = loss1 + loss2

        else:
            if self.concat:
                pred = self.m(self.forward(x, loc_tensor))
            else:
                out_sat, out_loc = self.forward(x, loc_tensor)
                pred = torch.multiply(self.m(out_sat), out_loc)  # self.forward(x)
            # range maps
            if self.opts.data.correction_factor.thresh:
                mask = correction
                cloned_pred = pred.clone().type_as(pred)
                print('predictons before: ', cloned_pred)
                cloned_pred *= mask.int()
                y *= mask.int()
                pred = cloned_pred
            else:
                y *= correction

            loss = self.criterion(y, pred)

        pred_ = pred.clone()

        if self.opts.data.target.type == "binary":
            pred_[pred_ > 0.5] = 1
            pred_[pred_ < 0.5] = 0

        for (name, _, scale) in self.metrics:
            nname = "train_" + name
            if name == "accuracy":
                getattr(self, name)(pred_, y.type(torch.uint8))
                # print(nname,getattr(self,name)(pred_,  y.type(torch.uint8)))

            else:

                getattr(self, name)(y, pred_)
                # print(nname,getattr(self,name)(y, pred_) )

            self.log(nname, getattr(self, name), on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int) -> None:

        """Validation step """

        x = batch['sat'].squeeze(1)
        loc_tensor = batch["loc"]
        y = batch['target']

        b, no_species = y.shape
        hotspot_id = batch['hotspot_id']

        correction = (self.correction_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
            columns=["index"]).iloc[:, self.subset].values
        correction = torch.tensor(correction, device=y.device)
        assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'

        # check weights are moving
        if self.opts.experiment.module.model == "inceptionv3":
            out_sat, aux_outputs, out_loc = self.forward(x, loc_tensor)
            y_hat = self.m(out_sat)
            aux_y_hat = self.m(aux_outputs)
            pred = torch.multiply(y_hat, out_loc)
            aux_pred = torch.multiply(aux_y_hat, out_loc)
            y *= correction
            loss1 = self.criterion(y, pred)
            loss2 = self.criterion(y, aux_pred)
            loss = loss1 + loss2

        else:
            if self.concat:
                pred = self.m(self.forward(x, loc_tensor))
            else:
                out_sat, out_loc = self.forward(x, loc_tensor)
                pred = torch.multiply(self.m(out_sat), out_loc)  # self.forward(x)
            # range maps
            if self.opts.data.correction_factor.thresh:
                mask = correction
                cloned_pred = pred.clone().type_as(pred)
                print('predictons before: ', cloned_pred)
                cloned_pred *= mask.int()
                y *= mask.int()
                pred = cloned_pred
            else:
                y *= correction
            loss = self.criterion(pred, y)
            print("val_loss", loss)
        pred_ = pred.clone()
        if self.opts.data.target.type == "binary":
            pred_[pred_ > 0.5] = 1
            pred_[pred_ < 0.5] = 0

        for (name, _, scale) in self.metrics:
            nname = "val_" + name
            if name == "accuracy":
                getattr(self, name)(pred_, y.type(torch.uint8))
            else:
                getattr(self, name)(y, pred_)

            self.log(nname, getattr(self, name), on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step """

        x = batch['sat'].squeeze(1)
        loc_tensor = batch["loc"]

        y = batch['target']
        b, no_species = y.shape
        hotspot_id = batch['hotspot_id']

        correction = (self.correction_data.reset_index().set_index('hotspot_id').loc[
            list(hotspot_id)]).drop(columns=["index"]).iloc[:, self.subset].values

        correction = torch.tensor(correction, device=y.device)
        assert correction.shape == (b, no_species), 'shape of correction factor is not as expected'

        if self.opts.experiment.module.model == "inceptionv3":
            out_sat, aux_outputs, out_loc = self.forward(x, loc_tensor)
            y_hat = self.m(out_sat)
            aux_y_hat = self.m(aux_outputs)
            pred = torch.multiply(y_hat, out_loc)
        else:
            if self.concat:
                y_hat = self.m(self.forward(x, loc_tensor))
            else:
                out_sat, out_loc = self.forward(x, loc_tensor)
                y_hat = torch.multiply(self.m(out_sat), out_loc)  # self.forward(x)

            pred = y_hat
            # range maps
            if self.opts.data.correction_factor.thresh == "after":
                mask = correction
                cloned_pred = pred.clone().type_as(pred)
                # just for debugging you can remove that later
                print('In test masking')
                cloned_before = copy.deepcopy(cloned_pred)
                print('does mask have zero values: ', (mask == 0).any())
                cloned_pred *= mask.int()
                y *= mask.int()
                print(mask, hotspot_id)
                pred = cloned_pred
            else:
                y = y * correction
        pred_ = pred.clone().cpu()
        if "target" in batch.keys():
            y = batch['target'].cpu()
            for (name, _, scale) in self.metrics:
                nname = "test_" + name
                getattr(self, name)(y, pred_)
                self.log(nname, getattr(self, name), on_step=True, on_epoch=True)

        for i, elem in enumerate(pred_):
            np.save(os.path.join(self.opts.save_preds_path, batch["hotspot_id"][i] + ".npy"),
                    elem.cpu().detach().numpy())
        print("saved elems")

    def get_optimizer(self, model, opts):
        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(  #
                model.parameters(),
                lr=self.learning_rate  # self.opts.experiment.module.lr,
            )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.opts.experiment.module.lr,
            )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate  # self.opts.experiment.module.lr,
            )
        else:
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return optimizer

    def get_optimizer_from_params(self, param, opts):

        if self.opts.optimizer == "Adam":
            optimizer = torch.optim.Adam(  #
                param,
                lr=self.learning_rate  # self.opts.experiment.module.lr,
            )
        elif self.opts.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                param,
                lr=self.opts.experiment.module.lr,
            )
        elif self.opts.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                param,
                lr=self.learning_rate  # self.opts.experiment.module.lr,
            )
        else:
            raise ValueError(f"Optimizer'{self.opts.optimizer}' is not valid")
        return optimizer

    def configure_optimizers(self):
        if not self.concat:
            self.optims = []
            self.scheds = []
            sat_opt = self.get_optimizer(self.encoders["sat"], self.opts)
            loc_opt = self.get_optimizer(self.encoders["loc"].model, self.opts)
            self.optims.append(sat_opt)
            self.optims.append(loc_opt)

            lr_scheduler = get_scheduler(self.optims[0], self.opts)
            self.scheds.append(lr_scheduler)

            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"
                , "frequency": 1, "monitor": "val_loss"}

            return self.optims, [lr_scheduler_config]

        else:

            parameters = (
                    list(self.encoders["sat"].parameters())
                    + list(self.encoders["loc"].parameters())
                    + list(self.linear_layer.parameters())
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
        self.loc_type = self.opts.loc.loc_type

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits"""

        self.all_train_dataset = EbirdVisionDataset(
            df_paths=self.df_train,
            bands=self.bands,
            env=self.env,
            transforms=get_transforms(self.opts, "train"),
            mode="train",
            datatype=self.datatype,
            target=self.target,
            subset=self.subset,
            use_loc=self.use_loc,
            loc_type=self.loc_type
        )

        self.all_test_dataset = EbirdVisionDataset(
            self.df_test,
            bands=self.bands,
            env=self.env,
            transforms=get_transforms(self.opts, "test"),
            mode="test",
            datatype=self.datatype,
            target=self.target,
            subset=self.subset,
            use_loc=self.use_loc,
            loc_type=self.loc_type
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
            use_loc=self.use_loc,
            loc_type=self.loc_type
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
