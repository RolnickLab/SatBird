"""
Trainer for the Rtran framework
"""
import os
import pickle
import pandas as pd
import numpy as np

from torch import optim, nn
import torch
import pytorch_lightning as pl
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader

from Rtran.dataloader import *
from src.transforms.transforms import get_transforms
from src.losses.metrics import get_metrics
from src.losses.losses import RMSLELoss, CustomFocalLoss, CustomCrossEntropyLoss
from Rtran.rtran import RTranModel
from Rtran.utils import *


class RegressionTransformerTask(pl.LightningModule):
    def __init__(self, opts, **kwargs: Any) -> None:
        """
        opts: configurations
        """
        super().__init__()
        self.config = opts

        self.num_species = self.config.data.total_species

        # model and optimizer utils
        self.learning_rate = self.config.experiment.module.lr
        self.criterion = self.__loss_mapping(self.config.losses.criterion)
        self.image_input_channels = len(self.config.data.bands)
        self.image_input_channels += sum(self.config.data.env_var_sizes) if len(self.config.data.env) > 0 else 0

        self.model = RTranModel(num_classes=self.num_species,
                                backbone=self.config.Rtran.backbone, pretrained_backbone=self.config.Rtran.pretrained_backbone,
                                input_channels=self.image_input_channels, d_hidden=self.config.Rtran.features_size,
                                scale_embeddings_by_labels=self.config.Rtran.scale_embeddings_by_labels)

        self.sigmoid_activation = nn.Sigmoid()

        # if using range maps (RM)
        if self.config.data.correction_factor.thresh:
            with open(os.path.join(self.config.data.files.base, self.config.data.files.correction_thresh), 'rb') as f:
                self.RM_correction_data = pickle.load(f)

        # metrics to report
        metrics = get_metrics(self.config)
        for (name, value, _) in metrics:
            setattr(self, "val_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "train_" + name, value)
        for (name, value, _) in metrics:
            setattr(self, "test_" + name, value)
        self.metrics = metrics

    def training_step(
            self, batch: Dict[str, Any], batch_idx: int):

        hotspot_id = batch["hotspot_id"]

        x = batch["sat"]
        y = batch["target"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, mask.clone()))

        # if using range maps
        if self.config.data.correction_factor.thresh:
            range_maps_correction_data = (self.RM_correction_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
                                        columns=["index"]).iloc[:, self.subset].values
            range_maps_correction_data = torch.tensor(range_maps_correction_data, device=y.device)
            y_pred *= range_maps_correction_data.int()
            y *= range_maps_correction_data.int()

        if self.config.Rtran.masked_loss:         # to train on unknown labels only
            unknown_mask = custom_replace(mask, 1, 0, 0)
            loss = self.criterion(y_pred[unknown_mask.bool()], y[unknown_mask.bool()], reduction='None')
            loss = loss.sum() / unknown_mask.sum().item()
        elif len(mask.unique()) > 3:
            unknown_mask = maksed_loss_custom_replace(mask, 0, 1, 1, 1)
            loss = self.criterion(y_pred[unknown_mask.bool()], y[unknown_mask.bool()], reduction='None')
            loss = loss.sum() / unknown_mask.sum().item()
        else:
            loss = self.criterion(y_pred, y)

        if batch_idx % 50 == 0:
            self.log("train_loss", loss, on_epoch=True)
            if len(self.config.data.species) > 1:
                self.log_metrics(mode="train", pred=y_pred, y=y, mask=mask)
            else:
                self.log_metrics(mode="train", pred=y_pred, y=y)

        return loss

    def validation_step(
            self, batch: Dict[str, Any], batch_idx: int) -> None:

        hotspot_id = batch["hotspot_id"]

        x = batch["sat"]
        y = batch["target"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, mask.clone()))
        # if using range maps
        if self.config.data.correction_factor.thresh:
            range_maps_correction_data = (self.RM_correction_data.reset_index().set_index('hotspot_id').loc[list(hotspot_id)]).drop(
                                        columns=["index"]).iloc[:, self.subset].values
            range_maps_correction_data = torch.tensor(range_maps_correction_data, device=y.device)
            y_pred *= range_maps_correction_data.int()
            y *= range_maps_correction_data.int()

        # print(mask.unique(), y.unique())
        if self.config.Rtran.mask_eval_metrics or len(self.config.data.species) > 1:
            self.log_metrics(mode="val", pred=y_pred, y=y, mask=mask)
        else:
            self.log_metrics(mode="val", pred=y_pred, y=y)

        # print(torch.topk(y_pred[0], k=30))
        # print(torch.topk(y[0], k=30))
        # exit(0)

    def test_step(
            self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step """
        hotspot_id = batch["hotspot_id"]

        x = batch["sat"]
        y = batch["target"]
        mask = batch["mask"].long()

        y_pred = self.sigmoid_activation(self.model(x, mask.clone()))

        # if using range maps
        if self.config.data.correction_factor.thresh:
            range_maps_correction_data = (self.RM_correction_data.reset_index().set_index('hotspot_id').loc[
                list(hotspot_id)]).drop(
                columns=["index"]).iloc[:, self.subset].values
            range_maps_correction_data = torch.tensor(range_maps_correction_data, device=y.device)
            y_pred *= range_maps_correction_data.int()
            y *= range_maps_correction_data.int()

        if self.config.Rtran.mask_eval_metrics or len(self.config.data.species) > 1:
            self.log_metrics(mode="test", pred=y_pred, y=y, mask=mask)
        else:
            self.log_metrics(mode="test", pred=y_pred, y=y)

        # saving model predictions
        if self.config.save_preds_path != "":
            for i, elem in enumerate(y_pred):
                np.save(os.path.join(self.config.base_dir, self.config.save_preds_path, batch["hotspot_id"][i] + ".npy"),
                        elem.cpu().detach().numpy())

    def __loss_mapping(self, loss_fn_name):
        loss_mapping = {
            "MSE": nn.MSELoss(),
            "MAE": nn.L1Loss(),
            "RMSLE": RMSLELoss(),
            "Focal": CustomFocalLoss(),
            "CE": CustomCrossEntropyLoss()
        }
        return loss_mapping.get(loss_fn_name)

    def configure_optimizers(self):
        optimizer_mapping = {
            "Adam": optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, weight_decay=0.01),
            "AdamW": optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, weight_decay=0.01),
            "SGD": optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate, momentum=0.9)
        }
        optimizer = optimizer_mapping.get(self.config.optimizer)

        if self.config.scheduler.name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.config.scheduler.reduce_lr_plateau.factor,
                              patience=self.config.scheduler.reduce_lr_plateau.lr_schedule_patience)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1
                }
            }
        else:
            return optimizer

    def __log_metric(self, mode, pred, y):
        for (name, _, scale) in self.metrics:
            nname = str(mode) + "_" + name
            if name == "accuracy":
                value = getattr(self, nname)(pred, y.type(torch.uint8))
            elif name == 'r2':
                value = torch.mean(getattr(self, nname)(y, pred))
            else:
                value = getattr(self, nname)(y, pred)

            self.log(nname, value, on_epoch=True)

    def log_metrics(self, mode, pred, y, mask=None):
        """
        log metrics through logger
        """
        if mask is not None:
            unknown_mask = mask
            if self.config.Rtran.mask_eval_metrics:
                if len(mask.unique()) > 3:
                    unknown_mask = maksed_loss_custom_replace(mask, 0, 1, 0, 0)
                else:
                    unknown_mask = custom_replace(mask, 1, 0, 0)

            unknown_mask[unknown_mask == -2] = 0
            unknown_mask[unknown_mask == -1] = 1

            loss = self.criterion(pred[unknown_mask.bool()], y[unknown_mask.bool()], reduction='None')
            loss = loss.sum() / unknown_mask.sum().item()

            pred = pred * unknown_mask
            y = y * unknown_mask

        else:
            loss = self.criterion(pred, y)

        self.__log_metric(mode, pred, y)
        self.log(str(mode) + "_loss", loss, on_epoch=True)


class SDMDataModule(pl.LightningDataModule):
    """
    SDM - Species Distribution Modeling: works for ebird or ebutterfly
    """
    def __init__(self, opts) -> None:
        super().__init__()
        self.config = opts

        self.seed = self.config.program.seed
        self.batch_size = self.config.data.loaders.batch_size
        self.num_workers = self.config.data.loaders.num_workers
        self.data_base_dir = self.config.data.files.base
        self.targets_folder = self.config.data.files.targets_folder
        self.env_data_folder = self.config.data.files.env_data_folder

        self.df_train = pd.read_csv(os.path.join(self.data_base_dir, self.config.data.files.train))
        self.df_val = pd.read_csv(os.path.join(self.data_base_dir, self.config.data.files.val))
        self.df_test = pd.read_csv(os.path.join(self.data_base_dir, self.config.data.files.test))

        self.bands = self.config.data.bands
        self.env = self.config.data.env
        self.env_var_sizes = self.config.data.env_var_sizes
        self.datatype = self.config.data.datatype

        self.subset = self.config.data.target.subset
        # self.predict_family = self.config.Rtran.predict_family
        self.num_species = self.config.data.total_species

        if len(self.config.data.species) > 1:
            self.dataloader_to_use = "SDMVJointDataset"
        else:
            self.dataloader_to_use = "SDMVisionMaskedDataset"

    def setup(self, stage: Optional[str] = None) -> None:
        """create the train/test/val splits and prepare the transforms for the multires"""
        self.all_train_dataset = globals()[self.dataloader_to_use](
            df=self.df_train,
            data_base_dir=self.data_base_dir,
            env=self.env,
            env_var_sizes=self.env_var_sizes,
            transforms=get_transforms(self.config, "train"),
            mode="train",
            datatype=self.datatype,
            targets_folder=self.targets_folder,
            env_data_folder=self.env_data_folder,
            maximum_unknown_labels_ratio=self.config.Rtran.train_unknown_ratio,
            subset=self.subset,
            num_species=self.num_species)

        self.all_val_dataset = globals()[self.dataloader_to_use](
            df=self.df_val,
            data_base_dir=self.data_base_dir,
            env=self.env,
            env_var_sizes=self.env_var_sizes,
            transforms=get_transforms(self.config, "val"),
            mode="val",
            datatype=self.datatype,
            targets_folder=self.targets_folder,
            env_data_folder=self.env_data_folder,
            maximum_unknown_labels_ratio=self.config.Rtran.eval_unknown_ratio,
            subset=self.subset,
            num_species=self.num_species)

        self.all_test_dataset = globals()[self.dataloader_to_use](
            df=self.df_test,
            data_base_dir=self.data_base_dir,
            env=self.env,
            env_var_sizes=self.env_var_sizes,
            transforms=get_transforms(self.config, "val"),
            mode="test",
            datatype=self.datatype,
            targets_folder=self.targets_folder,
            env_data_folder=self.env_data_folder,
            maximum_unknown_labels_ratio=self.config.Rtran.eval_unknown_ratio,
            subset=self.subset,
            num_species=self.num_species)

    def train_dataloader(self) -> DataLoader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.all_train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(
            self.all_val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Returns the test dataloader"""
        return DataLoader(
            self.all_test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
