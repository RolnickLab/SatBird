import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import models

from src.dataset.dataloader import EbirdVisionDataset

class EbirdTask(pl.LightningModule):
    def config_task(self, kwargs: Any) -> None:
        if kwargs["model"] == "resnet18":
            self.model = models.resnet18(pretrained=False, num_classes=1)
        else:
            raise ValueError(f"Model type '{kwargs['model']}' is not valid")

    def __init__(self, **kwargs: Any) -> None:
        """initializes a new Lightning Module to train"""

        super().__init__()
        self.save_hyperparameters()
        self.config_task(kwargs)

    def forward(self, x:Tensor) -> Any:
        return self.model(x)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    )-> Tensor:
    """Training step """
        
        x = batch["sat"]
        y = batch["target"]
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        
        return loss

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    )->None:
    """Validation step """
        x = batch['sat']
        y = batch['target']
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("Val Loss", loss)

    def test_step(
        self, batch: Dict[str, Any], batch_idx:int
    )-> None:
        """Test step """

        x = batch['sat']
        y = batch['target']
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("Test Loss", loss)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.hparams["learning_rate"],
        )
        return{
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateu(
                    optimizer,
                    patience = self.hparams["learning_rate_schedule_patience"]
            ),
            "monitor":"val_loss",
        }

class EbirdDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir=str,
        df = str,
        bands: list,
        seed: int, 
        batch_size=int=64,
        num_workers: int=4,
        **kwargs:Any,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = df
        self.bands = bands
    
    def custom_transform(self, sample: Dict[str, Any]) -> Dict [str, Any]:
        sample["sat"] = sample["sat"] / 255.0 ## TODO: Meli's transforms 
        sample["sat"] = (
            sample["sat"].unsqueeze(0).repeat(3,1,1)
        )#converting to 3 channel
        sample["target"] = torch.as_tensor(
            sample["target"]
        ).float()
        return sample

    def prepare_data(self) -> None:
        _ = EbirdVisionDataset(
            df, 
            split = "train",
            transforms = self.custom_transform
        )

    def setup(self, stage: Optional[str]=None)->None:
        """create the train/test/val splits"""
        self.all_train_dataset = EbirdVisionDataset(
            self.df, 
            split="train",
            transforms = self.custom_transform,

        )

        self.all_test_dataset = EbirdVisonDataset(
            self.df,
            split = "test",
            transforms = self.custom_transform,
        )

         self.all_val_dataset = EbirdVisonDataset(
            self.df,
            split = "val",
            transforms = self.custom_transform,
        )

        #TODO: Create subsets of the data
        self.train_dataset = self.all_train_dataset
        self.test_dataset = self.all_test_dataset
        self.val_dataset = self.all_val_dataset

    def train_dataloader(self) -> Dataloader[Any]:
        """Returns the actual dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )

    def val_dataloader(self) -> Dataloader[Any]:
        """Returns the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
        )

    def test_dataloader(self) -> Dataloader[Any]:
        """Returns the test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = False,
        )
