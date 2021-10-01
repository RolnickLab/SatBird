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

