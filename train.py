# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Tuple, Type, cast
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.trainer import EbirdTask, EbirdDataModule

print("training will go here :)")