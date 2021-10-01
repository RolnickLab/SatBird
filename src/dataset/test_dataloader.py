import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.dataset.dataloader import EbirdVisionDataset
from src.transforms.transforms import RandomCrop, RandomHorizontalFlip, Normalize
from torchvision import transforms as trsfs


r = "/miniscratch/akera/ecosystem-embedding/data/sentinel2/L595310_r.npy"
g = "/miniscratch/akera/ecosystem-embedding/data/sentinel2/L595310_g.npy"
b = "/miniscratch/akera/ecosystem-embedding/data/sentinel2/L595310_b.npy"
m = "/miniscratch/akera/ecosystem-embedding/data/sentinel2/L595310.json"
s = "/miniscratch/tengmeli/ecosystem-embedding/ebird_data_june/L595310.json"

r1="/miniscratch/akera/ecosystem-embedding/data/sentinel2/L999786_r.npy"
g1="/miniscratch/akera/ecosystem-embedding/data/sentinel2/L999786_g.npy"
b1="/miniscratch/akera/ecosystem-embedding/data/sentinel2/L999786_b.npy"
m1="/miniscratch/akera/ecosystem-embedding/data/sentinel2/L999786.json"
s1 = "/miniscratch/tengmeli/ecosystem-embedding/ebird_data_june/L595310.json"

df_paths = {'r': [r, r1], 'g': [g,g1], 'b':[b, b1], "metadata":[m,m1], "species":[s,s1]}

# print(df_paths)
df = pd.DataFrame(df_paths)

bands = ["r","g", "b"]

dataset = EbirdVisionDataset(df,bands, trsfs.Compose([RandomCrop((256,256)), RandomHorizontalFlip()]))
dataset.__getitem__(0)
import pdb;pdb.set_trace()