import pandas as pd
import os
import numpy as np 
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from PIL import Image, ImageStat

class csvDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None, batch_size, num_workers, raw_size, process_size):
        self.data = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.raw_size = raw_size
        self.processed_size = processed_size


    def load(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def build_transforms(self):
        transform_list = []
        transform_list.append(Resize((self.raw_size[0:2])))
        
        if self.is_training:
            transform_list.append(RandomCrop(self.processed_size[0:2]))
            transform_list.append(RandomHorizontalFlip())
            transform_list.append(ColorJitter())
            transform_list.append(RandomRotation(20))
        else:
            transform_list.append(CenterCrop(self.processed_size[0:2]))
            transform_list.append(ToTensor())
            mean, std = self.calc_mean_std()

            return Compose(transform_list)


    def calc_mean_std(self):
        cache_file = '.' + self.input_file + '_meanstd'+'.cache'
        if not os.path.exists(cache_file):
            print("Calculating mean and std")
            means = np.zeros((3))
            stds = np.zeros((3))
            sample_size = min(len(self.data), 1000)

            for i in range(sample_size):
                img_name = os.path.join(self.path_prefix, random.choice(self.data['image_name']))
                img = Image.open(img_name).convert('RGB')
                stat = ImageStat.Stat(img)
                means += np.array(stat.mean) /255.0
                stds += np.array(stat.stddev) /255.0

            means = means / sample_size
            stds = stds / sample_size

            np.savetxt(cache_file, [means, stds], delimiter=',')
        else:
            print("Load Mean and Std from " + cache_file)
            contents = np.loadtxt(cache_file, delimiter=',')
            means = contents[0,:]
            stds = contents[1,:]

        return means, stds

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.data['image_name'][idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            sample = self.transform(image)
        if not self.inference_only: 
            return sample, self.data['label'][idx]

        else:
            return sample, self.image_name[idx]
            