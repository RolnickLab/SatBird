"""Data transforms for the loaders
"""
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import rgba2rgb
from skimage.io import imread
from torchvision import transforms as trsfs
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
)

class CenterCrop:
    def __init__(self, size):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)

    def __call__(self, data):
        H, W = (
            data["x"].size()[-2:] if "x" in data else list(data.values())[0].size()[-2:]
        )


        top = (H - self.h) // 2
        left = (W - self.w) // 2

        return {
            task: tensor[:, :, top : top + self.h, left : left + self.w]
            for task, tensor in data.items()
        }
    
    
class RandomCrop:
    def __init__(self, size):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)


    def __call__(self, data):
        H, W = (
            data["x"].size()[-2:] if "x" in data else list(data.values())[0].size()[-2:]
        )

        top = np.random.randint(0, H - self.h)
        left = np.random.randint(0, W - self.w)

        return {
            task: tensor[:, :, top : top + self.h, left : left + self.w]
            for task, tensor in data.items()
        }

    
def get_transform(transform_item, mode):
    """Returns the torchivion transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomCrop(
            (transform_item.height, transform_item.width),
            center=transform_item.center == mode,
        )

    elif transform_item.name == "resize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return Resize(
            transform_item.new_size, transform_item.get("keep_aspect_ratio", False)
        )

    elif transform_item.name == "hflip" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "brightness" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandBrightness()

    elif transform_item.name == "saturation" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandSaturation()

    elif transform_item.name == "contrast" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandContrast()

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts, mode, domain):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    transforms = []
    color_jittering_transforms = ["brightness", "saturation", "contrast"]

    for t in opts.data.transforms:
        if t.name not in color_jittering_transforms:
            transforms.append(get_transform(t, mode))

    if "p" not in opts.tasks and mode == "train":
        for t in opts.data.transforms:
            if t.name in color_jittering_transforms:
                transforms.append(get_transform(t, mode))

    transforms += [Normalize(opts), BucketizeDepth(opts, domain)]
    transforms = [t for t in transforms if t is not None]

    return transforms