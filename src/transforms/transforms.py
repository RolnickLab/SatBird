from typing import Dict
from math import ceil, floor
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
import torchvision
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import collections
import numbers
import random

Module.__module__ = "torch.nn"

sat = ["sat"]
env = ["bioclim", "ped"]
landuse = ["landuse"]
all_data = sat + env + landuse


class RandomHorizontalFlip:  # type: ignore[misc,name-defined]
    """Horizontally flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.
        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.
        Args:
            sample: a single data sample
        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            for s in sample:
                if s in all_data:
                    sample[s] = sample[s].flip(-1)

                elif s == "boxes":
                    height, width = sample[s].shape[-2:]
                    sample["boxes"][:, [0, 2]] = width - sample["boxes"][:, [2, 0]]

        return sample


class RandomVerticalFlip:  # type: ignore[misc,name-defined]
    """Vertically flip the given sample randomly with a given probability."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize a new transform instance.
        Args:
            p: probability of the sample being flipped
        """
        super().__init__()
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Randomly flip the image and target tensors.
        Args:
            sample: a single data sample
        Returns:
            a possibly flipped sample
        """
        if torch.rand(1) < self.p:
            for s in sample:
                if s in all_data:
                    sample[s] = sample[s].flip(-2)

                elif s == "boxes":
                    height, width = sample[s].shape[-2:]
                    sample["boxes"][:, [1, 3]] = height - sample["boxes"][:, [3, 1]]

            # if "mask" in sample:
            #    sample["mask"] = sample["mask"].flip(-2)

        return sample


def normalize_custom(t, mini=0, maxi=1):
    if len(t.shape) == 3:
        return mini + (maxi - mini) * (t - t.min()) / (t.max() - t.min())

    batch_size = t.shape[0]
    min_t = t.reshape(batch_size, -1).min(1)[0].reshape(batch_size, 1, 1, 1)
    t = t - min_t
    max_t = t.reshape(batch_size, -1).max(1)[0].reshape(batch_size, 1, 1, 1)
    t = t / max_t
    return mini + (maxi - mini) * t


class Normalize:
    def __init__(self, maxchan=True, custom=None, subset=sat):
        """
        custom : ([means], [std])
        means =[r: 894.6719, g: 932.5726, b:693.2768, nir: 2817.9849]
        std = [r:883.9763, g:747.6857, b:749.3098, nir: 1342.6334]
        subset: set of inputs on which to apply the normalization (typically env variables and sat would require different normalizations)
        """
        self.maxchan = maxchan
        # TODO make this work with the values of the normalization values computed over the whole dataset
        self.subset = subset

        self.custom = custom

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:

        d = {}
        if self.maxchan:
            for task in self.subset:
                tensor = sample[task]
                sample[task] = tensor / torch.amax(tensor, dim=(-2, -1), keepdims=True)
        # TODO
        if self.custom:
            means, std = self.custom
            for task in self.subset:
                sample[task] = normalize(sample[task], means, std)
        return sample


class MatchRes:
    def __init__(self, target_size, custom):
        self.ped_res = 250
        self.bioclim_res = 1000
        self.sat_res = 10
        self.target_size = target_size
        self.custom = custom

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:

        H, W = self.target_size
        if "bioclim" in list(sample.keys()):
            # align bioclim with ped
            Hb, Wb = sample["bioclim"].shape[-2:]
            h = floor(Hb * self.sat_res / self.bioclim_res)
            w = floor(Wb * self.sat_res / self.bioclim_res)
            top = max(0, Hb // 2 - h // 2)
            left = max(0, Wb // 2 - w // 2)
            h, w = max(ceil(h), 1), max(ceil(w), 1)
            sample["bioclim"] = sample["bioclim"][:, int(top): int(top + h), int(left): int(left + w)]
        if "ped" in list(sample.keys()):
            # align bioclim with ped
            Hb, Wb = sample["ped"].shape[-2:]
            # print(Hb,Wb)
            h = floor(Hb * self.sat_res / self.ped_res)
            w = floor(Wb * self.sat_res / self.ped_res)
            top = max(0, Hb // 2 - h // 2)
            left = max(0, Wb // 2 - w // 2)
            h, w = max(ceil(h), 1), max(ceil(w), 1)
            sample["ped"] = sample["ped"][:, int(top): int(top + h), int(left): int(left + w)]

        means_bioclim, means_ped = self.custom

        for elem in list(sample.keys()):
            if elem in env:
                if ((sample[elem].shape[-1] == 0) or (sample[elem].shape[-2] == 0)):
                    if elem == "bioclim":
                        sample[elem] = torch.Tensor(means_bioclim).unsqueeze(-1).unsqueeze(-1)
                    elif elem == "ped":
                        sample[elem] = torch.Tensor(means_ped).unsqueeze(-1).unsqueeze(-1)

                sample[elem] = F.interpolate(sample[elem].unsqueeze(0).float(), size=(H, W))
        return sample


class RandomCrop:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def __init__(self, size, center=False, ignore_band=[], p=0.5):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)
        self.center = center
        self.ignore_band = ignore_band
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            the cropped input
        """

        H, W = (
            sample["sat"].shape[-2:] if "sat" in sample else list(sample.values())[0].shape[-2:]
        )
        for key in sample.keys():

            if (len(sample[key].shape) == 3):
                sample[key] = torch.unsqueeze(sample[key], 0)

        if torch.rand(1) > self.p:
            return (sample)
        else:
            if not self.center:

                top = max(0, np.random.randint(0, max(H - self.h, 1)))
                left = max(0, np.random.randint(0, max(W - self.w, 1)))
            else:
                top = max(0, (H - self.h) // 2)
                left = max(0, (W - self.w) // 2)

            item_ = {}
            for task, tensor in sample.items():
                if task in all_data and not task in self.ignore_band:
                    item_.update({task: tensor[:, :, top: top + self.h, left: left + self.w]})
                else:
                    item_.update({task: tensor})

            return item_


class Resize:
    def __init__(self, size):
        """
        size = (height, width) target size
        """
        self.h, self.w = size

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for s in sample:
            if s in sat:
                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode='bilinear')
            elif s in env or s in landuse:

                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode='nearest')
        return (sample)


class RandomGaussianNoise:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""

    def __init__(self, prob=0.5, max_noise=5e-2, std=1e-2):

        self.max = max_noise
        self.std = std
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            theinput with added gaussian noise
        """
        if random.random() < self.prob:
            for s in sample:
                if s in sat:

                    noise = torch.normal(0, self.std, sample[s].shape)
                    noise = torch.clamp(sample[s], min=0, max=self.max)
                    sample[s] += noise
        return sample

class RandBrightness:
    def __init__(self, prob=0.5, max_value=0):
        self.value = max_value
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s][:,0:3, :, :] = torchvision.transforms.functional.adjust_brightness(sample[s][:,0:3, :, :], self.value)
        return sample


class RandContrast:
    def __init__(self, prob=0.5, max_factor=0):
        self.factor = max_factor
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s][:,0:3, :, :] = torchvision.transforms.functional.adjust_contrast(sample[s][:,0:3, :, :], self.factor)
        return sample


class RandRotation:
    """random rotate the ndarray image with the degrees.

    Args:
        degrees (number or sequence): the rotate degree.
                                  If single number, it must be positive.
                                  if squeence, it's length must 2 and first number should small than the second one.

    Raises:
        ValueError: If degrees is a single number, it must be positive.
        ValueError: If degrees is a sequence, it must be of len 2.

    Returns:
        ndarray: return rotated ndarray image.
    """

    def __init__(self, degrees, prob=0.5, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            for s in sample:
                if s in sat:
                    sample[s] = torchvision.transforms.functional.rotate(sample[s], angle=angle, center=self.center)
        return sample


class GaussianBlurring:
    """Convert the input ndarray image to blurred image by gaussian method.

    Args:
        kernel_size (int): kernel size of gaussian blur method. (default: 3)

    Returns:
        ndarray: the blurred image.
    """

    def __init__(self, prob=0.5, kernel_size=3):
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if random.random() < self.prob:
            for s in sample:
                if s in sat:
                    sample[s] = torchvision.transforms.functional.gaussian_blur(sample[s], kernel_size=self.kernel_size)
        return sample


def get_transform(transform_item, mode):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """

    if transform_item.name == "crop" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomCrop(
            (transform_item.height, transform_item.width),
            center=(transform_item.center == mode or transform_item.center == True),
            ignore_band=transform_item.ignore_band or [], p=transform_item.p
        )
    elif transform_item.name == "matchres" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return MatchRes(transform_item.target_size, transform_item.custom_means)

    elif transform_item.name == "hflip" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomHorizontalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "vflip" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomVerticalFlip(p=transform_item.p or 0.5)

    elif transform_item.name == "randomnoise" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandomGaussianNoise(max_noise=transform_item.max_noise or 5e-2, std=transform_item.std or 1e-2)

    elif transform_item.name == "normalize" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):

        return Normalize(maxchan=transform_item.maxchan, custom=transform_item.custom or None,
                         subset=transform_item.subset)

    elif transform_item.name == "resize" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):

        return Resize(size=transform_item.size)

    elif transform_item.name == "blur" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return GaussianBlurring(prob=transform_item.p)
    elif transform_item.name == "rotate" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandRotation(prob=transform_item.p, degrees=transform_item.val)

    elif transform_item.name == "randomcontrast" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandContrast(prob=transform_item.p, max_factor=transform_item.val)

    elif transform_item.name == "randombrightness" and not (
            transform_item.ignore is True or transform_item.ignore == mode
    ):
        return RandBrightness(prob=transform_item.p, max_value=transform_item.val)

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts, mode):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    crop_transforms = []
    transforms = []

    for t in opts.data.transforms:
        if t.name == "normalize":
            if t.subset == ["sat"] and opts.data.datatype == "refl":
                t.custom = opts.variables.rgbnir_means, opts.variables.rgbnir_std
            if t.subset == ["sat"] and opts.data.datatype == "img":
                t.custom = opts.variables.visual_means, opts.variables.visual_stds
            elif t.subset == ["bioclim"]:
                t.custom = opts.variables.bioclim_means, opts.variables.bioclim_std
            elif t.subset == ["ped"]:
                t.custom = opts.variables.ped_means, opts.variables.ped_std
        if t.name == "matchres":
            t.custom_means = opts.variables.bioclim_means, opts.variables.ped_means

        # account for multires
        if t.name == 'crop' and len(opts.data.multiscale) > 1:
            for res in opts.data.multiscale:
                # adapt hight and width to vars in multires
                t.hight, t.width = res, res
                crop_transforms.append(get_transform(t, mode))
        else:
            transforms.append(get_transform(t, mode))
    transforms = [t for t in transforms if t is not None]
    if crop_transforms:
        crop_transforms = [t for t in crop_transforms if t is not None]
        print('crop transforms ', crop_transforms)
        return crop_transforms, transforms
    else:
        return transforms