from typing import Dict
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]
from torchvision.transforms.functional import normalize
# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

transformable = ["sat", "r", "g", "b", "ni"]

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
                if s in transformable:
                    sample[s] = sample[s].flip(-1)

                elif s=="boxes":
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
                if s in transformable:
                    sample[s] = sample[s].flip(-2)

                elif s == "boxes" :
                    height, width = sample[s].shape[-2:]
                    sample["boxes"][:, [1, 3]] = height - sample["boxes"][:, [3, 1]]

            #if "mask" in sample:
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
    def __init__(self, maxchan = True, custom = None):
        """
        custom : ([means], [std])
        means =[r: 894.6719, g: 932.5726, b:693.2768, nir: 2817.9849]
        std = [r:883.9763, g:747.6857, b:749.3098, nir: 1342.6334]
        """
        self.maxchan = maxchan
        #TODO make this work with the values of the normalization values computed over the whole dataset 
        self.custom = custom
    def __call__(self, sample: Dict[str, Tensor])-> Dict[str, Tensor]:
        
        d = {}
        if self.maxchan:
            d =  {
                task: tensor/ torch.amax(tensor, dim=(-2,-1), keepdims=True)
                for task, tensor in sample.items() if task in transformable
            }
        #TODO 
        if self.custom:
            means, std = self.custom
            
            d = {
                task: normalize(tensor.type(torch.FloatTensor), means, std)
                 for task, tensor in sample.items() if task in transformable
            }
        #    pass
        
        return(d)


class RandomCrop:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""
    
    def __init__(self, size, center=False):
        assert isinstance(size, (int, tuple, list))
        if not isinstance(size, int):
            assert len(size) == 2
            self.h, self.w = size
        else:
            self.h = self.w = size

        self.h = int(self.h)
        self.w = int(self.w)
        self.center = center
        
    def  __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            the cropped input
        """
        
        H, W = (
            sample["sat"].size()[-2:] if "sat" in sample else list(sample.values())[0].size()[-2:]
        )
        if (len(sample["sat"].size())==3):
            sample["sat"] = sample["sat"].unsqueeze(0)
           
        if not self.center:
            
            
            top = max(0, np.random.randint(0, max(H - self.h,1)))
            left = max(0, np.random.randint(0, max(W - self.w,1)))
        else:
            top = max(0, (H - self.h) // 2)
            left = max(0,(W - self.w) // 2)

        return {
            task: tensor[:, :, top : top + self.h, left : left + self.w]
            for task, tensor in sample.items() if task in transformable
        }

class RandomGaussianNoise:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""
    
    def __init__(self, max_noise = 5e-2, std = 1e-2):
       
        self.max = max_noise
        self.std = std
        
    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            theinput with added gaussian noise
        """
        
        for s in sample:
            if s in transformable : 
                noise = torch.normal(0,self.std,sample[s].size())
                noise = torch.clamp(sample[s], min=0, max=self.max)
                sample[s] += noise
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
            center=(transform_item.center == mode or transform_item == True),
        )

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
        return RandomGaussianNoise(max_noise = transform_item.max_noise or 5e-2, std = transform_item.std or 1e-2)
    
    elif transform_item.name == "normalize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        
        return Normalize(maxchan=transform_item.maxchan, custom=transform_item.custom or None)

    elif transform_item.ignore is True or transform_item.ignore == mode:
        return None

    raise ValueError("Unknown transform_item {}".format(transform_item))


def get_transforms(opts, mode):
    """Get all the transform functions listed in opts.data.transforms
    using get_transform(transform_item, mode)
    """
    transforms = []

    for t in opts.data.transforms:
        transforms.append(get_transform(t, mode))
    

    transforms = [t for t in transforms if t is not None]

    return transforms