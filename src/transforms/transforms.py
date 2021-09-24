from typing import Dict
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]

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

class Normalize:
    def __init__(self, maxchan = True, custom = None):
        """
        custom : ([means], [std])
        """
        self.maxchan = maxchan
        self.custom = custom
    def __call__(self, sample: Dict[str, Tensor])-> Dict[str, Tensor]:
        
        d = {}
        if self.maxchan:
            d =  {
                task: tensor/ torch.amax(tensor, dim=(-2,-1), keepdims=True)
                for task, tensor in sample.items() if task in transformable
            }
        #TODO 
        #if self.custom:
            
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

        if not self.center:
            top = np.random.randint(0, H - self.h)
            left = np.random.randint(0, W - self.w)
        else:
            top = (H - self.h) // 2
            left = (W - self.w) // 2

        return {
            task: tensor[:, :, top : top + self.h, left : left + self.w]
            for task, tensor in sample.items() if task in transformable
        }

class RandomGaussianNoise:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""
    
    def __init__(self, max_noise):
       
        self.max = max_noise
        
    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            theinput with added gaussian noise
        """
        
        for s in sample:
            if s in transformable : 
                noise = torch.rand(*sample[s].size())
                sample[s] += noise
        return sample