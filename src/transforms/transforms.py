from typing import Dict
from math import ceil
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module  # type: ignore[attr-defined]
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

satellite = ["sat", "r", "g", "b", "ni"]
sat = ["sat"]
env = ["bioclim", "ped"]
landuse = ["landuse"]
all_data = satellite + env + landuse


    
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
                if s in all_data:
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
    def __init__(self, maxchan = True, custom = None, subset = satellite):
        """
        custom : ([means], [std])
        means =[r: 894.6719, g: 932.5726, b:693.2768, nir: 2817.9849]
        std = [r:883.9763, g:747.6857, b:749.3098, nir: 1342.6334]
        subset: set of inputs on which to apply the normalization (typically env variables and sat would require different normalizations)
        """
        self.maxchan = maxchan
        #TODO make this work with the values of the normalization values computed over the whole dataset 
        self.subset = subset
        
        self.custom = custom
    def __call__(self, sample: Dict[str, Tensor])-> Dict[str, Tensor]:
        
        d = {}
        if self.maxchan:
            for task in self.subset: 
                tensor = sample[task]
                sample[task] = tensor/ torch.amax(tensor, dim=(-2,-1), keepdims=True)
            #d =  {
            #    task: tensor/ torch.amax(tensor, dim=(-2,-1), keepdims=True)
            #    for task, tensor in sample.items() if task in subset
            #}
        #TODO 
        if self.custom:
            means, std = self.custom
            for task in self.subset: 
                sample[task] = normalize(sample[task].type(torch.FloatTensor), means, std)
            #d = {
            #    task: normalize(tensor.type(torch.FloatTensor), means, std)
            #     for task, tensor in sample.items() if task in subset
            #}
        #    pass
        
        return(sample)

class MatchRes:
    def __init__(self, target_size):
        self.ped_res = 250
        self.bioclim_res = 1000
        self.sat_res = 10
        self.target_size = target_size
        
    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        
        H, W = self.target_size
               
        
        if "bioclim" in list(sample.keys()):
            #align bioclim with ped
            Hb, Wb = sample["bioclim"].size()[-2:]
            print(Hb,Wb)
            h = (Hb*self.sat_res/self.bioclim_res)
            w = (Wb*self.sat_res/self.bioclim_res)
            top = max(0, Hb//2 - h//2)
            left = max(0,Wb//2 - w//2)
            h,w = max(ceil(h),1), max(ceil(w),1)
            sample["bioclim"] = sample["bioclim"][ :, int(top) : int(top + h), int(left) : int(left + w)]
            print(sample["bioclim"].shape)
        if "ped" in list(sample.keys()):
            #align bioclim with ped
            Hb, Wb = sample["ped"].size()[-2:]
            print("ped")
            print(Hb,Wb)
            h = (Hb*self.sat_res/self.ped_res)
            w = (Wb*self.sat_res/self.ped_res)
            top = max(0, Hb//2 - h//2)
            left = max(0,Wb//2 - w//2)
            h,w = max(ceil(h),1), max(ceil(w),1)
            sample["ped"] = sample["ped"][ :, int(top) : int(top + h), int(left) : int(left + w)]
            print(sample["ped"].shape)
        for elem in list(sample.keys()):
            if elem in env:
                
                if ((sample[elem].size()[-1] == 0) or (sample[elem].size()[-2] == 0)):
                    if elem == "bioclim":
                        sample[elem] = torch.Tensor([ 11.99430391,  12.16226584,  36.94248176, 805.72045945,
        29.4489089 ,  -4.56172133,  34.01063026,  15.81641269,
         7.80845219,  21.77499491,   1.93990004, 902.9704986 ,
       114.61111788,  42.0276728 ,  37.11493781, 315.34206997,
       145.09703767, 231.19724491, 220.06619529]).unsqueeze(-1).unsqueeze(-1)
                sample[elem] = F.interpolate(sample[elem].unsqueeze(0).float(), size=(H, W))
        return (sample)



        
class RandomCrop:  # type: ignore[misc,name-defined]
    """Identity function used for testing purposes."""
    
    def __init__(self, size, center=False, ignore_band=None, p=0.5):
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
        
    def  __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Args:
            sample: the input
        Returns:
            the cropped input
        """
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
        for key in sample.keys():
            if (len(sample[key].size())==3):
                sample[key] = sample[key].unsqueeze(0)
        
        if torch.rand(1) > self.p:
            return(sample)
        else:
            if not self.center:

                top = max(0, np.random.randint(0, max(H - self.h,1)))
                left = max(0, np.random.randint(0, max(W - self.w,1)))
            else:
                top = max(0, (H - self.h) // 2)
                left = max(0,(W - self.w) // 2)

            item_ = {}
            for task, tensor in sample.items():
                
                if task in all_data and not task in self.ignore_band: 
                    item_.update({task: tensor[:, :, top : top + self.h, left : left + self.w]})
                else: 
                    item_.update({task: tensor})

            return(item_)
    
class Resize:
    def __init__(self, size):
        """
        size = (height, width) target size 
        """
        self.h, self.w = size
        
    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:    
        for s in sample:
            if s in satellite:
                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode = 'nearest')
            elif s in env or s in landuse:
                sample[s] = F.interpolate(sample[s].float(), size=(self.h, self.w), mode = 'nearest')
        return(sample)
        
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
            if s in satellite : 
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
            center=(transform_item.center == mode or transform_item.center == True), ignore_band = transform_item.ignore_band or None,p=transform_item.p
        )
    elif transform_item.name == "matchres" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        return MatchRes(transform_item.target_size)

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
       
        return Normalize(maxchan=transform_item.maxchan, custom=transform_item.custom or None, subset = transform_item.subset)
    
    elif transform_item.name == "resize" and not (
        transform_item.ignore is True or transform_item.ignore == mode
    ):
        
        return Resize(size=transform_item.size)

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
        if t.name == "normalize" and not (
        t.ignore is True or t.ignore == mode
    ) and t.subset==["sat"]:
            if opts.data.bands == ["r", "g", "b"]:
                print("only taking normalization values for r,g,b")
                means, std = t.custom
                t.custom = [means[:3], std[:3]]

            #assert (len(t.custom[0])== len(opts.data.bands))
        #account for multires
        if t.name=='crop' and len(opts.data.multiscale)>1:
            for res in opts.data.multiscale:
                #adapt hight and width to vars in multires
                t.hight, t.width=res, res
                crop_transforms.append(get_transform(t,mode))
        else:
             transforms.append(get_transform(t, mode))
    transforms = [t for t in transforms if t is not None]
    if crop_transforms:
        crop_transforms=[t for t in crop_transforms if t is not None]
        print('crop transforms ',crop_transforms)
        return crop_transforms,transforms
    else:
        return transforms