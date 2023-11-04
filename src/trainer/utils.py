# utility functions/models used by trainers
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from src.dataset.dataloader import get_subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def weights_init(module):
    """
    Initialize the weights
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def init_first_layer_weights(in_channels: int, rgb_weights, hs_weight_init: str = 'random'):
    '''Initializes the weights for filters in the first conv layer.
      If we are using RGB-only, then just initializes var to rgb_weights. Otherwise, uses
      hs_weight_init to determine how to initialize the weights for non-RGB bands.
      Args
      - int: in_channesl, input channels
          - in_channesl is  either 3 (RGB), 7 (lxv3), or 9 (Landsat7) or 2 (NL)
      - rgb_weights: ndarray of np.float32, shape [64, 3, F, F]
      - hs_weight_init: str, one of ['random', 'same', 'samescaled']
      Returs
      -torch tensor : final_weights
      '''

    out_channels, rgb_channels, H, W = rgb_weights.shape
    rgb_weights = torch.tensor(rgb_weights, device=rgb_weights.device)
    ms_channels = in_channels - rgb_channels
    if in_channels == 3:
        final_weights = rgb_weights

    elif in_channels < 3:
        with torch.no_grad():
            mean = rgb_weights.mean()
            std = rgb_weights.std()
            final_weights = torch.empty((out_channels, in_channels, H, W), device=rgb_weights.device)
            final_weights = torch.nn.init.trunc_normal_(final_weights, mean, std)
    elif in_channels > 3:
        # spectral images

        if hs_weight_init == 'same':

            with torch.no_grad():
                mean = rgb_weights.mean(dim=1, keepdim=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = mean

        elif hs_weight_init == 'random':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean()
                std = rgb_weights.std()
                ms_weights = torch.empty((out_channels, ms_channels, H, W), device=rgb_weights.device)
                ms_weights = torch.nn.init.trunc_normal_(ms_weights, mean, std)
            print(f'random: {time.time() - start}')

        elif hs_weight_init == 'samescaled':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean(dim=1, keepdim=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = (mean * 3) / (3 + ms_channels)
                # scale both rgb_weights and ms_weights
                rgb_weights = (rgb_weights * 3) / (3 + ms_channels)


        else:

            raise ValueError(f'Unknown hs_weight_init type: {hs_weight_init}')

        final_weights = torch.cat([rgb_weights, ms_weights], dim=1)
    return final_weights


def custom_replace(tensor, on_neg_1, on_zero, on_one):
    res = tensor.clone()
    res[tensor == -1] = on_neg_1
    res[tensor == 0] = on_zero
    res[tensor == 1] = on_one
    return res


def get_activation_fn(activation):
    """
    return activation function
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def get_nb_bands(bands):
    """
    return number of bands
    """
    n = 0
    for b in bands:
        if b in ["r", "g", "b", "nir", "landuse"]:
            n += 1
        elif b == "ped":
            n += 8
        elif b == "bioclim":
            n += 19
        elif b == "rgb":
            n += 3
    return n


def get_target_size(opts, subset=None):
    if subset is None:
        subset = get_subset(opts.data.target.subset)
    target_size = len(subset) if subset is not None else opts.data.total_species
    return target_size


def get_scheduler(optimizer, opts):
    if opts.scheduler.name == "ReduceLROnPlateau":
        return (ReduceLROnPlateau(optimizer, factor=opts.scheduler.reduce_lr_plateau.factor,
                                  patience=opts.scheduler.reduce_lr_plateau.lr_schedule_patience))
    elif opts.scheduler.name == "StepLR":
        return (StepLR(optimizer, opts.scheduler.step_lr.step_size, opts.scheduler.step_lr.gamma))
    elif opts.scheduler.name == "WarmUp":
        return (LinearWarmupCosineAnnealingLR(optimizer, opts.scheduler.warmup.warmup_epochs,
                                              opts.scheduler.warmup.max_epochs))
    elif opts.scheduler.name == "Cyclical":
        return (CosineAnnealingWarmRestarts(optimizer, opts.scheduler.cyclical.t0, opts.scheduler.cyclical.tmult))
    elif opts.scheduler.name == "":
        return (None)
    else:
        raise ValueError(f"Scheduler'{opts.scheduler.name}' is not valid")


def load_from_checkpoint(path, model):
    print(f'initializing model from pretrained weights at {path}')
    if 'moco' in path:
        # moco pretrained models need some weights renaming
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        loaded_dict = checkpoint['state_dict']

        model_dict = model.state_dict()
        del loaded_dict["module.queue"]
        del loaded_dict["module.queue_ptr"]
        # load state dict keys
        for key_model, key_seco in zip(model_dict.keys(), loaded_dict.keys()):
            if 'fc' in key_model:
                # ignore fc weight
                continue
            model_dict[key_model] = loaded_dict[key_seco]
        model.load_state_dict(model_dict)
    elif 'fmow_pretrain' in path:
        checkpoint = torch.load(path)
        checkpoint_model = checkpoint['model']
        print('fmow keys', checkpoint_model.keys())

        state_dict = model.state_dict()
        print('model keys', state_dict.keys())

        loaded_dict = checkpoint_model
        model_dict = model.state_dict()

        for key_model in model_dict.keys():
            if 'fc' in key_model or 'head' in key_model:
                #                 #ignore fc weight
                model_dict[key_model] = model_dict[key_model]
            else:
                model_dict[key_model] = loaded_dict[key_model]

        model.load_state_dict(model_dict)



    else:
        ckpt = torch.load(path)

        model.load_state_dict(torch.load(path))
    # model.eval()
    return model
