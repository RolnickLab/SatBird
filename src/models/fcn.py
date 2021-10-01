import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module


class FCN(Module):
    "A simple 5 layer CNN"
    def __init__(self, in_channels: int, classes: int, num_filters: int=64) -> None:
        super(FCN, self).__init__()
        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        self.backbone = nn.modules.Sequential(
            conv1,
            nn.modules.LeakyReLu(inplace=True),
            conv2,
            nn.modules.LeakyReLu(inplace=True),
            conv3,
            nn.modules.LeakyReLu(inplace=True),
            conv3,
            nn.modules.LeakyReLu(inplace=True),
            conv4,
            nn.modules.LeakyReLu(inplace=True),
            conv5,
            nn.modules.LeakyReLu(inplace=True),
            
        )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size = 1, stride=1, padding=0
        )

        def forward(slef, x: Tensor) -> Tensor:
            x = self.backbone(x)
            x = self.last(x)
            return x
            

