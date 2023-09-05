"""
NN models
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""
import torch.nn as nn
import torchvision.models as models
from src.trainer.utils import init_first_layer_weights


class Resnet50(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(Resnet50, self).__init__()
        self.x_input_channels = input_channels
        self.pretrained_backbone = pretrained

        self.freeze_base = False
        self.unfreeze_base_l4 = False

        self.base_network = models.resnet50(pretrained=self.pretrained_backbone)
        original_in_channels = self.base_network.conv1.in_channels

        # if input is not RGB
        if self.x_input_channels != original_in_channels:
            original_weights = self.base_network.conv1.weight.data.clone()
            self.base_network.conv1 = nn.Conv2d(self.x_input_channels, 64, kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            if self.pretrained_backbone:
                self.base_network.conv1.weight.data[:, :original_in_channels, :, :] = original_weights
                self.base_network.conv1.weight.data = init_first_layer_weights(self.x_input_channels, original_weights)

        if self.freeze_base:
            for param in self.base_network.parameters():
                param.requires_grad = False
        elif self.unfreeze_base_l4:
            for p in self.base_network.layer4.parameters():
                p.requires_grad = True

        #TODO: use original avgpool layer: original isA daptiveAvgPool2d(output_size=(1, 1))
        self.base_network.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        x = self.base_network.conv1(images)
        x = self.base_network.bn1(x)
        x = self.base_network.relu(x)
        x = self.base_network.maxpool(x)
        x = self.base_network.layer1(x)
        x = self.base_network.layer2(x)
        x = self.base_network.layer3(x)
        x = self.base_network.layer4(x)
        return x


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead)

    def forward(self, k, mask=None):
        k = k.transpose(0, 1)
        x = self.transformer_layer(k, src_mask=mask)
        x = x.transpose(0, 1)
        return x