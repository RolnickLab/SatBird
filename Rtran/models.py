"""
NN models
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""
import torch.nn as nn
import torchvision.models as models


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        self.freeze_base = False
        self.freeze_base_l4 = False

        self.base_network = models.resnet101(pretrained=True)

        #TODO: use original avgpool layer
        self.base_network.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)  # replace avg pool

        if self.freeze_base:
            for param in self.base_network.parameters():
                param.requires_grad = False
        elif self.freeze_base_l4:
            for p in self.base_network.layer4.parameters():
                p.requires_grad = True

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