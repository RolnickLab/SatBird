"""
Regression-Transformer model
Code is based on the C-tran paper: https://github.com/QData/C-Tran
"""
import torch
import torch.nn as nn
import numpy as np
from Rtran.utils import weights_init, custom_replace
from Rtran.models import *


class RTranModel(nn.Module):
    def __init__(self, num_classes, backbone='Resnet18', pretrained_backbone=True, input_channels=3, d_hidden=512, n_state=3, attention_layers=2, heads=2, dropout=0.2, scale_embeddings_by_labels=False):
        """
        pos_emb is false by default
        """
        super(RTranModel, self).__init__()
        self.d_hidden = d_hidden  # this should match the backbone output feature size (512 for Resnet18, 2048 for Resnet50)
        self.scale_embeddings_by_labels = scale_embeddings_by_labels

        # ResNet101 backbone
        self.backbone = globals()[backbone](input_channels=input_channels, pretrained=pretrained_backbone)
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_classes)).view(1, -1).long()

        # TODO: modify the labels to actually take text labels rather than label numbers
        self.label_embeddings = torch.nn.Embedding(num_classes, self.d_hidden, padding_idx=None)  # LxD

        # State Embeddings
        self.state_embeddings = torch.nn.Embedding(n_state, self.d_hidden, padding_idx=0) # Dx2 (known, unknown)

        # embedding for the regression labels
        # self.regression_embedding = torch.nn.Linear(num_classes, self.d_hidden)
        # TODO: Position Embeddings (for image features)
        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(self.d_hidden, heads, dropout) for _ in range(attention_layers)])

        # Classifier
        # Output is of size num_classes because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(self.d_hidden, num_classes)

        # Other
        self.LayerNorm = nn.LayerNorm(d_hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_embeddings.apply(weights_init)
        self.state_embeddings.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, mask, labels=None):
        z_features = self.backbone(images) # image: HxWxD

        const_label_input = self.label_input.repeat(images.size(0), 1).to(images.device) # LxD
        init_label_embeddings = self.label_embeddings(const_label_input)    # LxD

        z_features = z_features.view(z_features.size(0), z_features.size(1), -1).permute(0, 2, 1)

        # Get state embeddings (mask is 0 or regression value)
        # print(torch.unique(mask))
        # print(self.state_embeddings)
        # unknown_mask = custom_replace(mask, 1, 0, 0)
        label_feat_vec = custom_replace(mask, 0, 1, 2).long()
        state_embeddings = self.state_embeddings(label_feat_vec) # input: 3, output: 512

        # print(state_embeddings[5, 12])
        # print(labels[5, 12])
        # print(state_embeddings[5, 12])
        # Add state embeddings to label embeddings
        init_label_embeddings += state_embeddings
        # print(init_label_embeddings.size())
        # concatenate images features to label embeddings
        embeddings = torch.cat((z_features, init_label_embeddings), 1)
        # print(embeddings.size())
        # exit(0)
        # Feed all (image and label) embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        for layer in self.self_attn_layers:
            embeddings = layer(embeddings, mask=None)

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        output = self.output_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1), device=output.device).unsqueeze(0).repeat(output.size(0), 1, 1)
        output = (output * diag_mask).sum(-1)

        return output
