from typing import Tuple, Any

from .pointmlp import pointMLPMIL, pointMLPEliteMIL
import torch.nn.functional as F

import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum, Tensor

import torch
from torch import nn
import math

class InstancePooling(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes=2,
                 dropout=0.5,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(InstancePooling, self).__init__()
        self.non_linear = non_linear
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)


        if self.non_linear:
            self.node_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.node_out = nn.Linear(self.num_features, num_classes)

    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        isinstance_logits = self.node_out(features[1].transpose(2, 1))
        bag_logits = torch.mean(isinstance_logits, dim=1)

        return {
            'interpretation': isinstance_logits.transpose(2, 1),
            'bag_logits': bag_logits
        }

class AttentionPooling(nn.Module):
    def __init__(self, num_features,
                 num_classes=2,
                 dropout=0.5,
                 heads=8,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(AttentionPooling, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.non_linear = non_linear
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, heads),
            nn.Tanh(),
            nn.Linear(heads, 1),
            nn.Sigmoid(),
        )

        if self.non_linear:
            self.bag_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.BatchNorm1d(self.num_features * 2),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                nn.BatchNorm1d(self.num_features // 2),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.bag_out = nn.Linear(self.num_features, num_classes)

    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        attn_weights = self.attention_head(features[1].transpose(2, 1))
        bag_embedding = torch.mean(features[1].transpose(2, 1) * attn_weights, dim=1)
        bag_logits = self.bag_out(bag_embedding)

        return {
            'interpretation': attn_weights.repeat(1, 1, self.num_classes).transpose(2, 1),
            'bag_logits': bag_logits
        }

class AdditivePooling(nn.Module):
    def __init__(self,
                 num_features,
                 num_classes=2,
                 dropout=0.5,
                 heads=8,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(AdditivePooling, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.non_linear = non_linear
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)

        if self.non_linear:
            self.node_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.node_out = nn.Linear(self.num_features, num_classes)

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, heads),
            nn.Tanh(),
            nn.Linear(heads, 1),
            nn.Sigmoid(),
        )


    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        attn_weights = self.attention_head(features[1].transpose(2, 1))
        weighted_instance_features = features[1].transpose(2, 1) * attn_weights
        instance_logits = self.node_out(weighted_instance_features)
        bag_logits = torch.mean(instance_logits, dim=1)


        return {
            'interpretation': (instance_logits * attn_weights).transpose(2, 1),
            'bag_logits': bag_logits,
            'instance_logits': instance_logits.transpose(2, 1),
            'attention': attn_weights

        }


class ConjunctivePooling(nn.Module):
    def __init__(self, num_features,
                 num_classes=2,
                 dropout=0.5,
                 heads=8,
                 non_linear=True,
                 apply_pos_encoding=True,
                 **kwargs):
        super(ConjunctivePooling, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.non_linear = non_linear
        self.apply_pos_encoding = apply_pos_encoding
        if self.apply_pos_encoding:
            self.pos_enc = PositionalEncoding3D(num_features)

        # self.conv_raise = nn.Sequential(
        #         nn.Conv1d(self.num_features, self.num_features * 2, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(self.num_features * 2),
        #         nn.ReLU(True),
        #         nn.Conv1d(self.num_features * 2, self.num_features * 4, kernel_size=1, bias=False),
        #         nn.BatchNorm1d(self.num_features * 4),
        #         nn.ReLU(True))
        #
        # if self.non_linear:
        #     self.node_out = nn.Sequential(
        #         nn.Linear(self.num_features * 4, self.num_features * 2),
        #         nn.LayerNorm(self.num_features * 2),
        #         nn.ReLU(True),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(self.num_features * 2, self.num_features),
        #         nn.LayerNorm(self.num_features),
        #         nn.ReLU(True),
        #         nn.Dropout(p=dropout),
        #         nn.Linear(self.num_features, num_classes)
        #     )
        if self.non_linear:
            self.node_out = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Linear(self.num_features * 2, self.num_features // 2),
                nn.BatchNorm1d(1024),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(self.num_features // 2, num_classes)
            )
        else:
            self.node_out = nn.Linear(self.num_features, num_classes)

        self.attention_head = nn.Sequential(
            nn.Linear(num_features, heads),
            nn.Tanh(),
            nn.Linear(heads, 1),
            nn.Sigmoid(),
        )
        # self.attention_head = nn.Sequential(
        #     nn.Linear(num_features * 4, heads),
        #     nn.Tanh(),
        #     nn.Linear(heads, 1),
        #     nn.Sigmoid(),
        # )
    def forward(self, features):
        if self.apply_pos_encoding:
            features = self.pos_enc(features[0], features[1])
        # features_raised = self.conv_raise(features[1])
        # print(features_raised.shape)
        attn_weights = self.attention_head(features[1].transpose(2, 1))
        # print(features_raised.shape)

        instance_logits = self.node_out(features[1].transpose(2, 1))
        # print(instance_logits.shape)
        weighted_instance_logits = instance_logits * attn_weights
        bag_logits = torch.mean(weighted_instance_logits, dim=1)

        return {
            'interpretation': (instance_logits * attn_weights).transpose(2, 1),
            'bag_logits': bag_logits,
            'instance_logits': instance_logits.transpose(2, 1),
            'attention': attn_weights
        }



class PositionalEncoding3D(nn.Module):
    """
    A PyTorch module that generates sinusoidal positional encodings for 3D coordinates.
    """

    def __init__(self, num_features: int):
        """
        Initialize the positional encoding module.

        Args:
            num_features: Number of features to encode (should match the original feature dimension).
        """
        super(PositionalEncoding3D, self).__init__()
        self.num_features = num_features
        div_term = torch.exp(torch.arange(0, num_features, 2) * -(math.log(10000.0) / num_features))
        self.register_buffer('div_term', div_term)

    def forward(self, coords: torch.Tensor, features) -> Tuple[Tensor, Any]:
        """
        Generate sinusoidal positional encodings for 3D coordinates.

        Args:
            coords: Tensor of shape (batch_size, num_points, 3) representing (x, y, z).

        Returns:
            Tensor of shape (batch_size, num_points, num_features).
            :param coords:
            :param features:
        """
        batch_size, num_points, _ = coords.shape
        pe = torch.zeros(batch_size, num_points, self.num_features, device=coords.device)

        for i in range(3):  # Loop over x, y, z
            coord = coords[:, :, i].unsqueeze(-1)  # Shape: (batch_size, num_points, 1)
            pe[:, :, 0::2] += torch.sin(coord * self.div_term)
            pe[:, :, 1::2] += torch.cos(coord * self.div_term)

        return pe, features + pe.transpose(2, 1)


class PointMIL(nn.Module):
    def __init__(self,
                 feature_extractor=pointMLPEliteMIL(),
                 pooling=AttentionPooling(num_features=256,
                                          num_classes=40,
                                          apply_pos_encoding=False),
                 ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooling = pooling

    def interpret(self, model_output):
        return model_output['interpretation']
    def forward(self, x):
        # print(x.shape)
        features = self.feature_extractor(x)
        pooling = self.pooling(features)


        return (features[0], features[1] , features[2], pooling['interpretation']), pooling['bag_logits']