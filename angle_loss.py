import torch
from torch import nn
import torch.nn.functional as F


class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, angle, angle_hat):
        return torch.exp(F.mse_loss(angle_hat.float(), angle.float())) - 1
