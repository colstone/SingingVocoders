import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

LRELU_SLOPE = 0.1


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            x = torch.nan_to_num(x)
            fmap.append(x)

        x = self.conv_post(x)
        x = torch.nan_to_num(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []

        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)

            y_d_r, fmap_r = d(y)

            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs

