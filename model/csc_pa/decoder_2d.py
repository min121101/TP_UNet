"""DeepLabV3+ decoder for 2D full-supervised CSC-PA (no semi-supervised memobank / L-U split)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ASPP, get_syncbn
from .fpa import PrototypeCorrelationGeneration


class TemporalFiLM(nn.Module):
    """Modulate fused features with a pooled temporal prompt (zero-init last layer)."""

    def __init__(self, lang_dim, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lang_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, prompt):
        # prompt: [B, 1, lang_dim]
        gb = self.net(prompt.squeeze(1))
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + torch.tanh(gamma)) + beta


class DecDeepLabV3Plus2D(nn.Module):
    def __init__(self, in_planes, num_classes=1, inner_planes=256, sync_bn=False,
                 dilations=(12, 24, 36), use_pcg=True, fuse_dim=512, lang_dim=512):
        super().__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.fuse_dim = fuse_dim
        self.use_pcg = use_pcg

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations)
        self.head = nn.Sequential(
            nn.Conv2d(self.aspp.get_outplanes(), 256, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        if use_pcg:
            self.pcg = PrototypeCorrelationGeneration(fuse_dim)
        self.temporal_film = TemporalFiLM(lang_dim, fuse_dim)
        self.tp_gate = nn.Parameter(torch.tensor(0.12))

        self.bn_temporal = nn.Sequential(
            nn.Linear(1280, fuse_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fuse_dim, fuse_dim),
        )
        nn.init.zeros_(self.bn_temporal[-1].weight)
        nn.init.zeros_(self.bn_temporal[-1].bias)

        self.classifier = nn.Sequential(
            nn.Conv2d(fuse_dim, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def _fuse_encoder_feats(self, x):
        x1, _, _, x4 = x
        aspp_out = self.head(self.aspp(x4))
        low_feat = self.low_conv(x1)
        h, w = low_feat.shape[-2:]
        aspp_out = F.interpolate(aspp_out, size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat((low_feat, aspp_out), dim=1)

    def forward(self, x, temporal=None, prompt_skip=None):
        """
        x: [x1, x2, x3, x4] from ResNet FPN.
        temporal: optional [B, T, D] for full-sequence bottleneck bias (TP).
        prompt_skip: optional [B, 1, lang_dim] for FiLM (TP).
        """
        feat = self._fuse_encoder_feats(x)
        if self.use_pcg:
            feat = self.pcg(feat)

        if prompt_skip is not None:
            feat = self.temporal_film(feat, prompt_skip)

        if temporal is not None:
            b = temporal.shape[0]
            delta = self.bn_temporal(temporal.reshape(b, -1)).view(b, -1, 1, 1)
            feat = feat + self.tp_gate * delta

        return self.classifier(feat)
