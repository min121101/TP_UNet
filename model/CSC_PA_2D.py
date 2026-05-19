"""CSC-PA 2D for TP-UNet LITS pipeline (full supervised).

- ``CSCPA2D``    : ResNet50-FPN + DeepLabV3+ + prototype correlation (no temporal).
- ``CSCPA2D_TP`` : same + Electra temporal prompt (FiLM + gated channel bias on fused features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from csc_pa.resnet import resnet50
from csc_pa.decoder_2d import DecDeepLabV3Plus2D


class CSCPA2D(nn.Module):
    def __init__(self, num_classes=1, sync_bn=False, pretrained_encoder=False,
                 inner_planes=256, dilations=(12, 24, 36), use_pcg=True,
                 multi_grid=True, replace_stride_with_dilation=(False, True, True)):
        super().__init__()
        self.encoder = resnet50(
            pretrained=pretrained_encoder,
            sync_bn=sync_bn,
            multi_grid=multi_grid,
            zero_init_residual=True,
            fpn=True,
            replace_stride_with_dilation=list(replace_stride_with_dilation),
        )
        self.decoder = DecDeepLabV3Plus2D(
            in_planes=self.encoder.get_outplanes(),
            num_classes=num_classes,
            inner_planes=inner_planes,
            sync_bn=sync_bn,
            dilations=dilations,
            use_pcg=use_pcg,
            lang_dim=512,
        )

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out


class CSCPA2D_TP(nn.Module):
    def __init__(self, num_classes=1, sync_bn=False, pretrained_encoder=False,
                 inner_planes=256, dilations=(12, 24, 36), use_pcg=True,
                 multi_grid=True, replace_stride_with_dilation=(False, True, True),
                 temporal_hidden_dim=256, temporal_len=5, lang_dim=512,
                 tp_gate_init=0.12):
        super().__init__()
        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_len = temporal_len
        self.encoder = resnet50(
            pretrained=pretrained_encoder,
            sync_bn=sync_bn,
            multi_grid=multi_grid,
            zero_init_residual=True,
            fpn=True,
            replace_stride_with_dilation=list(replace_stride_with_dilation),
        )
        self.decoder = DecDeepLabV3Plus2D(
            in_planes=self.encoder.get_outplanes(),
            num_classes=num_classes,
            inner_planes=inner_planes,
            sync_bn=sync_bn,
            dilations=dilations,
            use_pcg=use_pcg,
            lang_dim=lang_dim,
        )
        self.decoder.tp_gate.data.fill_(float(tp_gate_init))
        self.temporal_token_proj = nn.Linear(temporal_hidden_dim, lang_dim)
        self.prompt_norm = nn.LayerNorm(lang_dim)
        flat_dim = temporal_len * temporal_hidden_dim
        if self.decoder.bn_temporal[0].in_features != flat_dim:
            self.decoder.bn_temporal[0] = nn.Linear(flat_dim, self.decoder.fuse_dim)

    def forward(self, x, temporal):
        if temporal.dim() != 3:
            raise ValueError(f'temporal expected [B,T,D], got {tuple(temporal.shape)}')
        b, t, d = temporal.shape
        if t != self.temporal_len or d != self.temporal_hidden_dim:
            raise ValueError(
                f'temporal shape mismatch: got T={t}, D={d}, '
                f'expected T={self.temporal_len}, D={self.temporal_hidden_dim}'
            )
        prompt_seq = self.prompt_norm(self.temporal_token_proj(temporal))
        prompt_skip = prompt_seq.mean(dim=1, keepdim=True)
        feats = self.encoder(x)
        out = self.decoder(feats, temporal=temporal, prompt_skip=prompt_skip)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out
