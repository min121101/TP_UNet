"""SKCDF 2D adaptation that plugs into TP-UNet's training pipeline.

Two model variants are exposed:

- ``SKCDF2D``           : monomodal supervised baseline. forward(images) -> (logits, logits_abc)
- ``SKCDF2D_TP``        : temporal Electra ``[B,T,D]`` -> per-step ``lang_dim`` (LayerNorm). ``Skip_CrossAtt`` uses
                          **one pooled skip token** (mean over time): its attention length is ``L*H*W`` and ``L>1``
                          explodes VRAM as ``(L*H*W)^2``. Full ``T*D`` is still used in the bottleneck bias MLP.
                          ``forward(images, temporal)`` -> ``(logits, logits_abc)`` — no text, no contrastive.

Compared to the original 3D ``VNet_Decouple_Attention_ABC`` we drop:
- cross-batch (labeled/unlabeled) attention (we are training with full supervision)
- the second decoder for unlabeled pseudo-labels
- 3D convs / dropout / ConvTranspose, replaced with the 2D counterparts

We keep:
- the V-Net style encoder/decoder (5 levels)
- the ABC double-head (``out_conv9`` for the main prediction and ``out_conv9_abc`` for
  the class-balanced auxiliary head)
- a self-attention global block at the bottleneck
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, LayerNorm

try:
    from unet.Cross_Attention import Skip_CrossAtt
except ImportError:
    Skip_CrossAtt = None


def _make_norm2d(name, ch):
    if name == 'batchnorm':
        return nn.BatchNorm2d(ch)
    if name == 'groupnorm':
        return nn.GroupNorm(num_groups=16, num_channels=ch)
    if name == 'instancenorm':
        return nn.InstanceNorm2d(ch)
    if name == 'none':
        return nn.Identity()
    raise ValueError(name)


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_in, n_out, normalization='batchnorm'):
        super().__init__()
        ops = []
        for i in range(n_stages):
            ic = n_in if i == 0 else n_out
            ops.append(nn.Conv2d(ic, n_out, 3, padding=1))
            if normalization != 'none':
                ops.append(_make_norm2d(normalization, n_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=2, normalization='batchnorm'):
        super().__init__()
        ops = [nn.Conv2d(n_in, n_out, kernel_size=stride, stride=stride, padding=0)]
        if normalization != 'none':
            ops.append(_make_norm2d(normalization, n_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=2, normalization='batchnorm'):
        super().__init__()
        ops = [nn.ConvTranspose2d(n_in, n_out, kernel_size=stride, stride=stride, padding=0)]
        if normalization != 'none':
            ops.append(_make_norm2d(normalization, n_out))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class Encoder2D(nn.Module):
    def __init__(self, n_channels=3, n_filters=32, normalization='batchnorm', has_dropout=True):
        super().__init__()
        self.has_dropout = has_dropout
        nf = n_filters

        self.block_one = ConvBlock(1, n_channels, nf, normalization)
        self.block_one_dw = DownsamplingConvBlock(nf, nf * 2, normalization=normalization)

        self.block_two = ConvBlock(2, nf * 2, nf * 2, normalization)
        self.block_two_dw = DownsamplingConvBlock(nf * 2, nf * 4, normalization=normalization)

        self.block_three = ConvBlock(3, nf * 4, nf * 4, normalization)
        self.block_three_dw = DownsamplingConvBlock(nf * 4, nf * 8, normalization=normalization)

        self.block_four = ConvBlock(3, nf * 8, nf * 8, normalization)
        self.block_four_dw = DownsamplingConvBlock(nf * 8, nf * 16, normalization=normalization)

        self.block_five = ConvBlock(3, nf * 16, nf * 16, normalization)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):
        x1 = self.block_one(x)
        x2 = self.block_two(self.block_one_dw(x1))
        x3 = self.block_three(self.block_two_dw(x2))
        x4 = self.block_four(self.block_three_dw(x3))
        x5 = self.block_five(self.block_four_dw(x4))
        if self.has_dropout:
            x5 = self.dropout(x5)
        return x1, x2, x3, x4, x5


class DecoderABC2D(nn.Module):
    """Single-branch decoder with a class-balanced ABC double head.

    Returns ``(out_main, out_abc)`` to mirror SKCDF's ``out_conv9`` / ``out_conv9_abc``
    behaviour.
    """

    def __init__(self, n_classes=2, n_filters=32, normalization='batchnorm', has_dropout=True):
        super().__init__()
        self.has_dropout = has_dropout
        nf = n_filters

        self.block_five_up = UpsamplingDeconvBlock(nf * 16, nf * 8, normalization=normalization)
        self.block_six = ConvBlock(3, nf * 8, nf * 8, normalization)
        self.block_six_up = UpsamplingDeconvBlock(nf * 8, nf * 4, normalization=normalization)
        self.block_seven = ConvBlock(3, nf * 4, nf * 4, normalization)
        self.block_seven_up = UpsamplingDeconvBlock(nf * 4, nf * 2, normalization=normalization)
        self.block_eight = ConvBlock(2, nf * 2, nf * 2, normalization)
        self.block_eight_up = UpsamplingDeconvBlock(nf * 2, nf, normalization=normalization)
        self.block_nine = ConvBlock(1, nf, nf, normalization)
        self.out_conv9 = nn.Conv2d(nf, n_classes, 1)
        self.out_conv9_abc = nn.Conv2d(nf, n_classes, 1)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        x = self.block_five_up(x5) + x4
        x = self.block_six(x)
        x = self.block_six_up(x) + x3
        x = self.block_seven(x)
        x = self.block_seven_up(x) + x2
        x = self.block_eight(x)
        x = self.block_eight_up(x) + x1
        x = self.block_nine(x)
        if self.has_dropout:
            x = self.dropout(x)
        return self.out_conv9(x), self.out_conv9_abc(x)


class SelfAttention2D(nn.Module):
    """Bottleneck self-attention: token-mix a (B,C,H,W) tensor."""

    def __init__(self, channels, num_heads=4, attn_dropout=0.1):
        super().__init__()
        assert channels % num_heads == 0, f'{channels} % {num_heads} != 0'
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = LayerNorm(channels, eps=1e-6)
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.out = nn.Linear(channels, channels, bias=False)
        self.attn_dropout = Dropout(attn_dropout)
        self.proj_dropout = Dropout(attn_dropout)
        self.softmax = Softmax(dim=-1)
        self.ffn_norm = LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        res = tokens
        tokens = self.norm(tokens)
        q = self.q(tokens).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(tokens).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(tokens).view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj_dropout(self.out(out)) + res
        out = self.ffn_norm(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class SKCDF2D(nn.Module):
    """SKCDF 2D — supervised baseline (no temporal prompt)."""

    def __init__(self, n_channels=3, n_classes=2, n_filters=32,
                 normalization='batchnorm', has_dropout=True, num_heads=4):
        super().__init__()
        self.encoder = Encoder2D(n_channels, n_filters, normalization, has_dropout)
        self.bottleneck_attn = SelfAttention2D(n_filters * 16, num_heads=num_heads)
        self.decoder = DecoderABC2D(n_classes, n_filters, normalization, has_dropout)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.bottleneck_attn(x5)
        return self.decoder((x1, x2, x3, x4, x5))


class SKCDF2D_TP(nn.Module):
    """SKCDF 2D + temporal prompt: shallow ``Skip_CrossAtt`` (pooled to L=1 for VRAM) + bottleneck bias from full ``T*D``."""

    def __init__(self, n_channels=3, n_classes=2, n_filters=32,
                 normalization='batchnorm', has_dropout=True, num_heads=4,
                 temporal_hidden_dim=256, temporal_len=5, lang_dim=512,
                 tp_bn_gate_init=0.12):
        super().__init__()
        if Skip_CrossAtt is None:
            raise ImportError('SKCDF2D_TP needs unet.Cross_Attention.Skip_CrossAtt')

        self.temporal_hidden_dim = temporal_hidden_dim
        self.temporal_len = temporal_len
        self.temporal_flat_dim = temporal_len * temporal_hidden_dim

        self.encoder = Encoder2D(n_channels, n_filters, normalization, has_dropout)
        self.bottleneck_attn = SelfAttention2D(n_filters * 16, num_heads=num_heads)
        self.decoder = DecoderABC2D(n_classes, n_filters, normalization, has_dropout)

        # Per-time-step projection; skip fusion pools to L=1 (Skip_CrossAtt uses seq len L*H*W -> O((L*HW)^2) memory).
        self.temporal_token_proj = nn.Linear(temporal_hidden_dim, lang_dim)
        self.prompt_norm = nn.LayerNorm(lang_dim)

        self.skip_att_shallow = Skip_CrossAtt(vis_dim=n_filters * 2, lang_dim=lang_dim,
                                              head_num=1, emb=n_filters * 2)

        c_bot = n_filters * 16
        self.bn_temporal = nn.Sequential(
            nn.Linear(self.temporal_flat_dim, c_bot),
            nn.ReLU(inplace=True),
            nn.Linear(c_bot, c_bot),
        )
        for m in self.bn_temporal.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Start training near the no-TP baseline (last layer only).
        nn.init.zeros_(self.bn_temporal[-1].weight)
        nn.init.zeros_(self.bn_temporal[-1].bias)

        self.tp_bn_gate = nn.Parameter(torch.tensor(float(tp_bn_gate_init)))

    def forward(self, x, temporal):
        # temporal: [B, T, D] from Electra(last_hidden_state), T/D match tokenizer max_length / Electra hidden.
        if temporal.dim() != 3:
            raise ValueError(f'temporal expected [B,T,D], got {tuple(temporal.shape)}')
        B, T, D = temporal.shape
        if T != self.temporal_len or D != self.temporal_hidden_dim:
            raise ValueError(
                f'temporal shape mismatch: got T={T}, D={D}, expected T={self.temporal_len}, D={self.temporal_hidden_dim}'
            )

        prompt_seq = self.prompt_norm(self.temporal_token_proj(temporal))
        # Mean over Electra time positions: uses all tokens, keeps L=1 inside Skip_CrossAtt (see Cross_Attention).
        prompt_skip = prompt_seq.mean(dim=1, keepdim=True)

        x1, x2, x3, x4, x5 = self.encoder(x)

        x2 = self.skip_att_shallow(x2, prompt_skip)

        delta = self.bn_temporal(temporal.reshape(B, -1)).view(B, -1, 1, 1)
        x5 = x5 + self.tp_bn_gate * delta

        x5 = self.bottleneck_attn(x5)

        out, out_abc = self.decoder((x1, x2, x3, x4, x5))
        return out, out_abc
