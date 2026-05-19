"""
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn


def contrastive_loss(cnn_code, rnn_code, eps=1e-6, temp3=4.0):
    """Image–text alignment. Run in float32 with bounded logits to avoid NaNs under AMP/fp16."""

    batch_size = cnn_code.shape[0]
    if batch_size < 2:
        zero = cnn_code.sum() * 0.0
        return zero, zero

    cnn_code = cnn_code.float()
    rnn_code = rnn_code.float()

    labels = torch.arange(batch_size, device=cnn_code.device, dtype=torch.long)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True).clamp(min=eps)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True).clamp(min=eps)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2)).clamp(min=eps ** 2)
    scores0 = (scores0 / norm0) * temp3
    scores0 = scores0.squeeze(0).clamp(min=-50.0, max=50.0)

    scores1 = scores0.transpose(0, 1).clamp(min=-50.0, max=50.0)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


