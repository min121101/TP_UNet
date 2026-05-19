"""CSC-PA building blocks (2D), adapted from https://github.com/... CSC-PA."""

from .decoder_2d import DecDeepLabV3Plus2D
from .resnet import resnet50

__all__ = ['DecDeepLabV3Plus2D', 'resnet50']
