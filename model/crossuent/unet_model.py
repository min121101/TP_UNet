""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from collections import OrderedDict
from unet.Cross_Attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.loss.contrastive_loss import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.Skip_Att = nn.Sequential(OrderedDict([
            ('unit1', Skip_CrossAtt(vis_dim=128, lang_dim=768, head_num=1, emb=128)),
            ('unit2', Skip_CrossAtt(vis_dim=1024, lang_dim=768, head_num=1, emb=1024)),
        ]))
        self.Down_Att = nn.Sequential(OrderedDict([
            ('unit1', Down_CrossAtt(vis_dim=256, lang_dim=768, head_num=1, emb=768)),
            ('unit2', Down_CrossAtt(vis_dim=512, lang_dim=768, head_num=1, emb=768)),
        ]))
        self.linear = torch.nn.Linear(4864, 768)
        # self.projection = torch.nn.Linear(768, 768)
        self.low_level_loss_conv1x1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.high_level_loss_conv1x1 = nn.Conv2d(in_channels=1024, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.low_level_loss_linear = nn.Linear(9216, 768)
        self.high_level_loss_linear = nn.Linear(4608, 768)
        self.Relu = nn.ReLU()

    def forward(self, x, text):
        text = text.flatten(1).unsqueeze(1)
        text_feature = self.linear(text)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # low_level_feat = self.low_level_loss_conv1x1(x2)
        # low_level_feat = self.Relu(low_level_feat)
        # low_level_feat = self.low_level_loss_linear(low_level_feat.flatten(1))
        # low_level_contrastive_loss = contrastive_loss(low_level_feat, text_feature.flatten(1))
        x2 = self.Skip_Att[0](x2, text_feature)
        x3 = self.down2(x2)
        # text_feature = self.Down_Att[0](x3, text_feature)
        x4 = self.down3(x3)
        # text_feature = self.Down_Att[1](x4, text_feature)
        x5 = self.down4(x4)
        # text_feature = self.projection(text_feature)
        # high_level_feat = self.high_level_loss_conv1x1(x5)
        # high_level_feat = self.Relu(high_level_feat)
        # high_level_feat = self.high_level_loss_linear(high_level_feat.flatten(1))
        # high_level_contrastive_loss = contrastive_loss(high_level_feat, text_feature.flatten(1))
        # x5 = self.Skip_Att[1](x5, text_feature)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        low_level_contrastive_loss = 0
        high_level_contrastive_loss = 0
        return logits, low_level_contrastive_loss, high_level_contrastive_loss

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)