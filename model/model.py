import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import segmentation_models_pytorch as smp
import torch
from torch import nn
import AttentionUnet
from torchinfo import summary
import Unet_Segmentation_Pytorch_Nest_of_Unets.Models as spm
from torch.backends import cudnn
import Unet_Segmentation_Pytorch_Nest_of_Unets.AtteffUnet as atteffunet
import SwinUnet
# from CrossUnet.vit_seg_modeling import VisionTransformer as ViT_seg
# from CrossUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from TransUnet.vit_seg_modeling import VisionTransformer as Trans_ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as Trans_CONFIGS_ViT_seg
import FCBFormer.Models as FCBFormer
from unet import UNet
from SKCDF_2D import SKCDF2D, SKCDF2D_TP
from CSC_PA_2D import CSCPA2D, CSCPA2D_TP

def build_model(CFG):
    if (CFG['model_name'] == 'Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        print('model is Unet')
    elif (CFG['model_name'] == 'new_Unet'):
        model = UNet(n_channels=3, n_classes=CFG['num_classes'], bilinear=False).to(CFG['device'])
        print('model is new_Unet')
    elif (CFG['model_name'] == 'SCSE_Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        decoder_attention_type=CFG['attention'],
                        decoder_use_DIA=False,
                    ).to(CFG['device'])
        # print('model is vgg16-SCSE_Unet')
        print('model is efficientnet-b0-SCSE_Unet')
    elif (CFG['model_name'] == 'UnetPlusPlus'):
        model = smp.UnetPlusPlus(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA = False,
                    ).to(CFG['device'])
        print('model is UnetPlusPlus')
    elif (CFG['model_name'] == 'Unet_3Plus'):
        model = torch.nn.DataParallel(Unet_3Plus.UNetPPP(
                        in_channels=3,
                        num_classes=1,
                    )).cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[CFG['local_rank']], output_device=CFG['local_rank'])
        print('model is Unet_3Plus')
    elif (CFG['model_name'] == 'DeepLabV3'):
        model = smp.DeepLabV3(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is DeepLabV3')
    elif (CFG['model_name'] == 'FPN'):
        model = smp.FPN(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is FPN')
    elif (CFG['model_name'] == 'SwinUnet'):
        model = SwinUnet.SwinUnet(
                    ).to(CFG['device'])
        print('model is SwinUnet')
    elif (CFG['model_name'] == 'MAnet'):
        model = smp.MAnet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is MAnet')
    elif (CFG['model_name'] == 'Linknet'):
        model = smp.Linknet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                    ).to(CFG['device'])
        print('model is Linknet')
    elif (CFG['model_name'] == 'AttentionUnet'):
        if (CFG['backbone'] == 'efficientnet-b0'):
            model = atteffunet.get_efficientunet_b0(out_channels=CFG['num_classes'], concat_input=True, pretrained=False).to(CFG['device'])
            print('model is efficientnet-b0-AttentionUnet')
        else:
            model = spm.AttU_Net(img_ch=3, output_ch=CFG['num_classes']).to(CFG['device'])
            print('model is vgg16-AttentionUnet')
    elif (CFG['model_name'] == 'SIA_Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA = CFG['use_channel_attention'],
                        # decoder_use_DBM = CFG['use_DBM'],
                        # decoder_use_timestamp = CFG['use_timestamp'],
                        # batch_size = CFG['train_bs']
                    ).to(CFG['device'])
        print('model is vgg16-DIAUnet')
    elif (CFG['model_name'] == 'TransUnet'):
        config_vit = Trans_CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = CFG['num_classes']
        model = Trans_ViT_seg(config_vit, img_size=CFG['img_size'][0], num_classes=config_vit.n_classes).cuda()
        print('model is TransUnet')
    elif (CFG['model_name'] == 'CrossUnet'):
        model = UNet(n_channels=3, n_classes=CFG['num_classes'], bilinear=False).to(CFG['device'])
        print('model is CrossUnet')
    elif (CFG['model_name'] == 'Dual_Stream_Unet'):
        model = smp.Unet(
                        encoder_name=CFG['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=CFG['num_classes'],  # model output channels (number of classes in your dataset)
                        activation=None,
                        # decoder_attention_type=CFG['attention'],
                        decoder_use_DIA=CFG['use_channel_attention'],
                        # decoder_use_DBM=CFG['use_DBM'],
                        # decoder_use_timestamp=CFG['use_timestamp'],
                        # batch_size=CFG['train_bs']
        ).to(CFG['device'])
        print('model is Dual_Stream_Unet')
    elif (CFG['model_name'] == 'FCBFormer'):
        model = FCBFormer.FCBFormer().cuda()
        print('model is CrossUnet')
    elif CFG['model_name'] in ('TP_UNet_Clip', 'TP_UNet_Electra'):
        model = UNet(n_channels=3, n_classes=CFG['num_classes'], bilinear=False).to(CFG['device'])
        print(f"model is {CFG['model_name']}")
    elif CFG['model_name'] == 'SKCDF':
        model = SKCDF2D(
            n_channels=3,
            n_classes=CFG['num_classes'],
            n_filters=CFG.get('skcdf_n_filters', 32),
            normalization=CFG.get('skcdf_norm', 'batchnorm'),
            has_dropout=CFG.get('skcdf_dropout', True),
            num_heads=CFG.get('skcdf_num_heads', 4),
        ).to(CFG['device'])
        print('model is SKCDF (2D supervised)')
    elif CFG['model_name'] == 'SKCDF_TP':
        th = CFG.get('skcdf_temporal_hidden_dim', 256)
        tl = CFG.get('skcdf_temporal_len', 5)
        tin = CFG.get('skcdf_temporal_in_dim', th * tl)
        if tin != th * tl:
            print(f'warning: skcdf_temporal_in_dim={tin} != skcdf_temporal_len*hidden={tl}*{th}={tl * th}; model uses len*hidden')
        model = SKCDF2D_TP(
            n_channels=3,
            n_classes=CFG['num_classes'],
            n_filters=CFG.get('skcdf_n_filters', 32),
            normalization=CFG.get('skcdf_norm', 'batchnorm'),
            has_dropout=CFG.get('skcdf_dropout', True),
            num_heads=CFG.get('skcdf_num_heads', 4),
            temporal_hidden_dim=th,
            temporal_len=tl,
            lang_dim=CFG.get('skcdf_lang_dim', 512),
            tp_bn_gate_init=CFG.get('skcdf_tp_bn_gate_init', 0.12),
        ).to(CFG['device'])
        print('model is SKCDF_TP (2D + temporal mean-pooled skip prompt + bottleneck bias)')
    elif CFG['model_name'] == 'CSC_PA':
        model = CSCPA2D(
            num_classes=CFG['num_classes'],
            sync_bn=CFG.get('cscpa_sync_bn', False),
            pretrained_encoder=CFG.get('cscpa_pretrained', False),
            inner_planes=CFG.get('cscpa_inner_planes', 256),
            dilations=tuple(CFG.get('cscpa_dilations', [12, 24, 36])),
            use_pcg=CFG.get('cscpa_use_pcg', True),
            multi_grid=CFG.get('cscpa_multi_grid', True),
            replace_stride_with_dilation=tuple(
                CFG.get('cscpa_replace_stride_with_dilation', [False, True, True])),
        ).to(CFG['device'])
        print('model is CSC_PA (2D DeepLabV3+ + prototype correlation)')
    elif CFG['model_name'] == 'CSC_PA_TP':
        th = CFG.get('cscpa_temporal_hidden_dim', 256)
        tl = CFG.get('cscpa_temporal_len', 5)
        model = CSCPA2D_TP(
            num_classes=CFG['num_classes'],
            sync_bn=CFG.get('cscpa_sync_bn', False),
            pretrained_encoder=CFG.get('cscpa_pretrained', False),
            inner_planes=CFG.get('cscpa_inner_planes', 256),
            dilations=tuple(CFG.get('cscpa_dilations', [12, 24, 36])),
            use_pcg=CFG.get('cscpa_use_pcg', True),
            multi_grid=CFG.get('cscpa_multi_grid', True),
            replace_stride_with_dilation=tuple(
                CFG.get('cscpa_replace_stride_with_dilation', [False, True, True])),
            temporal_hidden_dim=th,
            temporal_len=tl,
            lang_dim=CFG.get('cscpa_lang_dim', 512),
            tp_gate_init=CFG.get('cscpa_tp_gate_init', 0.12),
        ).to(CFG['device'])
        print('model is CSC_PA_TP (2D CSC-PA + Electra temporal FiLM/bias)')
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

