import torch
import torch.nn as nn

from .base import get_syncbn

__all__ = ['ResNet', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, sync_bn=False, multi_grid=False, fpn=False):
        super().__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.fpn = fpn
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            norm_layer(64),
            nn.ReLU(inplace=True),
            conv3x3(64, self.inplanes),
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], multi_grid=multi_grid)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_outplanes(self):
        return self.inplanes

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        grids = [2, 2, 4] if multi_grid else [1] * blocks
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                        previous_dilation * grids[0], norm_layer)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation * grids[i], norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        if self.fpn:
            return [x1, x2, x3, x4]
        return [x3, x4]


def _load_torchvision_resnet50(model):
    try:
        from torchvision.models import resnet50 as tv_resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        src = tv_resnet50(weights=weights)
    except Exception:
        try:
            from torchvision.models import resnet50 as tv_resnet50
            src = tv_resnet50(pretrained=True)
        except Exception as e:
            print(f'[CSC-PA] torchvision ResNet50 weights not loaded: {e}')
            return
    state = src.state_dict()
    own = model.state_dict()
    mapped = {}
    for k, v in state.items():
        if k.startswith('conv1.') or k == 'conv1.weight':
            continue
        if k in own and own[k].shape == v.shape:
            mapped[k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print(f'[CSC-PA] partial ImageNet init: loaded {len(mapped)} keys, missing={len(missing)}, unexpected={len(unexpected)}')


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_torchvision_resnet50(model)
    return model
