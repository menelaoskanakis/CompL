import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn import init
from torch.nn import functional as F
from vissl.models.additional_modules.pyramid_pooling import AtrousSpatialPyramidPoolingModule
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import os

try:
    from itertools import izip
except ImportError:  # python3.x
    izip = zip

affine_par = True  # Trainable Batchnorm for the pyramid pooling

MODEL_URLS= {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation=1, downsample=None, train_norm_layers=False,
                 reduction=16, norm_layers='BatchNorm2d'):
        super(Bottleneck, self).__init__()

        padding = dilation
        self.bnorm = getattr(torch.nn, norm_layers)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        # self.bn1 = self.bnorm(planes, affine=affine_par)
        if norm_layers == 'BatchNorm2d':
            self.bn1 = self.bnorm(planes, affine=affine_par)
        elif norm_layers == 'GroupNorm':
            self.bn1 = self.bnorm(32, planes, affine=affine_par)
        else:
            raise NotImplementedError

        for i in self.bn1.parameters():
            i.requires_grad = train_norm_layers

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation=dilation)
        # self.bn2 = self.bnorm(planes, affine=affine_par)
        if norm_layers == 'BatchNorm2d':
            self.bn2 = self.bnorm(planes, affine=affine_par)
        elif norm_layers == 'GroupNorm':
            self.bn2 = self.bnorm(32, planes, affine=affine_par)
        else:
            raise NotImplementedError

        for i in self.bn2.parameters():
            i.requires_grad = train_norm_layers

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = self.bnorm(planes * 4, affine=affine_par)
        if norm_layers == 'BatchNorm2d':
            self.bn3 = self.bnorm(planes * 4, affine=affine_par)
        elif norm_layers == 'GroupNorm':
            self.bn3 = self.bnorm(32, planes * 4, affine=affine_par)
        else:
            raise NotImplementedError

        for i in self.bn3.parameters():
            i.requires_grad = train_norm_layers

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, nInputChannels=3, pyramid_pooling="atrous-v3",
                 output_stride=16, decoder=True,
                 train_norm_layers=True, norm_layers='BatchNorm2d', **kwargs):

        super(ResNet, self).__init__()
        print("Constructing ResNet model...")
        print("Output stride: {}".format(output_stride))
        if norm_layers not in ['BatchNorm2d', 'GroupNorm']:
            raise NotImplementedError('Do not allow support for normalization named {}'.format(norm_layers))

        self.norm_layers = norm_layers
        self.train_norm_layers = train_norm_layers
        self.bnorm = getattr(torch.nn, norm_layers)

        v3_atrous_rates = [6, 12, 18]

        if output_stride == 8:
            dilations = (2, 4)
            strides = (2, 2, 2, 1, 1)
            v3_atrous_rates = [x * 2 for x in v3_atrous_rates]
        elif output_stride == 16:
            dilations = (1, 2)
            strides = (2, 2, 2, 2, 1)
        else:
            raise ValueError('Choose between output_stride 8 and 16')

        self.inplanes = 64
        self.pyramid_pooling = pyramid_pooling
        self.decoder = decoder

        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3,
                               bias=False)
        if norm_layers == 'BatchNorm2d':
            self.bn1 = self.bnorm(64, affine=affine_par)
        elif norm_layers == 'GroupNorm':
            self.bn1 = self.bnorm(32, 64, affine=affine_par)
        else:
            raise NotImplementedError

        for i in self.bn1.parameters():
            i.requires_grad = self.train_norm_layers

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[4])

        # Initialize weights
        self._initialize_weights()

        # Check if batchnorm parameters are trainable
        # self._verify_bnorm_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            if self.norm_layers == 'BatchNorm2d':
                kwargs = {"num_features": planes * block.expansion, "affine": affine_par}
            elif self.norm_layers == 'GroupNorm':
                kwargs = {"num_groups": 32, "num_channels": planes * block.expansion, "affine": affine_par}
            else:
                raise NotImplementedError

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.bnorm(**kwargs),
            )

            # Train batchnorm?
            for i in downsample._modules['1'].parameters():
                i.requires_grad = self.train_norm_layers

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            train_norm_layers=self.train_norm_layers, norm_layers=self.norm_layers))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                train_norm_layers=self.train_norm_layers, norm_layers=self.norm_layers))

        return nn.Sequential(*layers)

    def _verify_bnorm_params(self):
        verify_trainable = True
        a = 0
        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d):
                for y in x.parameters():
                    verify_trainable = (verify_trainable and y.requires_grad)
                a += isinstance(x, nn.BatchNorm2d)

        print("\nVerification: Trainable batchnorm parameters? Answer: {}\n".format(verify_trainable))
        print("bnorm layers: {}".format(a))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.decoder:
            x_low = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.layer5(x)

        if self.decoder:
            output = F.interpolate(output, size=(x_low.shape[2], x_low.shape[3]),
                              mode='bilinear', align_corners=False)
            x_low = self.low_level_reduce(x_low)
            output = torch.cat([output, x_low], dim=1)
            output = self.concat(output)
        return output


def get_state_dict(model_name, path=None):
    # Load checkpoint
    if model_name in MODEL_URLS:
        checkpoint = model_zoo.load_url(MODEL_URLS[model_name])
    elif 'resnet26' in model_name:
        if path is None:
            raise ValueError('Pretrained model path not specified')
        checkpoint = torch.load(os.path.join(path, '{}.pth'.format(model_name)),
                                map_location=lambda storage, loc: storage)['state_dict']
    else:
        raise NotImplementedError('Pretraining for {} not available'.format(model_name))

    # Handle DataParallel
    if 'module.' in list(checkpoint.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = checkpoint

    return new_state_dict


def resnet26(pretrained=None, nInputChannels=3, path=None,
             norm_layers='BatchNorm2d', **kwargs):
    """Constructs a ResNet-26 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], nInputChannels=nInputChannels, norm_layers=norm_layers, **kwargs)
    if pretrained:
        print('Loading resnet26 Imagenet')
        model_name = 'resnet26'

        if isinstance(pretrained, str):
            model_name += '_{}'.format(pretrained)

        if norm_layers == 'GroupNorm':
            model_name += '_gn'

        state_dict = get_state_dict(model_name, path)
        model.load_state_dict(state_dict, strict=False)
    else:
        print('Training from scratch')
    return model


# def resnet50(pretrained=False, nInputChannels=3, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], nInputChannels=nInputChannels, **kwargs)
#
#     if pretrained:
#         print('Loading resnet50 Imagenet')
#         model_name = 'resnet50'
#
#         state_dict = get_state_dict(model_name)
#         model.load_state_dict(state_dict, strict=False)
#     else:
#         print('Training from scratch')
#     return model
#
#
# def resnet101(pretrained=False, nInputChannels=3, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], nInputChannels=nInputChannels, **kwargs)
#
#     if pretrained:
#         print('Loading resnet101 Imagenet')
#         model_name = 'resnet101'
#
#         state_dict = get_state_dict(model_name)
#         model.load_state_dict(state_dict, strict=False)
#     else:
#         print('Training from scratch')
#     return model
