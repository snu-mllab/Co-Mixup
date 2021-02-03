import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ResNeXtBottleneck(nn.Module):
    expansion = 4
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self,
                 inplanes,
                 planes,
                 cardinality,
                 base_width,
                 stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        self.conv_reduce = nn.Conv2d(inplanes,
                                     D * C,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.bn_reduce = nn.BatchNorm2d(D * C)

        self.conv_conv = nn.Conv2d(D * C,
                                   D * C,
                                   kernel_size=3,
                                   stride=stride,
                                   padding=1,
                                   groups=cardinality,
                                   bias=False)
        self.bn = nn.BatchNorm2d(D * C)

        self.conv_expand = nn.Conv2d(D * C,
                                     planes * 4,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, block, depth, cardinality, base_width, num_classes,
                 dropout):
        super(CifarResNeXt, self).__init__()
        self.num_classes = num_classes
        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth -
                2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.cardinality, self.base_width,
                  stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, self.cardinality,
                      self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.conv_1_3x3(out)
        out = F.relu(self.bn_1(out), inplace=True)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)

        return out


def resnext29_4_24(num_classes=10, dropout=False, stride=None):
    """Constructs a ResNeXt-29, 4*24d model"""
    model = CifarResNeXt(ResNeXtBottleneck, 29, 4, 24, num_classes, dropout)
    return model
