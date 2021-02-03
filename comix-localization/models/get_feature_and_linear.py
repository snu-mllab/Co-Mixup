import torch.nn as nn


def get_feature_and_linear_resnet50(model: nn.Module):
    m = model
    feature = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1,
                            m.layer2, m.layer3, m.layer4)
    linear = m.fc
    factor = 32
    return feature, linear, factor


def get_feature_and_linear_preactresnet18(model: nn.Module):
    m = model
    feature = nn.Sequential(m.conv1, m.layer1, m.layer2, m.layer3, m.layer4)
    linear = m.linear
    factor = 16
    return feature, linear, factor
