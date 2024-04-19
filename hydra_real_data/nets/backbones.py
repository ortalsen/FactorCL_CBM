import torch.nn as nn
import torchvision.models as models


def resnet18_backbone(pretrained=True):
    resnet18 = models.resnet18(pretrained=pretrained)
    num_features = resnet18.fc.in_features
    backbone = nn.Sequential(*list(resnet18.children())[:-1])
    return backbone, num_features

def resnet50_backbone(pretrained=True):
    resnet50 = models.resnet50(pretrained=pretrained)
    num_features = resnet50.fc.in_features
    backbone = nn.Sequential(*list(resnet50.children())[:-1])
    return backbone, num_features