import torch.nn as nn
import torchvision.models as models


def resnet18_backbone():
    resnet18 = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
    num_features = resnet18.fc.in_features
    backbone = nn.Sequential(*list(resnet18.children())[:-1])
    return backbone, num_features

def resnet50_backbone():
    resnet50 = models.resnet50(weights=models.resnet.ResNet50_Weights.DEFAULT)
    num_features = resnet50.fc.in_features
    backbone = nn.Sequential(*list(resnet50.children())[:-1])
    return backbone, num_features

def vit_l_16_backbone():
    vit = models.vit_l_16(weights= models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    num_features = vit.heads.head.in_features  
    backbone = nn.Sequential(*list(vit.children())[:-1])
    return backbone, num_features


def custom_CNN():
    backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
    num_features = 128*7*14
    return backbone, num_features
