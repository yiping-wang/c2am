import torch.nn as nn
import torch
import torch.nn.functional as F
from misc import torchutils, pyutils
from net import resnet50
import os


class MLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=128):
        super().__init__()
        self.projection_head = nn.Sequential()
        self.projection_head.add_module('W1', nn.Linear(
            input_dim, input_dim))
        self.projection_head.add_module('BN1', nn.BatchNorm1d(input_dim))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            input_dim, output_dim))
        self.projection_head.add_module('BN2', nn.BatchNorm1d(output_dim))

    def forward(self, x):
        return self.projection_head(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(
            pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x).detach()
        x = self.stage3(x)
        x = self.stage4(x)
        feat = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(feat)
        x = x.view(-1, 20)
        return x, feat.squeeze()

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.resnet50 = resnet50.resnet50(
            pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class IntervenedCAM(nn.Module):
    def __init__(self, cam_dir, cam_square):
        super(IntervenedCAM, self).__init__()
        self.resnet50 = resnet50.resnet50(
            pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)

        self.backbone = nn.ModuleList(
            [self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

        cam_is_empty = len(os.listdir(cam_dir)) == 0
        if cam_is_empty:
            self.scams = torch.ones(20, cam_square, cam_square)
        else:
            self.scams = pyutils.sum_cams(self.cam_dir)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)

        x = x * F.interpolate(self.scams.unsqueeze(0), size=x.shape[2:],
                              mode='bilinear', align_corners=False).cuda(non_blocking=True)

        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
