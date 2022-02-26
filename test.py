import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from misc import pyutils, torchutils, imutils
from net.resnet50_cam import Net, CAM
from torch import nn
from torchvision.transforms import transforms
from voc12.dataloader import get_img_path

import matplotlib.pyplot as plt
from PIL import Image

train_list = '/Users/Andy/Projects/front_door_cam/voc12/val.txt'
voc12_root = '/Users/Andy/Projects/dataset/VOCdevkit/VOC2012'

train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list, voc12_root=voc12_root,
                                                            resize_long=(160, 320), hor_flip=True,
                                                            crop_size=256, crop_method="random")
train_data_loader = DataLoader(train_dataset,
                               batch_size=2,
                               shuffle=False,
                               num_workers=1,
                               pin_memory=True,
                               drop_last=True)

from net import resnet50

class NetDualHeads(nn.Module):
    def __init__(self):
        super(NetDualHeads, self).__init__()

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

        logit = torchutils.gap2d(x, keepdims=True)
        logit = self.classifier(logit)
        logit = logit.view(-1, 20)

        cam = F.conv2d(x, self.classifier.weight.detach())
        cam = F.relu(cam)
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
        return cam, logit

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


dm = NetDualHeads()

for pack in train_data_loader:
    imgs = pack['img']
    print(imgs.shape)
    labels = pack['label']
    cam, logit = dm(imgs)
    print(cam.shape)
    print(logit.shape)