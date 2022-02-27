import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class MLP(nn.Module):
    def __init__(self, input_dim=20, output_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)
        return x

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
        # Idea: juse stage 4 feat map (connected) and classifier.weights (disconnted) to generate CAM within batch
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

        # step: 6600/ 6610 Loss:0.1215 BCE:0.0950 CAM:0.0950 imps:336.7 lr: 0.0000 etc:Sat Feb 26 13:58:16 2022
        # validating ... Loss: 0.2115 | BCE: 0.1115 | CAM: 0.1115
        # cam = F.conv2d(x.detach(), self.classifier.weight)
        # ===
        #
        cam = F.conv2d(x.detach(), self.classifier.weight.detach())
        cam = F.relu(cam)
        cam = cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5)
        return logit, cam

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
