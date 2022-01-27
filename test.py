import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import glob
import os
import numpy as np
from torch.utils.data import DataLoader
from misc import pyutils, torchutils, imutils
from net.resnet50_cam import CAMDualHeads, Net, CAM
import itertools
import operator
import collections


class IterateCAM:
    def __init__(self, cam_dir):
        self.cam_list = glob.glob(os.path.join(cam_dir, '*.npy'))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.cam_list):
            raise StopIteration
        c = np.load(self.cam_list[self.index], allow_pickle=True).item()
        self.index += 1
        return c['cam']


def sum_cams(cam_dir):
    itcam = IterateCAM(cam_dir)
    running_sum = itertools.accumulate(itcam)
    running_mean = map(operator.truediv, running_sum, itertools.count(1))
    return collections.deque(running_mean, maxlen=1)[0]


cam_out_dir = "/data/home/yipingwang/data/CAM"
os.system('python3 make_cam.py')  # generate CAMs
scams = sum_cams(cam_out_dir).cpu().numpy()
np.save('scam.npy', scams)