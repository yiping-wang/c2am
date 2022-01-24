import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader
from misc import pyutils, torchutils
from net.resnet50_cam import CAM    
from misc import imutils
import matplotlib.pyplot as plt


val_list = '/data/home/yipingwang/front_door_cam/voc12/val.txt'
voc12_root = '/data/home/yipingwang/data/VOCdevkit/VOC2012'
cam_weight_path = '/data/home/yipingwang/data/Models/Classification/resnet50.pth'

val_dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(val_list,
                                                          voc12_root=voc12_root,
                                                          scales=(1.0, 0.5, 1.5, 2.0))
data_loader = DataLoader(val_dataset, shuffle=False)

model = CAM().cuda()
model.eval()
model.load_state_dict(torch.load(cam_weight_path, map_location='cpu'))


data_iter = iter(data_loader)
pack = next(data_iter)
pack = next(data_iter)
pack = next(data_iter)
pack = next(data_iter)
pack = next(data_iter)

import time

start = time.time()

with torch.no_grad():
    img_name = pack['name'][0]
    imgs = pack['img']
    label = pack['label'][0]
    size = pack['size']

    strided_size = imutils.get_strided_size(size, 4)
    strided_up_size = imutils.get_strided_up_size(size, 16)

    outputs = [model(img[0].cuda(non_blocking=True)) for img in pack['img']]

    strided_cam = torch.sum(torch.stack(
        [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
         in outputs]), 0)

#     highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
#                                  mode='bilinear', align_corners=False) for o in outputs]
#     highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]]

    valid_cat = torch.nonzero(label)[:, 0]
    print(valid_cat)
    print(strided_cam.shape)

    strided_cam = strided_cam[valid_cat]
    strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

#     highres_cam = highres_cam[valid_cat]
#     highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

print("hello")
end = time.time()
print(end - start)
