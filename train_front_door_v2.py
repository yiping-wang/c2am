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


def validate(cls_model, data_loader, cam_batch_size, logexpsum_r, cam_out_dir):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    # P(y|x, z)
    # os.system('python3 make_cam.py')  # generate CAMs
    scams = sum_cams(cam_out_dir)
    cls_model.eval()
    with torch.no_grad():
        for pack in data_loader:
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            # P(z|x)
            x = cls_model(imgs[:, 0])
            # x = F.softmax(x, dim=1)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(
                2) * scams.cuda(device, non_blocking=True)
            # loss
            x = torchutils.lse_agg(x, r=logexpsum_r)
            # x = x / (torch.sum(x, dim=1).unsqueeze(1) + 1e-5)
            loss1 = F.multilabel_soft_margin_loss(x, labels)
            val_loss_meter.add({'loss1': loss1.item()})

    cls_model.train()
    vloss = val_loss_meter.pop('loss1')
    print('loss: %.4f' % vloss)
    return vloss, scams


def train(config, device):
    seed = config['seed']
    train_list = config['train_list']
    val_list = config['val_list']
    voc12_root = config['voc12_root']
    cam_batch_size = config['cam_batch_size']
    cam_num_epoches = config['cam_num_epoches']
    cam_learning_rate = config['cam_learning_rate']
    cam_weight_decay = config['cam_weight_decay']
    model_root = config['model_root']
    cam_weights_name = config['cam_weights_name']
    cam_out_dir = config['cam_out_dir']
    image_size_height = config['image_size_height']
    image_size_width = config['image_size_width']
    logexpsum_r = config['logexpsum_r']

    num_workers = 1
    cam_weight_path = os.path.join(model_root, cam_weights_name + '.pth')
    pyutils.seed_all(seed)

    cls_model = Net().cuda(device)
    # load pre-trained classification network
    cls_model.load_state_dict(torch.load(cam_weight_path, map_location=device))

    # CAM generation dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDatasetFD(train_list,
                                                                  voc12_root=voc12_root,
                                                                  scales=(
                                                                      1.0,),
                                                                  size_h=image_size_height,
                                                                  size_w=image_size_width)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=cam_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)

    max_step = (len(train_dataset) // cam_batch_size) * cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDatasetFD(val_list,
                                                                voc12_root=voc12_root,
                                                                scales=(
                                                                    1.0,),
                                                                size_h=image_size_height,
                                                                size_w=image_size_width,
                                                                hor_flip=False,
                                                                crop_method="none")
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=cam_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    param_groups = cls_model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    # model = torch.nn.DataParallel(model).cuda(device)
    cls_model = cls_model.cuda(device)
    cls_model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    min_loss = float('inf')
    # P(y|x, z)
    os.system('python3 make_cam.py')  # generate CAMs
    scams = sum_cams(cam_out_dir)
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            # P(z|x)
            x = cls_model(imgs[:, 0])
            # x = F.softmax(x, dim=1)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(
                2) * scams.cuda(device, non_blocking=True)
            # loss
            x = torchutils.lse_agg(x, r=logexpsum_r)
            # x = x / (torch.sum(x, dim=1).unsqueeze(1) + 1e-5)
            loss = F.multilabel_soft_margin_loss(x, labels)
            # loss = nlll(x, labels)
            avg_meter.add({'loss1': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f' % (avg_meter.pop('loss1')),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                # validation
                vloss, vscams = validate(cls_model, val_data_loader,
                                         cam_batch_size, logexpsum_r, cam_out_dir)
                if vloss < min_loss:
                    torch.save(cls_model.state_dict(), cam_weight_path)
                    min_loss = vloss
                    scams = vscams

                timer.reset_stage()
        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/front_door_v2.yml')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = pyutils.set_gpus(n_gpus=1)
    else:
        device = 'cpu'
    config = pyutils.parse_config(args.config)
    train(config, device)
