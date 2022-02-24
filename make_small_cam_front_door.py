import torch
from torch import multiprocessing, cuda, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from net.resnet50_cam import Net
import argparse
import numpy as np
import os

import voc12.dataloader
from misc import torchutils, pyutils

cudnn.enabled = True


def _work(process_id, model, dataset, prev_scams, config):
    cam_out_dir = config['cam_out_dir']
    cam_square_shape = config['cam_square_shape']
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False,
                             num_workers=1, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        prev_scams = prev_scams.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]

            # if prev_scams:
            outputs = [model(img[0][0].unsqueeze(0).cuda(non_blocking=True))
                       for img in pack['img']]
            outputs = [l.unsqueeze(2).unsqueeze(
                2) * prev_scams for l in outputs]
            # else:
            #     outputs = [model(img[0].cuda(non_blocking=True))
            #                for img in pack['img']]

            # raw_outputs = torch.sum(torch.stack(
            #     [F.interpolate(torch.unsqueeze(o, 0),
            #                    (cam_square_shape, cam_square_shape), mode='bilinear', align_corners=False)[0]
            #         for o in outputs]), 0)
            raw_outputs = torch.sum(torch.stack(
                [F.interpolate(o,
                               (cam_square_shape, cam_square_shape), mode='bilinear', align_corners=False)[0]
                    for o in outputs]), 0)

            raw_outputs = raw_outputs / \
                (F.adaptive_max_pool2d(raw_outputs, (1, 1)) + 1e-5)
            valid_cat = torch.nonzero(label)[:, 0]

            # save cams
            np.save(os.path.join(cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat,
                    "raw_outputs": raw_outputs.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(config):
    train_list = config['train_list']
    voc12_root = config['voc12_root']
    model_root = config['model_root']
    cam_scales = config['cam_scales']
    cam_weights_name = config['cam_weights_name']
    scam_out_dir = config['scam_out_dir']
    scam_name = config['scam_name']
    scam_path = os.path.join(scam_out_dir, scam_name)

    if os.path.exists(scam_path):
        prev_scams = torch.from_numpy(np.load(scam_path))
    else:
        prev_scams = None

    model = Net()
    model.load_state_dict(torch.load(os.path.join(
        model_root, cam_weights_name), map_location='cpu'), strict=True)
    model.eval()
    model.cuda()

    n_gpus = torch.cuda.device_count() - 1

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(train_list,
                                                             voc12_root=voc12_root,
                                                             scales=cam_scales)

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(
        model, dataset, prev_scams, config), join=True)
    print(']')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/front_door.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    run(config)