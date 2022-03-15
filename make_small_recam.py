import torch
from torch import multiprocessing, cuda, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from net.resnet50_cam import ReCAM, Class_Predictor
import argparse
import numpy as np
import os

import voc12.dataloader
from misc import torchutils, pyutils

cudnn.enabled = True


def _work(process_id, model, recam_predictor, dataset, config):
    cam_out_dir = config['cam_out_dir']
    cam_square_shape = config['cam_square_shape']

    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False,
                             num_workers=1, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]

            outputs = [model.forward2(img[0].cuda(
                non_blocking=True), recam_predictor.classifier.weight) for img in pack['img']]  # b x 20 x w x h

            raw_outputs = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0),
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
    cam_weights_name = config['laste_cam_weights_name']
    recam_weights_name = config['laste_recam_weights_name']
    recam_weight_path = os.path.join(model_root, recam_weights_name)
    cam_weight_path = os.path.join(model_root, cam_weights_name)

    model = ReCAM()
    model.load_state_dict(torch.load(
        cam_weight_path, map_location='cpu'), strict=True)
    model.eval()

    recam_predictor = Class_Predictor(20, 2048)
    recam_predictor.load_state_dict(torch.load(
        recam_weight_path, map_location='cpu'), strict=True)
    recam_predictor.eval()

    n_gpus = torch.cuda.device_count() - 1

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(train_list,
                                                             voc12_root=voc12_root,
                                                             scales=cam_scales)

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(
        model, recam_predictor, dataset, config), join=True)
    print(']')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str, help='YAML config file path')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    run(config)
