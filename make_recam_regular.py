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
from misc import torchutils, imutils, pyutils

cudnn.enabled = True


def _work(process_id, model, recam_predictor, dataset, config):
    cam_out_dir = config['cam_out_dir']
    num_workers = config['num_workers']
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False,
                             num_workers=num_workers//n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        recam_predictor.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            valid_cat = torch.nonzero(label)[:, 0]

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [model.forward2(img[0].cuda(
                non_blocking=True), recam_predictor.classifier.weight) for img in pack['img']]  # b x 20 x w x h

            strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(
                o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)
            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = [F.interpolate(torch.unsqueeze(
                o, 1), strided_up_size, mode='bilinear', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[
                :, 0, :size[0], :size[1]]

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(os.path.join(cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat,
                     "cam": strided_cam.cpu(),
                     "high_res": highres_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(config):
    train_list = config['train_list']
    voc12_root = config['voc12_root']
    model_root = config['model_root']
    cam_scales = config['cam_scales']
    cam_weights_name = config['laste_cam_weights_name']
    recam_weights_name = config['laste_recam_weight_path']
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

    n_gpus = torch.cuda.device_count()

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
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/front_door_style_intervention.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    run(config)
