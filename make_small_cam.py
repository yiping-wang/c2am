import torch
from torch import multiprocessing, cuda, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from net.resnet50_cam import CAM
import argparse
import numpy as np
import os

import voc12.dataloader
from misc import torchutils, imutils, pyutils

cudnn.enabled = True


def _work(process_id, model, dataset, config):
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
            print(len(pack['img']))
            print(pack['img'][0].shape)
            outputs = model(pack['img'][0][0].cuda(non_blocking=True))

            outputs = F.interpolate(outputs.unsqueeze(
                0), (cam_square_shape, cam_square_shape), mode='bilinear', align_corners=False)[0]
            outputs = outputs / (F.adaptive_max_pool2d(outputs, (1, 1)) + 1e-5)
            valid_cat = torch.nonzero(label)[:, 0]

            # save cams
            # TODO: might improve "raw_outputs" to higher res
            np.save(os.path.join(cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat,
                    "raw_outputs": outputs.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(config):
    train_list = config['train_list']
    voc12_root = config['voc12_root']
    model_root = config['model_root']
    cam_scales = config['cam_scales']
    cam_weights_name = config['cam_weights_name']
    model = CAM()
    model.load_state_dict(torch.load(os.path.join(
        model_root, cam_weights_name) + '.pth', map_location='cpu'), strict=True)
    model.eval()
    model.cuda()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(train_list,
                                                             voc12_root=voc12_root,
                                                             scales=cam_scales)

    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(
        model, dataset, config), join=True)
    print(']')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/front_door_v2.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    run(config)
