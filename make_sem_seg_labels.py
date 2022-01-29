import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from net.resnet50_irn import EdgeDisplacement
import numpy as np
import importlib
import os
import imageio
from PIL import Image
import argparse
import voc12.dataloader
from misc import torchutils, indexing, pyutils, imutils

cudnn.enabled = True

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128,
           255, 255, 255]


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask.convert('RGB')


def _work(process_id, model, dataset, config):
    num_workers = config['num_workers']
    cam_out_dir = config['cam_out_dir']
    beta = config['beta']
    exp_times = config['exp_times']
    sem_seg_bg_thres = config['sem_seg_bg_thres']
    sem_seg_out_dir = config['sem_seg_out_dir']

    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=num_workers // n_gpus, pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            orig_img_size = np.asarray(pack['size'])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True))

            cam_dict = np.load(cam_out_dir + '/' + img_name +
                               '.npy', allow_pickle=True).item()

            cams = cam_dict['cam'][cam_dict['keys']]
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(
                cam_downsized_values, edge, beta=beta, exp_times=exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[
                ..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = imutils.crf_inference_label(
                pack['org_img'].squeeze(), rw_pred, n_labels=keys.shape[0])
            rw_pred = keys[rw_pred]

            # rw_pred = colorize_mask(rw_pred.astype(np.uint8))
            imageio.imsave(os.path.join(sem_seg_out_dir,
                           img_name + '.png'), rw_pred)

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(config):
    irn_weights_name = config['irn_weights_name']
    infer_list = config['infer_list']
    voc12_root = config['voc12_root']

    model = EdgeDisplacement()
    model.load_state_dict(torch.load(irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetFD(infer_list,
                                                            voc12_root=voc12_root,
                                                            scales=(
                                                                1.0,),
                                                            size_h=0,
                                                            size_w=0,
                                                            hor_flip=False,
                                                            crop_method="none")
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(
        model, dataset, config), join=True)
    print("]")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/ir_net.yml')
    args = parser.parse_args()
    # if torch.cuda.is_available():
    #     device = pyutils.set_gpus(n_gpus=1)
    # else:
    #     device = 'cpu'
    config = pyutils.parse_config(args.config)
    run(config)
