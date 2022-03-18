import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from misc import pyutils, torchutils
from net.resnet50_cam import Net, MLP
from voc12.dataloader import get_img_path


def validate(cls_model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss')

    cls_model.eval()
    with torch.no_grad():
        for pack in data_loader:
            imgs = pack['img'].cuda(non_blocking=True)
            labels = pack['label'].cuda(non_blocking=True)
            x, _ = cls_model(imgs)
            loss = torch.nn.BCEWithLogitsLoss()(x, labels)
            val_loss_meter.add({'loss': loss.item()})

    cls_model.train()
    loss = val_loss_meter.pop('loss')
    print('Loss: {:.4f}'.format(loss))
    return loss


def train(config, config_path):
    seed = config['seed']
    train_list = config['train_list']
    val_list = config['val_list']
    voc12_root = config['voc12_root']
    cam_batch_size = config['cam_batch_size']
    cam_num_epoches = config['cam_num_epoches']
    cam_learning_rate = config['cam_learning_rate']
    cam_weight_decay = config['cam_weight_decay']
    cam_crop_size = config['cam_crop_size']
    model_root = config['model_root']
    cam_weights_name = config['cam_weights_name']
    cam_out_dir = config['cam_out_dir']
    agg_smooth_r = config['agg_smooth_r']
    num_workers = config['num_workers']
    scam_name = config['scam_name']
    scam_out_dir = config['scam_out_dir']
    laste_cam_weights_name = config['laste_cam_weights_name']
    cam_weight_path = os.path.join(model_root, cam_weights_name)
    laste_cam_weight_path = os.path.join(model_root, laste_cam_weights_name)
    scam_path = os.path.join(scam_out_dir, scam_name)

    if cam_crop_size == 512:
        resize_long = (320, 640)
    else:
        resize_long = (160, 320)
    print('resize long: {}'.format(resize_long))

    pyutils.seed_all(seed)

    # CAM generation dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list, voc12_root=voc12_root,
                                                                resize_long=resize_long, hor_flip=True,
                                                                crop_size=cam_crop_size, crop_method="random")
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=cam_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)

    max_step = (len(train_dataset) // cam_batch_size) * cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(val_list, voc12_root=voc12_root,
                                                              crop_size=cam_crop_size)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=cam_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    cls_model = Net()

    # load the pre-trained weights
    cls_model.load_state_dict(torch.load(cam_weight_path), strict=True)

    param_groups = cls_model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    cls_model.train()

    # Parallel
    cls_model = torch.nn.DataParallel(cls_model).cuda()

    avg_meter = pyutils.AverageMeter('loss', 'bce')
    timer = pyutils.Timer()
    # P(y|x, z)
    # generate Global CAMs
    os.system('python3 make_small_cam.py --config {}'.format(config_path))
    gcams = pyutils.sum_cams(cam_out_dir).cuda(non_blocking=True)
    np.save(scam_path + '0', gcams.cpu().numpy())
    # ===
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            imgs = pack['img'].cuda(non_blocking=True)
            labels = pack['label'].cuda(non_blocking=True)
            # Front Door Adjustment
            # P(z|x)
            x, _ = cls_model(imgs)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(2) * gcams
            # Aggregate for classification
            # agg(P(z|x) * sum(P(y|x, z) * P(x)))
            x = torchutils.mean_agg(x, r=agg_smooth_r)
            # Entropy loss for Content Adjustment
            bce_loss = torch.nn.BCEWithLogitsLoss()(x, labels)
            # Loss
            loss = bce_loss
            avg_meter.add({'loss': loss.item(), 'bce': bce_loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                validate(cls_model, val_data_loader)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'BCE:%.4f' % (avg_meter.pop('bce')),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
            else:
                timer.reset_stage()
        torch.save(cls_model.module.state_dict(),
                   laste_cam_weight_path + str(ep + 1))
        torch.save(cls_model.module.state_dict(), laste_cam_weight_path)
        torch.save(cls_model.module.state_dict(), cam_weight_path)
        # P(y|x, z)
        # generate Global CAMs
        os.system('python3 make_small_cam.py --config {}'.format(config_path))
        gcams = pyutils.sum_cams(cam_out_dir).cuda(non_blocking=True)
        np.save(scam_path + str(ep + 1), gcams.cpu().numpy())
        # ===
        # Reinit optimizer
        param_groups = cls_model.module.trainable_parameters()
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': cam_learning_rate,
                'weight_decay': cam_weight_decay},
            {'params': param_groups[1], 'lr': cam_learning_rate,
                'weight_decay': cam_weight_decay},
        ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)
        # ===
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/fdsi.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    start_weight_name = config['start_weight_name']
    cam_weights_name = config['cam_weights_name']
    model_root = config['model_root']

    start_weight_path = os.path.join(model_root, start_weight_name)
    cam_weight_path = os.path.join(model_root, cam_weights_name)
    copy_weights = 'cp {} {}'.format(start_weight_path, cam_weight_path)

    print(copy_weights)
    print(args.config)
    os.system(copy_weights)
    train(config, args.config)
