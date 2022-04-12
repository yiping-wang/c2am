import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from misc import pyutils, torchutils
from net.resnet50_cam import Net_CAM_Feature, Class_Predictor


def validate(cls_model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss', 'bce')

    cls_model.eval()
    with torch.no_grad():
        for pack in data_loader:
            imgs = pack['img'].cuda(non_blocking=True)
            labels = pack['label'].cuda(non_blocking=True)
            x, _, _ = cls_model(imgs)
            bce_loss = torch.nn.BCEWithLogitsLoss()(x, labels)
            loss = bce_loss
            val_loss_meter.add(
                {'loss': loss.item(), 'bce': bce_loss.item()})

    cls_model.train()
    loss = val_loss_meter.pop('loss')
    bce = val_loss_meter.pop('bce')
    print('Loss: {:.4f} | BCE: {:.4f}'.format(loss, bce))
    return loss


def train(config):
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
    num_workers = config['num_workers']
    recam_loss_weight = config['recam_loss_weight']
    laste_cam_weights_name = config['laste_cam_weights_name']
    laste_recam_weights_name = config['laste_recam_weights_name']
    cam_weight_path = os.path.join(model_root, cam_weights_name)
    laste_cam_weight_path = os.path.join(model_root, laste_cam_weights_name)
    laste_recam_weight_path = os.path.join(
        model_root, laste_recam_weights_name)

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

    cls_model = Net_CAM_Feature()
    recam_predictor = Class_Predictor(20, 2048)

    # load the pre-trained weights
    cls_model.load_state_dict(torch.load(cam_weight_path), strict=True)

    param_groups = cls_model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 0.1 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': 0.1 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': recam_predictor.parameters(), 'lr': cam_learning_rate,
         'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    cls_model.train()
    recam_predictor.train()

    cls_model = torch.nn.DataParallel(cls_model).cuda()
    recam_predictor = torch.nn.DataParallel(recam_predictor).cuda()

    avg_meter = pyutils.AverageMeter('loss', 'bce', 'sce')
    timer = pyutils.Timer()

    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            imgs = pack['img'].cuda(non_blocking=True)
            labels = pack['label'].cuda(non_blocking=True)

            x, cam_feats, _ = cls_model(imgs)
            bce_loss = torch.nn.BCEWithLogitsLoss()(x * 0.2, labels)
            sce_loss, _ = recam_predictor(cam_feats, labels)

            sce_loss = sce_loss.mean()
            loss = bce_loss + recam_loss_weight * sce_loss

            avg_meter.add(
                {'loss': loss.item(), 'bce': bce_loss.item(), 'sce': sce_loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'BCE:%.4f' % (avg_meter.pop('bce')),
                      'SCE:%.4f' % (avg_meter.pop('sce')),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                # validation
                validate(cls_model, val_data_loader)
            else:
                timer.reset_stage()

        # empty cache
        torch.save(cls_model.module.state_dict(), laste_cam_weight_path)
        torch.save(recam_predictor.module.state_dict(),
                   laste_recam_weight_path)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', required=True)
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)

    model_root = config['model_root']
    start_weight_name = config['start_weight_name']
    cam_weights_name = config['cam_weights_name']
    start_weight_path = os.path.join(model_root, start_weight_name)
    cam_weight_path = os.path.join(model_root, cam_weights_name)

    copy_weights = 'cp {} {}'.format(start_weight_path, cam_weight_path)
    print(copy_weights)
    os.system(copy_weights)
    train(config)
