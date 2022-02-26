from torch.nn import functional as F
from cmath import log
import voc12.dataloader
import argparse
import torch
import os
from torch.utils.data import DataLoader
from misc import pyutils, torchutils
from net.resnet50_cam import NetDualHeads


def validate(cls_model, data_loader, device):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss', 'bce', 'cam')

    cls_model.eval()

    with torch.no_grad():
        for pack in data_loader:
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            logit, cam = cls_model(imgs)
            cam = torch.mean(cam, dim=0)
            # logit = F.softmax(logit, dim=1)
            mix_loss = torch.nn.BCEWithLogitsLoss()(
                torchutils.mean_agg(logit.unsqueeze(2).unsqueeze(2) * cam, r=1), labels)
            bce_loss = torch.nn.BCEWithLogitsLoss()(logit, labels)
            cam_loss = bce_loss
            loss = mix_loss
            val_loss_meter.add(
                {'loss': mix_loss.item(), 'bce': bce_loss.item(), 'cam': cam_loss.item()})

    cls_model.train()

    loss = val_loss_meter.pop('loss')
    bce = val_loss_meter.pop('bce')
    cam = val_loss_meter.pop('cam')
    print('Loss: {:.4f} | BCE: {:.4f} | CAM: {:.4f}'.format(loss, bce, cam))
    return loss


def train(config, device):
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
    cam_weight_path = os.path.join(model_root, cam_weights_name)

    pyutils.seed_all(seed)

    # CAM generation dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list, voc12_root=voc12_root,
                                                                resize_long=(160, 320), hor_flip=True,
                                                                crop_size=cam_crop_size, crop_method="random")
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=cam_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)

    max_step = (len(train_dataset) // cam_batch_size) * cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(val_list, voc12_root=voc12_root,
                                                              crop_size=512)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=cam_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    cls_model = NetDualHeads().cuda(device)

    param_groups = cls_model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    cls_model.train()

    avg_meter = pyutils.AverageMeter('loss', 'bce', 'cam')
    timer = pyutils.Timer()

    min_loss = float('inf')

    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)

            logit, cam = cls_model(imgs)
            cam = torch.mean(cam, dim=0)
            # logit = F.softmax(logit, dim=1)
            mix_loss = torch.nn.BCEWithLogitsLoss()(
                torchutils.mean_agg(logit.unsqueeze(2).unsqueeze(2) * cam, r=1), labels)
            bce_loss = torch.nn.BCEWithLogitsLoss()(logit, labels)
            cam_loss = bce_loss
            loss = mix_loss
            avg_meter.add(
                {'loss': mix_loss.item(), 'bce': bce_loss.item(), 'cam': cam_loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'BCE:%.4f' % (avg_meter.pop('bce')),
                      'CAM:%.4f' % (avg_meter.pop('cam')),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                # validation
                vloss = validate(cls_model, val_data_loader, device)
                if vloss < min_loss:
                    torch.save(cls_model.state_dict(), cam_weight_path)
                    min_loss = vloss

                timer.reset_stage()
        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/mix.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    device = torch.device('cuda:7')
    train(config, device)
    # os.system('python3 make_cam.py --config ./cfg/front_door.yml')
