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


def concat(names, aug_fn, voc12_root):
    ims = []
    for n in names:
        im = aug_fn(Image.open(get_img_path(n, voc12_root)
                               ).convert('RGB')).unsqueeze(0)
        ims += [im]
    return torch.cat(ims, dim=0)


def validate(cls_model, mlp, data_loader, logexpsum_r, cam_out_dir, data_aug_fn, voc12_root, alpha):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    # P(y|x, z)
    # generate CAMs
    os.system(
        'python3 make_small_cam.py --config ./cfg/front_door_style_intervention.yml')
    scams = pyutils.sum_cams(cam_out_dir).cuda(device, non_blocking=True)
    cls_model.eval()
    mlp.eval()
    bce_loss = torch.nn.BCELoss()

    with torch.no_grad():
        for pack in data_loader:
            names = pack['names']
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            # P(z|x)
            x = cls_model(imgs)
            x = F.softmax(x, dim=1)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(2) * scams
            # loss
            x = torchutils.mean_agg(x, r=logexpsum_r)
            # Style Intervention
            augs = [concat(names, data_aug_fn, voc12_root) for _ in range(4)]
            feats = [cls_model(aug) for aug in augs]
            projs = [mlp(feat) for feat in feats]
            proj_l, proj_k, proj_q, proj_t = projs
            score_lk = torch.matmul(proj_l, proj_k.permute(1, 0))
            score_qt = torch.matmul(proj_q, proj_t.permute(1, 0))
            logprob_lk = torch.nn.functional.log_softmax(score_lk, dim=1)
            logprob_qt = torch.nn.functional.log_softmax(score_qt, dim=1)
            kl_loss = torch.nn.KLDivLoss(
                log_target=True, reduction='batchmean')(logprob_lk, logprob_qt)
            # Loss
            # loss = F.multilabel_soft_margin_loss(x, labels)
            loss = bce_loss(x, labels) + alpha * kl_loss
            val_loss_meter.add({'loss1': loss.item()})

    cls_model.train()
    mlp.train()
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
    cam_crop_size = config['cam_crop_size']
    model_root = config['model_root']
    cam_weights_name = config['cam_weights_name']
    cam_out_dir = config['cam_out_dir']
    logexpsum_r = config['logexpsum_r']
    num_workers = config['num_workers']
    scam_name = config['scam_name']
    alpha = config['alpha']
    scam_out_dir = config['scam_out_dir']
    cam_weight_path = os.path.join(model_root, cam_weights_name)
    scam_path = os.path.join(scam_out_dir, scam_name)

    pyutils.seed_all(seed)

    data_aug_fn = torchutils.get_simclr_pipeline_transform(size=cam_crop_size)

    # CAM generation dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list, voc12_root=voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
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

    cls_model = Net().cuda(device)
    mlp = MLP().cuda(device)

    param_groups = cls_model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': mlp.parameters(), 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    cls_model.train()
    mlp.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    bce_loss = torch.nn.BCELoss()
    min_loss = float('inf')
    # P(y|x, z)
    # generate CAMs
    os.system(
        'python3 make_small_cam.py --config ./cfg/front_door_style_intervention.yml')
    scams = pyutils.sum_cams(cam_out_dir).cuda(device, non_blocking=True)
    np.save(scam_path, scams.cpu().numpy())
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            names = pack['name']
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            # P(z|x)
            x = cls_model(imgs)
            x = F.softmax(x, dim=1)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(2) * scams
            # Aggregation
            x = torchutils.mean_agg(x, r=logexpsum_r)
            # Style Intervention
            augs = [concat(names, data_aug_fn, voc12_root) for _ in range(4)]
            feats = [cls_model(aug) for aug in augs]
            projs = [mlp(feat) for feat in feats]
            proj_l, proj_k, proj_q, proj_t = projs
            score_lk = torch.matmul(proj_l, proj_k.permute(1, 0))
            score_qt = torch.matmul(proj_q, proj_t.permute(1, 0))
            logprob_lk = torch.nn.functional.log_softmax(score_lk, dim=1)
            logprob_qt = torch.nn.functional.log_softmax(score_qt, dim=1)
            kl_loss = torch.nn.KLDivLoss(
                log_target=True, reduction='batchmean')(logprob_lk, logprob_qt)
            # Loss
            # loss = F.multilabel_soft_margin_loss(x, labels)
            loss = bce_loss(x, labels) + alpha * kl_loss
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
                vloss, vscams = validate(
                    cls_model, mlp, val_data_loader, logexpsum_r, cam_out_dir, data_aug_fn, voc12_root, alpha)
                if vloss < min_loss:
                    torch.save(cls_model.state_dict(), cam_weight_path)
                    min_loss = vloss
                    scams = vscams
                    np.save(scam_path, scams.cpu().numpy())

                timer.reset_stage()
        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/front_door_style_intervention.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    device = torch.device('cuda:7')
    train(config, device)
    # os.system('python3 make_cam.py --config ./cfg/front_door.yml')
