import re
import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
from torch.utils.data import DataLoader
from misc import pyutils, torchutils
from net.resnet50_cam import Net, MLP
from voc12.dataloader import get_img_path


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')
    model.eval()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img'].cuda(non_blocking=True)
            label = pack['label'].cuda(non_blocking=True)
            x, _ = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)
            val_loss_meter.add({'loss1': loss1.item()})
    model.train()
    vloss = val_loss_meter.pop('loss1')
    print('loss: %.4f' % vloss)
    return vloss


def train(config):
    seed = config['seed']
    train_list = config['train_list']
    val_list = config['val_list']
    voc12_root = config['voc12_root']
    cam_batch_size = config['cam_batch_size']
    cam_num_epoches = config['cam_num_epoches']
    cam_learning_rate = config['cam_learning_rate']
    sty_learning_rate = cam_learning_rate
    cam_weight_decay = config['cam_weight_decay']
    model_root = config['model_root']
    cam_weights_name = config['laste_cam_weights_name']
    num_workers = config['num_workers']
    cam_crop_size = config['cam_crop_size']
    alpha = config['alpha']
    cam_weight_path = os.path.join(model_root, cam_weights_name)
    pyutils.seed_all(seed)
    data_aug_fn = torchutils.get_simclr_pipeline_transform(size=cam_crop_size)

    model = Net()
    mlp = MLP()

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list,
                                                                voc12_root=voc12_root,
                                                                resize_long=(
                                                                    320, 640),
                                                                hor_flip=True,
                                                                crop_size=cam_crop_size,
                                                                crop_method="random")
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=cam_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)

    max_step = (len(train_dataset) // cam_batch_size) * cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(val_list,
                                                              voc12_root=voc12_root,
                                                              crop_size=cam_crop_size)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=cam_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': mlp.parameters(), 'lr': sty_learning_rate,
            'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    cls_model.train()
    mlp.train()

    cls_model = torch.nn.DataParallel(cls_model).cuda()
    mlp = torch.nn.DataParallel(mlp).cuda() if alpha > 0 else MLP()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()
    kl_loss = torch.tensor(0.)

    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            img = pack['img'].cuda(non_blocking=True)
            label = pack['label'].cuda(non_blocking=True)
            x, _ = model(img)
            if alpha > 0:
                # Style Intervention from Eq. 3 at 2010.07922
                names = pack['name']
                augs = torch.cat([torchutils.get_style_variants(
                    names, data_aug_fn, voc12_root, get_img_path) for _ in range(4)], dim=0)
                _, feats = cls_model(augs)
                projs = mlp(feats)
                norms = F.normalize(projs, dim=1)
                proj_l, proj_k, proj_q, proj_t = torch.split(
                    norms, split_size_or_sections=cam_batch_size, dim=0)
                score_lk = torch.matmul(proj_l, proj_k.T)
                score_qt = torch.matmul(proj_q, proj_t.T)
                logprob_lk = F.log_softmax(score_lk, dim=1)
                prob_qt = F.softmax(score_qt, dim=1)
                kl_loss = alpha * \
                    torch.nn.KLDivLoss(reduction='batchmean')(
                        logprob_lk, prob_qt)

            bce_loss = F.multilabel_soft_margin_loss(x, label)
            loss = bce_loss + kl_loss if alpha > 0 else bce_loss
            avg_meter.add(
                {'loss': loss.item(), 'bce': bce_loss.item(), 'kl': kl_loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'bce:%.4f' % (avg_meter.pop('bce')),
                      'kl:%.4f' % (avg_meter.pop('kl')),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                # validation
                validate(model, val_data_loader)
            else:
                timer.reset_stage()
        # empty cache
        torch.save(model.state_dict(), cam_weight_path)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', required=True)
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    train(config)
