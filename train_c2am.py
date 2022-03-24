import torch.nn.functional as F
import numpy as np
import voc12.dataloader
import argparse
import torch
import os
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
    agg_smooth_r = config['agg_smooth_r']
    num_workers = config['num_workers']
    scam_name = config['scam_name']
    alpha = config['alpha']
    cam_out_dir = config['cam_out_dir']
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
    data_aug_fn = torchutils.get_simclr_pipeline_transform(size=cam_crop_size)

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
    mlp = MLP()

    # load the pre-trained weights
    cls_model.load_state_dict(torch.load(cam_weight_path), strict=True)

    param_groups = cls_model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': param_groups[1], 'lr': 10 * cam_learning_rate,
            'weight_decay': cam_weight_decay},
        {'params': mlp.parameters(), 'lr': 0.01,
            'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    cls_model.train()
    mlp.train()

    # Parallel
    cls_model = torch.nn.DataParallel(cls_model).cuda()
    mlp = torch.nn.DataParallel(mlp).cuda() if alpha > 0 else MLP()

    avg_meter = pyutils.AverageMeter('loss', 'bce', 'kl')
    timer = pyutils.Timer()
    # P(y|x, z)
    # generate Global CAMs
    # os.system('python3 make_square_cam.py --config {}'.format(config_path))
    # global_cams = pyutils.sum_cams(cam_out_dir).cuda(non_blocking=True)
    # np.save(scam_path, global_cams.cpu().numpy())
    global_cams = torch.from_numpy(np.load(os.path.join(
        scam_out_dir, 'global_cam.npy'))).cuda(non_blocking=True)
    # global_cams = torch.from_numpy(
    #     np.load(os.path.join(scam_out_dir, 'global_cam_01.npy')))
    # global_cams = (global_cams - global_cams.min()) / \
    #     (global_cams.max() - global_cams.min())
    # global_cams = global_cams.cuda(non_blocking=True)
    # ===
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            names = pack['name']
            imgs = pack['img'].cuda(non_blocking=True)
            labels = pack['label'].cuda(non_blocking=True)
            # Front Door Adjustment
            # P(z|x)
            x, _ = cls_model(imgs)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(2) * global_cams
            # Aggregate for classification
            # agg(P(z|x) * sum(P(y|x, z) * P(x)))
            x = torchutils.mean_agg(x, r=agg_smooth_r)
            # Entropy loss for Content Adjustment
            bce_loss = torch.nn.BCEWithLogitsLoss()(x, labels)
            kl_loss = torch.tensor(0.).cuda(
            ) if alpha > 0 else torch.tensor(0.)
            # Style Intervention from Eq. 3 at 2010.07922
            if alpha > 0:
                augs = torch.cat([torchutils.get_style_variants(
                    names, data_aug_fn, voc12_root, get_img_path) for _ in range(4)], dim=0)
                _, feats = cls_model(augs)
                projs = mlp(feats)
                norms = F.normalize(projs, dim=1)
                proj_l, proj_k, proj_q, proj_t = torch.split(
                    norms, split_size_or_sections=cam_batch_size, dim=0)
                # cosine similarity of each feature in l to all features in k
                # each row of score_lk is an image in k and its similarity to all features in l
                # each col of score_lk is an image in l and its similarity to all features in k
                # applies the same to another set of augmentations qt
                # then minimizes the KL-divergence of kl and qt
                score_lk = torch.matmul(proj_l, proj_k.T)
                score_qt = torch.matmul(proj_q, proj_t.T)
                logprob_lk = F.log_softmax(score_lk, dim=1)
                prob_qt = F.softmax(score_qt, dim=1)
                # KL-divergence loss for Style Intervention
                kl_loss = alpha * \
                    torch.nn.KLDivLoss(reduction='batchmean')(
                        logprob_lk, prob_qt)
            # Loss
            loss = bce_loss + kl_loss
            avg_meter.add(
                {'loss': loss.item(), 'bce': bce_loss.item(), 'kl': kl_loss.item()})
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Progress
            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'BCE:%.4f' % (avg_meter.pop('bce')),
                      'KL:%.4f' % (avg_meter.pop('kl')),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                validate(cls_model, val_data_loader)
            else:
                timer.reset_stage()
        # Empty cache
        torch.save(cls_model.module.state_dict(), laste_cam_weight_path)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', required=True)
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
