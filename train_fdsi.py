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


def concat(names, aug_fn, voc12_root, device):
    return torch.cat([aug_fn(Image.open(get_img_path(n, voc12_root)).convert('RGB')).unsqueeze(0) for n in names], dim=0).cuda(device, non_blocking=True)


def validate(cls_model, mlp, data_loader, agg_smooth_r, data_aug_fn, voc12_root, alpha, device, scams):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss', 'bce', 'kl')

    cls_model.eval()
    mlp.eval()
    with torch.no_grad():
        for pack in data_loader:
            names = pack['name']
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            x, _ = cls_model(imgs)
            x = x.unsqueeze(2).unsqueeze(2) * scams
            x = torchutils.mean_agg(x, r=agg_smooth_r)
            augs = [concat(names, data_aug_fn, voc12_root, device)
                    for _ in range(4)]
            feats = [cls_model(aug)[1] for aug in augs]
            projs = [mlp(feat) for feat in feats]
            norms = [F.normalize(proj, dim=1) for proj in projs]
            proj_l, proj_k, proj_q, proj_t = norms
            score_lk = torch.matmul(proj_l, proj_k.T)
            score_qt = torch.matmul(proj_q, proj_t.T)
            logprob_lk = F.log_softmax(score_lk, dim=1)
            prob_qt = F.softmax(score_qt, dim=1)
            kl_loss = alpha * \
                torch.nn.KLDivLoss(reduction='batchmean')(logprob_lk, prob_qt)
            bce_loss = torch.nn.BCEWithLogitsLoss()(x, labels)
            loss = bce_loss + kl_loss
            val_loss_meter.add(
                {'loss': loss.item(), 'bce': bce_loss.item(), 'kl': kl_loss.item()})

    cls_model.train()
    mlp.train()

    loss = val_loss_meter.pop('loss')
    bce = val_loss_meter.pop('bce')
    kl = val_loss_meter.pop('kl')
    print('Loss: {:.4f} | BCE: {:.4f} | KL: {:.4f}'.format(loss, bce, kl))
    return loss, bce, kl


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
    agg_smooth_r = config['agg_smooth_r']
    num_workers = config['num_workers']
    scam_name = config['scam_name']
    alpha = config['alpha']
    scam_out_dir = config['scam_out_dir']
    cam_weight_path = os.path.join(model_root, cam_weights_name)
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

    cls_model = Net().cuda(device)
    mlp = MLP().cuda(device)

    # load the pre-trained weights
    cls_model.load_state_dict(torch.load(os.path.join(
        model_root, cam_weights_name)), strict=True)

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

    avg_meter = pyutils.AverageMeter('loss', 'bce', 'kl')
    timer = pyutils.Timer()

    # P(y|x, z)
    # generate CAMs
    # Using the pre-trained weights
    os.system('python3 make_small_cam.py --config ./cfg/fdsi.yml')
    scams = pyutils.sum_cams(cam_out_dir).cuda(device, non_blocking=True)
    np.save(scam_path, scams.cpu().numpy())
    # ===
    min_loss = float('inf')
    min_bce = float('inf')
    min_kl = float('inf')
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            names = pack['name']
            imgs = pack['img'].cuda(device, non_blocking=True)
            labels = pack['label'].cuda(device, non_blocking=True)
            # Front Door Adjustment
            # P(z|x)
            x, _ = cls_model(imgs)
            # P(y|do(x))
            x = x.unsqueeze(2).unsqueeze(2) * scams
            # Aggregate for classification
            # agg(P(z|x) * sum(P(y|x, z) * P(x)))
            x = torchutils.mean_agg(x, r=agg_smooth_r)
            # Style Intervention from Eq. 3 at 2010.07922
            augs = [concat(names, data_aug_fn, voc12_root, device)
                    for _ in range(4)]
            feats = [cls_model(aug)[1] for aug in augs]
            projs = [mlp(feat) for feat in feats]
            norms = [F.normalize(proj, dim=1) for proj in projs]
            proj_l, proj_k, proj_q, proj_t = norms
            # cosine similarity of each feature in l to all features in k
            # each row of score_lk is an image in k and its similarity to all features in l
            # each col of score_lk is an image in l and its similarity to all features in k
            # applies the same to another set of augmentations qt
            # then minimizes the KL-divergence of kl and qt
            score_lk = torch.matmul(proj_l, proj_k.T)
            score_qt = torch.matmul(proj_q, proj_t.T)
            logprob_lk = F.log_softmax(score_lk, dim=1)
            prob_qt = F.softmax(score_qt, dim=1)
            # Loss
            # KL-divergence loss for Style Intervention
            kl_loss = alpha * \
                torch.nn.KLDivLoss(reduction='batchmean')(logprob_lk, prob_qt)
            # Entropy loss for Content Adjustment
            bce_loss = torch.nn.BCEWithLogitsLoss()(x, labels)
            loss = bce_loss + kl_loss
            avg_meter.add(
                {'loss': loss.item(), 'bce': bce_loss.item(), 'kl': kl_loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'Loss:%.4f' % (avg_meter.pop('loss')),
                      'BCE:%.4f' % (avg_meter.pop('bce')),
                      'KL:%.4f' % (avg_meter.pop('kl')),
                      'Min:%.4f' % (min_loss),
                      'imps:%.1f' % (
                          (step + 1) * cam_batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
                # validation
                vloss, vbce, vkl = validate(cls_model, mlp, val_data_loader, agg_smooth_r,
                                            data_aug_fn, voc12_root, alpha, device, scams)
                if vloss < min_loss:
                    torch.save(cls_model.state_dict(), cam_weight_path)
                    min_loss = vloss
                    min_bce = vbce
                    min_kl = vkl
                    # P(y|x, z)
                    # generate CAMs
                    # Using the current best weights
                    os.system(
                        'python3 make_small_cam.py --config ./cfg/fdsi.yml')
                    scams = pyutils.sum_cams(cam_out_dir).cuda(
                        device, non_blocking=True)
                    np.save(scam_path, scams.cpu().numpy())
                    # ===

                timer.reset_stage()
        # empty cache
        torch.cuda.empty_cache()

    with open(cam_weights_name + '.txt', 'w') as f:
        f.write('Min Validation Loss: {:.4f}'.format(min_loss) + '\n')
        f.write('Min BCE Loss: {:.4f}'.format(min_bce) + '\n')
        f.write('Min KL Loss: {:.4f}'.format(min_kl) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/fdsi.yml')
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    copy_weights = 'cp /data/home/yipingwang/data/Models/Classification/resnet50_baseline_{}.pth /data/home/yipingwang/data/Models/Classification/{}'.format(
        config['cam_crop_size'], config['cam_weights_name'])
    print(copy_weights)
    os.system(copy_weights)
    device = torch.device('cuda:7')
    train(config, device)
    # os.system('python3 make_cam.py --config ./cfg/front_door.yml')
