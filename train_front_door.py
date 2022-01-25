import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
from torch.utils.data import DataLoader
from misc import pyutils, torchutils, imutils
from net.resnet50_cam import CAM


def validate(model, data_loader, image_size_height, image_size_width, cam_batch_size, logexpsum_r):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')
    model.eval()
    with torch.no_grad():
        for pack in data_loader:
            imgs = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            # P(y|x, z)
            strided_size = imutils.get_strided_size(
                (image_size_height, image_size_width), 4)
            cams = []
            for b in range(cam_batch_size):
                img = imgs[b].unsqueeze(0)
                outputs = [model(i) for i in img]
                strided_cam = torch.sum(torch.stack([F.interpolate(torch.unsqueeze(
                    o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o in outputs]), 0)
                strided_cam /= F.adaptive_max_pool2d(
                    strided_cam, (1, 1)) + 1e-5
                cams += [strided_cam.unsqueeze(0)]
            cams = torch.cat(cams, dim=0)  # B * 20 * H * W
            # P(z|x)
            p = F.softmax(torchutils.lse_agg(
                cams.detach(), r=logexpsum_r), dim=1)
            # P(y|do(x))
            scams = torch.mean(cams, dim=0)
            C = cams.shape[1]
            wcams = torch.zeros_like(cams)
            for c in range(C):
                wcams += p[:, c].unsqueeze(1).unsqueeze(1).unsqueeze(1) * scams
            # loss
            x = torchutils.lse_agg(scams, r=logexpsum_r)
            loss1 = F.multilabel_soft_margin_loss(x, label)
            val_loss_meter.add({'loss1': loss1.item()})
    model.train()
    vloss = val_loss_meter.pop('loss1')
    print('loss: %.4f' % vloss)
    return vloss


def train(config, device):
    seed = config['seed']
    train_list = config['train_list']
    val_list = config['val_list']
    voc12_root = config['voc12_root']
    cam_batch_size = config['cam_batch_size']
    cam_num_epoches = config['cam_num_epoches']
    cam_learning_rate = config['cam_learning_rate']
    cam_weight_decay = config['cam_weight_decay']
    model_root = config['model_root']
    cam_weights_name = config['cam_weights_name']

    image_size_height = config['image_size_height']
    image_size_width = config['image_size_width']
    logexpsum_r = config['logexpsum_r']

    num_workers = 1
    cam_weight_path = os.path.join(model_root, cam_weights_name + '.pth')
    pyutils.seed_all(seed)

    model = CAM().cuda(device)
    # load pre-trained classification network
    model.load_state_dict(torch.load(cam_weight_path))

    # CAM generation dataset
    train_dataset = voc12.dataloader.VOC12ClassificationDatasetFD(train_list,
                                                                  voc12_root=voc12_root,
                                                                  scales=(
                                                                      1.0,),
                                                                  size_h=image_size_height,
                                                                  size_w=image_size_width)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=cam_batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=True)

    max_step = (len(train_dataset) // cam_batch_size) * cam_num_epoches

    val_dataset = voc12.dataloader.VOC12ClassificationDataset(val_list,
                                                              voc12_root=voc12_root,
                                                              crop_size=512)
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
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    # model = torch.nn.DataParallel(model).cuda(device)
    model = model.cuda(device)
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    min_loss = float('inf')
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            imgs = pack['img']
            labels = pack['label'].cuda(device, non_blocking=True)
            # P(y|x, z)
            strided_size = imutils.get_strided_size(
                (image_size_height, image_size_width), 4)
            acams = torch.zeros(
                (cam_batch_size, 20, strided_size[0]//4, strided_size[1]//4)).cuda(device)
            for b in range(cam_batch_size):
                img = imgs[b].cuda(device, non_blocking=True)
                strided_cam = model(img).unsqueeze(0)
                # strided_cam = F.interpolate(torch.unsqueeze(
                #     outputs, 0), strided_size, mode='bilinear', align_corners=False)
                strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
                acams[b] = strided_cam
            # P(z|x)
            p = F.softmax(torchutils.lse_agg(acams.detach(), r=logexpsum_r), dim=1)
            # P(y|do(x))
            scams = torch.mean(acams, dim=0)
            C = acams.shape[1]
            wcams = torch.zeros_like(acams)
            for c in range(C):
                wcams += p[:, c].unsqueeze(1).unsqueeze(1).unsqueeze(1) * scams
            # loss
            x = torchutils.lse_agg(wcams, r=logexpsum_r)
            loss = F.multilabel_soft_margin_loss(x, labels)
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
                vloss = validate(model, val_data_loader, image_size_height,
                                 image_size_width, cam_batch_size, logexpsum_r)
                if vloss < min_loss:
                    torch.save(model.module.state_dict(),
                               cam_weight_path + '_fd.pth')
                    min_loss = vloss
                timer.reset_stage()
        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/front_door.yml')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = pyutils.set_gpus(n_gpus=1)
    else:
        device = 'cpu'
    config = pyutils.parse_config(args.config)
    train(config, device)
