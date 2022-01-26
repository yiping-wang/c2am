import torch.nn.functional as F
import voc12.dataloader
import argparse
import torch
import os
from torch.utils.data import DataLoader
from misc import pyutils, torchutils
from net.resnet50_cam import Net


def validate(model, data_loader):
    print('validating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')
    model.eval()
    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)
            x = model(img)
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
    num_workers = 1
    cam_weight_path = os.path.join(model_root, cam_weights_name)
    pyutils.seed_all(seed)

    model = Net().cuda(device)

    model.load_state_dict(torch.load(cam_weight_path + '.pth'))

    train_dataset = voc12.dataloader.VOC12ClassificationDataset(train_list,
                                                                voc12_root=voc12_root,
                                                                resize_long=(
                                                                    320, 640),
                                                                hor_flip=True,
                                                                crop_size=512, crop_method="random")
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
        # {'params': param_groups[1], 'lr': 10 * cam_learning_rate,
        #     'weight_decay': cam_weight_decay},
    ], lr=cam_learning_rate, weight_decay=cam_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda(device)
    model.train()

    avg_meter = pyutils.AverageMeter()
    timer = pyutils.Timer()

    min_loss = float('inf')
    for ep in range(cam_num_epoches):
        print('Epoch %d/%d' % (ep+1, cam_num_epoches))
        for step, pack in enumerate(train_data_loader):
            img = pack['img']
            label = pack['label'].cuda(device, non_blocking=True)
            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)
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
                vloss = validate(model, val_data_loader)
                if vloss < min_loss:
                    torch.save(model.module.state_dict(),
                               cam_weight_path + '.pth')
                    min_loss = vloss
                timer.reset_stage()
        # empty cache
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/base.yml')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = pyutils.set_gpus(n_gpus=1)
    else:
        device = 'cpu'
    config = pyutils.parse_config(args.config)
    train(config, device)
