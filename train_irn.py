from statistics import mode
import torch
from torch.backends import cudnn
import argparse
from torch.utils.data import DataLoader
import voc12.dataloader
from misc import pyutils, torchutils, indexing
import importlib
from net.resnet50_irn import AffinityDisplacementLoss


def train(config, device):
    train_list = config['train_list']
    ir_label_out_dir = config['ir_label_out_dir']
    infer_list = config['infer_list']
    voc12_root = config['voc12_root']
    irn_crop_size = config['irn_crop_size']
    irn_batch_size = config['irn_batch_size']
    irn_num_epoches = config['irn_num_epoches']
    irn_learning_rate = config['irn_learning_rate']
    irn_weight_decay = config['irn_weight_decay']
    irn_weights_name = config['irn_weights_name']

    path_index = indexing.PathIndex(radius=10, default_size=(
        irn_crop_size // 4, irn_crop_size // 4))

    model = AffinityDisplacementLoss(path_index)

    train_dataset = voc12.dataloader.VOC12AffinityDataset(train_list,
                                                          label_dir=ir_label_out_dir,
                                                          voc12_root=voc12_root,
                                                          indices_from=path_index.src_indices,
                                                          indices_to=path_index.dst_indices,
                                                          hor_flip=True,
                                                          crop_size=irn_crop_size,
                                                          crop_method="random",
                                                          rescale=(0.5, 1.5)
                                                          )
    train_data_loader = DataLoader(train_dataset, batch_size=irn_batch_size,
                                   shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    max_step = (len(train_dataset) // irn_batch_size) * irn_num_epoches

    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': 1 *
            irn_learning_rate, 'weight_decay': irn_weight_decay},
        {'params': param_groups[1], 'lr': 10 *
            irn_learning_rate, 'weight_decay': irn_weight_decay}
    ], lr=irn_learning_rate, weight_decay=irn_weight_decay, max_step=max_step)

    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    for ep in range(irn_num_epoches):

        print('Epoch %d/%d' % (ep+1, irn_num_epoches))

        for iter, pack in enumerate(train_data_loader):

            img = pack['img'].cuda(non_blocking=True)
            bg_pos_label = pack['aff_bg_pos_label'].cuda(non_blocking=True)
            fg_pos_label = pack['aff_fg_pos_label'].cuda(non_blocking=True)
            neg_label = pack['aff_neg_label'].cuda(non_blocking=True)

            pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss = model(
                img, True)

            bg_pos_aff_loss = torch.sum(
                bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
            fg_pos_aff_loss = torch.sum(
                fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
            pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
            neg_aff_loss = torch.sum(
                neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

            dp_fg_loss = torch.sum(
                dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)
            dp_bg_loss = torch.sum(
                dp_bg_loss * torch.unsqueeze(bg_pos_label, 1)) / (2 * torch.sum(bg_pos_label) + 1e-5)

            avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item(), 'loss4': dp_bg_loss.item()})

            total_loss = (pos_aff_loss + neg_aff_loss) / \
                2 + (dp_fg_loss + dp_bg_loss) / 2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % (
                      avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3'), avg_meter.pop('loss4')),
                      'imps:%.1f' % ((iter + 1) * irn_batch_size /
                                     timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()

    infer_dataset = voc12.dataloader.VOC12ImageDataset(args.infer_list,
                                                       voc12_root=args.voc12_root,
                                                       crop_size=args.irn_crop_size,
                                                       crop_method="top_left")

    infer_data_loader = DataLoader(infer_dataset, batch_size=irn_batch_size,
                                   shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    model.eval()
    print('Analyzing displacements mean ... ', end='')

    dp_mean_list = []

    with torch.no_grad():
        for iter, pack in enumerate(infer_data_loader):
            img = pack['img'].cuda(non_blocking=True)

            aff, dp = model(img[:, 0], False)

            dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())

        model.module.mean_shift.running_mean = torch.mean(
            torch.stack(dp_mean_list), dim=0)
    print('done.')

    torch.save(model.module.state_dict(), irn_weights_name)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/ir_net.yml')
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = pyutils.set_gpus(n_gpus=1)
    else:
        device = 'cpu'
    config = pyutils.parse_config(args.config)
    train(config, device)
