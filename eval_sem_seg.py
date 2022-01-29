import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import imageio
from misc import pyutils
import argparse
from PIL import Image


def run(config):
    chainer_eval_set = config['chainer_eval_set']
    voc12_root = config['voc12_root']
    sem_seg_out_dir = config['sem_seg_out_dir']

    dataset = VOCSemanticSegmentationDataset(
        split=chainer_eval_set, data_dir=voc12_root)
    labels = [dataset.get_example_by_keys(
        i, (1,))[0] for i in range(len(dataset))]

    preds = []
    i = 0
    for id in dataset.ids:
        cls_labels = imageio.imread(os.path.join(
            sem_seg_out_dir, id + '.png')).astype(np.uint8)
        cls_labels[cls_labels == 255] = 0
        cls_labels = np.asarray(Image.fromarray(cls_labels).resize(labels[i].shape, Image.NEAREST)).transpose(1,0)
        # print(cls_labels.shape, labels[i].shape)
        i += 1
        preds.append(cls_labels.copy())

    confusion = calc_semantic_segmentation_confusion(preds, labels)[:21, :21]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator

    print(fp[0], fn[0])
    print(np.mean(fp[1:]), np.mean(fn[1:]))

    print({'iou': iou, 'miou': np.nanmean(iou)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', default='./cfg/ir_net.yml')
    args = parser.parse_args()
    # if torch.cuda.is_available():
    #     device = pyutils.set_gpus(n_gpus=1)
    # else:
    #     device = 'cpu'
    config = pyutils.parse_config(args.config)
    run(config)
