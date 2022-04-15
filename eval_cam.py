import numpy as np
import os
import argparse
from misc import pyutils
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

def run(config, cam_eval_thres):
    chainer_eval_set = config['chainer_eval_set']
    voc12_root = config['voc12_root']
    cam_out_dir = config['cam_out_dir']

    dataset = VOCSemanticSegmentationDataset(split=chainer_eval_set, data_dir=voc12_root)
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

    preds = []
    for id in dataset.ids:
        cam_dict = np.load(os.path.join(cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())

    print(preds[0].shape)
    print(labels[0].shape)
    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print({'iou': iou, 'miou': np.nanmean(iou)})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Front Door Semantic Segmentation')
    parser.add_argument('--config', type=str,
                        help='YAML config file path', required=True)
    parser.add_argument('--cam_eval_thres', type=float, required=True)
    args = parser.parse_args()
    config = pyutils.parse_config(args.config)
    run(config, args.cam_eval_thres)