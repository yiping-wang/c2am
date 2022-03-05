import numpy as np
import time
import sys
import random
import yaml
import torch
import os
from pynvml import *

import itertools
import operator
import collections
import glob

class IterateCAM:
    def __init__(self, cam_dir):
        self.cam_list = glob.glob(os.path.join(cam_dir, '*.npy'))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.cam_list):
            raise StopIteration
        c = np.load(self.cam_list[self.index], allow_pickle=True).item()
        self.index += 1
        return c['raw_outputs']


def sum_cams(cam_dir):
    itcam = IterateCAM(cam_dir)
    running_sum = itertools.accumulate(itcam)
    running_mean = map(operator.truediv, running_sum, itertools.count(1))
    return torch.from_numpy(collections.deque(running_mean, maxlen=1)[0])

def set_gpus(n_gpus, verbose=False):
    selected_gpu = []
    gpu_free_mem = {}

    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_usage = nvmlDeviceGetMemoryInfo(handle)
        gpu_free_mem[i] = mem_usage.free
        if verbose:
            print("GPU: {} \t Free Memory: {}".format(i, mem_usage.free))

    res = sorted(gpu_free_mem.items(), key=lambda x: x[1], reverse=True)
    res = res[:n_gpus]
    selected_gpu = [r[0] for r in res]

    print("Using GPU {}".format(','.join([str(s) for s in selected_gpu])))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(s) for s in selected_gpu])
    return torch.device('cuda:{}'.format(str(selected_gpu[0])))


def seed_all(seed=0):
    torch.manual_seed(0)
    np.random.seed(seed)
    random.seed(seed)


def parse_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class Timer:
    def __init__(self, starting_msg=None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out


def to_one_hot(sparse_integers, maximum_val=None, dtype=np.bool):

    if maximum_val is None:
        maximum_val = np.max(sparse_integers) + 1

    src_shape = sparse_integers.shape

    flat_src = np.reshape(sparse_integers, [-1])
    src_size = flat_src.shape[0]

    one_hot = np.zeros((maximum_val, src_size), dtype)
    one_hot[flat_src, np.arange(src_size)] = 1

    one_hot = np.reshape(one_hot, [maximum_val] + list(src_shape))

    return one_hot
