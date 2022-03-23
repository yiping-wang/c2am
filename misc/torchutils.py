import torch
from torch.utils.data import Subset
import numpy as np
import math
from torch import nn
from PIL import Image
from torchvision.transforms import transforms


def get_style_variants(names, aug_fn, voc12_root, get_img_path):
    return torch.cat([aug_fn(Image.open(get_img_path(n, voc12_root)).convert('RGB')).unsqueeze(0) for n in names], dim=0).cuda(non_blocking=True)


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        super().step(closure)
        self.global_step += 1


class SGDROptimizer(torch.optim.SGD):
    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0
        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult
        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step /
                   self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)
    return out


def mean_agg(cam, r):
    return torch.mean(cam * r, dim=(2, 3))


def max_agg(cam, r=None):
    return torch.max(torch.max(cam, dim=2)[0], dim=2)[0]


def lse_agg(cam, r):
    return (1/r) * torch.logsumexp(cam * r, dim=(2, 3))


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply(
                                              [color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(
                                              kernel_size=int(0.1 * size)),
                                          # transforms.RandomSolarize(threshold=0.5, p=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
        mean=torch.tensor(
            [0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]))])
    return data_transforms
