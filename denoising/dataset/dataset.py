
import os
import glob
import cv2
import imageio
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise(object):
    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [float(p) / 255.0 for p in style.replace('gauss', '', 1).split('_')]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [float(p) for p in style.replace('poisson', '', 1).split('_')]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1), device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised

    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std, dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


class Train400(object):
    def __init__(self, data_dir, sigma=25, is_target_noisy=False):
        self.file_list = glob.glob(data_dir + '/*.png')
        self.sigma = sigma
        self.is_target_noisy = is_target_noisy
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        batch_y = self.transform(img)
        noise = torch.randn(batch_y.size()).mul_(self.sigma / 255.0)
        batch_x = batch_y + noise
        if self.is_target_noisy:
            noise = torch.randn(batch_y.size()).mul_(self.sigma / 255.0)
            batch_y = batch_y + noise
        return batch_x, batch_y

    def __len__(self):
        return len(self.file_list)

class ImageNetValDataset(object):
    def __init__(self, dir):
        super(ImageNetValDataset, self).__init__()
        self.dir = dir
        file = open(os.path.join(self.dir, 'Dn_ILSVRC2012_img_val.txt'), 'r')
        self.file_list = file.read()
        self.file_list = self.file_list.strip().split('\n')
        self.transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.file_list[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.file_list)

class TestDataset(object):
    def __init__(self, dir, format='*.png'):
        super(TestDataset, self).__init__()
        self.dir = dir
        self.file_list = glob.glob(os.path.join(dir, format))
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __get_item__(self, index):
        img_path = os.path.join(self.dir, self.file_list[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img)
        return img

if __name__ == '__main__':
    noiser = AugmentNoise(style='gauss25')
    x = torch.randn(1, 3, 3, 3).cuda()
    y = noiser.add_train_noise(x)
    print(y)