# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob
import cv2
import numpy as np
from copy import deepcopy
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
        Arguments:
            xs (Tensor): clean image patches
            style: noise type and noise level, e.g., gauss25, poisson25, pulse0.1, bernoulli0.1
            mode: training mode, e.g., n2c, n2s, n2n
        """
    def __init__(self, xs, style, mode):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.style = style
        self.mode = mode
        self.add_noise = self.__get_noise_adder()

    def __getitem__(self, index):
        batch_x = self.xs[index]
        batch_y = self.add_noise(batch_x)
        if self.mode in ['n2c', 'n2s']:
            return batch_y, batch_x
        elif self.mode == 'n2n':
            batch_x = self.add_noise(batch_x)
            return batch_y, batch_x
        else:
            raise NotImplementedError

    def __len__(self):
        return self.xs.size(0)

    def gauss_adder(self, x, sigma):
        noise = torch.randn(x.size()).mul_(sigma / 255.0)
        return x + noise

    def poisson_adder(self, x, lamb):
        shape = x.shape
        lamb = lamb * torch.ones((shape[0], 1, 1, ))
        return torch.poisson(lamb * x) / lamb

    def bernoulli_adder(self, x, p):
        prob = torch.rand_like(x)
        mask = (prob < p).float()
        return x * mask

    def saltpepper_adder(self, x, p):
        prob = torch.rand_like(x)
        y = x + 0.0
        y[prob < p] = 0
        y[prob > 1-p] = 1
        return y

    def impulse_adder(self, x, p):
        prob = torch.rand_like(x)
        noise = torch.rand_like(x)
        y = x + 0.0
        y[prob < p] = noise[prob<p]
        return y

    def __get_noise_adder(self):
        if self.style.startswith('gauss'):
            sigma = float(self.style.split('s')[-1])
            return lambda x: self.gauss_adder(x, sigma)
        elif self.style.startswith('poisson'):
            lamb = float(self.style.split('n')[-1])
            return lambda x: self.poisson_adder(x, lamb)
        elif self.style.startswith('bernoulli'):
            p = float(self.style.split('i')[-1])
            return lambda x: self.bernoulli_adder(x, p)
        elif self.style.startswith('saltpepper'):
            p = float(self.style.split('r')[-1])
            return lambda x: self.saltpepper_adder(x, p)
        elif self.style.startswith('impulse'):
            p = float(self.style.split('e')[-1])
            return lambda x: self.impulse_adder(x, p)
        else:
            raise NotImplementedError


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data


if __name__ == '__main__': 

    data = datagenerator(data_dir='data/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       