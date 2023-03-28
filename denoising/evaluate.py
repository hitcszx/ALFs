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

# run this to test the model

import argparse
import os, time, datetime
import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from copy import deepcopy

from skimage.io import imread, imsave
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()



class Evaluate():
    def __init__(self, set_dir='data/Test', set_names=None, style='gauss25', n_channels=1, result_dir='results', save_result=0, tf_writer=None):
        if set_names is None:
            set_names = ['Set68', 'Set12']
        self.set_dir = set_dir
        self.set_names = set_names
        self.style = style
        self.result_dir = result_dir
        self.save_result = save_result
        self.tf_writer = tf_writer
        self.n_channels = n_channels
        self.add_noise = self.__get_noise_adder()

    def gauss_adder(self, x, sigma):
        noise = np.random.randn(* x.shape) * sigma / 255.0
        return x + noise

    def poisson_adder(self, x, lamb):
        shape = x.shape
        lamb = lamb * torch.ones((shape[0], 1, 1,))
        return torch.poisson(lamb * x) / lamb

    def bernoulli_adder(self, x, p):
        prob = np.random.rand(*x.shape)
        mask = np.float32(prob < p)
        return x * mask

    def saltpepper_adder(self, x, p):
        prob = np.random.rand(*x.shape)
        y = deepcopy(x)
        y[prob < p] = 0
        y[prob > 1 - p] = 1
        return y

    def impulse_adder(self, x, p):
        prob = np.random.rand(*x.shape)
        noise = np.random.rand(*x.shape)
        y = deepcopy(x)
        y[prob < p] = noise[prob < p]
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


    def evaluate(self, model, epoch):
        model.eval()

        for set_cur in self.set_names:
            if not os.path.exists(os.path.join(self.result_dir, set_cur)):
                os.makedirs(os.path.join(self.result_dir, set_cur))
            psnrs = []
            ssims = []

            for im in os.listdir(os.path.join(self.set_dir, set_cur)):
                if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                    clean_img = np.array(imread(os.path.join(self.set_dir, set_cur, im)), dtype=np.float32)
                    if len(clean_img.shape) < 3 and self.n_channels == 3:
                        img = Image.fromarray(clean_img).convert('RGB')
                        clean_img = np.array(img, dtype=np.float32) / 255.0
                    elif len(clean_img.shape) > 2 and self.n_channels == 1:
                        img = Image.fromarray(clean_img).convert('L')
                        clean_img = np.array(img, dtype=np.float32) / 255.0
                    else:
                        clean_img = clean_img / 255.0

                    w, h = clean_img.shape[0], clean_img.shape[1]
                    clean_img = clean_img[: w // 32 * 32, : h // 32 * 32]

                    np.random.seed(seed=0)  # for reproducibility
                    noisy_img = self.add_noise(clean_img)
                    noisy_img = noisy_img.astype(np.float32)
                    noisy_img = torch.from_numpy(noisy_img).view(1, -1, noisy_img.shape[0], noisy_img.shape[1])

                    torch.cuda.synchronize()
                    start_time = time.time()
                    noisy_img = noisy_img.cuda()
                    denoised_img = model(noisy_img)  # inference
                    denoised_img = denoised_img.view(*clean_img.shape)
                    denoised_img = denoised_img.cpu().detach().numpy().astype(np.float32)
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start_time
                    print('%10s : %10s : %2.4f second' % (set_cur, im, elapsed_time))


                    denoised_img = np.clip(denoised_img, 0, 1)
                    psnr = compare_psnr(denoised_img, clean_img)
                    if self.n_channels > 1:
                        ssim = compare_ssim(denoised_img, clean_img,multichannel=True)
                    else:
                        ssim = compare_ssim(denoised_img, clean_img)
                    if self.save_result:
                        name, ext = os.path.splitext(im)
                        show(np.hstack((noisy_img, clean_img)))  # show the image
                        save_result(clean_img, path=os.path.join(self.result_dir, set_cur,
                                                          name + '_dncnn' + ext))  # save the denoised image
                    psnrs.append(psnr)
                    ssims.append(ssim)
            psnr_avg = np.mean(psnrs)
            ssim_avg = np.mean(ssims)
            psnrs.append(psnr_avg)
            ssims.append(ssim_avg)
            if self.save_result:
                save_result(np.hstack((psnrs, ssims)), path=os.path.join(self.result_dir, set_cur, 'results.txt'))
            log('Datset: {0:10s} \n  PSNR = {1:2.4f}dB, SSIM = {2:1.6f}'.format(set_cur, psnr_avg, ssim_avg))
            if self.tf_writer:
                self.tf_writer.add_scalar('psnr/' + set_cur, psnr_avg, epoch)
                self.tf_writer.add_scalar('ssim/' + set_cur, ssim_avg, epoch)