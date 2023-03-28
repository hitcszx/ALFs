import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from evaluate import Evaluate


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


root = '/data/denoising/gmodels/'
exps = ['n2c', 'n2n', 'n2s']
styles = ['gauss20', 'bernoulli0.3', 'saltpepper0.3', 'impulse0.3']
losses = ['lp2', 'heat0.2', 'poisson0.2']
reduction = 'sum'
for exp in exps:
    for style in styles:
        for loss in losses:
            sign = exp + '_' + style + '_'  + loss
            path = os.path.join(root, exp + '_' + style +  '/' + '_'.join([loss, reduction, 'DnCNN']))
            model = DnCNN()
            model = torch.load(os.path.join(path, 'model_current.pth.tar'))
            model = model.cuda()
            evaluator = Evaluate(n_channels=1, style=style, result_dir='/data/denoising/results', save_result=True, verbose=sign)
            evaluator.evaluate(model)

            model = DnCNN()
            model = torch.load(os.path.join(path, 'model_current.psnr.best.pth.tar'))
            model = model.cuda()
            evaluator = Evaluate(n_channels=1, style=style, result_dir='/data/denoising/results', save_result=True, verbose=sign + '_psnr_best')
            evaluator.evaluate(model)

            model = DnCNN()
            model = torch.load(os.path.join(path, 'model_current.ssim.best.pth.tar'))
            model = model.cuda()
            evaluator = Evaluate(n_channels=1, style=style, result_dir='/data/denoising/results', save_result=True, verbose=sign + '_ssim_best')
            evaluator.evaluate(model)
