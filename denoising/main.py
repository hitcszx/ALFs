
import argparse
import re
import os, glob, datetime, time
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from dataset import data_generator as dg
from dataset import *
from tensorboardX import SummaryWriter
from evaluate import Evaluate
from utils import *
from losses import *
from dataset import Masker
import gc
import shutil

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--exp', default='n2c', type=str)
parser.add_argument('--loss', default='heat0.1', type=str)
parser.add_argument('--reduction', default='sum', type=str)
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')
parser.add_argument('--style', default='gauss25', help='the noise style')
parser.add_argument('--epoch', default=120, type=int, help='number of train epochs')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for the optimizer')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='the weight decay of the optimizer')
parser.add_argument('--gpu', default='0', type=str, help='the choice of gpu')
parser.add_argument('--resume', default=0, type=int, help='resume')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
if args.style.startswith('gauss'):
    args.sigma = float(args.style.split('s')[-1])
elif args.style.startswith('poisson'):
    args.sigma = float(args.style.split('n')[-1])


save_dir = os.path.join('./models/' + args.exp + '_' + args.style +  '/' + '_'.join([args.loss, args.reduction, args.model]))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


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

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0 and args.resume:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    else:
        initial_epoch = 0
    model.train()
    if args.loss.startswith('lp'):
        criterion = LpLoss(p=float(args.loss.split('p')[-1]), reduction=args.reduction)
    elif args.loss.startswith('heat'):
        criterion = NegHeatKernelLoss(a=float(args.loss.split('t')[-1]), reduction=args.reduction)
    elif args.loss.startswith('poisson'):
        criterion = NegPoissonKernelLoss(a=float(args.loss.split('n')[-1]), reduction=args.reduction)
    else:
        criterion = nn.MSELoss()

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epoch * 238336 / 128, eta_min=0.0)  # learning rates
    tf_writer = SummaryWriter(log_dir='./logs/' + '_'.join([args.exp, args.style, args.loss, args.reduction]))
    evaluator = Evaluate(tf_writer=tf_writer, n_channels=1, style=args.style)
    best_psnr = 0.0
    best_ssim = 0.0
    for epoch in range(initial_epoch, n_epoch):
        model.train()
        xs = dg.datagenerator(data_dir=args.train_data)
        xs = xs.astype('float32') / 255.0
        xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        Dataset = DenoisingDataset(xs, args.style, mode=args.exp)
        DLoader = DataLoader(dataset=Dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        masker = Masker(width=5, mode='random')
        for n_count, batch_xy in enumerate(DLoader):
            optimizer.zero_grad()
            batch_x, batch_y = batch_xy[0].cuda(), batch_xy[1].cuda()
            if args.exp == 'n2s':
                input, mask = masker.mask(batch_x, n_count)
                loss = criterion(model(input) * mask, batch_x * mask)
            else:
                loss = criterion(model(batch_x), batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            if n_count % 20 == 0:
                print('%4d %4d / %4d loss = %2.6f' % (
                epoch + 1, n_count, xs.size(0) // batch_size, loss.item() / batch_size))

            scheduler.step()  # step to the learning rate in this epcoh
        elapsed_time = time.time() - start_time
        total_psnr, total_ssim = evaluator.evaluate(model, epoch)

        tf_writer.add_scalar('loss/train', epoch_loss / n_count / batch_size, epoch)

        log('epoch = %4d , loss = %4.6f , time = %4.2f s' % (epoch + 1, epoch_loss / n_count / batch_size, elapsed_time))
        filename = os.path.join(save_dir, 'model_current.pth.tar')
        torch.save(model, filename)
        gc.collect()