
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from data.cifar import CIFAR10, CIFAR100
# from data.mnist import MNIST
from dataset import DatasetGenerator
# from torchvision.models import resnet34
from models import *
from losses import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import *
import random
from utils import *

import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('darkgrid')
plt.switch_backend('agg')
plt.figure(figsize=(20, 20), dpi=600)


parser = argparse.ArgumentParser(description='Robust loss for learning with noisy labels')
parser.add_argument('--dataset', type=str, default="CIFAR100", metavar='DATA', help='Dataset name (default: CIFAR10)')
parser.add_argument('--root', type=str, default="../database/", help='the data root')
parser.add_argument('--noise_type', type=str, default='symmetric', help='the noise type: clean, symmetric, pairflip, asymmetric')
parser.add_argument('--noise_rate', type=float, default=0.4, help='the noise rate')

# learning settings
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=6, help='the number of worker for loading data')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--times', type=int, default=3)
parser.add_argument('--grad_bound', type=float, default=5., help='the gradient norm bound')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--loss', type=str, default='CE')


args = parser.parse_args()
parser.loss = parser.loss.upper()
parser.dataset = parser.dataset.upper()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

gpu_ids = ['0']
device = 'cuda' if torch.cuda.is_available() and len(gpu_ids) > 0 else 'cpu'
print('We are using', device)


seed = 123
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print(args)
def evaluate(loader, model):
    model.eval()
    correct = 0.
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        probs = F.softmax(z, dim=1)
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc


if args.dataset == 'MNIST':
    in_channels = 1
    num_classes = 10
    weight_decay = 1e-3
    lr = 0.01
    epochs = 50
elif args.dataset == 'CIFAR10':
    in_channels = 3
    num_classes = 10
    weight_decay = 1e-4
    lr = 0.01
    epochs=120
elif args.dataset == 'CIFAR100':
    in_channels = 3
    num_classes = 100
    weight_decay = 1e-5
    lr = 0.1
    epochs=200
else:
    raise ValueError('Invalid value {}'.format(args.dataset))


data_loader = DatasetGenerator(data_path=os.path.join(args.root, args.dataset),
                               num_of_workers=args.num_workers,
                               seed=args.seed,
                               asym=args.noise_type=='asymmetric',
                               dataset_type=args.dataset,
                               noise_rate=args.noise_rate
                               )

data_loader = data_loader.getDataLoader()
train_loader = data_loader['train_dataset']
test_loader = data_loader['test_dataset']


path = './results/' + args.dataset +'/' + args.noise_type + '/' +str(args.noise_rate)
if not os.path.exists(path):
    os.mkdir(path)

log_file = open('./results/'+args.dataset + args.loss + '_'+args.noise_type+'_'+str(epochs)+'_'+str(args.noise_rate)+'.log', 'w')
criterion = get_loss_config(args.dataset, train_loader, args.loss)


log(log_file, args.loss)
accs = np.zeros((args.times, epochs))
for i in range(args.times):
    if args.dataset != 'CIFAR100':
        model = CNN(type=args.dataset).to(device)
    else:
        model = ResNet34(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    for ep in range(epochs):
        model.train()
        total_loss = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x)
            model.zero_grad()
            optimizer.zero_grad()
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        test_acc = evaluate(test_loader, model)
        accs[i, ep] = test_acc
        log(log_file, 'Iter {}: loss={:.4f}, test_acc={:.4f}'.format(ep, total_loss, test_acc))
    log(log_file)
save_accs(path, args.loss, accs)
if log_file is not None:
    log_file.close()

