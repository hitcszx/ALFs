
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet34
from dataset import DatasetGenerator
from models import *
from losses import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from utils import *
from sklearn import manifold

import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('darkgrid')
plt.switch_backend('agg')
plt.figure(figsize=(20, 20), dpi=600)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
gpu_ids = ['0']
device = 'cuda' if torch.cuda.is_available() and len(gpu_ids) > 0 else 'cpu'
print('We are using', device)


class SharpCE(nn.Module):
    def __init__(self, T=0.5):
        super(SharpCE, self).__init__()
        self.T = T
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        pred = pred / self.T
        return self.ce(pred, labels)


from visdom import Visdom


parser = argparse.ArgumentParser(description='Robust loss for learning with noisy labels')
parser.add_argument('--dataset', type=str, default="CIFAR10", metavar='DATA', help='Dataset name (default: CIFAR10)')
parser.add_argument('--root', type=str, default="../database/", help='the data root')
parser.add_argument('--noise_type', type=str, default='symmetric', help='the noise type: clean, symmetric, pairflip, asymmetric')
parser.add_argument('--noise_rate', type=float, default=0.0, help='the noise rate')

# learning settings
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=10, help='the number of worker for loading data')
parser.add_argument('--grad_bound', type=float, default=5., help='the gradient norm bound')
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()

# if device == 'cuda':
#     torch.cuda.manual_seed(args.seed)
# else:
#     torch.manual_seed(args.seed)

markercolor = np.array(
    [
        [118, 80, 200],
        [197, 192, 199],
        [178, 173, 180],
        [177, 165, 201],
        [108, 150, 191],
        [99, 99, 99],
        [78, 101, 200],
        [203, 157, 162],
        [211, 205, 160],
        [176, 214, 171]
    ],
    dtype=np.uint8
)

if args.noise_rate == 0.0:
    args.noise_type = 'clean'

viz = Visdom(env='SR--Dataset: {}, noise type: {}, noise rate: {:.2f}, ns'.format(args.dataset, args.noise_type, args.noise_rate))


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
        z, _ = model(x)
        probs = F.softmax(z, dim=1)
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc

def show_points(loader, model):
    model.eval()
    flag = 0
    for x, y in loader:
        x = x.to(device)
        _, z = model(x)
        z = z.detach().cpu().numpy()
        y = y.numpy()
        if flag == 0:
            X = z
            Y = y
            flag = 1
        else:
            X = np.r_[X, z]
            Y = np.r_[Y, y]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=123)
    X_tsne = tsne.fit_transform(X)
    np.save('./tsne/'+ args.dataset + '/' + label  + '_' + str(args.noise_rate) + '.npy', X_tsne)
    np.save('./tsne/'+ args.dataset + '/' + label + '_' + str(args.noise_rate) + '_y.npy', Y)
    X_norm = (X_tsne - np.min(X_tsne)) / (np.max(X_tsne) - np.min(X_tsne))
    win1 = viz.scatter(X=X_norm, Y=Y + 1, opts=dict(title=label, markersize=5, markercolor=markercolor))


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

# f = open('./results/'+args.dataset+'_'+args.noise_type+'_'+str(args.epochs)+'_'+str(args.noise_rate)+'.log', 'w')
f = None
#
criterions = [GCELoss(num_classes=10, q=0.7), AUELoss(num_classes=10, a=3, q=0.1), AExpLoss(num_classes=10, a=3.5)]
labels = ['GCE', 'AUL', 'AEL']

for criterion, label in zip(criterions, labels):
    print(label)
    accs = []
    if args.dataset != 'CIFAR100':
        model = CNN(type=args.dataset, show=True).to(device)
    else:
        model = ResNet34(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    for ep in range(epochs):
        model.train()
        total_loss = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            out, _ = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
            if label.startswith('N'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        test_acc = evaluate(test_loader, model)
        if ep == 0:
            win = viz.line(X=np.array([0]), Y=np.array([test_acc]), opts=dict(title=label))
        else:
            viz.line(X=np.array([ep]), Y=np.array([test_acc]), win=win, update='append')
        accs.append(test_acc)
        log(f, 'Iter {}: loss={:.4f}, test_acc={:.4f}'.format(ep, total_loss, test_acc))
    log(f)
    show_points(test_loader, model)
if f is not None:
    f.close()

