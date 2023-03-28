import numpy as np
import torch
import torch.nn.functional as F
from numpy import clip, exp
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def expand(x, r):
    return np.repeat(np.repeat(x, r, axis=0), r, axis=1)

def tensor_to_numpy(x):
    x = x.detach().cpu().numpy()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        return x

    if x.shape[0] == 1:
        return x[0]
    elif x.shape[0] == 3:
        return x.transpose((1, 2, 0))
    else:
        raise

def scale_tensor(x):
    return (x - x.min()) / (x.max() - x.min())

def clamp_tensor(x):
    return torch.clamp(x, 0, 1)

def random_noise(img, params):
    """Parameters for random noise include the mode and the type.
    mode: gaussian, poisson, or gaussian_poisson noise type
    std: std of gaussian
    photons_at_max: at image with intensity 1 has this many photons on average
    clamp: clamp result to [0,1]
    """
    noisy = img

    if params['mode'] == 'poisson' or params['mode'] == 'gaussian_poisson':
        noisy = torch.poisson(noisy * params['photons_at_max']) / params['photons_at_max']

    if params['mode'] == 'gaussian' or params['mode'] == 'gaussian_poisson':
        noise = torch.randn(img.size()).to(img.device) * params['std']
        noisy = noise + noisy

    if params['mode'] == 'bernoulli':
        noisy = noisy * torch.bernoulli(torch.ones(noisy.shape) * params['p'])

    if 'clamp' in params and params['clamp']:
        noisy = torch.clamp(noisy, 0, 1)

    return noisy

def test_bernoulli_noise():
    torch.manual_seed(2018)
    p = 0.2
    shape = (10, 1, 100, 100)
    n = 10 * 100 * 100
    img = torch.ones(shape)
    noisy = random_noise(img, {'mode': 'bernoulli', 'p': p})

    var = n * p * (1 - p)

    assert torch.abs(noisy.sum() - p * img.sum()) < 3 * (var ** 0.5)

def mse(x, y, pad=None, rescale=False):
    if pad:
        x = x[:, :, pad:-pad, pad:-pad]
        y = y[:, :, pad:-pad, pad:-pad]

    def batchwise_mean(z):
        return z.reshape(z.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1)

    if rescale:
        x = x - batchwise_mean(x)
        y = y - batchwise_mean(y)
        a = batchwise_mean(x * y) / batchwise_mean(x * x)
        x = a * x

    return batchwise_mean((x - y) ** 2).reshape(-1)


def psnr(x, x_true, max_intensity=1.0, pad=None, rescale=False):
    '''A function computing the PSNR of a noisy tensor x approximating a tensor x_true.
    It vectorizes over the batch.
    PSNR := 10*log10 (MAX^2/MSE)
    where the MSE is the averaged squared error over all pixels and channels.
    '''
    return 10 * torch.log10((max_intensity ** 2) / mse(x, x_true, pad=pad, rescale=rescale))

def test_psnr():
    std = 0.1
    noise = torch.randn(10, 3, 100, 100) * std
    x_true = torch.ones(10, 3, 100, 100) / 2
    x = x_true + noise
    # MSE should be 0.01. PSNR should be 20.
    assert (torch.abs(psnr(x, x_true) - 20) < 0.1).all()

    x = 256 * x
    x_true = 256 * x_true
    assert (torch.abs(psnr(x, x_true, 256) - 20) < 0.2).all()

def test_mse_rescale():
    y = torch.randn(10, 3, 10, 10)
    x = 10 * y + 7
    assert (mse(x, y, rescale=True) < 1e-5).all()

    # Normalized values are (1, 1, 0, -2) and (1, 1, -1, -1)
    y = torch.Tensor([3, 3, 2, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x = torch.Tensor([5, 5, 0, 0]).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    assert mse(x, y, rescale=True).sum() == 0.5

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def smooth(tensor):
    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 2.0, 1.0], (0.5, 1.0, 0.5)])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(tensor.device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)
    return filtered_tensor

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def gpuinfo(gpuid):
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q', '-i', str(gpuid), '-d', 'MEMORY'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass
    return out_dict

def getfreegpumem(id):
    return int(gpuinfo(id)['Free'].replace('MiB', '').strip())

def getbestgpu():
    freememlist = []
    for id in range(4):
        freemem = getfreegpumem(id)
        print("GPU device %d has %d MiB left." % (id, freemem))
        freememlist.append(freemem)
    idbest = freememlist.index(max(freememlist))
    print("--> GPU device %d was chosen" % idbest)
    return idbest



class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)