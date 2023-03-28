import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss

pi = np.pi

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

class sum_abs_error(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(sum_abs_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


class Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Loss, self).__init__()
        self.reduction = reduction
    
    def _calculate_loss(self, x, y):
        pass
    
    def forward(self, x, y):
        loss = self._calculate_loss(x, y)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class LpLoss(Loss):
    def __init__(self, p=1, reduction='mean', epsilon=1e-7):
        super(LpLoss, self).__init__(reduction=reduction)
        self.p = p
        self.epsilon = epsilon
    
    def _calculate_loss(self, x, y):
        diff = torch.abs(x-y) + self.epsilon
        loss = diff ** self.p
        return loss


class NegHeatKernelLoss(Loss):
    def __init__(self, a=1, reduction='sum'):
        super(NegHeatKernelLoss, self).__init__(reduction=reduction)
        self.a = a
        assert a > 0

    def _calculate_loss(self, x, y):
        diff = (x-y) ** 2
        loss = 1. / (self.a * np.sqrt(pi)) - 1. / (self.a * np.sqrt(pi)) * torch.exp(-diff / self.a**2)
        return loss


class NegPoissonKernelLoss(Loss):
    def __init__(self, a=1, reduction='sum'):
        super(NegPoissonKernelLoss, self).__init__(reduction=reduction)
        self.a = a
        assert a > 0

    def _calculate_loss(self, x, y):
        diff = (x - y) ** 2
        loss = self.a / pi / (self.a**2)-self.a / pi / (self.a**2 + diff)
        return loss
