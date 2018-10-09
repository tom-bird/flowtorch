import torch
from torch import nn
from torch.distributions import Normal
import torch.functional as F
from torchvision.utils import save_image
import numpy as np

from flows import SequentialFlow, AffineCouplingLayer


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(*layer) for layer in layers)
        self.bn = nn.ModuleList(nn.BatchNorm1d(layer[-1]) for layer in layers)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        x_shape = x.shape
        h = x.view(x_shape[0], -1)
        for layer, bn in zip(self.layers, self.bn):
            h = layer(h)
            h = bn(h)
            h = self.nonlin(h)
        return h.view(*x_shape)


class ConvResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        self.nonlin = nn.ReLU()

    def forward(self, x):
        res = x
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.nonlin(h)

        h = self.conv2(h)
        h = self.bn2(h)

        h = h + res
        h = self.nonlin(h)
        return h


class StackedAffineCouplingFlow(nn.Module):
    def __init__(self, x_shape, cuda=False):
        super().__init__()
        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.x_shape = x_shape
        self.sigmoid = nn.Sigmoid()
        self.cuda = cuda

        # checkerboard mask
        dims = int(np.prod(x_shape))
        mask1 = [1 if n % 2 else 0 for n in range(dims)]
        mask1 = torch.tensor(mask1).view(x_shape)
        mask2 = 1 - mask1

        mask1 = mask1.float()
        mask2 = mask2.float()

        if self.cuda:
            mask1 = mask1.cuda()
            mask2 = mask2.cuda()

        # get_net = lambda: nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), ConvResBlock(16),
        #                                 ConvResBlock(16), nn.Conv2d(16, 1, 3, padding=1))
        get_net = lambda: MLP([(784, 256), (256, 256), (256, 784)])

        self.flow = SequentialFlow([AffineCouplingLayer(mask=mask1, s=get_net(), t=get_net(), cuda=cuda),
                                    AffineCouplingLayer(mask=mask2, s=get_net(), t=get_net(), cuda=cuda),
                                    AffineCouplingLayer(mask=mask1, s=get_net(), t=get_net(), cuda=cuda),
                                    AffineCouplingLayer(mask=mask2, s=get_net(), t=get_net(), cuda=cuda),
                                    AffineCouplingLayer(mask=mask1, s=get_net(), t=get_net(), cuda=cuda),
                                    AffineCouplingLayer(mask=mask2, s=get_net(), t=get_net(), cuda=cuda),
                                    ], cuda=cuda)

    def forward(self, x):
        return self.flow.forward(x)

    def backward(self, y):
        return self.flow.backward(y)

    def loss(self, x):
        """Additive coupling layers have unit jacobian so we can just optimise the log probs"""
        y, logdet_j = self.forward(x)
        return - torch.mean(torch.sum(Normal(self.prior_mean, self.prior_std).log_prob(y), dim=[1, 2, 3]) + logdet_j) / 784.

    def sample(self, device, n, epoch):
        y = Normal(0, 1).sample((n, 1, 28, 28)).to(device)
        x = self.backward(y)
        x = self.sigmoid(x)  # move back from logit space to x space
        save_image(x.view(-1, *self.x_shape), 'results/samples_epoch{}.png'.format(epoch))
