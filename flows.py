import torch
from torch import nn


class SequentialFlow(nn.Module):
    def __init__(self, flows, cuda=False):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.cuda = cuda

    def forward(self, x):
        h = x
        batch_size, *_ = x.shape
        logdet_j = torch.zeros(batch_size)
        if self.cuda:
            logdet_j = logdet_j.cuda()

        for flow in self.flows:
            h, logdet_j_inc = flow.forward(h)
            logdet_j = logdet_j + logdet_j_inc
        return h, logdet_j

    def backward(self, y):
        h = y
        for flow in self.flows[::-1]:
            h = flow.backward(h)
        return h


class AffineCouplingLayer(nn.Module):
    def __init__(self, mask, s, t, cuda=False):
        super().__init__()
        self.mask = mask
        self.s = s
        self.t = t
        self.tanh = nn.Tanh()
        self.tanh_scale = nn.Parameter(torch.tensor(1.))
        self.cuda = cuda

    def forward(self, x):
        """
        Perform an affine coupling
        y1 = x1
        y2 = x2.*exp(s(x1)) + t(x1)

        Return the transformed tensor plus log(det(J))
        """
        x1 = (1 - self.mask) * x
        x2 = self.mask * x

        y1 = x1
        s = self.s(x1)
        s = self.tanh_scale * self.tanh(s)
        y2 = torch.exp(s)*x2 + self.t(x1)

        y = y1 + y2 * self.mask
        logdet_j = torch.sum(s, dim=[1, 2, 3])
        return y, logdet_j

    def backward(self, y):
        """
        Invert an affine coupling
        x1 = y1
        x2 = (y2 - t(y1)) .* exp(-s(y1))
        """
        y1 = (1 - self.mask) * y
        y2 = self.mask * y

        x1 = y1
        s = self.s(y1)
        s = self.tanh_scale * self.tanh(s)
        x2 = torch.exp(-s) * (y2 - self.t(y1))

        x = x1 + x2 * self.mask
        return x


class BatchNormLayer(nn.Module):
    def __init__(self, momentum=0.9):
        super().__init__()
        self.mu = 0
        self.sigma_sq = 1
        self.momentum = momentum

    def forward(self, x, eps=1e-6):
        mu = torch.mean(x, dim=0).detach()
        sigma_sq = torch.mean((x-mu)**2, dim=0).detach()
        self.mu = self.momentum*self.mu + (1-self.momentum)*mu
        self.sigma_sq = self.momentum*self.sigma_sq + (1-self.momentum)*sigma_sq

        y = (x - self.mu) / (self.sigma_sq + eps)**0.5
        # logdet_j = -0.5 * torch.sum(torch.log(sigma_sq))
        logdet_j = 0  # there are no trainable params in this, so dont bother
        return y, logdet_j

    def backward(self, y, eps=1e-6):
        """We will only do this to generate samples.
        Can use the average stats, ie not just for minibatch"""
        # mu = self.acc_x / float(self.n)
        # sigma_sq = self.acc_x_sq / float(self.n) - mu**2

        return y * (self.sigma_sq + eps)**0.5 + self.mu
