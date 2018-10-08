import torch
from torch import nn


class SequentialFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        h = x
        batch_size, _ = x.shape
        logdet_j = torch.zeros(batch_size)
        for flow in self.flows:
            h, logdet_j_inc = flow.forward(h)
            logdet_j = logdet_j + logdet_j_inc
        return h, logdet_j

    def backward(self, y):
        h = y
        for flow in self.flows[::-1]:
            h = flow.backward(h)
        return h


class AdditiveCouplingLayer(nn.Module):
    def __init__(self, mask, conditioner):
        super().__init__()
        self.mask = mask  # binary mask partitioning the space
        self.conditioner = conditioner

    def forward(self, x):
        """
        Perform an additive coupling
        y1 = x1
        y2 = x2 + m(x1)
        """
        batch_size, _ = x.shape
        x1 = torch.masked_select(x, 1 - self.mask).view(batch_size, -1)
        x2 = torch.masked_select(x, self.mask).view(batch_size, -1)

        y1 = x1
        y2 = x2 + self.conditioner(x1)
        y = torch.zeros(x.shape)
        y.masked_scatter_(1 - self.mask, y1)
        y.masked_scatter_(self.mask, y2)
        return y

    def backward(self, y):
        """
        Invert an additive coupling
        x1 = y1
        x2 = y2 - m(y1)
        """
        batch_size, _ = y.shape
        y1 = torch.masked_select(y, 1 - self.mask).view(batch_size, -1)
        y2 = torch.masked_select(y, self.mask).view(batch_size, -1)

        x1 = y1
        x2 = y2 - self.conditioner(y1)
        x = torch.zeros(y.shape)
        x.masked_scatter_(1 - self.mask, x1)
        x.masked_scatter_(self.mask, x2)
        return x


class AffineCouplingLayer(nn.Module):
    def __init__(self, mask, s, t):
        super().__init__()
        self.mask = mask
        self.s = s
        self.t = t
        self.tanh = nn.Tanh()
        self.tanh_scale = 2

    def forward(self, x):
        """
        Perform an affine coupling
        y1 = x1
        y2 = x2.*exp(s(x1)) + t(x1)

        Return the transformed tensor plus log(det(J))
        """
        batch_size, _ = x.shape
        x1 = torch.masked_select(x, 1 - self.mask).view(batch_size, -1)
        x2 = torch.masked_select(x, self.mask).view(batch_size, -1)

        y1 = x1
        s = self.s(x1)
        s = self.tanh_scale * self.tanh(s)
        y2 = torch.exp(s)*x2 + self.t(x1)

        y = torch.zeros(x.shape)
        y.masked_scatter_(1 - self.mask, y1)
        y.masked_scatter_(self.mask, y2)

        logdet_j = torch.sum(s, dim=1)
        return y, logdet_j

    def backward(self, y):
        """
        Invert an affine coupling
        x1 = y1
        x2 = (y2 - t(y1)) .* exp(-s(y1))
        """
        batch_size, _ = y.shape
        y1 = torch.masked_select(y, 1 - self.mask).view(batch_size, -1)
        y2 = torch.masked_select(y, self.mask).view(batch_size, -1)

        x1 = y1
        s = self.s(y1)
        s = self.tanh_scale * self.tanh(s)
        x2 = torch.exp(-s) * (y2 - self.t(y1))
        x = torch.zeros(y.shape)

        x.masked_scatter_(1 - self.mask, x1)
        x.masked_scatter_(self.mask, x2)
        return x


if __name__ == '__main__':
    mask = torch.tensor([1, 1, 0, 0]).byte()
    conditioner = nn.Linear(2, 2)
    affine_layer = AdditiveCouplingLayer(mask, conditioner)
    x = torch.tensor([0.1, 0.5, 1.0, 0.2])
    y = affine_layer.forward(x)
    print(y)
    recon_x = affine_layer.backward(y)
    print(recon_x)