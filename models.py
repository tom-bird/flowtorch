import torch
from torch import nn
from torch.distributions import Normal

from flows import AdditiveCouplingLayer


class StackedAdditiveCouplingFlow(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.x_dim = x_dim
        mid = x_dim // 2
        mask1 = torch.cat((torch.ones(mid), torch.zeros(x_dim - mid)))
        mask2 = 1 - mask1

        conditioner1 = nn.Linear(mid, mid)
        conditioner2 = nn.Linear(x_dim - mid, x_dim - mid)

        self.flow1 = AdditiveCouplingLayer(mask=mask1.byte(), conditioner=conditioner1)
        self.flow2 = AdditiveCouplingLayer(mask=mask2.byte(), conditioner=conditioner2)

    def forward(self, x):
        h = self.flow1(x)
        y = self.flow2(h)
        return y

    def backward(self, y):
        h = self.flow2.backward(y)
        x = self.flow1.backward(h)
        return x

    def loss(self, x):
        """Additive coupling layers have unit jacobian so we can just optimise the log probs"""
        y = self.forward(x.view(-1, self.x_dim))
        return - torch.mean(torch.sum(Normal(0, 1).log_prob(y), dim=1)) / 784.

    def sample(self, n):
        y = Normal(0, 1).sample_n(n)
        x = self.backward(y)
        return x