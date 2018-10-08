import torch
from torch import nn
from torch.distributions import Normal
from torchvision.utils import save_image

from flows import AdditiveCouplingLayer, SequentialFlow


class SimpleConditioner(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x)
        h = self.nonlin(h)
        return self.fc2(h)


class StackedAdditiveCouplingFlow(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.x_dim = x_dim
        mid = x_dim // 2
        mask1 = torch.cat((torch.ones(mid), torch.zeros(x_dim - mid)))
        mask2 = 1 - mask1

        mask1 = mask1.byte()
        mask2 = mask2.byte()

        self.flow = SequentialFlow([AdditiveCouplingLayer(mask=mask1, conditioner=SimpleConditioner(mid, 50)),
                                    AdditiveCouplingLayer(mask=mask2, conditioner=SimpleConditioner(x_dim - mid, 50)),
                                    AdditiveCouplingLayer(mask=mask1, conditioner=SimpleConditioner(mid, 50)),
                                    AdditiveCouplingLayer(mask=mask2, conditioner=SimpleConditioner(x_dim - mid, 50)),
                                    AdditiveCouplingLayer(mask=mask1, conditioner=SimpleConditioner(mid, 50)),
                                    AdditiveCouplingLayer(mask=mask2, conditioner=SimpleConditioner(x_dim - mid, 50)),
                                    AdditiveCouplingLayer(mask=mask1, conditioner=SimpleConditioner(mid, 50)),
                                    AdditiveCouplingLayer(mask=mask2, conditioner=SimpleConditioner(x_dim - mid, 50))
                                    ])

    def forward(self, x):
        return self.flow.forward(x)

    def backward(self, y):
        return self.flow.backward(y)

    def loss(self, x):
        """Additive coupling layers have unit jacobian so we can just optimise the log probs"""
        y = self.forward(x.view(-1, self.x_dim))
        return - torch.mean(torch.sum(Normal(0, 1).log_prob(y), dim=1)) / 784.

    def sample(self, n, epoch):
        y = Normal(0, 1).sample((n, 784))
        x = self.backward(y)
        save_image(x.view(-1, 1, 28, 28), 'results/samples_epoch{}.png'.format(epoch))
