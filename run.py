import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

from models import StackedAdditiveCouplingFlow

torch.manual_seed(17)


def train(model, epoch, data_loader, optimizer, log_interval=10):
    model.train()
    losses = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, np.mean(losses)))


def test(model, epoch, data_loader):
    model.eval()
    losses = []
    for data, _ in data_loader:
        loss = model.loss(data)
        losses.append(loss.item())
    print('\nEpoch: {}\tTest loss: {:.6f}\n\n'.format(
        epoch, np.mean(losses)
    ))


if __name__ == '__main__':
    epochs = 20
    batch_size = 100
    randomise_data = True

    model = StackedAdditiveCouplingFlow(784)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(1, epochs + 1):
            train(model, epoch, train_loader, optimizer)
            test(model, epoch, test_loader)
            model.sample(n=64, epoch=epoch)
    except KeyboardInterrupt:
        pass
    #     torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
    # torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
