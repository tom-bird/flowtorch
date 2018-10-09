import torch.utils.data
from torch import optim
from torchvision import datasets, transforms
import numpy as np

from models import StackedAffineCouplingFlow

torch.manual_seed(0)


def train(model, device, epoch, data_loader, optimizer, log_interval=10):
    model.train()
    losses = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
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


def test(model, device, epoch, data_loader):
    model.eval()
    losses = []
    for data, _ in data_loader:
        data = data.to(device)
        loss = model.loss(data)
        losses.append(loss.item())
    print('\nEpoch: {}\tTest loss: {:.6f}\n\n'.format(
        epoch, np.mean(losses)
    ))


class Dequantise:
    def __call__(self, pic):
        t = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        t = t.float()
        t = t.view(-1, *pic.size)

        # add uniform noise to each pixel
        eps = 1e-3  # if we dont clamp then we can get divergent logit
        noise = torch.clamp(torch.rand(t.shape), eps, 1 - eps)
        t = t + noise

        # rescale
        alpha = 0.05
        t = alpha + (1 - alpha) * t/256.

        # convert to logit
        t = torch.log(t) - torch.log(1 - t)
        return t


if __name__ == '__main__':
    epochs = 20
    batch_size = 100

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = StackedAffineCouplingFlow(x_shape=(1, 28, 28), cuda=use_cuda).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=Dequantise()),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=Dequantise()),
        batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(1, epochs + 1):
            train(model, device, epoch, train_loader, optimizer)
            test(model, device, epoch, test_loader)
            model.sample(device, n=64, epoch=epoch)
    except KeyboardInterrupt:
        pass
    #     torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
    # torch.save(model.state_dict(), 'saved_params/torch_binary_vae_params_new')
