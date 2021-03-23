import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from photoDataset import PhotoDataset
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHES = 1000
BATCH_SIZE = 16
LEARN_RATE = 0.0002
print(f'Using device: {DEVICE}')

transform = Compose([ToTensor()])
train_set = PhotoDataset(transform, transform)
trans_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,
                      padding=2, padding_mode='replicate'),
            nn.PReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1,
                      padding=0, padding_mode='replicate'),
            nn.PReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.PReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.PReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.PReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.PReLU(),



            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4,
                               stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        return self.main(x)


model = SuperResolution().to(DEVICE)
lossFunc = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARN_RATE, betas=(0.9,0.999))

def showImage(t):
    tmp = next(iter(trans_loader))
    t1 = tmp[0][0]
    t2 = tmp[1][0]
    t3 = model(t1.reshape(1, 3, 250, 250).to(DEVICE))
    t3 = t3.detach().cpu()[0]

    t1 = np.rollaxis(t1.numpy(), 0, 3)
    t2 = np.rollaxis(t2.numpy(), 0, 3)
    t3 = np.rollaxis(t3.numpy(), 0, 3)
    plt.figure(num=None, dpi=500)
    plt.subplot(1, 3, 1)
    plt.title('small')
    plt.imshow(t1)

    plt.subplot(1, 3, 2)
    plt.title('big')
    plt.imshow(t2)

    plt.subplot(1, 3, 3)
    plt.title('product')
    plt.imshow(t3)
    # plt.show()

    plt.savefig(f'target/test_{t+1}.png')
    plt.close()

def train(dataloader:DataLoader, model:nn.Module, optimizer:torch.optim.Optimizer):
    size = len(dataloader.dataset)
    for batch, (small, big) in enumerate(dataloader):
        batchsize = small.shape[0]
        small = small.to(DEVICE)
        big = big.to(DEVICE)

        product = model(small)
        loss = lossFunc(product, big)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch % 20 == 0:
            current = batch * batchsize
            print(f'loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]')

for t in range(100):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(trans_loader, model, optimizer)
    showImage(t)