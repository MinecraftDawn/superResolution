import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
from imageDataset import PhotoDataset, TestDataset
from models import FSRCNN, VDSR, UnetSR, SRDenseNet
import os
import math
import cv2 as cv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHES = 100
BATCH_SIZE = 16
LEARN_RATE = 0.0002
# LEARN_RATE = 0.0005
imageSize = (160, 160)
print(f'Using device: {DEVICE}')

transform = Compose([ToTensor()])
train_set = PhotoDataset(img_dir="./archive/", img_big_dir=".", img_small_dir=".",
                         transform=transform, splitSize=imageSize)
trans_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = TestDataset(transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)



model = SRDenseNet().to(DEVICE)
# model.load_state_dict(torch.load("VDSR.weight"))

lossFunc = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARN_RATE, betas=(0.9, 0.999))

def getPSNR(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR

def showTestImage(small, super, t):
    small = small.detach().cpu()[0]
    super = super.detach().cpu()[0]

    small = np.rollaxis(small.numpy(), 0, 3)
    super = np.rollaxis(super.numpy(), 0, 3)

    plt.figure(num=None, dpi=1000)
    plt.subplot(1, 3, 1)
    plt.title('small')
    plt.imshow(cv.cvtColor(small, cv.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Super Rsoulution')
    plt.imshow(cv.cvtColor(super, cv.COLOR_BGR2RGB))


    plt.savefig(f'target/test_{t + 1}.png')
    plt.close()


def showImage(t):
    tmp = next(iter(trans_loader))
    t1 = tmp[0][0]
    t2 = tmp[1][0]
    t3 = model(t1.reshape(1, 3, imageSize[0]//2, imageSize[1]//2).to(DEVICE))
    # t3 = model(t1.reshape(1, *t1.shape).to(DEVICE))
    t3 = t3.detach().cpu()[0]

    t1 = np.rollaxis(t1.numpy(), 0, 3)
    t2 = np.rollaxis(t2.numpy(), 0, 3)
    t3 = np.rollaxis(t3.numpy(), 0, 3)
    plt.figure(num=None, dpi=320)
    plt.subplot(1, 3, 1)
    plt.title('small')
    plt.imshow(cv.cvtColor(t1, cv.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Super Rsoulution')
    plt.imshow(cv.cvtColor(t3, cv.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title('big')
    plt.imshow(cv.cvtColor(t2, cv.COLOR_BGR2RGB))
    # plt.show()

    plt.savefig(f'target/train_{t+1}.png')
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

        if batch % 200 == 0:
            current = batch * batchsize
            print(f'loss: {loss.item():>7f} PSNR:{getPSNR(product, big)} [{current:>5d}/{size:>5d}]')


def test(dataloader:DataLoader, model:nn.Module):
    size = len(dataloader.dataset)
    model.eval()
    count = 0
    with torch.no_grad():
        for batch, (small, big) in enumerate(dataloader):
            batchsize = small.shape[0]
            small = small.to(DEVICE)
            big = big.to(DEVICE)

            product = model(small)
            loss = lossFunc(product, big)
            psnr = getPSNR(product, big)

            product = model(product)
            product = model(product)

            if batch % 1 == 0:
                count += 1
                showTestImage(small, product, count)
                current = batch * batchsize
                print(f'loss: {loss.item():>7f} PSNR:{psnr} [{current:>5d}/{size:>5d}]')


for t in range(EPOCHES):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(trans_loader, model, optimizer)
    showImage(t)

# test(test_loader, model)