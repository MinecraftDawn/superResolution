import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
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


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                              padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,
                               padding=1, bias=False)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3,
                                padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.convTran = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=4,
                                           stride=2, padding=1)

    def make_layer(self, block, nums):
        layers = []
        for _ in range(nums):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        out = self.convTran(out)
        return out

