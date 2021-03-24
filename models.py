import torch
import torch.nn as nn
from torchsummary import summary


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


class Conv_Normal_Relu_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Normal_Relu_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.normal1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.normal2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.PReLU()

    def forward(self, x):
        x = self.relu1(self.normal1(self.conv1(x)))
        x = self.relu2(self.normal2(self.conv2(x)))
        return x


class UnetSR(nn.Module):
    def __init__(self):
        super(UnetSR, self).__init__()
        # Down
        self.input = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv1 = Conv_Normal_Relu_Block(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = Conv_Normal_Relu_Block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = Conv_Normal_Relu_Block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = Conv_Normal_Relu_Block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = Conv_Normal_Relu_Block(512, 1024)

        # Up
        self.cvtr6 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
                                        kernel_size=2, stride=2, padding=0)
        self.conv6 = Conv_Normal_Relu_Block(1024 + 512, 512)
        self.cvtr7 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=2, stride=2, padding=0)
        self.conv7 = Conv_Normal_Relu_Block(512 + 256, 256)
        self.cvtr8 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0)
        self.conv8 = Conv_Normal_Relu_Block(256 + 128, 128)
        self.cvtr9 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0)
        self.conv9 = Conv_Normal_Relu_Block(128 + 64, 64)

    def forward(self, x):
        x = self.input(x)
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        cvtr6 = self.cvtr6(conv5)

        merge6 = torch.cat([cvtr6, conv4], dim=1)
        conv6 = self.conv6(merge6)

        # conv6 = self.conv6(cvtr6)
        cvtr7 = self.cvtr7(conv6)

        merge7 = torch.cat([cvtr7, conv3], dim=1)
        conv7 = self.conv7(merge7)

        # conv7 = self.conv7(cvtr7)
        cvtr8 = self.cvtr8(conv7)

        merge8 = torch.cat([cvtr8, conv2], dim=1)
        conv8 = self.conv8(merge8)

        # conv8 = self.conv8(cvtr8)
        cvtr9 = self.cvtr9(conv8)

        merge9 = torch.cat([cvtr9, conv1], dim=1)
        conv9 = self.conv9(merge9)

        # conv9 = self.conv9(cvtr9)
        return conv9
