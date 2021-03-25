import math
import numpy as np
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
        # self.pool4 = nn.MaxPool2d(2)
        # self.conv5 = Conv_Normal_Relu_Block(512, 1024)

        # Up
        # self.cvtr6 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
        #                                 kernel_size=2, stride=2, padding=0)
        # self.conv6 = Conv_Normal_Relu_Block(1024 + 512, 512)
        self.cvtr7 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=2, stride=2, padding=0)
        self.conv7 = Conv_Normal_Relu_Block(512 + 256, 256)
        self.cvtr8 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        kernel_size=2, stride=2, padding=0)
        self.conv8 = Conv_Normal_Relu_Block(256 + 128, 128)
        self.cvtr9 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        kernel_size=2, stride=2, padding=0)
        self.conv9 = Conv_Normal_Relu_Block(128 + 64, 64)
        self.conv10 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                         kernel_size=2, stride=2, padding=0)
        self.output = nn.Conv2d(in_channels=32, out_channels=3,
                                kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input(x)
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        # pool4 = self.pool4(conv4)
        # conv5 = self.conv5(pool4)
        # cvtr6 = self.cvtr6(conv5)


        # merge6 = torch.cat([cvtr6, conv4], dim=1)
        # conv6 = self.conv6(merge6)
        # cvtr7 = self.cvtr7(conv6)
        cvtr7 = self.cvtr7(conv4)
        merge7 = torch.cat([cvtr7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        cvtr8 = self.cvtr8(conv7)
        merge8 = torch.cat([cvtr8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        cvtr9 = self.cvtr9(conv8)
        merge9 = torch.cat([cvtr9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        output = self.output(conv10)
        return output


class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=224, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))

        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1))

        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1))

        return cout8_dense


class SRDenseNet(nn.Module):
    def __init__(self):
        super(SRDenseNet, self).__init__()

        self.relu = nn.PReLU()
        self.lowlevel = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(in_channels=2304, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.denseblock1 = self.make_layer(DenseBlock, 256)
        self.denseblock2 = self.make_layer(DenseBlock, 512)
        self.denseblock3 = self.make_layer(DenseBlock, 768)
        self.denseblock4 = self.make_layer(DenseBlock, 1024)
        self.denseblock5 = self.make_layer(DenseBlock, 1280)
        self.denseblock6 = self.make_layer(DenseBlock, 1536)
        self.denseblock7 = self.make_layer(DenseBlock, 1792)
        self.denseblock8 = self.make_layer(DenseBlock, 2048)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
            nn.PReLU(),
        )

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.relu(self.lowlevel(x))

        out = self.denseblock1(residual)
        concat = torch.cat([residual, out], 1)

        out = self.denseblock2(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock3(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock4(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock5(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock6(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock7(concat)
        concat = torch.cat([concat, out], 1)

        out = self.denseblock8(concat)
        out = torch.cat([concat, out], 1)

        out = self.bottleneck(out)

        out = self.deconv(out)

        out = self.reconstruction(out)

        return out