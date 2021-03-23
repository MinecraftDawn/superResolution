import torch.nn as nn


# class SuperResolution(nn.Module):
#     def __init__(self):
#         super(SuperResolution, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,
#                       padding=2, padding_mode='replicate'),
#             nn.PReLU(),
#
#             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1,
#                       padding=0, padding_mode='replicate'),
#             nn.PReLU(),
#
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
#                       padding=1, padding_mode='replicate'),
#             nn.PReLU(),
#
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
#                       padding=1, padding_mode='replicate'),
#             nn.PReLU(),
#
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
#                       padding=1, padding_mode='replicate'),
#             nn.PReLU(),
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
#                       padding=1, padding_mode='replicate'),
#             nn.PReLU(),
#
#             nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4,
#                                stride=2, padding=1, bias=False),
#         )
#
#     def forward(self, x):
#         return self.main(x)


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
