from torch.utils.data import Dataset
import os
import cv2 as cv


class PhotoDataset(Dataset):
    def __init__(self, transform=None, targetTransform=None):
        self.imgDirBig = './images/big/'
        self.imgDirSmall = './images/small/'
        self.transform = transform
        self.targetTransform = targetTransform

        # self.paths = os.listdir(self.imgDirBig)
        paths = os.listdir(self.imgDirBig)
        self.images = []
        self.__loadImages(paths)

    def __len__(self):
        return len(self.images)

    # def __getitem__(self, index):
    #     path = self.paths[index]
    #     small = cv.imread(self.imgDirSmall + path, cv.IMREAD_COLOR)
    #     big = cv.imread(self.imgDirBig + path, cv.IMREAD_COLOR)
    #     if self.transform:
    #         small = self.targetTransform(small)
    #     if self.targetTransform:
    #         big = self.transform(big)
    #
    #     return small, big

    def __loadImages(self, paths: list):
        for path in paths:
            smallImage = cv.imread(self.imgDirSmall + path, cv.IMREAD_COLOR)
            bigImage = cv.imread(self.imgDirBig + path, cv.IMREAD_COLOR)
            self.images.append((smallImage, bigImage))

    def __getitem__(self, index):
        small, big = self.images[index]
        if self.transform:
            small = self.targetTransform(small)
        if self.targetTransform:
            big = self.transform(big)

        return (small, big)
