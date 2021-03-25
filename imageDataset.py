from torch.utils.data import Dataset
import os
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
import numpy as np


class PhotoDataset(Dataset):
    def __init__(self, img_dir, img_big_dir, img_small_dir,
                 transform=None, save=False, splitSize=(100, 100)):
        self.imgDir = img_dir
        self.imgDirBig = img_big_dir
        self.imgDirSmall = img_small_dir
        self.__createFolder()
        self.transform = transform
        self.splieSize = splitSize
        self.save = save

        self.pool = ThreadPool(256)
        self.paths = os.listdir(self.imgDir)
        self.images = []
        self.resizePath = []
        self.__loadImage()

    def __len__(self):
        return len(self.images) or len(self.resizePath)

    def __createFolder(self):
        if not os.path.exists(self.imgDirBig):
            os.makedirs(self.imgDirBig)
        if not os.path.exists(self.imgDirSmall):
            os.makedirs(self.imgDirSmall)

    def __loadImage(self):
        for path in self.paths:
            originImage = cv.imread(f'{self.imgDir}{path}')
            self.pool.apply_async(self.__splitImage, args=(originImage, path))
        self.pool.close()
        self.pool.join()

    def __splitImage(self, origin: np.ndarray, name: str):
        rows, cols, _ = origin.shape
        rs, cs = self.splieSize
        count = 0
        for row in range(rows // rs):
            for col in range(cols // cs):
                bigImage = origin[row * rs:(row + 1) * rs, col * cs:(col + 1) * cs]
                smallImage = cv.resize(bigImage, (rs // 2, cs // 2))
                if self.save:
                    name, subname = name.split('.')
                    cv.imwrite(f'{self.imgDirBig}{name}_{str(count).zfill(3)}.{subname}', bigImage)
                    cv.imwrite(f'{self.imgDirSmall}{name}_{str(count).zfill(3)}.{subname}', smallImage)
                    self.resizePath.append(f'{name}_{str(count).zfill(3)}.{subname}')
                    count += 1
                else:
                    self.images.append((smallImage, bigImage))

    def __getitem__(self, index):
        if self.save:
            path = self.resizePath[index]
            small = cv.imread(self.imgDirSmall + path, cv.IMREAD_COLOR)
            big = cv.imread(self.imgDirBig + path, cv.IMREAD_COLOR)
        else:
            small, big = self.images[index]

        if self.transform:
            small = self.transform(small)
        if self.transform:
            big = self.transform(big)

        return small, big


class TestDataset(Dataset):
    def __init__(self, transform=None, save=False, splitSize=(100, 100)):
        self.images = []
        self.transform = transform

        img = cv.imread("./img/CAT_00/00000001_000.jpg", cv.IMREAD_COLOR)
        x, y, _ = img.shape
        m = min(x, y)
        m = m if m % 2 == 0 else m - 1
        self.images.append(img[:m, :m])

        img = cv.imread("./img/CAT_00/00000001_005.jpg", cv.IMREAD_COLOR)
        x, y, _ = img.shape
        m = min(x, y)
        m = m if m % 2 == 0 else m - 1
        self.images.append(img[:m, :m])

    def __len__(self):
        return 2

    def __getitem__(self, index):
        img = self.images[index]
        half = img.shape[0] // 2
        small = img[:half, :half]
        img = img.astype(np.float32) / 255
        small = small.astype(np.float32) / 255
        if self.transform:
            img = self.transform(img)
            small = self.transform(small)

        return small, img
