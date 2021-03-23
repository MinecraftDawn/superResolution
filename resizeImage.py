import os
import cv2 as cv
import numpy as np
from multiprocessing.pool import ThreadPool

img_dir = './archive/'
target_dir = './images/'

paths = os.listdir(img_dir)

pool = ThreadPool(32)

def resizeImage(image: np.ndarray, filename:str):
    fullImage = cv.resize(image, (500, 500))
    smallImage = cv.resize(fullImage, (250, 250))
    cv.imwrite(target_dir + 'big/' + filename, fullImage)
    cv.imwrite(target_dir + 'small/' + filename, smallImage)

def loadImages(paths: list):
    for path in paths:
        originImage = cv.imread(img_dir + path, cv.IMREAD_COLOR)
        size = min(originImage.shape[:2])
        originImage = originImage[:size, :size]
        pool.apply_async(resizeImage, args=(originImage, path))
    pool.close()
    pool.join()
    print("Resize done")



loadImages(paths)