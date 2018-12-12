import glob
import numpy as np
import random

# from matplotlib.image import imread
from scipy.misc import imread
import os
from functools import reduce


def rlistdir(path):
    childs = [os.path.join(path, child) for child in os.listdir(path)]
    dirs = list(filter(os.path.isdir, childs))
    files = list(filter(os.path.isfile, childs))
    if len(dirs) != 0:
        return files + reduce(lambda x, y: x + y, list(map(rlistdir, dirs)))
    else:
        return files


class ClothGenerator:
    def __init__(self, img_path='/home/annajung/hard_symlink/hojeon/clothes'):
        folders = os.listdir(img_path)
        folders.sort()
        self.img_path = img_path
        self.folders = folders
        self.image_names = {folder: os.listdir(os.path.join(img_path, folder)) for folder in folders}

    def generate(self, cloth_type, shape=None):
        folder = self.folders[cloth_type]
        image_name = np.random.choice(self.image_names[folder])
        image_path = os.path.join(*[self.img_path, folder, image_name])
        return imread(image_path)
