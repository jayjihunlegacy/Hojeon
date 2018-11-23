import glob
import numpy as np
import random

from matplotlib.image import imread


class ClothGenerator:
    def __init__(self, img_path='/home/annajung/hard_symlink/hojeon/clothes/military_crop2'):
        self.img_list = glob.glob("%s/*" % img_path)
        self.len_img = len(self.img_list)

    def generate(self, shape=None):
        ran_number = random.randint(0, self.len_img)
        img = imread(self.img_list[ran_number])
        return img
