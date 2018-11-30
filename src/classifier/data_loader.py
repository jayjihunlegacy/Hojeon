from PIL import Image
import numpy as np
import os
import random

class Dataset:
    def __init__(self):
        self.savedir = '/home/annajung/PycharmProjects/hojeon/poisson_editing/pb_image/'
        self.get_file_list()

    def get_file_list(self):
        train_defect1 = self.savedir + 'train/'
        train_defect2 = self.savedir + 'train_dif_coordinates/'
        train_normal = self.savedir + 'train_normal/'
        test_defect1 = self.savedir + 'test/'
        test_defect2 = self.savedir + 'test_dif_coordinates/'
        test_normal = self.savedir + 'test_normal/'
        f = open('/home/annajung/PycharmProjects/hojeon/poisson_editing/pb_image_size_300/train_data.txt', 'r')
        self.train_total_image_list = f.readlines()
        self.train_defect_list = [train_defect1 + s for s in os.listdir(train_defect1) if '.jpg' in s] + [train_defect2 + s for s in os.listdir(train_defect2) if '.jpg' in s]
        self.train_no_defect_list = [train_normal + s for s in os.listdir(train_normal) if '.jpg' in s]
        self.test_defect_list = [test_defect1 + s for s in os.listdir(test_defect1) if '.jpg' in s] + [test_defect2 + s for s in os.listdir(test_defect2) if '.jpg' in s]
        self.test_no_defect_list = [test_normal + s for s in os.listdir(test_normal) if '.jpg' in s]

    def get_train_batch(self, batch_size):
        imgs = np.zeros([batch_size, 3, 32, 32])
        labels = np.zeros([batch_size], dtype=np.int8)
        for i in range(0, int(batch_size / 2)):
            idx = random.randint(0, len(self.train_defect_list) - 1)
            im = Image.open(self.train_defect_list[idx])
            imgs[i] = np.transpose(np.array(im), (2,0,1))
            labels[i] = 1 # defect
        for i in range(int(batch_size / 2), batch_size):
            idx = random.randint(0, len(self.train_no_defect_list) - 1)
            im = Image.open(self.train_no_defect_list[idx])
            imgs[i] = np.transpose(np.array(im), (2,0,1))
            labels[i] = 0 # no_defect

        return imgs, labels

    def get_test_batch(self, first_index, last_index, type = 'defect'):
        imgs = np.zeros([last_index - first_index, 3, 32, 32])
        labels = np.zeros([last_index - first_index], dtype=np.int8)

        if type == 'defect':
            for i in range(first_index, last_index):
                im = Image.open(self.test_defect_list[i])
                imgs[i - first_index] = np.transpose(np.array(im), (2,0,1))
                labels[i - first_index] = 1
        if type == 'no_defect':
            for i in range(first_index, last_index):
                im = Image.open(self.test_no_defect_list[i])
                imgs[i - first_index] = np.transpose(np.array(im), (2,0,1))
                labels[i - first_index] = 0

        return imgs, labels


