# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 20:47:46 2014

@author: huajh
"""

# 300x300 size dataset generation

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.sparse import linalg

import pathlib
import random
from random import randint
import numpy

import scipy.misc

from random import shuffle
import shutil
import os


from os import listdir
from os.path import isfile, join


def lined_target(target):
    image_size = 250
    target_np = np.array(Image.open(target))
    target_np = target_np[1300:1550, 1500:1750]

    target = Image.fromarray(target_np, 'RGB')
    pix = target.load()

    draw = ImageDraw.Draw(target)
    oy = randint(0, 150)
    ox = randint(0, 150)
    tny = randint(50, image_size-oy)
    tnx = randint(50, image_size-ox)
    # oy, tny, ox, tnx

    red_diff = randint(0, 20)
    #green_diff = randint(0, 20)
    #blue_diff = randint(0, 20)

    width = randint(0, 5)

    draw.line((oy, ox, tny, tnx), fill=(max(pix[10, 10][0]-red_diff, 0), max(pix[10, 10][1]-red_diff, 0), max(pix[10, 10][2]-red_diff, 0)), width=width)
    return target, oy, tny, ox, tnx


class SeamlessEditingTool:

    def __init__(self, ref, target, mask, f, file_index, iter_making_train_from_one_image, iter):


        self.target = np.array(Image.open(target))
        self.target = self.target[1300:1550, 1500:1750]

        rotate_angle = randint(0, 180)

        ref_width = randint(100, 200)
        ref_hieght = randint(100, 200)
        #self.ref = np.array(Image.open(ref))
        self.mask = np.array(Image.open(mask).resize((ref_width, ref_hieght)).rotate(rotate_angle))

        img_temp = Image.fromarray(self.target, 'RGB')
        pix = img_temp.load()
        print(pix[10, 10])

        #img_temp.show()

        self.ref = np.array(Image.open(ref).resize((ref_width, ref_hieght)).rotate(rotate_angle).convert('RGBA'))
        #self.mask = np.array(Image.open(mask).convert('RGBA'))
        red, green, blue, alpha = self.ref.T

        red_diff = randint(0, 20)
        #green_diff = randint(0, 20)
        #blue_diff = randint(0, 20)

        black_areas = (red == 0) & (blue == 0) & (green == 0)
        self.ref[..., :-1][black_areas.T] = (max(pix[10, 10][0]-red_diff, 0), max(pix[10, 10][1]-red_diff, 0), max(pix[10, 10][2]-red_diff, 0))
        #self.mask[..., :-1][black_areas.T] = (max(pix[10, 10][0] - 20, 0), max(pix[10, 10][1] - 20, 0), max(pix[10, 10][2] - 20, 0))

        #im = Image.fromarray(self.target)
        #im.show()

        #im2 = Image.fromarray(self.ref)
        #im2 = im2.rotate(randint(0, 180))
        #im2.show()

        #im3 = Image.fromarray(self.mask)
        #im3.show()

        self.target_size_y = self.target.shape[0] - 5
        self.target_size_x = self.target.shape[1] - 5
        #resize_random = random.uniform(1, 1.3)

        #self.ref = scipy.misc.imresize(self.ref, resize_random, interp="bicubic")
        #self.mask = scipy.misc.imresize(self.mask, resize_random, interp="bicubic")

        flip_or_not = random.randint(0, 1)

        if flip_or_not is 0:
            self.ref = np.flip(self.ref, axis=1)
            self.mask = np.flip(self.mask, axis=1)

        self.tny = self.ref.shape[0]
        self.tnx = self.ref.shape[1]

        self.oy = random.randint(0, self.target_size_y - self.tny)
        self.ox = random.randint(0, self.target_size_x - self.tnx)

        #           defect_place_x:defect_place_x + random_defect_size_x, :]
        f.write('%s.jpg, %s, %s, (%d, %d, %d, %d)\n' % (str((file_index*iter_making_train_from_one_image)+iter), target,
                                                        ref, self.oy, self.oy+self.tny, self.ox, self.ox+self.tnx))

        zeros = numpy.zeros_like(self.target)
        self.ref_resized = zeros.copy()
        self.ref_resized[self.oy:self.oy + self.tny, self.ox:self.ox + self.tnx, 0] = self.ref[:, :, 0]
        self.ref_resized[self.oy:self.oy + self.tny, self.ox:self.ox + self.tnx, 1] = self.ref[:, :, 1]
        self.ref_resized[self.oy:self.oy + self.tny, self.ox:self.ox + self.tnx, 2] = self.ref[:, :, 2]

        mask = numpy.zeros_like(self.target)
        self.mask_resized = mask.copy()
        self.mask_resized[self.oy:self.oy + self.tny, self.ox:self.ox + self.tnx, 0] = self.mask[:, :, 0]
        self.mask_resized[self.oy:self.oy + self.tny, self.ox:self.ox + self.tnx, 1] = self.mask[:, :, 1]
        self.mask_resized[self.oy:self.oy + self.tny, self.ox:self.ox + self.tnx, 2] = self.mask[:, :, 2]

        self.height, self.width, blank = self.ref_resized.shape
        # (width, height)-tuple
        self.newImage = Image.new('RGB', (self.width, self.height))
        # index of mask
        # map coordinate of pixels to be calculated to index_map according to
        # mask
        self.maskidx2Corrd = []
        # map coordinates of neigbourhoods to mask indices
        self.Coord2indx = -1 * np.ones([self.height, self.width])

        # True  if  q \in N_p \bigcap \Sigma
        # False elsewise
        # at boundary
        self.if_strict_interior = []  # left, right, top, botton
        idx = 0

        for i in range(self.height):
            for j in range(self.width):
                if self.mask_resized[i, j, 0] == 255:
                    self.maskidx2Corrd.append([i, j])
                    self.if_strict_interior.append([
                        i > 0 and self.mask_resized[i - 1, j, 0] == 255,
                        i < self.height - 1 and self.mask_resized[i + 1, j, 0] == 255,
                        j > 0 and self.mask_resized[i, j - 1, 0] == 255,
                        j < self.width - 1 and self.mask_resized[i, j + 1, 0] == 255
                    ])
                    self.Coord2indx[i][j] = idx
                    idx += 1

        # number of mask
        N = idx
        self.b = np.zeros([N, 3])
        self.A = np.zeros([N, N])

    def return_mura_size(self):
        return (self.oy, self.tny, self.ox, self.tnx)

    def create_possion_equation(self):

        # Using the finite difference method
        N = self.b.shape[0]
        for i in range(N):
            # for every pixel in interior and boundary
            self.A[i, i] = 4
            x, y = self.maskidx2Corrd[i]
            if self.if_strict_interior[i][0]:
                self.A[i, int(self.Coord2indx[x - 1, y])] = -1
            if self.if_strict_interior[i][1]:
                self.A[i, int(self.Coord2indx[x + 1, y])] = -1
            if self.if_strict_interior[i][2]:
                self.A[i, int(self.Coord2indx[x, y - 1])] = -1
            if self.if_strict_interior[i][3]:
                self.A[i, int(self.Coord2indx[x, y + 1])] = -1

        # Row-based linked list sparse matrix
        # This is an efficient structure for
        # constructing sparse matrices incrementally.
        self.A = sparse.lil_matrix(self.A, dtype=int)

        for i in range(N):
            flag = np.mod(
                np.array(self.if_strict_interior[i], dtype=int) + 1, 2)
            x, y = self.maskidx2Corrd[i]
            for j in range(3):

                self.b[i, j] = 4 * self.ref_resized[x, y, j] - self.ref_resized[x - 1, y, j] - \
                    self.ref_resized[x + 1, y, j] - self.ref_resized[x, y - 1, j] - self.ref_resized[x, y + 1, j]
                self.b[i, j] += flag[0] * self.target[x - 1, y, j] + \
                    flag[1] * self.target[x + 1, y, j] + flag[2] * \
                    self.target[x, y - 1, j] + \
                    flag[3] * self.target[x, y + 1, j]

    def possion_solver(self):

        self.create_possion_equation()

        # Use Conjugate Gradient iteration to solve A x = b
        x_r = linalg.cg(self.A, self.b[:, 0])[0]
        x_g = linalg.cg(self.A, self.b[:, 1])[0]
        x_b = linalg.cg(self.A, self.b[:, 2])[0]

        self.newImage = self.target

        for i in range(self.b.shape[0]):
            x, y = self.maskidx2Corrd[i]
            self.newImage[x, y, 0] = np.clip(x_r[i], 0, 255)
            self.newImage[x, y, 1] = np.clip(x_g[i], 0, 255)
            self.newImage[x, y, 2] = np.clip(x_b[i], 0, 255)

        self.newImage = Image.fromarray(self.newImage)
        return self.newImage


if __name__ == "__main__":

    if os.path.exists('./pb_image_line/defect'):
        shutil.rmtree('./pb_image_line/defect')

    if os.path.exists('./pb_image_line/normal'):
        shutil.rmtree('./pb_image_line/normal')

    os.makedirs('./pb_image_line/defect', exist_ok=True)
    os.makedirs('./pb_image_line/normal', exist_ok=True)

    #mypath = '/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/simple/regular_pattern_small_repetitive'

    #files_1 = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    mypath = '/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/simple/textureless'

    files_2 = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

    files = files_2 # files_1 +

    defects = []

    defects = defects + ["/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/zigzag.png"]
    #defects = defects + ["/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/slim_zigzag.png"]
    #defects = defects + ["/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/spray_line.png"]
    #defects = defects + ["/media/hdd/annajung/DAGM/DAGM_crop/Class5/0589_crop.PNG"]
    #defects = defects + ["/media/hdd/annajung/DAGM/DAGM_crop/Class6/0728_crop.PNG"]
    #defects = defects + ["/media/hdd/annajung/DAGM/DAGM_crop/Class7/1287_crop.PNG"]
    #defects = defects + ["/media/hdd/annajung/DAGM/DAGM_crop/Class9/1397_crop.PNG"]
    # /media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/line.png
    # Image.open(defects[0]).show()
    # Image.open(defects[1]).show()
    # Image.open(defects[2]).show()
    # Image.open(defects[3]).show()
    # Image.open(defects[4]).show()
    # Image.open(defects[5]).show()
    # Image.open(defects[6]).show()

    masks = []

    masks = masks + ["/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/zigzag_mask.png"]
    #masks = masks + ["/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/slim_zigzag_mask.png"]
    #masks = masks + ["/media/ssd/annajung/PycharmProjects/hojeon/poisson_editing/line_defect/spray_line_mask.png"]
    #masks = masks + ["/media/hdd/annajung/DAGM/DAGM_crop/Class5/0589_crop_mask.PNG"]
    #masks = masks + ["/media/hdd/annajung/DAGM/DAGM_crop/Class6/0728_crop_mask.PNG"]
    #masks = masks + ["/media/hdd/annajung/DAGM/DAGM_crop/Class7/1287_crop_mask.PNG"]
    #masks = masks + ["/media/hdd/annajung/DAGM/DAGM_crop/Class9/1397_crop_mask.PNG"]

    # Image.open(masks[0]).show()
    # Image.open(masks[1]).show()
    # Image.open(masks[2]).show()
    # Image.open(masks[3]).show()
    # Image.open(masks[4]).show()
    # Image.open(masks[5]).show()
    # Image.open(masks[6]).show()

    num_defects = len(defects)

    with open('./pb_image_line/data.txt', 'w') as f:
        for train_file in files:
            f.write("%s\n" % train_file)

    iter_making_train_from_one_image = 5

    iter_making_normal_from_one_image = 1

    generate_normal = 1

    f = open('./pb_image_line/defect/coordinates.txt', 'w')

    for file_index in range(len(files)):
        target = files[file_index]

        target_list = target.split("/")

        only_file_name = target_list[len(target_list) - 1]

        files_jpg = only_file_name.split('.')

        only_file_name = files_jpg[0]

        os.makedirs('./pb_image_line/defect/' + only_file_name, exist_ok=True)

        for iter in range(iter_making_train_from_one_image):

            print(only_file_name)

            if randint(0, 1) is 0:
                defect_index = random.randint(0, num_defects - 1)

                ref = defects[defect_index]
                mask = masks[defect_index]

                tools = SeamlessEditingTool(ref, target, mask, f, file_index, iter_making_train_from_one_image, iter)

                newImage = tools.possion_solver()

                (oy, tny, ox, tnx) = tools.return_mura_size()
            else:
                (newImage, oy, tny, ox, tnx) = lined_target(target)

            image_name = 'def_' + str(ox) + '_' + str(oy) + '_' + str(tnx) + '_' + str(tny) + '_' + only_file_name

            newImage.save('./pb_image_line/%s/%s/%s.jpg' % ('defect', only_file_name, image_name))

        ######################################################################################################################################

        target = np.array(Image.open(target))
        target = target[1300:1550, 1500:1750]
        target = Image.fromarray(target)
        target.save('./pb_image_line/%s/%s.jpg' % ('normal', only_file_name))

    f.close()

    # f = open('./pb_image_size_300/test/coordinates.txt', 'w')
    #
    # for file_index in range(len(test_files)):
    #     target = test_files[file_index]
    #
    #     for iter in range(iter_making_train_from_one_image):
    #         defect_index = random.randint(0, num_defects - 1)
    #
    #         ref = defects[defect_index]
    #         mask = masks[defect_index]
    #
    #         tools = SeamlessEditingTool(ref, target, mask, f, file_index, iter_making_train_from_one_image, iter)
    #
    #         newImage = tools.possion_solver()
    #
    #         newImage.save('./pb_image_size_300/%s/%s.jpg' % ('test', str((file_index * iter_making_train_from_one_image) + iter)))
    #
    #     ########################################################################################################
    #
    #     target = np.array(Image.open(target))
    #     target = target[1100:1400, 1300:1600]
    #     target = Image.fromarray(target)
    #     target.save('./pb_image_size_300/%s/%s.jpg' % ('test_normal', str(file_index)))
    #
    # f.close()