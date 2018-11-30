import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import os
import numpy as np
from math import ceil

from PIL import Image
from artificial_data_total_image import generate_data
from model import model

isPoissonTest = True
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# gData = generate_data()
mymodel = model().cuda().eval()
mymodel.load_state_dict(torch.load('/home/dlwjdqja1958/PycharmProjects/hojeon/models/model_acc'))


train_defect_path = '/home/annajung/PycharmProjects/hojeon/poisson_editing/pb_image_size_300/train/'
train_normal_path = '/home/annajung/PycharmProjects/hojeon/poisson_editing/pb_image_size_300/train_normal/'
test_data_path = '/media/ssd/dlwjdqja1958/Hojeon/dataset/'


# train_normal_list = [train_normal_path + s for s in os.listdir(train_normal_path) if '.jpg' in s]
# train_defect_list = [train_defect_path + s for s in os.listdir(train_defect_path) if '.jpg' in s]

# test_defect_list_DSL = sorted([test_data_path + 'defect_DSL/' + s for s in os.listdir(test_data_path + 'defect_DSL/') if '.jpg' in s])
# test_defect_list_ISPL = sorted([test_data_path + 'defect_ISPL/' + s for s in os.listdir(test_data_path + 'defect_ISPL/') if '.jpg' in s])
# test_normal_list_DSL = sorted([test_data_path + 'normal_DSL/' + s for s in os.listdir(test_data_path + 'normal_DSL/') if '.jpg' in s])
# test_normal_list_ISPL = sorted([test_data_path + 'normal_ISPL/' + s for s in os.listdir(test_data_path + 'normal_ISPL/') if '.jpg' in s])
all_test_defect_list_DSL = sorted([test_data_path + 'defect_DSL/' + s for s in os.listdir(test_data_path + 'defect_DSL/') if '.jpg' in s])
all_test_defect_list_ISPL = sorted([test_data_path + 'defect_ISPL/' + s for s in os.listdir(test_data_path + 'defect_ISPL/') if '.jpg' in s])
all_test_normal_list_DSL = sorted([test_data_path + 'normal_DSL/' + s for s in os.listdir(test_data_path + 'normal_DSL/') if '.jpg' in s])
all_test_normal_list_ISPL = sorted([test_data_path + 'normal_ISPL/' + s for s in os.listdir(test_data_path + 'normal_ISPL/') if '.jpg' in s])
all_test_normal_list_total = sorted([test_data_path + 'normal_total/' + s for s in os.listdir(test_data_path + 'normal_ISPL/') if '.jpg' in s])
test_defect_list_DSL = []
test_defect_list_ISPL = []
test_normal_list_DSL = []
test_normal_list_ISPL = []
test_normal_list_total = []
remove_list = ['170941','171001','171022','171119','171154','171249','171309','171401','171448','171507','171513','171522','171735','171843','171929','171954','172135','172150','172346','172427','172507','172524','172543','172543','172555','172633','172652','172744','173147','173154','173320','173356','173514','173536','173553','173735','173815','173904','174420','174446','174724','174737','174817','174823','174840','174856','174912','174937','175010','175021','175141','175152','175204','175210','175337','175346','175350','175403','175406','175543','175558','175756','175836','175931','180001','180013','180018','180033','180316']
print(len(remove_list))
for list in all_test_defect_list_DSL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_defect_list_DSL.append(list)
for list in all_test_defect_list_ISPL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_defect_list_ISPL.append(list)
for list in all_test_normal_list_ISPL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_normal_list_ISPL.append(list)
for list in all_test_normal_list_DSL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_normal_list_DSL.append(list)

for list in all_test_normal_list_total:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_normal_list_total.append(list)
softmax = nn.Softmax()

# train_normal = np.zeros([len(train_normal_list), int(300 / 16) - 1, int(300 / 16) - 1])
# for idx, t in enumerate(train_normal_list):
#     cropped_img = Image.open(t)
#     for y in range(int(cropped_img.size[1] / 16) - 1):
#         imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
#         for x in range(int(cropped_img.size[0] / 16) - 1):
#             i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
#             i = np.transpose(np.array(i), (2,0,1))
#             imgs[x] = i
#         imgs = torch.Tensor(imgs)
#         inputs = Variable(imgs.cuda())
#         # print(inputs.shape)
#         outputs = nn.parallel.data_parallel(mymodel, inputs, [0, 1])
#         outputs = softmax(outputs)
#
#         for x in range(int(cropped_img.size[0] / 16) - 1):
#             train_normal[idx][x][y] = outputs[x][1]
#
#
# train_defect = np.zeros([len(train_defect_list), int(300 / 16) - 1, int(300 / 16) - 1])
# for idx, t in enumerate(train_defect_list):
#     cropped_img = Image.open(t)
#     for y in range(int(cropped_img.size[1] / 16) - 1):
#         imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
#         for x in range(int(cropped_img.size[0] / 16) - 1):
#             i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
#             i = np.transpose(np.array(i), (2,0,1))
#             imgs[x] = i
#         imgs = torch.Tensor(imgs)
#         inputs = Variable(imgs.cuda())
#         # print(inputs.shape)
#         outputs = nn.parallel.data_parallel(mymodel, inputs, [0, 1])
#         outputs = softmax(outputs)
#
#         for x in range(int(cropped_img.size[0] / 16) - 1):
#             train_defect[idx][x][y] = outputs[x][1]

test_defect_DSL = np.zeros([len(test_defect_list_DSL), int(300 / 16) - 1, int(300 / 16) - 1])
test_defect_ISPL = np.zeros([len(test_defect_list_ISPL), int(300 / 16) - 1, int(300 / 16) - 1])

for idx, t in enumerate(test_defect_list_DSL):
    cropped_img = Image.open(t)
    for y in range(int(cropped_img.size[1] / 16) - 1):
        imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
        for x in range(int(cropped_img.size[0] / 16) - 1):
            i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
            i = np.transpose(np.array(i), (2,0,1))
            imgs[x] = i
        imgs = torch.Tensor(imgs)
        inputs = Variable(imgs.cuda())
        # print(inputs.shape)
        outputs = nn.parallel.data_parallel(mymodel, inputs, [0])
        outputs = softmax(outputs)

        for x in range(int(cropped_img.size[0] / 16) - 1):
            test_defect_DSL[idx][x][y] = outputs[x][1]

for idx, t in enumerate(test_defect_list_ISPL):
    cropped_img = Image.open(t)
    for y in range(int(cropped_img.size[1] / 16) - 1):
        imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
        for x in range(int(cropped_img.size[0] / 16) - 1):
            i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
            i = np.transpose(np.array(i), (2,0,1))
            imgs[x] = i
        imgs = torch.Tensor(imgs)
        inputs = Variable(imgs.cuda())
        # print(inputs.shape)
        outputs = nn.parallel.data_parallel(mymodel, inputs, [0])
        outputs = softmax(outputs)

        for x in range(int(cropped_img.size[0] / 16) - 1):
            test_defect_ISPL[idx][x][y] = outputs[x][1]




test_normal_DSL = np.zeros([len(test_normal_list_DSL), int(300 / 16) - 1, int(300 / 16) - 1])
test_normal_ISPL = np.zeros([len(test_normal_list_ISPL), int(300 / 16) - 1, int(300 / 16) - 1])

for idx, t in enumerate(test_normal_list_DSL):
    cropped_img = Image.open(t)
    for y in range(int(cropped_img.size[1] / 16) - 1):
        imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
        for x in range(int(cropped_img.size[0] / 16) - 1):
            i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
            i = np.transpose(np.array(i), (2,0,1))
            imgs[x] = i
        imgs = torch.Tensor(imgs)
        inputs = Variable(imgs.cuda())
        # print(inputs.shape)
        outputs = nn.parallel.data_parallel(mymodel, inputs, [0])
        outputs = softmax(outputs)

        for x in range(int(cropped_img.size[0] / 16) - 1):
            test_normal_DSL[idx][x][y] = outputs[x][1]

for idx, t in enumerate(test_normal_list_ISPL):
    cropped_img = Image.open(t)
    for y in range(int(cropped_img.size[1] / 16) - 1):
        imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
        for x in range(int(cropped_img.size[0] / 16) - 1):
            i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
            i = np.transpose(np.array(i), (2, 0, 1))
            imgs[x] = i
        imgs = torch.Tensor(imgs)
        inputs = Variable(imgs.cuda())
        # print(inputs.shape)
        outputs = nn.parallel.data_parallel(mymodel, inputs, [0])
        outputs = softmax(outputs)

        for x in range(int(cropped_img.size[0] / 16) - 1):
            test_normal_ISPL[idx][x][y] = outputs[x][1]

test_normal_total = np.zeros([len(test_normal_list_total), int(300 / 16) - 1, int(300 / 16) - 1])

for idx, t in enumerate(test_normal_list_total):
    cropped_img = Image.open(t)
    for y in range(int(cropped_img.size[1] / 16) - 1):
        imgs = np.zeros([int(cropped_img.size[0] / 16) - 1, 3, 32, 32])
        for x in range(int(cropped_img.size[0] / 16) - 1):
            i = cropped_img.crop((x * 16, y * 16, x * 16 + 32, y * 16 + 32))
            i = np.transpose(np.array(i), (2,0,1))
            imgs[x] = i
        imgs = torch.Tensor(imgs)
        inputs = Variable(imgs.cuda())
        # print(inputs.shape)
        outputs = nn.parallel.data_parallel(mymodel, inputs, [0])
        outputs = softmax(outputs)

        for x in range(int(cropped_img.size[0] / 16) - 1):
            test_normal_total[idx][x][y] = outputs[x][1]
# np.save('/media/hdd/dlwjdqja1958/Hojeon/train_normal.npy', train_normal)
# np.save('/media/hdd/dlwjdqja1958/Hojeon/train_defect.npy', train_defect)
np.save('/media/ssd/dlwjdqja1958/Hojeon/removed_test_normal_total.npy', test_normal_DSL)
np.save('/media/ssd/dlwjdqja1958/Hojeon/removed_test_defect_DSL.npy', test_defect_DSL)
np.save('/media/ssd/dlwjdqja1958/Hojeon/removed_test_normal_ISPL.npy', test_normal_ISPL)
np.save('/media/ssd/dlwjdqja1958/Hojeon/removed_test_defect_ISPL.npy', test_defect_ISPL)
np.save('/media/ssd/dlwjdqja1958/Hojeon/removed_test_defect_total.npy', test_normal_total)
