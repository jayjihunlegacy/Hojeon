import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel
import os
import numpy as np
from PIL import Image
from model import model
from data_for_total import Dataset
import time
import matplotlib.pyplot as plt

data_dir = '/media/ssd/dlwjdqja1958/Hojeon/'
test_defect_DSL = np.load(data_dir + 'removed_test_defect_DSL.npy')
test_normal_DSL = np.load(data_dir + 'removed_test_normal_DSL.npy')
test_defect_ISPL = np.load(data_dir + 'removed_test_defect_ISPL.npy')
test_normal_ISPL = np.load(data_dir + 'removed_test_normal_ISPL.npy')

test_data_path = '/media/ssd/dlwjdqja1958/Hojeon/dataset/'
all_test_defect_list_DSL = sorted([test_data_path + 'defect_DSL/' + s for s in os.listdir(test_data_path + 'defect_DSL/') if '.jpg' in s])
all_test_defect_list_ISPL = sorted([test_data_path + 'defect_ISPL/' + s for s in os.listdir(test_data_path + 'defect_ISPL/') if '.jpg' in s])
all_test_normal_list_DSL = sorted([test_data_path + 'normal_DSL/' + s for s in os.listdir(test_data_path + 'normal_DSL/') if '.jpg' in s])
all_test_normal_list_ISPL = sorted([test_data_path + 'normal_ISPL/' + s for s in os.listdir(test_data_path + 'normal_ISPL/') if '.jpg' in s])
test_defect_list_DSL = []
test_defect_list_ISPL = []
test_normal_list_DSL = []
test_normal_list_ISPL = []
remove_list = ['170941','171001','171022','171119','171154','171249','171309','171401','171448','171507','171513','171522','171735','171843','171929','171954','172135','172150','172346','172427','172507','172524','172543','172543','172555','172633','172652','172744','173147','173154','173320','173356','173514','173536','173553','173735','173815','173904','174420','174446','174724','174737','174817','174823','174840','174856','174912','174937','175010','175021','175141','175152','175204','175210','175337','175346','175350','175403','175406','175543','175558','175756','175836','175931','180001','180013','180018','180033','180316']
print(len(remove_list))
for list in all_test_defect_list_DSL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_defect_list_DSL.append(list)
for list in all_test_defect_list_ISPL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_defect_list_ISPL.append(list)
for list in all_test_normal_list_DSL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_normal_list_DSL.append(list)
for list in all_test_normal_list_ISPL:
    if list.split('_')[-1].split('.jpg')[0] not in remove_list:
        test_normal_list_ISPL.append(list)

thresholds = []

Recalls = []
Precisions = []
permax = 0


for threshold in range(1, 1000):
    threshold = threshold / 1000
    thresholds.append(threshold)
    defect_right_DSL = 0
    normal_right_DSL = 0
    defect_right_ISPL = 0
    normal_right_ISPL = 0

    for i in range(test_defect_DSL.shape[0]):
        if np.max(test_defect_DSL[i]) > threshold:
            defect_right_DSL += 1

    for i in range(test_normal_DSL.shape[0]):
        if np.max(test_normal_DSL[i]) < threshold:
            normal_right_DSL += 1

    for i in range(test_defect_ISPL.shape[0]):
        if np.max(test_defect_ISPL[i]) > threshold:
            defect_right_ISPL += 1

    for i in range(test_normal_ISPL.shape[0]):
        if np.max(test_normal_ISPL[i]) < threshold:
            normal_right_ISPL += 1

    if defect_right_DSL + defect_right_ISPL + normal_right_DSL + normal_right_ISPL > permax:
        permax = defect_right_DSL + defect_right_ISPL + normal_right_DSL + normal_right_ISPL
        print(threshold)


threshold = 0.275
defect_right_DSL = 0
normal_right_DSL = 0
defect_right_ISPL = 0
normal_right_ISPL = 0

incorrect_defect_DSL = []
incorrect_defect_ISPL = []
incorrect_normal_DSL = []
incorrect_normal_ISPL = []
print(test_defect_DSL.shape)
defect = 0
normal = 0
for i in range(test_defect_DSL.shape[0]):
    # print(np.max(test_defect[i]))
    if np.max(test_defect_DSL[i]) > threshold:
        defect_right_DSL += 1
        defect += 1
    else:
        normal += 1
        incorrect_defect_DSL.append(test_defect_list_DSL[i].split('/')[-1])
        print()
for i in range(test_normal_DSL.shape[0]):
    if np.max(test_normal_DSL[i]) < threshold:
        normal += 1
        normal_right_DSL += 1
    else:
        defect += 1
        incorrect_normal_DSL.append(test_normal_list_DSL[i].split('/')[-1])

for i in range(test_defect_ISPL.shape[0]):
    # print(np.max(test_defect[i]))
    if np.max(test_defect_ISPL[i]) > threshold:
        defect += 1
        defect_right_ISPL += 1
    else:
        normal += 1
        incorrect_defect_ISPL.append(test_defect_list_ISPL[i].split('/')[-1])


for i in range(test_normal_ISPL.shape[0]):
    if np.max(test_normal_ISPL[i]) < threshold:
        normal += 1
        normal_right_ISPL += 1
    else:
        defect += 1
        incorrect_normal_ISPL.append(test_normal_list_ISPL[i].split('/')[-1])


print("DSL")
print(defect_right_DSL/test_defect_DSL.shape[0])
print(normal_right_DSL/test_normal_DSL.shape[0])
print("ISPL")
print(defect_right_ISPL/test_defect_ISPL.shape[0])
print(normal_right_ISPL/test_normal_ISPL.shape[0])

x = [incorrect_defect_DSL, incorrect_defect_ISPL, incorrect_normal_DSL, incorrect_normal_ISPL]
print(incorrect_defect_DSL)
print(incorrect_defect_ISPL)
print(incorrect_normal_DSL)
print(incorrect_normal_ISPL)
print(len(x[0]))
print(len(x[1]))
print(len(x[2]))
print(len(x[3]))


print("=======================================")
print("gumchulryul : %f" % ((defect_right_DSL + defect_right_ISPL) / (test_defect_DSL.shape[0] + test_defect_ISPL.shape[0])))
print("ohinsickryul : %f" % (1 - (normal_right_DSL + normal_right_ISPL) / (test_normal_DSL.shape[0] + test_normal_ISPL.shape[0])))
print(defect_right_DSL + defect_right_ISPL)
print(test_defect_DSL.shape[0] + test_defect_ISPL.shape[0])
print(normal_right_DSL + normal_right_ISPL)
print(test_normal_DSL.shape[0] + test_normal_ISPL.shape[0])
print(test_normal_DSL.shape[0])

