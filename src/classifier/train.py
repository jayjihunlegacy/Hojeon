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
from data_loader import Dataset
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logdir = '/home/dlwjdqja1958/PycharmProjects/hojeon/logs'
data = Dataset()
mymodel = model().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=0.0001)

total_iter = 100000
batch_size = 32
total_loss = 0
display_step = 100
test_step = 100

gpu = []
# max_acc = 0.98
start_time = time.time()
for iter in range(total_iter):
    # mymodel.train()
    imgs, labels = data.get_train_batch(batch_size=batch_size)
    imgs = torch.Tensor(imgs)
    labels = torch.LongTensor(labels.tolist())
    inputs, labels = Variable(imgs.cuda()), Variable(labels.cuda())

    optimizer.zero_grad()
    outputs = nn.parallel.data_parallel(mymodel, inputs, gpu)
    # print(outputs)
    loss = criterion(outputs, labels)
    total_loss += loss.data[0]
    loss.backward()
    optimizer.step()

    if (iter + 1) % display_step == 0:
        mean_loss = total_loss / display_step
        _, pr = torch.max(outputs.data, 1)
        # print("Train Accuracy : %f" % ((pr == labels.data).sum() / 16))
        print('[%5d] loss: %.12f' % (iter + 1, mean_loss))
        print("Train Accuracy : %f" % ((pr == labels.data).sum() / batch_size))
        total_loss = 0

    if (iter + 1) % test_step == 0:

        num_of_test_defect = len(data.test_defect_list)
        num_of_test_no_defect = len(data.test_no_defect_list)
        test_iter_defect = round(num_of_test_defect / batch_size)
        test_iter_no_defect = round(num_of_test_no_defect / batch_size)
        # print(num_of_test_defect + num_of_test_no_defect)
        correct1 = 0
        total1 = 0
        # mymodel.eval()
        for i in range(test_iter_defect):
            if i != test_iter_defect - 1:
                test_imgs, test_label = data.get_test_batch(batch_size * i, batch_size * (i + 1), 'defect')
            else:
                test_imgs, test_label = data.get_test_batch(batch_size * i, num_of_test_defect, 'defect')

            imgs = torch.Tensor(test_imgs)
            labels = torch.LongTensor(test_label.tolist())
            inputs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
            outputs = nn.parallel.data_parallel(mymodel, inputs, gpu)
            _, pr = torch.max(outputs.data, 1)
            correct1 += (pr == labels.data).sum()
            total1 += labels.size(0)

        correct2 = 0
        total2 = 0

        for i in range(test_iter_no_defect):
            if i != test_iter_no_defect - 1:
                test_imgs, test_label = data.get_test_batch(batch_size * i, batch_size * (i + 1), 'no_defect')
            else:
                test_imgs, test_label = data.get_test_batch(batch_size * i, num_of_test_no_defect, 'no_defect')

            imgs = torch.Tensor(test_imgs)
            labels = torch.LongTensor(test_label.tolist())
            inputs, labels = Variable(imgs.cuda()), Variable(labels.cuda())
            outputs = nn.parallel.data_parallel(mymodel, inputs, gpu)
            _, pr = torch.max(outputs.data, 1)
            correct2 += (pr == labels.data).sum()

            total2 += labels.size(0)

        print("Test Accuracy with %d iters, defect accuracy : %f, no_defect accuracy : %f, total_accuracy : %f" %(iter, correct1/total1, correct2/total2, (correct1+correct2)/(total1+total2)))
        if (correct1+correct2)/(total1+total2) > 0.98:
            torch.save(mymodel.state_dict(), '/home/dlwjdqja1958/PycharmProjects/hojeon/models/model_acc')
            max_acc = (correct1+correct2)/(total1+total2)
            print("Model Saved !!")

        # print("%s seconds until now" % (time.time() - start_time))
