import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GaussianNoise
from sklearn import svm
from fl_mnist_implementation_tutorial_utils import *
import pandas as pd
from sklearn import preprocessing
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision


def load(paths, verbose=-1):
    '''expects images for each class in seperate dir,
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        # image = np.array(im_gray).flatten()
        image = im_gray
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)

        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


def split_data_list_label(clients_all, client_names):
    client = clients_all[client_names]
    size = len(client)
    image_list_all = list()
    label_list_all = list()
    for i in range(size):
        image_list_all.append(client[i][0])
        cc = client[i][1]
        for j in range(10):
            if cc[j] == 1:
                label_list_all.append(j)
    return image_list_all, label_list_all


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization




img_path = 'E:/myself/FedGitHubDate/tutorial-master/tutorial-master/archive/trainingSet/trainingSet'
# get the path list using the path object
image_paths = list(paths.list_images(img_path))

# apply our function
image_list, label_list = load(image_paths, verbose=10000)
# binarize the labels
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

# split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list,
                                                    label_list,
                                                    test_size=0.1,
                                                    random_state=42)

# create clients
clients = create_clients(X_train, y_train, num_clients=10, initial='client')
# declear path to your mnist data folder

EPOCH = 10
# 节点1
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list1, label_list1 = split_data_list_label(clients, 'client_1')

# binarize the labels
# lb0 = LabelBinarizer()
# label_list0 = lb0.fit_transform(label_list0)

# split data into training and test set
x_train_s1, x_test_ss1, y_train_s1, y_test_ss1 = train_test_split(image_list1,
                                                                  label_list1,
                                                                  test_size=0.1,
                                                                  random_state=42)
train1 = list(zip(x_train_s1, y_train_s1))
x_test_s1 = torch.tensor(x_test_ss1).to(torch.float32).unsqueeze(1)
train_loader1 = Data.DataLoader(dataset = train1, batch_size=BATCH_SIZE, shuffle=True)
y_test_s1 = torch.tensor(y_test_ss1)

for epoch in range(EPOCH):
    for step1, (bb_x1, b_y1) in enumerate(train_loader1):   # gives batch data, normalize x when iterate train_loader
        bbb_x1 = bb_x1.unsqueeze(1)
        b_x1 = bbb_x1.to(torch.float32)
        output1 = cnn(b_x1)[0]               # cnn output
        loss1 = loss_func(output1, b_y1)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss1.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step1 % 50 == 0:
            test_output1, last_layer1 = cnn(x_test_s1)
            pred_y1 = torch.max(test_output1, 1)[1].data.numpy()
            accuracy1 = float((pred_y1 == y_test_s1.data.numpy()).astype(int).sum()) / float(y_test_s1.size(0))
            print('Epoch1: ', epoch, '| train loss: %.4f' % loss1.data.numpy(), '| test accuracy: %.4f' % accuracy1)


# train_batched1 = tf.data.Dataset.from_tensor_slices((x_train_s1, y_train_s1)).batch(len(y_train_s1))
# 节点2
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list2, label_list2 = split_data_list_label(clients, 'client_2')
# split data into training and test set
x_train_s2, x_test_ss2, y_train_s2, y_test_ss2 = train_test_split(image_list2,
                                                                label_list2,
                                                                test_size=0.1,
                                                                random_state=42)
train2 = list(zip(x_train_s2, y_train_s2))
x_test_s2 = torch.tensor(x_test_ss2).to(torch.float32).unsqueeze(1)
train_loader2 = Data.DataLoader(dataset = train2, batch_size=BATCH_SIZE, shuffle=True)
y_test_s2 = torch.tensor(y_test_ss2)

for epoch in range(EPOCH):
    for step2, (bb_x2, b_y2) in enumerate(train_loader2):   # gives batch data, normalize x when iterate train_loader
        bbb_x2 = bb_x2.unsqueeze(1)
        b_x2 = bbb_x2.to(torch.float32)
        output2 = cnn(b_x2)[0]               # cnn output
        loss2 = loss_func(output2, b_y2)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss2.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step2 % 50 == 0:
            test_output2, last_layer2 = cnn(x_test_s2)
            pred_y2 = torch.max(test_output2, 1)[1].data.numpy()
            accuracy2 = float((pred_y2 == y_test_s2.data.numpy()).astype(int).sum()) / float(y_test_s2.size(0))
            print('Epoch2: ', epoch, '| train loss: %.4f' % loss2.data.numpy(), '| test accuracy: %.4f' % accuracy2)
# 节点3
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list3, label_list3 = split_data_list_label(clients, 'client_3')
# split data into training and test set
x_train_s3, x_test_ss3, y_train_s3, y_test_ss3 = train_test_split(image_list3,
                                                                label_list3,
                                                                test_size=0.1,
                                                                random_state=42)
train3 = list(zip(x_train_s3, y_train_s3))
x_test_s3 = torch.tensor(x_test_ss3).to(torch.float32).unsqueeze(1)
train_loader3 = Data.DataLoader(dataset = train3, batch_size=BATCH_SIZE, shuffle=True)
y_test_s3 = torch.tensor(y_test_ss3)

for epoch in range(EPOCH):
    for step3, (bb_x3, b_y3) in enumerate(train_loader3):   # gives batch data, normalize x when iterate train_loader
        bbb_x3 = bb_x3.unsqueeze(1)
        b_x3 = bbb_x3.to(torch.float32)
        output3 = cnn(b_x3)[0]               # cnn output
        loss3 = loss_func(output3, b_y3)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss3.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step3 % 50 == 0:
            test_output3, last_layer3 = cnn(x_test_s3)
            pred_y3 = torch.max(test_output3, 1)[1].data.numpy()
            accuracy3 = float((pred_y3 == y_test_s3.data.numpy()).astype(int).sum()) / float(y_test_s3.size(0))
            print('Epoch3: ', epoch, '| train loss: %.4f' % loss3.data.numpy(), '| test accuracy: %.4f' % accuracy3)
# 节点4
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list4, label_list4 = split_data_list_label(clients, 'client_4')
# split data into training and test set
x_train_s4, x_test_ss4, y_train_s4, y_test_ss4 = train_test_split(image_list4,
                                                                label_list4,
                                                                test_size=0.1,
                                                                random_state=42)
train4 = list(zip(x_train_s4, y_train_s4))
x_test_s4 = torch.tensor(x_test_ss4).to(torch.float32).unsqueeze(1)
train_loader4 = Data.DataLoader(dataset = train4, batch_size=BATCH_SIZE, shuffle=True)
y_test_s4 = torch.tensor(y_test_ss4)

for epoch in range(EPOCH):
    for step4, (bb_x4, b_y4) in enumerate(train_loader4):   # gives batch data, normalize x when iterate train_loader
        bbb_x4 = bb_x4.unsqueeze(1)
        b_x4 = bbb_x4.to(torch.float32)
        output4 = cnn(b_x4)[0]               # cnn output
        loss4 = loss_func(output4, b_y4)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss4.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step4 % 50 == 0:
            test_output4, last_layer4 = cnn(x_test_s4)
            pred_y4 = torch.max(test_output4, 1)[1].data.numpy()
            accuracy4 = float((pred_y4 == y_test_s4.data.numpy()).astype(int).sum()) / float(y_test_s4.size(0))
            print('Epoch4: ', epoch, '| train loss: %.4f' % loss4.data.numpy(), '| test accuracy: %.4f' % accuracy4)
# 节点5
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list5, label_list5 = split_data_list_label(clients, 'client_5')
# split data into training and test set
x_train_s5, x_test_ss5, y_train_s5, y_test_ss5 = train_test_split(image_list5,
                                                                label_list5,
                                                                test_size=0.1,
                                                                random_state=42)
train5 = list(zip(x_train_s5, y_train_s5))
x_test_s5 = torch.tensor(x_test_ss5).to(torch.float32).unsqueeze(1)
train_loader5 = Data.DataLoader(dataset = train5, batch_size=BATCH_SIZE, shuffle=True)
y_test_s5 = torch.tensor(y_test_ss5)

for epoch in range(EPOCH):
    for step5, (bb_x5, b_y5) in enumerate(train_loader5):   # gives batch data, normalize x when iterate train_loader
        bbb_x5 = bb_x5.unsqueeze(1)
        b_x5 = bbb_x5.to(torch.float32)
        output5 = cnn(b_x5)[0]               # cnn output
        loss5 = loss_func(output5, b_y5)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss5.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step5 % 50 == 0:
            test_output5, last_layer5 = cnn(x_test_s5)
            pred_y5 = torch.max(test_output5, 1)[1].data.numpy()
            accuracy5 = float((pred_y5 == y_test_s5.data.numpy()).astype(int).sum()) / float(y_test_s5.size(0))
            print('Epoch5: ', epoch, '| train loss: %.4f' % loss5.data.numpy(), '| test accuracy: %.4f' % accuracy5)
# 节点6
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list6, label_list6 = split_data_list_label(clients, 'client_6')
# split data into training and test set
x_train_s6, x_test_ss6, y_train_s6, y_test_ss6 = train_test_split(image_list6,
                                                                label_list6,
                                                                test_size=0.1,
                                                                random_state=42)
train6 = list(zip(x_train_s6, y_train_s6))
x_test_s6 = torch.tensor(x_test_ss6).to(torch.float32).unsqueeze(1)
train_loader6 = Data.DataLoader(dataset = train6, batch_size=BATCH_SIZE, shuffle=True)
y_test_s6 = torch.tensor(y_test_ss6)

for epoch in range(EPOCH):
    for step6, (bb_x6, b_y6) in enumerate(train_loader6):   # gives batch data, normalize x when iterate train_loader
        bbb_x6 = bb_x6.unsqueeze(1)
        b_x6 = bbb_x6.to(torch.float32)
        output6 = cnn(b_x6)[0]               # cnn output
        loss6 = loss_func(output6, b_y6)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss6.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step6 % 50 == 0:
            test_output6, last_layer6 = cnn(x_test_s6)
            pred_y6 = torch.max(test_output6, 1)[1].data.numpy()
            accuracy6 = float((pred_y6 == y_test_s6.data.numpy()).astype(int).sum()) / float(y_test_s6.size(0))
            print('Epoch6: ', epoch, '| train loss: %.4f' % loss6.data.numpy(), '| test accuracy: %.4f' % accuracy6)
# 节点7
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list7, label_list7 = split_data_list_label(clients, 'client_7')
# split data into training and test set
x_train_s7, x_test_ss7, y_train_s7, y_test_ss7 = train_test_split(image_list7,
                                                                label_list7,
                                                                test_size=0.1,
                                                                random_state=42)
train7 = list(zip(x_train_s7, y_train_s7))
x_test_s7 = torch.tensor(x_test_ss7).to(torch.float32).unsqueeze(1)
train_loader7 = Data.DataLoader(dataset = train7, batch_size=BATCH_SIZE, shuffle=True)
y_test_s7 = torch.tensor(y_test_ss7)

for epoch in range(EPOCH):
    for step7, (bb_x7, b_y7) in enumerate(train_loader1):   # gives batch data, normalize x when iterate train_loader
        bbb_x7 = bb_x7.unsqueeze(1)
        b_x7 = bbb_x7.to(torch.float32)
        output7 = cnn(b_x7)[0]               # cnn output
        loss7 = loss_func(output7, b_y7)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss7.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step7 % 50 == 0:
            test_output7, last_layer7 = cnn(x_test_s7)
            pred_y7 = torch.max(test_output7, 1)[1].data.numpy()
            accuracy7 = float((pred_y7 == y_test_s7.data.numpy()).astype(int).sum()) / float(y_test_s7.size(0))
            print('Epoch7: ', epoch, '| train loss: %.4f' % loss7.data.numpy(), '| test accuracy: %.4f' % accuracy7)
# 节点8
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list8, label_list8 = split_data_list_label(clients, 'client_8')
# split data into training and test set
x_train_s8, x_test_ss8, y_train_s8, y_test_ss8 = train_test_split(image_list8,
                                                                label_list8,
                                                                test_size=0.1,
                                                                random_state=42)
train8 = list(zip(x_train_s8, y_train_s8))
x_test_s8 = torch.tensor(x_test_ss8).to(torch.float32).unsqueeze(1)
train_loader8 = Data.DataLoader(dataset = train8, batch_size=BATCH_SIZE, shuffle=True)
y_test_s8 = torch.tensor(y_test_ss8)

for epoch in range(EPOCH):
    for step8, (bb_x8, b_y8) in enumerate(train_loader8):   # gives batch data, normalize x when iterate train_loader
        bbb_x8 = bb_x8.unsqueeze(1)
        b_x8 = bbb_x8.to(torch.float32)
        output8 = cnn(b_x8)[0]               # cnn output
        loss8 = loss_func(output8, b_y8)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss8.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step8 % 50 == 0:
            test_output8, last_layer8 = cnn(x_test_s8)
            pred_y8 = torch.max(test_output8, 1)[1].data.numpy()
            accuracy8 = float((pred_y8 == y_test_s8.data.numpy()).astype(int).sum()) / float(y_test_s8.size(0))
            print('Epoch8: ', epoch, '| train loss: %.4f' % loss8.data.numpy(), '| test accuracy: %.4f' % accuracy8)
# 节点9
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list9, label_list9 = split_data_list_label(clients, 'client_9')
# split data into training and test set
x_train_s9, x_test_ss9, y_train_s9, y_test_ss9 = train_test_split(image_list9,
                                                                label_list9,
                                                                test_size=0.1,
                                                                random_state=42)
train9 = list(zip(x_train_s9, y_train_s9))
x_test_s9 = torch.tensor(x_test_ss9).to(torch.float32).unsqueeze(1)
train_loader9 = Data.DataLoader(dataset = train9, batch_size=BATCH_SIZE, shuffle=True)
y_test_s9 = torch.tensor(y_test_ss9)

for epoch in range(EPOCH):
    for step9, (bb_x9, b_y9) in enumerate(train_loader9):   # gives batch data, normalize x when iterate train_loader
        bbb_x9 = bb_x9.unsqueeze(1)
        b_x9 = bbb_x9.to(torch.float32)
        output9 = cnn(b_x9)[0]               # cnn output
        loss9 = loss_func(output9, b_y9)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss9.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step9 % 50 == 0:
            test_output9, last_layer9 = cnn(x_test_s9)
            pred_y9 = torch.max(test_output9, 1)[1].data.numpy()
            accuracy9 = float((pred_y9 == y_test_s9.data.numpy()).astype(int).sum()) / float(y_test_s9.size(0))
            print('Epoch9: ', epoch, '| train loss: %.4f' % loss9.data.numpy(), '| test accuracy: %.4f' % accuracy9)
# 节点10
cnn = CNN()
print(cnn)  # net architecture
LR = 0.001              # learning rate
BATCH_SIZE = 50
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
image_list10, label_list10 = split_data_list_label(clients, 'client_10')
# split data into training and test set
x_train_s10, x_test_ss10, y_train_s10, y_test_ss10 = train_test_split(image_list10,
                                                                    label_list10,
                                                                    test_size=0.1,
                                                                    random_state=42)
train10 = list(zip(x_train_s10, y_train_s10))
x_test_s10 = torch.tensor(x_test_ss10).to(torch.float32).unsqueeze(1)
train_loader10 = Data.DataLoader(dataset = train10, batch_size=BATCH_SIZE, shuffle=True)
y_test_s10 = torch.tensor(y_test_ss10)

for epoch in range(EPOCH):
    for step10, (bb_x10, b_y10) in enumerate(train_loader10):   # gives batch data, normalize x when iterate train_loader
        bbb_x10 = bb_x10.unsqueeze(1)
        b_x10 = bbb_x10.to(torch.float32)
        output10 = cnn(b_x10)[0]               # cnn output
        loss10 = loss_func(output10, b_y10)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss10.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step10 % 50 == 0:
            test_output10, last_layer10 = cnn(x_test_s10)
            pred_y10 = torch.max(test_output10, 1)[1].data.numpy()
            accuracy10 = float((pred_y10 == y_test_s10.data.numpy()).astype(int).sum()) / float(y_test_s10.size(0))
            print('Epoch10: ', epoch, '| train loss: %.4f' % loss10.data.numpy(), '| test accuracy: %.4f' % accuracy10)



