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


def load(paths, verbose=-1):
    '''expects images for each class in seperate dir,
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
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
# 节点1
image_list1, label_list1 = split_data_list_label(clients, 'client_1')

# binarize the labels
# lb0 = LabelBinarizer()
# label_list0 = lb0.fit_transform(label_list0)

# split data into training and test set
x_train_s1, x_test_s1, y_train_s1, y_test_s1 = train_test_split(image_list1,
                                                                label_list1,
                                                                test_size=0.1,
                                                                random_state=42)
# test_batched1 = tf.data.Dataset.from_tensor_slices((x_test_s1, y_test_s1)).batch(len(y_test_s1))
# 节点2
image_list2, label_list2 = split_data_list_label(clients, 'client_2')
# split data into training and test set
x_train_s2, x_test_s2, y_train_s2, y_test_s2 = train_test_split(image_list2,
                                                                label_list2,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点3
image_list3, label_list3 = split_data_list_label(clients, 'client_3')
# split data into training and test set
x_train_s3, x_test_s3, y_train_s3, y_test_s3 = train_test_split(image_list3,
                                                                label_list3,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点4
image_list4, label_list4 = split_data_list_label(clients, 'client_4')
# split data into training and test set
x_train_s4, x_test_s4, y_train_s4, y_test_s4 = train_test_split(image_list4,
                                                                label_list4,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点5
image_list5, label_list5 = split_data_list_label(clients, 'client_5')
# split data into training and test set
x_train_s5, x_test_s5, y_train_s5, y_test_s5 = train_test_split(image_list5,
                                                                label_list5,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点6
image_list6, label_list6 = split_data_list_label(clients, 'client_6')
# split data into training and test set
x_train_s6, x_test_s6, y_train_s6, y_test_s6 = train_test_split(image_list6,
                                                                label_list6,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点7
image_list7, label_list7 = split_data_list_label(clients, 'client_7')
# split data into training and test set
x_train_s7, x_test_s7, y_train_s7, y_test_s7 = train_test_split(image_list7,
                                                                label_list7,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点8
image_list8, label_list8 = split_data_list_label(clients, 'client_8')
# split data into training and test set
x_train_s8, x_test_s8, y_train_s8, y_test_s8 = train_test_split(image_list8,
                                                                label_list8,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点9
image_list9, label_list9 = split_data_list_label(clients, 'client_9')
# split data into training and test set
x_train_s9, x_test_s9, y_train_s9, y_test_s9 = train_test_split(image_list9,
                                                                label_list9,
                                                                test_size=0.1,
                                                                random_state=42)
# 节点10
image_list10, label_list10 = split_data_list_label(clients, 'client_10')
# split data into training and test set
x_train_s10, x_test_s10, y_train_s10, y_test_s10 = train_test_split(image_list10,
                                                                    label_list10,
                                                                    test_size=0.1,
                                                                    random_state=42)
# initialize global model
s_acc_s1 = list()
s_acc_s2 = list()
s_acc_s3 = list()
s_acc_s4 = list()
s_acc_s5 = list()
s_acc_s6 = list()
s_acc_s7 = list()
s_acc_s8 = list()
s_acc_s9 = list()
s_acc_s10 = list()
# 节点1，单个节点训练

clf1 = svm.LinearSVC()
clf1.fit(x_train_s1, y_train_s1)

# 节点2，单个节点训练
clf2 = svm.LinearSVC()
clf2.fit(x_train_s2, y_train_s2)

# 节点3，单个节点训练
clf3 = svm.LinearSVC()
clf3.fit(x_train_s3, y_train_s3)

# 节点4，单个节点训练
clf4 = svm.LinearSVC()
clf4.fit(x_train_s4, y_train_s4)

# 节点5，单个节点训练
clf5 = svm.LinearSVC()
clf5.fit(x_train_s5, y_train_s5)

# 节点6，单个节点训练
clf6 = svm.LinearSVC()
clf6.fit(x_train_s6, y_train_s6)

# 节点7，单个节点训练
clf7 = svm.LinearSVC()
clf7.fit(x_train_s7, y_train_s7)

# 节点8，单个节点训练
clf8 = svm.LinearSVC()
clf8.fit(x_train_s8, y_train_s8)

# 节点9，单个节点训练
clf9 = svm.LinearSVC()
clf9.fit(x_train_s9, y_train_s9)

# 节点10，单个节点训练
clf10 = svm.LinearSVC()
clf10.fit(x_train_s10, y_train_s10)

# 节点1的测试集分别在单个节点数据集模型上的准确率
s1_train_acc = clf1.score(x_train_s1, y_train_s1)  # 精度
clf1.predict(x_train_s1)

s1_test_acc = clf1.score(x_test_s1, y_test_s1)
clf1.predict(x_test_s1)

# 节点2的测试集分别在单个节点数据集模型上的准确率
s2_train_acc = clf2.score(x_train_s2, y_train_s2)  # 精度
clf2.predict(x_train_s2)

s2_test_acc = clf2.score(x_test_s2, y_test_s2)
clf2.predict(x_test_s2)

# 节点3的测试集分别在单个节点数据集模型上的准确率
s3_train_acc = clf3.score(x_train_s3, y_train_s3)  # 精度
clf3.predict(x_train_s3)

s3_test_acc = clf1.score(x_test_s3, y_test_s3)
clf3.predict(x_test_s3)

# 节点4的测试集分别在单个节点数据集模型上的准确率
s4_train_acc = clf4.score(x_train_s4, y_train_s4)  # 精度
clf4.predict(x_train_s4)

s4_test_acc = clf4.score(x_test_s4, y_test_s4)
clf4.predict(x_test_s4)

# 节点5的测试集分别在单个节点数据集模型上的准确率
s5_train_acc = clf5.score(x_train_s5, y_train_s5)  # 精度
clf5.predict(x_train_s5)

s5_test_acc = clf5.score(x_test_s5, y_test_s5)
clf5.predict(x_test_s5)

# 节点6的测试集分别在单个节点数据集模型上的准确率
s6_train_acc = clf6.score(x_train_s6, y_train_s6)  # 精度
clf6.predict(x_train_s6)

s6_test_acc = clf6.score(x_test_s6, y_test_s6)
clf6.predict(x_test_s6)

# 节点7的测试集分别在单个节点数据集模型上的准确率
s7_train_acc = clf7.score(x_train_s7, y_train_s7)  # 精度
clf7.predict(x_train_s7)

s7_test_acc = clf7.score(x_test_s7, y_test_s7)
clf7.predict(x_test_s7)

# 节点8的测试集分别在单个节点数据集模型上的准确率
s8_train_acc = clf8.score(x_train_s8, y_train_s8)  # 精度
clf8.predict(x_train_s8)

s8_test_acc = clf8.score(x_test_s8, y_test_s8)
clf8.predict(x_test_s8)

# 节点9的测试集分别在单个节点数据集模型上的准确率
s9_train_acc = clf9.score(x_train_s9, y_train_s9)  # 精度
clf9.predict(x_train_s9)

s9_test_acc = clf9.score(x_test_s9, y_test_s9)
clf9.predict(x_test_s9)

# 节点10的测试集分别在单个节点数据集模型上的准确率
s10_train_acc = clf10.score(x_train_s10, y_train_s10)  # 精度
clf10.predict(x_train_s10)

s10_test_acc = clf10.score(x_test_s10, y_test_s10)
clf10.predict(x_test_s10)

