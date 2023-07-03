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
# from __future__ import print_function

import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
# import matplotlib.pyplot as plt
import numpy as np


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
acc1=[]
acc2=[]
acc3=[]
acc4=[]
acc5=[]
acc6=[]
acc7=[]
acc8=[]
acc9=[]
acc10=[]
for j in range(10):
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
    accuracies1 = []
    for k1 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model1 = KNeighborsClassifier(n_neighbors=k1)
        model1.fit(x_train_s1, y_train_s1)

        # evaluate the model and print the accuracies list
        score1 = model1.score(x_test_s1, y_test_s1)
        print("k=%d, accuracy=%.2f%%" % (k1, score1 * 100))
        accuracies1.append(score1)
        localtime1 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime1)
    kVals1 = range(1, 30, 2)
    i = np.argmax(accuracies1)
    acc1.append(accuracies1[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals1[i],
        accuracies1[i] * 100))
    model1 = KNeighborsClassifier(n_neighbors=kVals1[i])
    model1.fit(x_train_s1, y_train_s1)
    predictions1 = model1.predict(x_test_s1)
    # p1 = classification_report(y_test_s1, predictions1)
    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_1")
    print(classification_report(y_test_s1, predictions1))

    # test_batched1 = tf.data.Dataset.from_tensor_slices((x_test_s1, y_test_s1)).batch(len(y_test_s1))
    # 节点2
    image_list2, label_list2 = split_data_list_label(clients, 'client_2')
    # split data into training and test set
    x_train_s2, x_test_s2, y_train_s2, y_test_s2 = train_test_split(image_list2,
                                                                    label_list2,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies2 = []
    for k2 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model2 = KNeighborsClassifier(n_neighbors=k2)
        model2.fit(x_train_s2, y_train_s2)

        # evaluate the model and print the accuracies list
        score2 = model2.score(x_test_s2, y_test_s2)
        print("k=%d, accuracy=%.2f%%" % (k2, score2 * 100))
        accuracies2.append(score2)
        localtime2 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime2)
    kVals2 = range(1, 30, 2)
    i = np.argmax(accuracies2)
    acc2.append(accuracies2[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals2[i],
        accuracies2[i] * 100))
    model2 = KNeighborsClassifier(n_neighbors=kVals2[i])
    model2.fit(x_train_s2, y_train_s2)
    predictions2 = model2.predict(x_test_s2)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_2")
    print(classification_report(y_test_s2, predictions2))
    # 节点3
    image_list3, label_list3 = split_data_list_label(clients, 'client_3')
    # split data into training and test set
    x_train_s3, x_test_s3, y_train_s3, y_test_s3 = train_test_split(image_list3,
                                                                    label_list3,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies3 = []
    for k3 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model3 = KNeighborsClassifier(n_neighbors=k3)
        model3.fit(x_train_s3, y_train_s3)

        # evaluate the model and print the accuracies list
        score3 = model3.score(x_test_s3, y_test_s3)
        print("k=%d, accuracy=%.2f%%" % (k3, score3 * 100))
        accuracies3.append(score3)
        localtime3 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime3)
    kVals3 = range(1, 30, 2)
    i = np.argmax(accuracies3)
    acc3.append(accuracies3[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals3[i],
        accuracies3[i] * 100))
    model3 = KNeighborsClassifier(n_neighbors=kVals3[i])
    model3.fit(x_train_s3, y_train_s3)
    predictions3 = model3.predict(x_test_s3)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_3")
    print(classification_report(y_test_s3, predictions3))
    # 节点4
    image_list4, label_list4 = split_data_list_label(clients, 'client_4')
    # split data into training and test set
    x_train_s4, x_test_s4, y_train_s4, y_test_s4 = train_test_split(image_list4,
                                                                    label_list4,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies4 = []
    for k4 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model4 = KNeighborsClassifier(n_neighbors=k4)
        model4.fit(x_train_s4, y_train_s4)

        # evaluate the model and print the accuracies list
        score4 = model4.score(x_test_s4, y_test_s4)
        print("k=%d, accuracy=%.2f%%" % (k4, score4 * 100))
        accuracies4.append(score4)
        localtime4 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime4)
    kVals4 = range(1, 30, 2)
    i = np.argmax(accuracies4)
    acc4.append(accuracies4[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals4[i],
        accuracies4[i] * 100))
    model4 = KNeighborsClassifier(n_neighbors=kVals4[i])
    model4.fit(x_train_s4, y_train_s4)
    predictions4 = model4.predict(x_test_s4)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_4")
    print(classification_report(y_test_s4, predictions4))
    # 节点5
    image_list5, label_list5 = split_data_list_label(clients, 'client_5')
    # split data into training and test set
    x_train_s5, x_test_s5, y_train_s5, y_test_s5 = train_test_split(image_list5,
                                                                    label_list5,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies5 = []
    for k5 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model5 = KNeighborsClassifier(n_neighbors=k5)
        model5.fit(x_train_s5, y_train_s5)

        # evaluate the model and print the accuracies list
        score5 = model5.score(x_test_s5, y_test_s5)
        print("k=%d, accuracy=%.2f%%" % (k5, score5 * 100))
        accuracies5.append(score5)
        localtime5 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime5)
    kVals5 = range(1, 30, 2)
    i = np.argmax(accuracies5)
    acc5.append(accuracies5[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals5[i],
        accuracies5[i] * 100))
    model5 = KNeighborsClassifier(n_neighbors=kVals5[i])
    model5.fit(x_train_s5, y_train_s5)
    predictions5 = model5.predict(x_test_s5)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_5")
    print(classification_report(y_test_s5, predictions5))
    # 节点6
    image_list6, label_list6 = split_data_list_label(clients, 'client_6')
    # split data into training and test set
    x_train_s6, x_test_s6, y_train_s6, y_test_s6 = train_test_split(image_list6,
                                                                    label_list6,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies6 = []
    for k6 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model6 = KNeighborsClassifier(n_neighbors=k6)
        model6.fit(x_train_s6, y_train_s6)

        # evaluate the model and print the accuracies list
        score6 = model6.score(x_test_s6, y_test_s6)
        print("k=%d, accuracy=%.2f%%" % (k6, score6 * 100))
        accuracies6.append(score6)
        localtime6 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime6)
    kVals6 = range(1, 30, 2)
    i = np.argmax(accuracies6)
    acc6.append(accuracies6[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals6[i],
        accuracies6[i] * 100))
    model6 = KNeighborsClassifier(n_neighbors=kVals6[i])
    model6.fit(x_train_s6, y_train_s6)
    predictions6 = model6.predict(x_test_s6)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_6")
    print(classification_report(y_test_s6, predictions6))
    # 节点7
    image_list7, label_list7 = split_data_list_label(clients, 'client_7')
    # split data into training and test set
    x_train_s7, x_test_s7, y_train_s7, y_test_s7 = train_test_split(image_list7,
                                                                    label_list7,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies7 = []
    for k7 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model7 = KNeighborsClassifier(n_neighbors=k7)
        model7.fit(x_train_s7, y_train_s7)

        # evaluate the model and print the accuracies list
        score7 = model7.score(x_test_s7, y_test_s7)
        print("k=%d, accuracy=%.2f%%" % (k7, score7 * 100))
        accuracies7.append(score7)
        localtime7 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime7)
    kVals7 = range(1, 30, 2)
    i = np.argmax(accuracies7)
    acc7.append(accuracies7[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals7[i],
        accuracies7[i] * 100))
    model7 = KNeighborsClassifier(n_neighbors=kVals7[i])
    model7.fit(x_train_s7, y_train_s7)
    predictions7 = model7.predict(x_test_s7)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_7")
    print(classification_report(y_test_s7, predictions7))
    # 节点8
    image_list8, label_list8 = split_data_list_label(clients, 'client_8')
    # split data into training and test set
    x_train_s8, x_test_s8, y_train_s8, y_test_s8 = train_test_split(image_list8,
                                                                    label_list8,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies8 = []
    for k8 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model8 = KNeighborsClassifier(n_neighbors=k8)
        model8.fit(x_train_s8, y_train_s8)

        # evaluate the model and print the accuracies list
        score8 = model8.score(x_test_s8, y_test_s8)
        print("k=%d, accuracy=%.2f%%" % (k8, score8 * 100))
        accuracies8.append(score8)
        localtime8 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime8)
    kVals8 = range(1, 30, 2)
    i = np.argmax(accuracies8)
    acc8.append(accuracies8[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals8[i],
        accuracies8[i] * 100))
    model8 = KNeighborsClassifier(n_neighbors=kVals8[i])
    model8.fit(x_train_s8, y_train_s8)
    predictions8 = model8.predict(x_test_s8)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_8")
    print(classification_report(y_test_s8, predictions8))
    # 节点9
    image_list9, label_list9 = split_data_list_label(clients, 'client_9')
    # split data into training and test set
    x_train_s9, x_test_s9, y_train_s9, y_test_s9 = train_test_split(image_list9,
                                                                    label_list9,
                                                                    test_size=0.1,
                                                                    random_state=42)
    accuracies9 = []
    for k9 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model9 = KNeighborsClassifier(n_neighbors=k9)
        model9.fit(x_train_s9, y_train_s9)

        # evaluate the model and print the accuracies list
        score9 = model9.score(x_test_s9, y_test_s9)
        print("k=%d, accuracy=%.2f%%" % (k9, score9 * 100))
        accuracies9.append(score9)
        localtime9 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime9)
    kVals9 = range(1, 30, 2)
    i = np.argmax(accuracies9)
    acc9.append(accuracies9[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals9[i],
        accuracies9[i] * 100))
    model9 = KNeighborsClassifier(n_neighbors=kVals9[i])
    model9.fit(x_train_s9, y_train_s9)
    predictions9 = model9.predict(x_test_s9)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_9")
    print(classification_report(y_test_s9, predictions9))
    # 节点10
    image_list10, label_list10 = split_data_list_label(clients, 'client_10')
    # split data into training and test set
    x_train_s10, x_test_s10, y_train_s10, y_test_s10 = train_test_split(image_list10,
                                                                        label_list10,
                                                                        test_size=0.1,
                                                                        random_state=42)
    accuracies10 = []
    for k10 in range(1, 30, 2):
        # train the classifier with the current value of `k`
        model10 = KNeighborsClassifier(n_neighbors=k10)
        model10.fit(x_train_s10, y_train_s10)

        # evaluate the model and print the accuracies list
        score10 = model10.score(x_test_s10, y_test_s10)
        print("k=%d, accuracy=%.2f%%" % (k10, score10 * 100))
        accuracies10.append(score10)
        localtime10 = time.asctime(time.localtime(time.time()))
        print("本地时间为 :", localtime10)
    kVals10 = range(1, 30, 2)
    i = np.argmax(accuracies10)
    acc10.append(accuracies10[i])
    print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals10[i],
        accuracies10[i] * 100))
    model10 = KNeighborsClassifier(n_neighbors=kVals10[i])
    model10.fit(x_train_s10, y_train_s10)
    predictions10 = model10.predict(x_test_s10)

    # Evaluate performance of model for each of the digits
    print("EVALUATION ON TESTING DATA:client_10")
    print(classification_report(y_test_s10, predictions10))
ac1 = np.mean(acc1)
ac2 = np.mean(acc2)
ac3 = np.mean(acc3)
ac4 = np.mean(acc4)
ac5 = np.mean(acc5)

ac6 = np.mean(acc6)
ac7 = np.mean(acc7)
ac8 = np.mean(acc8)
ac9 = np.mean(acc9)
ac10 = np.mean(acc10)

