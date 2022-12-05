import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
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
from fl_mnist_implementation_tutorial_utils import *
import pandas as pd


def tm_local(X_test1, Y_test1, model1):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits1 = model1.predict(X_test1)
    loss1 = cce(Y_test1, logits1)
    acc1 = accuracy_score(tf.argmax(logits1, axis=1), tf.argmax(Y_test1, axis=1))
    pre1, rec1, f_score1, true_sum = precision_recall_fscore_support(tf.argmax(logits1, axis=1), tf.argmax(Y_test1, axis=1))
    return acc1, loss1, pre1, rec1, f_score1


from sklearn import preprocessing

# declear path to your mnist data folder

img_path = 'C:/Users/a/Desktop/test/data.xls'
sheet_n = ['client1', 'client2', 'client3', 'client4', 'client5', 'client6', 'client7', 'client8']

X_train = list()
X_test = list()
y_train = list()
y_test = list()
client_num = 8
client_names = ['{}_{}'.format('client', i + 1) for i in range(client_num)]
res = []

# 节点1
data_frame_1 = pd.read_excel(img_path, sheet_name=sheet_n[0], header=None)

image_list1_1 = data_frame_1.iloc[:, 1:]

label_list1_1 = data_frame_1.iloc[:, 0]
image_list_1 = list()
length_1 = len(label_list1_1)
label_list_1 = np.zeros((length_1, 2), int)
for i in np.arange(len(label_list1_1)):

    label_1 = label_list1_1[i]
    label_1 = label_1.astype(str)
    label_1 = str(label_1)
    if label_1 == '0':
        label_list_1[i, 0] = 1
    else:
        label_list_1[i, 1] = 1
    d_list_1 = image_list1_1.loc[i]
    d_list1_1 = d_list_1.tolist()
    image_list_1.append(d_list1_1)

# split data into training and test set
# 拆分数据，按0.1的测试集分开，
x_train_s1, x_test_s1, y_train_s1, y_test_s1 = train_test_split(image_list_1,
                                                        label_list_1,
                                                        test_size=0.1,
                                                        random_state=30)
test_batched1 = tf.data.Dataset.from_tensor_slices((x_test_s1, y_test_s1)).batch(len(y_test_s1))
# 节点2

data_frame_2 = pd.read_excel(img_path, sheet_name=sheet_n[1], header=None)

image_list1_2 = data_frame_2.iloc[:, 1:]

label_list1_2 = data_frame_2.iloc[:, 0]
image_list_2 = list()
length_2 = len(label_list1_2)
label_list_2 = np.zeros((length_2, 2), int)
for i in np.arange(len(label_list1_2)):

    label_2 = label_list1_2[i]
    label_2 = label_2.astype(str)
    label_2 = str(label_2)
    if label_2 == '0':
        label_list_2[i, 0] = 1
    else:
        label_list_2[i, 1] = 1
    d_list_2 = image_list1_2.loc[i]
    d_list1_2 = d_list_2.tolist()
    image_list_2.append(d_list1_2)

# split data into training and test set
x_train_s2, x_test_s2, y_train_s2, y_test_s2 = train_test_split(image_list_2,
                                                    label_list_2,
                                                    test_size=0.1,
                                                    random_state=30)
test_batched2 = tf.data.Dataset.from_tensor_slices((x_test_s2, y_test_s2)).batch(len(y_test_s2))
# 节点3

data_frame_3 = pd.read_excel(img_path, sheet_name=sheet_n[2], header=None)

image_list1_3 = data_frame_3.iloc[:, 1:]

label_list1_3 = data_frame_3.iloc[:, 0]
image_list_3 = list()
length_3 = len(label_list1_3)
label_list_3 = np.zeros((length_3, 2), int)
for i in np.arange(len(label_list1_3)):

    label_3 = label_list1_3[i]
    label_3 = label_3.astype(str)
    label_3 = str(label_3)
    if label_3 == '0':
        label_list_3[i, 0] = 1
    else:
        label_list_3[i, 1] = 1
    d_list_3 = image_list1_3.loc[i]
    d_list1_3 = d_list_3.tolist()
    image_list_3.append(d_list1_3)

# split data into training and test set
x_train_s3, x_test_s3, y_train_s3, y_test_s3 = train_test_split(image_list_3,
                                                    label_list_3,
                                                    test_size=0.9,
                                                    random_state=30)
test_batched3 = tf.data.Dataset.from_tensor_slices((x_test_s3, y_test_s3)).batch(len(y_test_s3))
# 节点4

data_frame_4 = pd.read_excel(img_path, sheet_name=sheet_n[3], header=None)

image_list1_4 = data_frame_4.iloc[:, 1:]

label_list1_4 = data_frame_4.iloc[:, 0]
image_list_4 = list()
length_4 = len(label_list1_4)
label_list_4 = np.zeros((length_4, 2), int)
for i in np.arange(len(label_list1_4)):

    label_4 = label_list1_4[i]
    label_4 = label_4.astype(str)
    label_4 = str(label_4)
    if label_4 == '0':
        label_list_4[i, 0] = 1
    else:
        label_list_4[i, 1] = 1
    d_list_4 = image_list1_4.loc[i]
    d_list1_4 = d_list_4.tolist()
    image_list_4.append(d_list1_4)

# split data into training and test set
x_train_s4, x_test_s4, y_train_s4, y_test_s4 = train_test_split(image_list_4,
                                                    label_list_4,
                                                    test_size=0.9,
                                                    random_state=30)
test_batched4 = tf.data.Dataset.from_tensor_slices((x_test_s4, y_test_s4)).batch(len(y_test_s4))
# 节点5

data_frame_5 = pd.read_excel(img_path, sheet_name=sheet_n[4], header=None)

image_list1_5 = data_frame_5.iloc[:, 1:]

label_list1_5 = data_frame_5.iloc[:, 0]
image_list_5 = list()
length_5 = len(label_list1_5)
label_list_5 = np.zeros((length_5, 2), int)
for i in np.arange(len(label_list1_5)):

    label_5 = label_list1_5[i]
    label_5 = label_5.astype(str)
    label_5 = str(label_5)
    if label_5 == '0':
        label_list_5[i, 0] = 1
    else:
        label_list_5[i, 1] = 1
    d_list_5 = image_list1_5.loc[i]
    d_list1_5 = d_list_5.tolist()
    image_list_5.append(d_list1_5)

# split data into training and test set
x_train_s5, x_test_s5, y_train_s5, y_test_s5 = train_test_split(image_list_5,
                                                    label_list_5,
                                                    test_size=0.9,
                                                    random_state=30)
test_batched5 = tf.data.Dataset.from_tensor_slices((x_test_s5, y_test_s5)).batch(len(y_test_s5))
# 节点6

data_frame_6 = pd.read_excel(img_path, sheet_name=sheet_n[5], header=None)

image_list1_6 = data_frame_6.iloc[:, 1:]

label_list1_6 = data_frame_6.iloc[:, 0]
image_list_6 = list()
length_6 = len(label_list1_6)
label_list_6 = np.zeros((length_6, 2), int)
for i in np.arange(len(label_list1_6)):

    label_6 = label_list1_6[i]
    label_6 = label_6.astype(str)
    label_6 = str(label_6)
    if label_6 == '0':
        label_list_6[i, 0] = 1
    else:
        label_list_6[i, 1] = 1
    d_list_6 = image_list1_6.loc[i]
    d_list1_6 = d_list_6.tolist()
    image_list_6.append(d_list1_6)

# split data into training and test set
x_train_s6, x_test_s6, y_train_s6, y_test_s6 = train_test_split(image_list_6,
                                                    label_list_6,
                                                    test_size=0.9,
                                                    random_state=30)
test_batched6 = tf.data.Dataset.from_tensor_slices((x_test_s6, y_test_s6)).batch(len(y_test_s6))
# 节点7

data_frame_7 = pd.read_excel(img_path, sheet_name=sheet_n[6], header=None)

image_list1_7 = data_frame_7.iloc[:, 1:]

label_list1_7 = data_frame_7.iloc[:, 0]
image_list_7 = list()
length_7 = len(label_list1_7)
label_list_7 = np.zeros((length_7, 2), int)
for i in np.arange(len(label_list1_7)):

    label_7 = label_list1_7[i]
    label_7 = label_7.astype(str)
    label_7 = str(label_7)
    if label_7 == '0':
        label_list_7[i, 0] = 1
    else:
        label_list_7[i, 1] = 1
    d_list_7 = image_list1_7.loc[i]
    d_list1_7 = d_list_7.tolist()
    image_list_7.append(d_list1_7)

# split data into training and test set
x_train_s7, x_test_s7, y_train_s7, y_test_s7 = train_test_split(image_list_7,
                                                    label_list_7,
                                                    test_size=0.9,
                                                    random_state=30)
test_batched7 = tf.data.Dataset.from_tensor_slices((x_test_s7, y_test_s7)).batch(len(y_test_s7))
# 节点8

data_frame_8 = pd.read_excel(img_path, sheet_name=sheet_n[7], header=None)

image_list1_8 = data_frame_8.iloc[:, 1:]

label_list1_8 = data_frame_8.iloc[:, 0]
image_list_8 = list()
length_8 = len(label_list1_8)
label_list_8 = np.zeros((length_8, 2), int)
for i in np.arange(len(label_list1_8)):

    label_8 = label_list1_8[i]
    label_8 = label_8.astype(str)
    label_8 = str(label_8)
    if label_8 == '0':
        label_list_8[i, 0] = 1
    else:
        label_list_8[i, 1] = 1
    d_list_8 = image_list1_8.loc[i]
    d_list1_8 = d_list_8.tolist()
    image_list_8.append(d_list1_8)

# split data into training and test set
x_train_s8, x_test_s8, y_train_s8, y_test_s8 = train_test_split(image_list_8,
                                                    label_list_8,
                                                    test_size=0.9,
                                                    random_state=30)
test_batched8 = tf.data.Dataset.from_tensor_slices((x_test_s8, y_test_s8)).batch(len(y_test_s8))







# create optimizer
lr = 0.01
comms_round = 100
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr,
                decay=lr / comms_round,
                momentum=0.9
               )

# initialize global model

g_acc = list()
g_pre = list()
g_rec = list()
g_f_score = list()

g_ac = list()
g_pr = list()
g_re = list()
g_f_s = list()
repeat = 10
repeat_client1_acc = list()
repeat_client2_acc = list()
repeat_client3_acc = list()
repeat_client4_acc = list()
repeat_client5_acc = list()
repeat_client6_acc = list()
repeat_client7_acc = list()
repeat_client8_acc = list()
repeat_client1_loss = list()
repeat_client2_loss = list()
repeat_client3_loss = list()
repeat_client4_loss = list()
repeat_client5_loss = list()
repeat_client6_loss = list()
repeat_client7_loss = list()
repeat_client8_loss = list()

s_acc_s1 = list()
s_acc_s2 = list()
s_acc_s3 = list()
s_acc_s4 = list()
s_acc_s5 = list()
s_acc_s6 = list()
s_acc_s7 = list()
s_acc_s8 = list()

s_pre_s1 = list()
s_pre_s2 = list()
s_pre_s3 = list()
s_pre_s4 = list()
s_pre_s5 = list()
s_pre_s6 = list()
s_pre_s7 = list()
s_pre_s8 = list()

s_rec_s1 = list()
s_rec_s2 = list()
s_rec_s3 = list()
s_rec_s4 = list()
s_rec_s5 = list()
s_rec_s6 = list()
s_rec_s7 = list()
s_rec_s8 = list()

s_f_score_s1 = list()
s_f_score_s2 = list()
s_f_score_s3 = list()
s_f_score_s4 = list()
s_f_score_s5 = list()
s_f_score_s6 = list()
s_f_score_s7 = list()
s_f_score_s8 = list()

g_acc_s1 = list()
g_acc_s2 = list()
g_acc_s3 = list()
g_acc_s4 = list()
g_acc_s5 = list()
g_acc_s6 = list()
g_acc_s7 = list()
g_acc_s8 = list()

g_pre_s1 = list()
g_pre_s2 = list()
g_pre_s3 = list()
g_pre_s4 = list()
g_pre_s5 = list()
g_pre_s6 = list()
g_pre_s7 = list()
g_pre_s8 = list()

g_rec_s1 = list()
g_rec_s2 = list()
g_rec_s3 = list()
g_rec_s4 = list()
g_rec_s5 = list()
g_rec_s6 = list()
g_rec_s7 = list()
g_rec_s8 = list()

g_f_score_s1 = list()
g_f_score_s2 = list()
g_f_score_s3 = list()
g_f_score_s4 = list()
g_f_score_s5 = list()
g_f_score_s6 = list()
g_f_score_s7 = list()
g_f_score_s8 = list()

S_acc = list()
S_pre = list()
S_rec = list()
S_f_score = list()

S_acc_s1 = list()
S_acc_s2 = list()
S_acc_s3 = list()
S_acc_s4 = list()
S_acc_s5 = list()
S_acc_s6 = list()
S_acc_s7 = list()
S_acc_s8 = list()

S_pre_s1 = list()
S_pre_s2 = list()
S_pre_s3 = list()
S_pre_s4 = list()
S_pre_s5 = list()
S_pre_s6 = list()
S_pre_s7 = list()
S_pre_s8 = list()

S_rec_s1 = list()
S_rec_s2 = list()
S_rec_s3 = list()
S_rec_s4 = list()
S_rec_s5 = list()
S_rec_s6 = list()
S_rec_s7 = list()
S_rec_s8 = list()

S_f_score_s1 = list()
S_f_score_s2 = list()
S_f_score_s3 = list()
S_f_score_s4 = list()
S_f_score_s5 = list()
S_f_score_s6 = list()
S_f_score_s7 = list()
S_f_score_s8 = list()
for j in range(repeat):

    smlp_global = SimpleMLP()
    global_model = smlp_global.build(27, 2)
    # commence global training loop
    print(j)
    k_add = np.arange(580, 590, 10)
    client_1_acc = list()
    client_2_acc = list()
    client_3_acc = list()
    client_4_acc = list()
    client_5_acc = list()
    client_6_acc = list()
    client_7_acc = list()
    client_8_acc = list()

    client_1_pre = list()
    client_2_pre = list()
    client_3_pre = list()
    client_4_pre = list()
    client_5_pre = list()
    client_6_pre = list()
    client_7_pre = list()
    client_8_pre = list()

    client_1_rec = list()
    client_2_rec = list()
    client_3_rec = list()
    client_4_rec = list()
    client_5_rec = list()
    client_6_rec = list()
    client_7_rec = list()
    client_8_rec = list()

    client_1_f_score = list()
    client_2_f_score = list()
    client_3_f_score = list()
    client_4_f_score = list()
    client_5_f_score = list()
    client_6_f_score = list()
    client_7_f_score = list()
    client_8_f_score = list()

    client_1_loss = list()
    client_2_loss = list()
    client_3_loss = list()
    client_4_loss = list()
    client_5_loss = list()
    client_6_loss = list()
    client_7_loss = list()
    client_8_loss = list()
    g_c = list()
    g_p = list()
    g_r = list()
    g_f = list()
    for comm_round in range(comms_round):
        X_train = list()
        X_test = list()
        y_train = list()
        y_test = list()
        client_num = 8
        client_names = ['{}_{}'.format('client', i + 1) for i in range(client_num)]
        res = []
        for k in np.arange(client_num):
            data_frame = pd.read_excel(img_path, sheet_name=sheet_n[k], header=None)
            data_frame_Rows = data_frame.shape[0]
            cr = (comm_round+1)*10
            # if data_frame_Rows >= cr:
            #     N = random.sample(range(0, data_frame_Rows), k_add[comm_round])
            #     image_list0 = data_frame.iloc[:, 1:]
            #     image_list1 = image_list0.iloc[N, :]
            #     label_list0 = data_frame.iloc[:, 0]
            #     label_list1 = label_list0[N]
            # else:
            image_list1 = data_frame.iloc[:, 1:]
            label_list1 = data_frame.iloc[:, 0]
            image_list = list()
            length = len(label_list1)
            label_list = np.zeros((length, 2), int)
            label_list1 = label_list1.values.tolist()
            for i in np.arange(len(label_list1)):

                label = label_list1[i]
                # label = label.astype(str)
                label = str(label)
                if label == '0':
                    label_list[i, 0] = 1
                else:
                    label_list[i, 1] = 1
                d_list = image_list1.values.tolist()
                d_list1 = d_list[i]
                image_list.append(d_list1)

            # split data into training and test set
            x_train, x_test, yy_train, yy_test = train_test_split(image_list,
                                                                  label_list,
                                                                  test_size=0.1,
                                                                  random_state=30)

            # randomize the data
            shards = list(zip(x_train, yy_train))

            X_train.extend(x_train)
            X_test.extend(x_test)
            y_train.extend(yy_train)
            y_test.extend(yy_test)
            res.append(shards)
        clients = {client_names[i]: res[i] for i in range(client_num)}

        # create clients
        # clients = create_clients(X_train, y_train, num_clients=8, initial='client')

        # process and batch the training data for each client
        clients_batched = dict()
        for (client_name, data) in clients.items():
            clients_batched[client_name] = batch_data(data)

        # process and batch the test set

        test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()

        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        random.shuffle(client_names)

        # loop through each client and create new local model
        for client in client_names:
            smlp_local = SimpleMLP()
            local_model = smlp_local.build(27, 2)
            local_model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # fit local model with client's data
            local_model.fit(clients_batched[client], epochs=5, verbose=0)
            local_model.add(GaussianNoise(0.5))
            if client == 'client_1':
                test_batched1 = tf.data.Dataset.from_tensor_slices((x_test_s1, y_test_s1)).batch(len(y_test_s1))
                for (x_test_s1, y_test_s1) in test_batched1:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s1, y_test_s1, local_model)
                client1_acc = local_acc
                client1_loss = local_loss
                client1_pre = local_pre
                client1_rec = local_rec
                client1_f_score = local_f_score
                client_1_acc.append(client1_acc)
                client_1_loss.append(client1_loss)
                client_1_pre.append(client1_pre)
                client_1_rec.append(client1_rec)
                client_1_f_score.append(client1_f_score)
            elif client == 'client_2':
                test_batched2 = tf.data.Dataset.from_tensor_slices((x_test_s2, y_test_s2)).batch(len(y_test_s2))
                for (x_test_s2, y_test_s2) in test_batched2:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s2, y_test_s2, local_model)

                client2_acc = local_acc
                client2_loss = local_loss
                client2_pre = local_pre
                client2_rec = local_rec
                client2_f_score = local_f_score
                client_2_acc.append(client2_acc)
                client_2_loss.append(client2_loss)
                client_2_pre.append(client2_pre)
                client_2_rec.append(client2_rec)
                client_2_f_score.append(client2_f_score)
            elif client == 'client_3':
                test_batched3 = tf.data.Dataset.from_tensor_slices((x_test_s3, y_test_s3)).batch(len(y_test_s3))
                for (x_test_s3, y_test_s3) in test_batched3:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s3, y_test_s3, local_model)

                client3_acc = local_acc
                client3_loss = local_loss
                client3_pre = local_pre
                client3_rec = local_rec
                client3_f_score = local_f_score
                client_3_acc.append(client3_acc)
                client_3_loss.append(client3_loss)
                client_3_pre.append(client3_pre)
                client_3_rec.append(client3_rec)
                client_3_f_score.append(client3_f_score)
            elif client == 'client_4':
                test_batched4 = tf.data.Dataset.from_tensor_slices((x_test_s4, y_test_s4)).batch(len(y_test_s4))
                for (x_test_s4, y_test_s4) in test_batched4:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s4, y_test_s4, local_model)

                client4_acc = local_acc
                client4_loss = local_loss
                client4_pre = local_pre
                client4_rec = local_rec
                client4_f_score = local_f_score
                client_4_acc.append(client4_acc)
                client_4_loss.append(client4_loss)
                client_4_pre.append(client4_pre)
                client_4_rec.append(client4_rec)
                client_4_f_score.append(client4_f_score)
            elif client == 'client_5':
                test_batched5 = tf.data.Dataset.from_tensor_slices((x_test_s5, y_test_s5)).batch(len(y_test_s5))
                for (x_test_s5, y_test_s5) in test_batched5:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s5, y_test_s5, local_model)

                client5_acc = local_acc
                client5_loss = local_loss
                client5_pre = local_pre
                client5_rec = local_rec
                client5_f_score = local_f_score
                client_5_acc.append(client5_acc)
                client_5_loss.append(client5_loss)
                client_5_pre.append(client5_pre)
                client_5_rec.append(client5_rec)
                client_5_f_score.append(client5_f_score)
            elif client == 'client_6':
                test_batched6 = tf.data.Dataset.from_tensor_slices((x_test_s6, y_test_s6)).batch(len(y_test_s6))
                for (x_test_s6, y_test_s6) in test_batched6:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s6, y_test_s6, local_model)

                client6_acc = local_acc
                client6_loss = local_loss
                client6_pre = local_pre
                client6_rec = local_rec
                client6_f_score = local_f_score
                client_6_acc.append(client6_acc)
                client_6_loss.append(client6_loss)
                client_6_pre.append(client6_pre)
                client_6_rec.append(client6_rec)
                client_6_f_score.append(client6_f_score)
            elif client == 'client_7':
                test_batched7 = tf.data.Dataset.from_tensor_slices((x_test_s7, y_test_s7)).batch(len(y_test_s7))
                for (x_test_s7, y_test_s7) in test_batched7:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s7, y_test_s7, local_model)

                client7_acc = local_acc
                client7_loss = local_loss
                client7_pre = local_pre
                client7_rec = local_rec
                client7_f_score = local_f_score
                client_7_acc.append(client7_acc)
                client_7_loss.append(client7_loss)
                client_7_pre.append(client7_pre)
                client_7_rec.append(client7_rec)
                client_7_f_score.append(client7_f_score)
            elif client == 'client_8':
                test_batched8 = tf.data.Dataset.from_tensor_slices((x_test_s8, y_test_s8)).batch(len(y_test_s8))
                for (x_test_s8, y_test_s8) in test_batched8:
                    local_acc, local_loss, local_pre, local_rec, local_f_score = tm_local(x_test_s8, y_test_s8, local_model)

                client8_acc = local_acc
                client8_loss = local_loss
                client8_pre = local_pre
                client8_rec = local_rec
                client8_f_score = local_f_score
                client_8_acc.append(client8_acc)
                client_8_loss.append(client8_loss)
                client_8_pre.append(client8_pre)
                client_8_rec.append(client8_rec)
                client_8_f_score.append(client8_f_score)
            # scale the model weights and add to list
            scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()
        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)

        # test global model and print out metrics after each communications round
        for(X_test, Y_test) in test_batched:
            global_acc, global_loss, global_pre, global_rec, global_f_score = test_model(X_test, Y_test, global_model, comm_round)
        g_c.append(global_acc)
        g_p.append(global_pre)
        g_r.append(global_rec)
        g_f.append(global_f_score)

    g_ac.append(g_c)
    g_pr.append(g_p)
    g_re.append(g_r)
    g_f_s.append(g_f)
    g_acc.append(global_acc)
    g_pre.append(global_pre)
    g_rec.append(global_rec)
    g_f_score.append(global_f_score)
    SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)

    smlp_SGD = SimpleMLP()
    SGD_model = smlp_SGD.build(27, 2)

    SGD_model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
    SGD_model.fit(SGD_dataset, epochs=100, verbose=0)
    # 节点1，单个节点训练
    s1_dataset = tf.data.Dataset.from_tensor_slices((x_train_s1, y_train_s1)).shuffle(len(y_train_s1)).batch(320)
    smlp_s1 = SimpleMLP()
    s1_model = smlp_SGD.build(27, 2)

    s1_model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)

    # fit the SGD training data to model
    s1_model.fit(s1_dataset, epochs=100, verbose=0)

    # 节点2，单个节点训练
    s2_dataset = tf.data.Dataset.from_tensor_slices((x_train_s2, y_train_s2)).shuffle(len(y_train_s2)).batch(320)
    smlp_s2 = SimpleMLP()
    s2_model = smlp_SGD.build(27, 2)

    s2_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s2_model.fit(s2_dataset, epochs=100, verbose=0)

    # 节点3，单个节点训练
    s3_dataset = tf.data.Dataset.from_tensor_slices((x_train_s3, y_train_s3)).shuffle(len(y_train_s3)).batch(320)
    smlp_s3 = SimpleMLP()
    s3_model = smlp_SGD.build(27, 2)

    s3_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s3_model.fit(s3_dataset, epochs=100, verbose=0)

    # 节点4，单个节点训练
    s4_dataset = tf.data.Dataset.from_tensor_slices((x_train_s4, y_train_s4)).shuffle(len(y_train_s4)).batch(320)
    smlp_s1 = SimpleMLP()
    s4_model = smlp_SGD.build(27, 2)

    s4_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s4_model.fit(s4_dataset, epochs=100, verbose=0)

    # 节点5，单个节点训练
    s5_dataset = tf.data.Dataset.from_tensor_slices((x_train_s5, y_train_s5)).shuffle(len(y_train_s5)).batch(320)
    smlp_s5 = SimpleMLP()
    s5_model = smlp_SGD.build(27, 2)

    s5_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s5_model.fit(s5_dataset, epochs=100, verbose=0)

    # 节点6，单个节点训练
    s6_dataset = tf.data.Dataset.from_tensor_slices((x_train_s6, y_train_s6)).shuffle(len(y_train_s6)).batch(320)
    smlp_s6 = SimpleMLP()
    s6_model = smlp_SGD.build(27, 2)

    s6_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s6_model.fit(s6_dataset, epochs=100, verbose=0)

    # 节点7，单个节点训练
    s7_dataset = tf.data.Dataset.from_tensor_slices((x_train_s7, y_train_s7)).shuffle(len(y_train_s7)).batch(320)
    smlp_s7 = SimpleMLP()
    s7_model = smlp_SGD.build(27, 2)

    s7_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s7_model.fit(s7_dataset, epochs=100, verbose=0)

    # 节点8，单个节点训练
    s8_dataset = tf.data.Dataset.from_tensor_slices((x_train_s8, y_train_s8)).shuffle(len(y_train_s8)).batch(320)
    smlp_s8 = SimpleMLP()
    s8_model = smlp_SGD.build(27, 2)

    s8_model.compile(loss=loss,
                     optimizer=optimizer,
                     metrics=metrics)

    # fit the SGD training data to model
    s8_model.fit(s8_dataset, epochs=100, verbose=0)



    # 节点1的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s1, y_test_s1) in test_batched1:
        s1_acc, s1_loss, s1_pre, s1_rec, s1_f_score = test_model(x_test_s1, y_test_s1, s1_model, 1)
        global_acc_s1, global_loss_s1, global_pre_s1, global_rec_s1, global_f_score_s1 = test_model(x_test_s1, y_test_s1, global_model, j)
        SGD_acc_s1, SGD_loss_s1, SGD_pre_s1, SGD_rec_s1, SGD_f_score_s1 = test_model(x_test_s1, y_test_s1, SGD_model, 1)
    s_acc_s1.append(s1_acc)
    g_acc_s1.append(global_acc_s1)
    S_acc_s1.append(SGD_acc_s1)

    s_pre_s1.append(s1_pre)
    g_pre_s1.append(global_pre_s1)
    S_pre_s1.append(SGD_pre_s1)

    s_rec_s1.append(s1_rec)
    g_rec_s1.append(global_rec_s1)
    S_rec_s1.append(SGD_rec_s1)

    s_f_score_s1.append(s1_f_score)
    g_f_score_s1.append(global_f_score_s1)
    S_f_score_s1.append(SGD_f_score_s1)

    # 节点2的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s2, y_test_s2) in test_batched2:
        s2_acc, s2_loss, s2_pre, s2_rec, s2_f_score = test_model(x_test_s2, y_test_s2, s2_model, 1)
        global_acc_s2, global_loss_s2, global_pre_s2, global_rec_s2, global_f_score_s2 = test_model(x_test_s2, y_test_s2, global_model, j)
        SGD_acc_s2, SGD_loss_s2, SGD_pre_s2, SGD_rec_s2, SGD_f_score_s2 = test_model(x_test_s2, y_test_s2, SGD_model, 1)
    s_acc_s2.append(s2_acc)
    g_acc_s2.append(global_acc_s2)
    S_acc_s2.append(SGD_acc_s2)

    s_pre_s2.append(s2_pre)
    g_pre_s2.append(global_pre_s2)
    S_pre_s2.append(SGD_pre_s2)

    s_rec_s2.append(s2_rec)
    g_rec_s2.append(global_rec_s2)
    S_rec_s2.append(SGD_rec_s2)

    s_f_score_s2.append(s2_f_score)
    g_f_score_s2.append(global_f_score_s2)
    S_f_score_s2.append(SGD_f_score_s2)
    # 节点3的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s3, y_test_s3) in test_batched3:
        s3_acc, s3_loss, s3_pre, s3_rec, s3_f_score = test_model(x_test_s3, y_test_s3, s3_model, 1)
        global_acc_s3, global_loss_s3, global_pre_s3, global_rec_s3, global_f_score_s3 = test_model(x_test_s3, y_test_s3, global_model, j)
        SGD_acc_s3, SGD_loss_s3, SGD_pre_s3, SGD_rec_s3, SGD_f_score_s3 = test_model(x_test_s3, y_test_s3, SGD_model, 1)
    s_acc_s3.append(s3_acc)
    g_acc_s3.append(global_acc_s3)
    S_acc_s3.append(SGD_acc_s3)

    s_pre_s3.append(s3_pre)
    g_pre_s3.append(global_pre_s3)
    S_pre_s3.append(SGD_pre_s3)

    s_rec_s3.append(s3_rec)
    g_rec_s3.append(global_rec_s3)
    S_rec_s3.append(SGD_rec_s3)

    s_f_score_s3.append(s3_f_score)
    g_f_score_s3.append(global_f_score_s3)
    S_f_score_s3.append(SGD_f_score_s3)
    # 节点4的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s4, y_test_s4) in test_batched4:
        s4_acc, s4_loss, s4_pre, s4_rec, s4_f_score = test_model(x_test_s4, y_test_s4, s4_model, 1)
        global_acc_s4, global_loss_s4, global_pre_s4, global_rec_s4, global_f_score_s4 = test_model(x_test_s4, y_test_s4, global_model, j)
        SGD_acc_s4, SGD_loss_s4, SGD_pre_s4, SGD_rec_s4, SGD_f_score_s4 = test_model(x_test_s4, y_test_s4, SGD_model, 1)
    s_acc_s4.append(s4_acc)
    g_acc_s4.append(global_acc_s4)
    S_acc_s4.append(SGD_acc_s4)

    s_pre_s4.append(s4_pre)
    g_pre_s4.append(global_pre_s4)
    S_pre_s4.append(SGD_pre_s4)

    s_rec_s4.append(s4_rec)
    g_rec_s4.append(global_rec_s4)
    S_rec_s4.append(SGD_rec_s4)

    s_f_score_s4.append(s4_f_score)
    g_f_score_s4.append(global_f_score_s4)
    S_f_score_s4.append(SGD_f_score_s4)
    # 节点5的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s5, y_test_s5) in test_batched5:
        s5_acc, s5_loss, s5_pre, s5_rec, s5_f_score = test_model(x_test_s5, y_test_s5, s5_model, 1)
        global_acc_s5, global_loss_s5, global_pre_s5, global_rec_s5, global_f_score_s5 = test_model(x_test_s5, y_test_s5, global_model, j)
        SGD_acc_s5, SGD_loss_s5, SGD_pre_s5, SGD_rec_s5, SGD_f_score_s5 = test_model(x_test_s5, y_test_s5, SGD_model, 1)
    s_acc_s5.append(s5_acc)
    g_acc_s5.append(global_acc_s5)
    S_acc_s5.append(SGD_acc_s5)

    s_pre_s5.append(s5_pre)
    g_pre_s5.append(global_pre_s5)
    S_pre_s5.append(SGD_pre_s5)

    s_rec_s5.append(s5_rec)
    g_rec_s5.append(global_rec_s5)
    S_rec_s5.append(SGD_rec_s5)

    s_f_score_s5.append(s5_f_score)
    g_f_score_s5.append(global_f_score_s5)
    S_f_score_s5.append(SGD_f_score_s5)
    # 节点6的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s6, y_test_s6) in test_batched6:
        s6_acc, s6_loss, s6_pre, s6_rec, s6_f_score = test_model(x_test_s6, y_test_s6, s6_model, 1)
        global_acc_s6, global_loss_s6, global_pre_s6, global_rec_s6, global_f_score_s6 = test_model(x_test_s6, y_test_s6, global_model, j)
        SGD_acc_s6, SGD_loss_s6, SGD_pre_s6, SGD_rec_s6, SGD_f_score_s6 = test_model(x_test_s6, y_test_s6, SGD_model, 1)
    s_acc_s6.append(s6_acc)
    g_acc_s6.append(global_acc_s6)
    S_acc_s6.append(SGD_acc_s6)

    s_pre_s6.append(s6_pre)
    g_pre_s6.append(global_pre_s6)
    S_pre_s6.append(SGD_pre_s6)

    s_rec_s6.append(s6_rec)
    g_rec_s6.append(global_rec_s6)
    S_rec_s6.append(SGD_rec_s6)

    s_f_score_s6.append(s6_f_score)
    g_f_score_s6.append(global_f_score_s6)
    S_f_score_s6.append(SGD_f_score_s6)
    # 节点7的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s7, y_test_s7) in test_batched7:
        s7_acc, s7_loss, s7_pre, s7_rec, s7_f_score = test_model(x_test_s7, y_test_s7, s7_model, 1)
        global_acc_s7, global_loss_s7, global_pre_s7, global_rec_s7, global_f_score_s7 = test_model(x_test_s7, y_test_s7, global_model, j)
        SGD_acc_s7, SGD_loss_s7, SGD_pre_s7, SGD_rec_s7, SGD_f_score_s7 = test_model(x_test_s7, y_test_s7, SGD_model, 1)
    s_acc_s7.append(s7_acc)
    g_acc_s7.append(global_acc_s7)
    S_acc_s7.append(SGD_acc_s7)

    s_pre_s7.append(s7_pre)
    g_pre_s7.append(global_pre_s7)
    S_pre_s7.append(SGD_pre_s7)

    s_rec_s7.append(s7_rec)
    g_rec_s7.append(global_rec_s7)
    S_rec_s7.append(SGD_rec_s7)

    s_f_score_s7.append(s7_f_score)
    g_f_score_s7.append(global_f_score_s7)
    S_f_score_s7.append(SGD_f_score_s7)
    # 节点8的测试集分别在单个节点数据集模型上、中心模型上、在total的模型上的准确率
    for (x_test_s8, y_test_s8) in test_batched8:
        s8_acc, s8_loss, s8_pre, s8_rec, s8_f_score = test_model(x_test_s8, y_test_s8, s8_model, 1)
        global_acc_s8, global_loss_s8, global_pre_s8, global_rec_s8, global_f_score_s8 = test_model(x_test_s8, y_test_s8, global_model, j)
        SGD_acc_s8, SGD_loss_s8, SGD_pre_s8, SGD_rec_s8, SGD_f_score_s8 = test_model(x_test_s8, y_test_s8, SGD_model, 1)
    s_acc_s8.append(s8_acc)
    g_acc_s8.append(global_acc_s8)
    S_acc_s8.append(SGD_acc_s8)

    s_pre_s8.append(s8_pre)
    g_pre_s8.append(global_pre_s8)
    S_pre_s8.append(SGD_pre_s8)

    s_rec_s8.append(s8_rec)
    g_rec_s8.append(global_rec_s8)
    S_rec_s8.append(SGD_rec_s8)

    s_f_score_s8.append(s8_f_score)
    g_f_score_s8.append(global_f_score_s8)
    S_f_score_s8.append(SGD_f_score_s8)


    # test the SGD global model and print out metrics
    for(X_test, Y_test) in test_batched:
        SGD_acc, SGD_loss, SGD_pre, SGD_rec, SGD_f_score = test_model(X_test, Y_test, SGD_model, 1)
    S_acc.append(SGD_acc)
    S_pre.append(SGD_pre)
    S_rec.append(SGD_rec)
    S_f_score.append(SGD_f_score)

av_acc = sum(g_acc)/repeat
all_av_acc = sum(S_acc)/repeat
print('FL_moxing: {:.3%}'.format(av_acc, g_acc))
print('Zong_moxing: {:.3%}'.format(all_av_acc, S_acc))
print(g_acc)
print(S_acc)
path_load = "C:/Users/a/Desktop/test/fl_result/"
df = pd.DataFrame(s_acc_s1)
df.to_excel(path_load+"single_acc1.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s1)
df.to_excel(path_load+"single_pre1.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s1)
df.to_excel(path_load+"single_rec1.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s1)
df.to_excel(path_load+"single_f_score1.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(s_acc_s2)
df.to_excel(path_load+"single_acc2.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s2)
df.to_excel(path_load+"single_pre2.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s2)
df.to_excel(path_load+"single_rec2.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s2)
df.to_excel(path_load+"single_f_score2.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(s_acc_s3)
df.to_excel(path_load+"single_acc3.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s3)
df.to_excel(path_load+"single_pre3.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s3)
df.to_excel(path_load+"single_rec3.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s3)
df.to_excel(path_load+"single_f_score3.xlsx", sheet_name="f_score", index=False)


df = pd.DataFrame(s_acc_s4)
df.to_excel(path_load+"single_acc4.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s4)
df.to_excel(path_load+"single_pre4.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s4)
df.to_excel(path_load+"single_rec4.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s4)
df.to_excel(path_load+"single_f_score4.xlsx", sheet_name="f_score", index=False)


df = pd.DataFrame(s_acc_s5)
df.to_excel(path_load+"single_acc5.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s5)
df.to_excel(path_load+"single_pre5.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s5)
df.to_excel(path_load+"single_rec5.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s5)
df.to_excel(path_load+"single_f_score5.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(s_acc_s6)
df.to_excel(path_load+"single_acc6.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s6)
df.to_excel(path_load+"single_pre6.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s6)
df.to_excel(path_load+"single_rec6.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s6)
df.to_excel(path_load+"single_f_score6.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(s_acc_s7)
df.to_excel(path_load+"single_acc7.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s7)
df.to_excel(path_load+"single_pre7.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s7)
df.to_excel(path_load+"single_rec7.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s7)
df.to_excel(path_load+"single_f_score7.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(s_acc_s8)
df.to_excel(path_load+"single_acc8.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(s_pre_s8)
df.to_excel(path_load+"single_pre8.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(s_rec_s8)
df.to_excel(path_load+"single_rec8.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(s_f_score_s8)
df.to_excel(path_load+"single_f_score8.xlsx", sheet_name="f_score", index=False)


df = pd.DataFrame(g_acc_s1)
df.to_excel(path_load+"fls_acc1.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s1)
df.to_excel(path_load+"fls_pre1.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s1)
df.to_excel(path_load+"fls_rec1.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s1)
df.to_excel(path_load+"fls_f_score1.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s2)
df.to_excel(path_load+"fls_acc2.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s2)
df.to_excel(path_load+"fls_pre2.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s2)
df.to_excel(path_load+"fls_rec2.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s2)
df.to_excel(path_load+"fls_f_score2.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s3)
df.to_excel(path_load+"fls_acc3.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s3)
df.to_excel(path_load+"fls_pre3.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s3)
df.to_excel(path_load+"fls_rec3.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s3)
df.to_excel(path_load+"fls_f_score3.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s4)
df.to_excel(path_load+"fls_acc4.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s4)
df.to_excel(path_load+"fls_pre4.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s4)
df.to_excel(path_load+"fls_rec4.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s4)
df.to_excel(path_load+"fls_f_score4.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s5)
df.to_excel(path_load+"fls_acc5.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s5)
df.to_excel(path_load+"fls_pre5.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s5)
df.to_excel(path_load+"fls_rec5.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s5)
df.to_excel(path_load+"fls_f_score5.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s6)
df.to_excel(path_load+"fls_acc6.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s6)
df.to_excel(path_load+"fls_pre6.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s6)
df.to_excel(path_load+"fls_rec6.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s6)
df.to_excel(path_load+"fls_f_score6.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s7)
df.to_excel(path_load+"fls_acc7.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s7)
df.to_excel(path_load+"fls_pre7.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s7)
df.to_excel(path_load+"fls_rec7.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s7)
df.to_excel(path_load+"fls_f_score7.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(g_acc_s8)
df.to_excel(path_load+"fls_acc8.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(g_pre_s8)
df.to_excel(path_load+"fls_pre8.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(g_rec_s8)
df.to_excel(path_load+"fls_rec8.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(g_f_score_s8)
df.to_excel(path_load+"fls_f_score8.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s1)
df.to_excel(path_load+"totals_acc1.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s1)
df.to_excel(path_load+"totals_pre1.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s1)
df.to_excel(path_load+"totals_rec1.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s1)
df.to_excel(path_load+"totals_f_score1.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s2)
df.to_excel(path_load+"totals_acc2.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s2)
df.to_excel(path_load+"totals_pre2.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s2)
df.to_excel(path_load+"totals_rec2.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s2)
df.to_excel(path_load+"totals_f_score2.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s3)
df.to_excel(path_load+"totals_acc3.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s3)
df.to_excel(path_load+"totals_pre3.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s3)
df.to_excel(path_load+"totals_rec3.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s3)
df.to_excel(path_load+"totals_f_score3.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s4)
df.to_excel(path_load+"totals_acc4.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s4)
df.to_excel(path_load+"totals_pre4.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s4)
df.to_excel(path_load+"totals_rec4.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s4)
df.to_excel(path_load+"totals_f_score4.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s5)
df.to_excel(path_load+"totals_acc5.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s5)
df.to_excel(path_load+"totals_pre5.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s5)
df.to_excel(path_load+"totals_rec5.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s5)
df.to_excel(path_load+"totals_f_score5.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s6)
df.to_excel(path_load+"totals_acc6.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s6)
df.to_excel(path_load+"totals_pre6.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s6)
df.to_excel(path_load+"totals_rec6.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s6)
df.to_excel(path_load+"totals_f_score6.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s7)
df.to_excel(path_load+"totals_acc7.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s7)
df.to_excel(path_load+"totals_pre7.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s7)
df.to_excel(path_load+"totals_rec7.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s7)
df.to_excel(path_load+"totals_f_score7.xlsx", sheet_name="f_score", index=False)

df = pd.DataFrame(S_acc_s8)
df.to_excel(path_load+"totals_acc8.xlsx", sheet_name="acc", index=False)
df = pd.DataFrame(S_pre_s8)
df.to_excel(path_load+"totals_pre8.xlsx", sheet_name="pre", index=False)
df = pd.DataFrame(S_rec_s8)
df.to_excel(path_load+"totals_rec8.xlsx", sheet_name="rec", index=False)
df = pd.DataFrame(S_f_score_s8)
df.to_excel(path_load+"totals_f_score8.xlsx", sheet_name="f_score", index=False)

df11 = pd.DataFrame(list(global_loss))
df11.to_excel(path_load+"global_loss.xlsx", sheet_name="Sheet1", index=False)
