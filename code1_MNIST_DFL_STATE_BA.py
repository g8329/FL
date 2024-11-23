import time
import functools
import torch
import random
import copy
import math
import statistics
import numpy as np
import tensorflow as tf
import cv2
import os
from imutils import paths
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import networkx as nx  # 导入networkx包，命名为nx
import random
import matplotlib.animation as animation
warnings.filterwarnings("ignore")


def datapreprocessing(raw_data):
    data_list0 = raw_data.iloc[:, 1:]  # 将数据中的特征提取
    label_list0 = raw_data.iloc[:, 0]  # 将数据中的标签提取
    data_list_0 = list()
    length_0 = len(label_list0)
    label_list_0 = np.zeros((length_0, 2), int)
    # 将数据和标签的格式转换
    for i in np.arange(length_0):
        label_0 = label_list0[i]
        label_0 = label_0.astype(str)
        label_0 = str(label_0)
        if label_0 == '0':
            label_list_0[i, 0] = 1
        else:
            label_list_0[i, 1] = 1
        d_list_0 = data_list0.loc[i]
        d_list0 = d_list_0.tolist()
        data_list_0.append(d_list0)
    return data_list_0, label_list_0


# 计算三类人群的数量
def countSARS(G,num_nodes):
    I = 0
    R = 0
    for k in G:
        if G.nodes[k]["state"] == "I":
            I = I + 1
        # if G.nodes[k]["state"] =="A":
        #     A = A + 1
        if G.nodes[k]["state"] =="R":
            R = R + 1
    return I, num_nodes-I-R,R


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def tm_local(X_test1, Y_test1, model1):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits1 = model1.predict(X_test1)
    loss1 = cce(Y_test1, logits1)
    acc1 = accuracy_score(tf.argmax(logits1, axis=1), tf.argmax(Y_test1, axis=1))
    return acc1, loss1


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
        label_list_all.append(client[i][1])
        # cc = client[i][1]
        # for j in range(10):
        #     if cc[j] == 1:
        #         label_list_all.append(j)
    return image_list_all, label_list_all


def preprocess_data(clients, client_name):
    # 节点数据预处理
    image_list, label_list = split_data_list_label(clients, client_name)
    image_array = np.array(image_list)
    label_array = np.array(label_list)

    # 将数据分割成训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(image_array,
                                                        label_array,
                                                        test_size=0.2,
                                                        random_state=42)
    test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))
    shards = list(zip(x_train, y_train))

    return shards, test_batched


def evaluate_clients(client_names, all_model, test_batchedacc, tm_local):
    client_acc = list()
    client_loss = list()

    for c_num in range(client_names):
        local_model.set_weights(all_model[c_num])
        acc_list = list()
        loss_list = list()
        for (x_test, y_test) in test_batchedacc:
            local_acc, local_loss = tm_local(x_test, y_test, local_model)
            acc_list.append(local_acc)
            loss_list.append(local_loss)
        client_acc.append(acc_list)
        client_loss.append(loss_list)

    return client_acc, client_loss


def count_neighbors(graph, node):
    non_infected_count = 0
    neighbors = graph.neighbors(node)
    for neighbor in neighbors:
        if graph.nodes[neighbor]["state"] != "I":
            non_infected_count += 1
    return non_infected_count


def updateNodeState(G, acc_all_node, t, state_change,first_all_A_iteration,t_first):
    G1 = G.copy()
    for k in G1:
        if G1.nodes[k]["state"] == "A":    # 感染者
            nerbor2=0
            for neibor1 in G1.adj[k]:
                if G1.nodes[neibor1]["state"] != "I":
                    nerbor2=nerbor2+1
            if nerbor2==G1.degree[k]:
                first_all_A_iteration[k]=first_all_A_iteration[k]+1
            else:
                state_change[k] = 0
            if first_all_A_iteration[k]==1:  # 如果这是该节点第一次满足条件
                t_first[k]=t
            if t_first[k]<=t and t>0:
                p = random.random()  # 生成一个0到1的随机数
                blA = acc_all_node[k][t] - acc_all_node[k][t-1]
                blA1 = 1 / ((1 + np.exp(blA))*50)
                if p<blA1:
                    G.nodes[k]["state"] = "R"
                    print('blA1=',blA1)
                    state_change[k]=1
        if G1.nodes[k]["state"] == "I":     # 易感者
            p = random.random()    # 生成一个0到1的随机数
            beta_list=list()  # 计算邻居中的感染者数量
            for nb in G1.adj[k]:  # 查看所有邻居状态，遍历邻居用 G.adj[node]
                bl = acc_all_node[nb][t] - acc_all_node[k][t]
                bl1 = 1.5-1.5 / (1 + np.exp(- bl))
                # bl2=acc_all_node[nb][t]-acc_all_node[nb][t-1]-acc_all_node[k][t]+acc_all_node[k][t-1]
                # print('bl1=',bl1)
                if G1.nodes[nb]["state"] == "A" and bl > 0:
                # if G1.nodes[nb]["state"] == "A" and bl > 0 and bl2>0:  # 如果这个邻居是感染者，则k加1
                    print('beta=',1-bl1,'node=',k)
                    # print('bl2=',bl2)
                    beta_list.append(bl1)
            l1=len(beta_list)
            if l1>0:
                beta = functools.reduce(lambda x, y: x * y, beta_list)
                print('1-betai=',1-beta)
                # beta = 1
                # for bb in beta_list:
                #     beta *= bb
                if p < 1 - beta:  # 易感者被感染
                    G.nodes[k]["state"] = "A"
                    state_change[k]=0
        if G1.nodes[k]["state"] == "R":  # 易感者
            p = random.random()  # 生成一个0到1的随机数
            beta_list = list()  # 计算邻居中的感染者数量
            for nb in G1.adj[k]:  # 查看所有邻居状态，遍历邻居用 G.adj[node]
                bl = acc_all_node[nb][t] - acc_all_node[k][t]
                bl1 =1.5-1.5 / (1 + np.exp(- bl))
                # bl2=acc_all_node[nb][t]-acc_all_node[nb][t-1]-acc_all_node[k][t]+acc_all_node[k][t-1]
                # print('bl1=',bl1)
                if G1.nodes[nb]["state"] == "A" and bl > 0:
                    # if G1.nodes[nb]["state"] == "A" and bl > 0 and bl2>0:  # 如果这个邻居是感染者，则k加1
                    print('bl1r=', bl1, 'node=', k)
                    # print('bl2=',bl2)
                    beta_list.append(bl1)
            l1 = len(beta_list)
            if l1 > 0:
                beta = functools.reduce(lambda x, y: x * y, beta_list)
                print('1-betar=', 1 - beta)
                # beta = 1
                # for bb in beta_list:
                #     beta *= bb
                if p < 1 - beta:  # 易感者被感染
                    G.nodes[k]["state"] = "A"
                    state_change[k] = 0


w_factor = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
for m in range(1,10,1):
    T=list()
    print(m)
    for m1 in range(1,11,1):
        print(m1)
        img_path = 'E:/OldComputer/myself/FedGitHubDate/tutorial-master/tutorial-master/archive/trainingSet/trainingSet'
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
        test_batched_all = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
        # create clients
        clients = create_clients(X_train, y_train, num_clients=10, initial='client')

        res = []
        test_batched = []
        # 节点1的数据预处理
        shards_1, test_batched_1 = preprocess_data(clients, 'client_1')
        res.append(shards_1)
        test_batched.append(test_batched_1)

        # 节点2的数据预处理
        shards_2, test_batched_2 = preprocess_data(clients, 'client_2')
        res.append(shards_2)
        test_batched.append(test_batched_2)
        # 节点3的数据预处理
        shards_3, test_batched_3 = preprocess_data(clients, 'client_3')
        res.append(shards_3)
        test_batched.append(test_batched_3)
        # 节点4的数据预处理
        shards_4, test_batched_4 = preprocess_data(clients, 'client_4')
        res.append(shards_4)
        test_batched.append(test_batched_4)
        # 节点5的数据预处理
        shards_5, test_batched_5 = preprocess_data(clients, 'client_5')
        res.append(shards_5)
        test_batched.append(test_batched_5)
        # 节点6的数据预处理
        shards_6, test_batched_6 = preprocess_data(clients, 'client_6')
        res.append(shards_6)
        test_batched.append(test_batched_6)
        # 节点7的数据预处理
        shards_7, test_batched_7 = preprocess_data(clients, 'client_7')
        res.append(shards_7)
        test_batched.append(test_batched_7)
        # 节点8的数据预处理
        shards_8, test_batched_8 = preprocess_data(clients, 'client_8')
        res.append(shards_8)
        test_batched.append(test_batched_8)
        # 节点9
        shards_9, test_batched_9 = preprocess_data(clients, 'client_9')
        res.append(shards_9)
        test_batched.append(test_batched_9)
        # 节点10
        shards_10, test_batched_10 = preprocess_data(clients, 'client_10')
        res.append(shards_10)
        test_batched.append(test_batched_10)

        num_nodes = 10
        client_names = ['{}_{}'.format('client', i + 1) for i in range(num_nodes)]
        clients = {client_names[i]: res[i] for i in range(num_nodes)}
        # process and batch the training data for each client
        clients_batched = dict()
        for (client_name, data) in clients.items():
            clients_batched[client_name] = batch_data(data)

        lr = 0.01
        comms_round = 500
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
        optimizer = SGD(lr=lr,
                        decay=lr / comms_round,
                        momentum=0.9
                        )
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
        initialization_local_model = local_model.get_weights()
        seed = 123  # 本案例中将会使用的随机数种子

        G = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        nodelist = sorted(G.nodes())
        # 转换为邻接矩阵，指定节点顺序
        H = nx.to_numpy_matrix(G, nodelist=nodelist)

        df_H = pd.DataFrame(H)
        df_H.to_excel('E:/OldComputer/guan/test/StatusChangeIA/MNIST/BA/'+ str(m) + "/"+'H'+str(m1)+ ".xlsx")
        # 初始化节点 state 属性
        for node in G:
            G.nodes[node]["state"] = "I"

        # n_p= np.zeros((num_nodes))
        # 随机选择一个节点并将其状态更改为"A"
        # random_node_index = random.randint(0, num_nodes-1)
        random_node_index = max(G.degree, key=lambda x: x[1])[0]
        # 随机选取一个节点为初始感染者
        G.nodes[random_node_index]["state"] = "A"
        # n_p[random_node_index] = 1
        print(list(G.nodes))
        all_local_model = list()
        all_ave_model = [None] * num_nodes
        all_client_acc = []
        all_client_loss = []
        IA_list = []
        IA_state = []
        # 记录程序开始时间
        start_time = time.time()
        acc_all=[[] for _ in range(num_nodes)]
        loss_all=[[] for _ in range(num_nodes)]
        ppp=list()
        # beta=0.042
        mm=list()
        state_change=np.zeros(num_nodes)
        state_change_record_list=list()
        first_all_A_iteration=np.zeros(num_nodes)
        t_first = [comms_round]*num_nodes
        for t in range(comms_round):
            IAR_num = countSARS(G, num_nodes)
            IA_list.append(list(countSARS(G, num_nodes)))
            node_state_rd = list()
            for node in range(num_nodes):
                node_state_rd.append(G.nodes[node]["state"])  # 记录一天的节点状态
            IA_state.append(node_state_rd)  # 记录N天的每个节点的状态
            if IAR_num[1] == 0:
                print("IAR的个数", IAR_num)
                print("通信次数为" + str(t))
                break
            state_change_record_list.append(state_change)
            print("di" + str(t + 1) + "ci" )
            print(node_state_rd)
            for client in client_names:
                client_num = int(client.split('_')[1]) - 1
                if G.nodes[client_num]["state"] == "A":
                    nerbor3 = 0
                    for neibor1 in G.adj[client_num]:
                        if G.nodes[neibor1]["state"] == "R":
                            nerbor3 = nerbor3 + 1
                    if nerbor3 == G.degree[client_num]:
                        state_change[client_num] = 1
                        print("通信次数为核对" + str(t+1))
                # if G.nodes[client_num]["state"]=='A':
                if t == 0:
                    local_model.set_weights(initialization_local_model)
                    local_model.fit(clients_batched[client], epochs=1, verbose=0)
                    all_local_model.append(local_model.get_weights())
                else:
                    if state_change[client_num]==0:
                        # client_num = int(client.split('_')[1]) - 1
                        local_model.set_weights(all_local_model[client_num])
                        local_model.fit(clients_batched[client], epochs=1, verbose=0)
                        all_local_model[client_num] = local_model.get_weights()
            mm.append(all_local_model)
            all_local_model1 = all_local_model.copy()
            # I节点间的参数交换
            # nodes1 = nodes.copy()
            # print("nodes",list(G.nodes))
            for i in range(num_nodes):
                local_model.set_weights(all_local_model1[i])
                for (x_test, y_test) in test_batched[i]:
                    local_acc_A_self, local_loss_A_self = tm_local(x_test, y_test, local_model)
                    acc_all[i].append(local_acc_A_self)
                    loss_all[i].append(local_loss_A_self)

                if G.nodes[i]["state"] == "A":
                    nodes_state_andneibor111 = list()
                    # 获取当前节点的模型参数
                    node_state = all_local_model1[i]
                    nodes_state_andneibor = list()
                    nodes_state_andneibor.append(all_local_model1[i])
                    # 初始化邻居参数列表，包括当前节点的参数
                    # count_neibor = count_neighbors(G, i)
                    # nodes_state_andneibor = [[np.copy(layer_weights) for layer_weights in node_state] for _ in range(count_neibor+1)]
                    # 对于每个节点
                    weight_record = list()
                    weight_factor = list()
                    weight_record.append(all_local_model1[i])
                    weight_factor.append(w_factor[i])
                    for i1 in G.adj[i]:
                        if G.nodes[i1]["state"] == 'A':
                            weight_record.append(all_local_model1[i1])
                            weight_factor.append(w_factor[i1])
                            # p = n_p[i1]
                            node_state2 = all_local_model1[i1]
                            # if p == 1:
                            nodes_state_andneibor.append(node_state2)
                            # else:
                            #     nodes_state_andneibor.append(copy_weights(node_state2, node_state, p))
                    l_neibor = np.size(nodes_state_andneibor, 0)
                    new_weight_factor = [xx / sum(weight_factor) for xx in weight_factor]
                    for jj11 in range(0, l_neibor):
                        nodes_state_andneibor11 = scale_model_weights(nodes_state_andneibor[jj11],
                                                                      new_weight_factor[jj11])
                        nodes_state_andneibor111.append(nodes_state_andneibor11)
                    avg_params = sum_scaled_weights(nodes_state_andneibor111)
                    # avg_params = sum_scaled_weights(nodes_state_andneibor111)
                    all_local_model[i]=avg_params


            client_acc, client_loss = evaluate_clients(num_nodes, all_local_model, test_batched_all, tm_local)
            all_client_acc.append(client_acc)
            all_client_loss.append(client_loss)
            if t>=0:
                updateNodeState(G, acc_all, t,state_change,first_all_A_iteration,t_first)
              # 计算更新后三种节点的数量
        end_time = time.time()
        file_path = 'E:/OldComputer/guan/test/StatusChangeIA/MNIST/BA/'+ str(m) + "/"
        data_no_brackets = [[str(item).strip('[]') for item in row] for row in all_client_acc]
        df = pd.DataFrame(data_no_brackets)
        # 尝试将字符串转换为数字
        df = df.apply(pd.to_numeric, errors='ignore')
        lujing = file_path + "all_acc_client" + str(m1)
        df.to_excel(lujing + ".xlsx", index=False)



        df_accall = pd.DataFrame(acc_all)
        lujing_accall = file_path + "accall" + str(m1)
        df_accall.to_excel(lujing_accall + ".xlsx", index=False)


        df1 = pd.DataFrame(ppp)
        lujing1 = file_path + "pp"+str(m1)
        df1.to_excel(lujing1 + ".xlsx", index=False)


        df2 = pd.DataFrame(IA_list)
        lujing2 = file_path + "IA_list" + str(m1)
        df2.to_excel(lujing2 + ".xlsx", index=False)
        df3 = pd.DataFrame(IA_state)
        lujing3 = file_path + "IA_state" + str(m1)
        df3.to_excel(lujing3 + ".xlsx", index=False)
        # 计算运行时长
        duration = (end_time - start_time)/60
        T.append(duration)
        print("程序运行时长：", duration, "分钟")
    file_path = 'E:/OldComputer/guan/test/StatusChangeIA/MNIST/BA/' + str(m) + "/"
    df = pd.DataFrame(T)
    lujing_T = file_path + "T"
    df.to_excel(lujing_T + ".xlsx", index=False)