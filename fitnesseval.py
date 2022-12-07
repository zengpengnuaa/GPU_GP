import math

import torch
import torchvision
from numba import cuda
import numpy as np
from sklearn import preprocessing
import importlib
import cuda_func
import cuda_func_1
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

# load dataset
import test

dataSetName = 'kth'
x_train = np.load('./Dataset/KTH/' + dataSetName + '_train_data.npy')
y_train = np.load('./Dataset/KTH/' + dataSetName + '_train_label.npy')
x_test = np.load('./Dataset/KTH/' + dataSetName + '_test_data.npy')
y_test = np.load('./Dataset/KTH/' + dataSetName + '_test_label.npy')

# cifar10
dataset_path = './Dataset/cifar'
from torchvision import transforms

gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
])
train_dataset = torchvision.datasets.CIFAR10(dataset_path, train=True)
test_dataset = torchvision.datasets.CIFAR10(dataset_path, train=False)
train_data = train_dataset.data
train_data = np.transpose(torch.tensor(train_data), (0, 3, 1, 2))
train_data = gray_transform(train_data)
train_data = train_data.float()
train_data = torch.reshape(train_data, shape=[50000, 32, 32])
train_data = train_data.detach().numpy()
train_target = train_dataset.targets
train_target = np.asarray(train_target)
test_data = test_dataset.data
test_data = np.transpose(torch.tensor(test_data), (0, 3, 1, 2))
test_data = gray_transform(test_data)
test_data = test_data.float()
test_data = torch.reshape(test_data, shape=[10000, 32, 32])
test_data = test_data.detach().numpy()
test_target = test_dataset.targets
test_target = np.asarray(test_target)

# data process
train_data = test.data_process(train_data)
test_data = test.data_process(test_data)


def CzekanowskiDistance(u, v):
    uv = np.matrix([u, v])
    uv = np.min(uv, axis=0)
    sum = 2 * np.sum(uv)
    den = np.sum(u) + np.sum(v)
    dis = 1.0 - 1.0 * sum / den
    return dis


def FitnessEvaluation(train_samples, toolbox, individual):
    nClasses = len(train_samples)
    nImages = len(train_samples[1])

    indi_lambda = toolbox.compile(individual)

    train_tf = []
    for i in range(0, len(nClasses * nImages)):
        result = np.asarray(indi_lambda(train_samples))
        train_tf.append(result)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    train_norm = np.reshape((10, nClasses * nImages))


# fitness evaluating
def fitness_func(indi, lock):
    try:
        innerfunc_string = 'lambda ARG0: ' + str(indi)
        file_path = '../XSGP/cuda_func.py'
        lock.acquire()
        fd = open(file_path, mode='r')
        content = fd.read()
        pos = content.find('innerfunc')
        content = content[:pos]
        s = 'innerfunc = ' + innerfunc_string + '\n    feature_vec = innerfunc(imgs[x])\n    for i in range(10):\n        ret_vectors[x][i] = feature_vec[i]'
        content += s
        fd = open(file_path, mode='w')
        fd.write(content)
        fd.close()
        importlib.reload(cuda_func)
        lock.release()

        threads_num = 480
        vector_dim = 10
        total_x = threads_num
        total_y = 1
        threadDim = 256
        threadsperblock = (threadDim, 1)
        blockspergrid_x = int(math.ceil(total_x / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(total_y / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        oral_data = x_train[:, :, :]
        d_imgs = cuda.to_device(oral_data)
        ret_vectors = cuda.device_array((threads_num, vector_dim))
        cuda_func.cudafunc[blockspergrid, threadsperblock](d_imgs, ret_vectors)
        results = ret_vectors.copy_to_host()
        cuda.close()
        train_tf = np.asarray(results)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        # pred_value = numpy.mean(train_norm, axis=1)
        # pred_labels = []
        # for i in range(0, len(y_train)):
        #     pred_labels.append(bisect_left(labels, pred_value[i]))
        # acc = metrics.accuracy_score(y_train, pred_labels)
        lsvm = LinearSVC()
        acc = round(100 * cross_val_score(lsvm, train_norm, y_train, cv=5).mean(), 2)
    except:
        acc = 0.
    return acc,


# fitness evaluating
def fitness_func_1(indi, lock):
    try:
        innerfunc_string = 'lambda ARG0: ' + str(indi)
        file_path = '../XSGP/cuda_func.py'
        lock.acquire()
        fd = open(file_path, mode='r')
        content = fd.read()
        pos = content.find('innerfunc')
        content = content[:pos]
        s = 'innerfunc = ' + innerfunc_string + '\n    feature_vec = innerfunc(imgs[x])\n    for i in range(10):\n        ret_vectors[x][i] = feature_vec[i]'
        content += s
        fd = open(file_path, mode='w')
        fd.write(content)
        fd.close()
        importlib.reload(cuda_func_1)
        lock.release()

        threads_num = 50000
        vector_dim = 10
        total_x = threads_num
        total_y = 1
        threadDim = 256
        threadsperblock = (threadDim, 1)
        blockspergrid_x = int(math.ceil(total_x / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(total_y / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        oral_data = train_data[:, :, :]
        d_imgs = cuda.to_device(oral_data)
        ret_vectors = cuda.device_array((threads_num, vector_dim))
        cuda_func_1.cudafunc[blockspergrid, threadsperblock](d_imgs, ret_vectors)
        results = ret_vectors.copy_to_host()
        cuda.close()
        train_tf = np.asarray(results)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        # pred_value = numpy.mean(train_norm, axis=1)
        # pred_labels = []
        # for i in range(0, len(train_target)):
        #     pred_labels.append(bisect_left(labels, pred_value[i]))
        # acc = metrics.accuracy_score(train_target, pred_labels)
        lsvm = LinearSVC()
        acc = round(100 * cross_val_score(lsvm, train_norm, train_target, cv=5).mean(), 2)
    except:
        acc = 0.
    return acc,
