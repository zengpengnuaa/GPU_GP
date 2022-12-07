import numpy as np
# from sklearn import metrics
from bisect import bisect_left

from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
# import cuml
import random
from sklearn import metrics


def img_norm(x):
    return x / 255.0


def flatten(x):
    return x.reshape(x.shape[0], -1)


def feat_norm(x):
    return 2 * (x - np.min(x) / (np.max(x) - np.min(x))) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def predict(Bestindi, testdata, testlabel=None, toolbox=None, labels=[]):
#     pred_labels, _ = predict_indi(Bestindi, testdata, toolbox, labels)
#     acc = None if testlabel is None else metrics.accuracy_score(pred_labels, testlabel)
#     return pred_labels, acc
def predict(toolbox, individual, trainData, trainLabel, testData, testLabel):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(trainLabel)):
        train_tf.append(np.asarray(func(trainData[i, :, :])))
    for j in range(0, len(testLabel)):
        test_tf.append(np.asarray(func(testData[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)
    lsvm = LinearSVC()
    lsvm.fit(train_norm, trainLabel)
    accuracy = round(100 * lsvm.score(test_norm, testLabel), 2)
    return accuracy


def predict_DC(toolbox, best_trees, weights, trainData, trainLabel, testData, testLabel):
    accuracies = []
    for individual in best_trees:
        func = toolbox.compile(expr=individual)
        train_tf = []
        test_tf = []
        for i in range(0, len(trainLabel)):
            train_tf.append(np.asarray(func(trainData[i, :, :])))
        for j in range(0, len(testLabel)):
            test_tf.append(np.asarray(func(testData[j, :, :])))
        train_tf = np.asarray(train_tf)
        test_tf = np.asarray(test_tf)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(train_tf)
        test_norm = min_max_scaler.transform(test_tf)
        lsvm = LinearSVC()
        lsvm.fit(train_norm, trainLabel)
        accuracy = round(100 * lsvm.score(test_norm, testLabel), 2)
        accuracies.append(accuracy)
    final_acc = 0.
    for i in range(len(weights)):
        final_acc += accuracies[i] * weights[i]
    return final_acc


def data_process(data):
    data = img_norm(data)
    # data = flatten(data).tolist()
    # data = np.vstack(data)
    # data = feat_norm(data).tolist()
    return data


def get_label_by_output(output, labels):
    index = bisect_left(labels, output) - 1
    if index < 0:
        return 0
    if index > len(labels):
        return len(labels)
    return index


def predict_indi(indi, data, toolbox, labels):
    func = toolbox.compile(expr=indi)
    # terms = data_process(data)
    output_values = [sigmoid(func(*term)) for term in data]
    pred_labels = [get_label_by_output(output, labels) for output in output_values]
    return np.asarray(pred_labels), np.asarray(output_values)
