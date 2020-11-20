import numpy as np

def mse(y_true, y_predict):
    return np.mean(np.square(y_true - y_predict))

def mae(y_true, y_predict):
    return np.mean(np.abs(y_true - y_predict))

def acc(y_true, y_predict):
    return np.sum(y_true == y_predict)/len(y_true)

def _acc_diff(y_true, y_predict, diff):
    return np.sum(np.abs(y_true - y_predict) <= diff)/len(y_true)

def acc_diff1(y_true, y_predict):
    return _acc_diff(y_true, y_predict, 1)

def acc_diff2(y_true, y_predict):
    return _acc_diff(y_true, y_predict, 2)