import numpy as np
import random

def train_val_test_split(X, y, train_ratio, val_ratio):
    n_train = int(train_ratio * len(X))
    n_val = int(val_ratio * len(X))
    sequences = np.arange(len(X)).tolist()
    train_indices_c = random.sample(sequences, n_train)
    train_indices = [i for i in sequences if i not in train_indices_c]
    val_indices = random.sample(train_indices_c, n_val)
    test_indices = [i for i in train_indices_c if i not in val_indices]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_val = [X[i] for i in val_indices]
    y_val = [y[i] for i in val_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, y_train, X_val, y_val, X_test, y_test