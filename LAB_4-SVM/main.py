import json
from functools import partial
import pandas as pd
from numpy import ndarray
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from LAB_4.svm import SVM


def linear_kernel(x1: ndarray, x2: ndarray):
    return np.dot(x1, x2)


def RFB_kernel(x1: ndarray, x2: ndarray, sigma=1.0):
    return np.exp(-(np.linalg.norm(x1 - x2) ** 2) /
                  (2 * (sigma ** 2)))


def prepare_data(path) -> tuple[ndarray, ndarray]:
    data = pd.read_csv(path, delimiter=';', low_memory=False)
    data['quality'].values[data['quality'] <= 5] = -1
    data['quality'].values[data['quality'] > 5] = 1

    y = data["quality"]
    x = data.drop(columns=["quality"])

    x = preprocessing.normalize(x, axis=0)
    return x, y


def kernels_comparison(kernel_type: str, data_path: str, sigma=1.0):
    if kernel_type == "linear":
        kernel_function = linear_kernel
    elif kernel_type == "RFB":
        kernel_function = partial(RFB_kernel, sigma=sigma)
    else:
        raise ValueError("Invalid kernel.")

    x, y = prepare_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2137)
    C_values = [10 ** i for i in range(-4, 6)]
    kernel_results = {}
    for c in C_values:
        model = SVM(kernel_function=kernel_function, C=c, min_lagrange_multiplier=1e-5)
        model.train(x_train, y_train.to_numpy())
        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        kernel_results.update({c: score})

    return kernel_results


def sigma_comparison(data_path):
    x, y = prepare_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2137)
    sigma_values = [10 ** i for i in range(-4, 6)]
    C = 1000.0
    kernel_results = {}
    for sigma in sigma_values:
        kernel_function = partial(RFB_kernel, sigma=sigma)
        model = SVM(kernel_function=kernel_function, C=C, min_lagrange_multiplier=1e-5)
        model.train(x_train, y_train.to_numpy())
        y_pred = model.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        kernel_results.update({sigma: score})
    return kernel_results


if __name__ == "__main__":
    comparison = {
        "white_lin": kernels_comparison(kernel_type="linear", data_path="winequality-white.csv"),
        "white_rfb": kernels_comparison(kernel_type="RFB", data_path="winequality-white.csv"),
        "red_lin": kernels_comparison(kernel_type="linear", data_path="winequality-red.csv"),
        "red_rfb": kernels_comparison(kernel_type="RFB", data_path="winequality-red.csv")
    }
    with open("comparison.json", "w") as file:
        json.dump(comparison, file)

    sigma_results = {
        "white_rfb": sigma_comparison("winequality-white.csv"),
        "red_rfb": sigma_comparison("winequality-red.csv"),
    }
    with open("sigma_comparison.json", "w") as file:
        json.dump(sigma_results, file)
