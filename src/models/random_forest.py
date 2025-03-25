import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_curve(curve_path):
    reading_data = False
    features = []
    label = None

    with open(curve_path) as file:
        for line in file:
            if reading_data:
                features.append(float(line.strip().split()[1]))
            elif line.startswith('#'):
                reading_data = True
            elif line.startswith('Glint:'):
                label = int(line.strip().split()[1] == 'True')

    return np.array(features), label


def load_dataset(dataset_path):
    files = os.listdir(dataset_path)
    X_ = np.zeros((len(files), 300))
    y_ = np.zeros(len(files))

    x = np.arange(0, 1.00001, 1 / (300 - 1))
    for index, filename in enumerate(files):
        x_row, y_col = load_curve(os.path.join(dataset_path, filename))
        x_row_fit = fourier(x, x_row, 8)
        X_[index] = x_row - x_row_fit
        y_[index] = y_col

    return X_, y_


def fourier(x, y, n):
    N = x.shape[0]
    fft = np.fft.fft(y)
    fft[n:-n] = 0
    res = sp.fft.ifft(fft)

    return np.abs(res)


if __name__ == '__main__':
    dat_path = r'C:\Users\13and\PycharmProjects\DP\data\dataset\01\exports'
    X, y = load_dataset(dat_path)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_c = RandomForestClassifier(n_estimators=300)
    rf_c.fit(X_train, y_train)

    predictions = rf_c.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(accuracy)
    print(report)
