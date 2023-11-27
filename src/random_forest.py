import os

import matplotlib.pyplot as plt
import numpy as np
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

    return features, label


def load_dataset(dataset_path):
    files = os.listdir(dataset_path)
    X_ = np.zeros((len(files), 300))
    y_ = np.zeros(len(files))

    for index, filename in enumerate(files):
        x_row, y_col = load_curve(os.path.join(dataset_path, filename))
        X_[index] = x_row
        y_[index] = y_col

    return X_, y_


if __name__ == '__main__':
    X, y = load_dataset(r'C:\Users\13and\PycharmProjects\DP\data\dataset\exports')
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_c = RandomForestClassifier(n_estimators=100)
    rf_c.fit(X_train, y_train)

    predictions = rf_c.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(accuracy)
    print(report)
