import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.utils.file_handler import load_dataset, load_dataset3, load_model, save_model


def train():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    save_model(model, model_path)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(accuracy)
    print(report)


if __name__ == '__main__':
    dataset_directory1 = r'C:\Users\13and\PycharmProjects\DP\data\dataset\02'
    dataset_directory2 = r'C:\Users\13and\PycharmProjects\DP\data\dataset\04'
    model_path = r'C:\Users\13and\PycharmProjects\DP\data\model_test.pkl'

    X, y = load_dataset3(dataset_directory1)
    X2, y2 = load_dataset(dataset_directory2, mode=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model(model_path)

    # for i in range(len(X_test)):
    #     t = np.linspace(0, 1, len(X_test[i]))
    #     y1 = y_test[i].reshape((300,))
    #     y2 = model.predict(X_test[i].reshape(1, 300)).reshape((300, ))
    #
    #     plt.gca().invert_yaxis()
    #     plt.plot(t, y1, linewidth=10.0)
    #     plt.plot(t, y2, color='lime')
    #     plt.scatter(t, X_test[i])
    #     plt.show()
    #     input()

    for i in range(len(X2)):
        t = np.linspace(0, 1, len(X2[i]))
        y2 = model.predict(X2[i].reshape(1, 300)).reshape((300,))

        plt.gca().invert_yaxis()
        plt.plot(t, y2, color='lime')
        plt.scatter(t, X2[i])
        plt.show()
        input()
