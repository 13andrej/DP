import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.datasets import load_unit_test # A small example dataset

from src.utils.constants import dataset_directory, data_directory
from src.utils.functions import fourier
from src.utils.file_handler import load_dataset, load_model, save_model, read_light_curve


if __name__ == '__main__':
    # Load example data (replace with your own data)
    # X_train, y_train = load_unit_test(split="train", return_X_y=True)
    # X_test, y_test = load_unit_test(split="test", return_X_y=True)
    #
    # print(X_train.shape)
    # print(y_train.shape)
    #
    # print(type(X_train))
    # for i in X_train:
    #     print(type(i))
    #
    # print(X_train['dim_0'][20])
    # exit(77)

    X, y = load_dataset(dataset_directory)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Time Series Forest classifier
    # You can adjust parameters like n_estimators (number of trees)
    tsf = TimeSeriesForestClassifier(n_estimators=200, random_state=42)

    # Train the classifier
    tsf.fit(X_train, y_train)

    # Make predictions
    y_pred = tsf.predict(X_test)

    # Evaluate the performance (e.g., using accuracy)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # You can also get prediction probabilities
    y_proba = tsf.predict_proba(X_test)
    print("Prediction Probabilities:")
    print(y_proba[:5])  # Print probabilities for the first 5 test instances
