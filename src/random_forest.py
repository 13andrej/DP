import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

if __name__ == '__main__':
    X = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf_c = RandomForestClassifier(n_estimators=10)
    rf_c.fit(X_train, y_train)

    predictions = rf_c.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(accuracy)
    print(report)