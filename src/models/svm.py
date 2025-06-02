import os

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

from src.utils.constants import dataset_directory, synthetic_dataset_directory, data_directory, param_grid_svm
from src.utils.file_handler import load_dataset, save_model


best_params = {
    0: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'},
    1: {'C': 0.5, 'gamma': 'auto', 'kernel': 'rbf'},
    2: {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'},
    3: {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},
    4: {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'},
    5: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
}


def grid_search(param_grid=param_grid_svm):
    model = SVC(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print('Best parameters found: ', grid_search.best_params_)
    print('Best cross-validation accuracy: {:.2f}'.format(grid_search.best_score_))

    test_score = grid_search.score(X_test, y_test)
    print('Test set accuracy: {:.2f}'.format(test_score))


def randomized_search(n=20, param_grid=param_grid_svm):
    model = SVC(random_state=42)
    rand_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n, cv=5)
    rand_search.fit(X_train, y_train)

    print('Best parameters found: ', rand_search.best_params_)
    print('Best cross-validation accuracy: {:.2f}'.format(rand_search.best_score_))

    test_score = rand_search.score(X_test, y_test)
    print('Test set accuracy: {:.2f}'.format(test_score))


def cross_val(C=1.0, gamma='scale', kernel='rbf'):
    model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print('Cross-validation scores:', scores)
    print('Average accuracy:', scores.mean())

    return scores


def train_model(C=1.0, gamma='scale', kernel='rbf', save=False, show=True):
    model = SVC(C=C, gamma=gamma, kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    if show:
        print(accuracy)
        print(report)

        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.show()

    if save:
        save_model(model, model_path)

    return accuracy


def my_grid():
    res = {}
    for c in param_grid_svm['C']:
        for gamma in param_grid_svm['gamma']:
            for kernel in param_grid_svm['kernel']:
                a = train_model(C=c, gamma=gamma, kernel=kernel, show=False)
                res[f'C={c}, gamma={gamma}, kernel={kernel}'] = a
                print(f'C={c}, gamma={gamma}, kernel={kernel}', a)

    best = None
    best_a = 0

    for k, v in res.items():
        if v > best_a:
            best_a = v
            best = k

    return best, best_a


if __name__ == '__main__':
    model_path = os.path.join(data_directory, 'rf_c.pkl')
    features_mode = 2
    train_mode = 1

    if train_mode:
        X_train, y_train = load_dataset(synthetic_dataset_directory, mode=features_mode)
        X_test, y_test = load_dataset(dataset_directory, mode=features_mode)
    else:
        X, y = load_dataset(dataset_directory, mode=features_mode)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # grid_search()
    # print(my_grid())
    # train_model()
    train_model(C=0.1, gamma='auto', kernel='rbf')
