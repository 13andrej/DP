import os

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.utils.constants import dataset_directory, synthetic_dataset_directory, data_directory, param_grid_rf
from src.utils.file_handler import load_dataset, load_model, save_model, use_model_dir


best_params = {
    0: {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 300},
    1: {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 50},
    2: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300},
    3: {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 100},
    4: {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 300},
    5: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
}


def grid_search(param_grid=param_grid_rf):
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print('Best parameters found: ', grid_search.best_params_)
    print('Best cross-validation accuracy: {:.2f}'.format(grid_search.best_score_))

    test_score = grid_search.score(X_test, y_test)
    print('Test set accuracy: {:.2f}'.format(test_score))


def randomized_search(n=20, param_grid=param_grid_rf):
    model = RandomForestClassifier(random_state=42)
    rand_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n, cv=5)
    rand_search.fit(X_train, y_train)

    print('Best parameters found: ', rand_search.best_params_)
    print('Best cross-validation accuracy: {:.2f}'.format(rand_search.best_score_))

    test_score = rand_search.score(X_test, y_test)
    print('Test set accuracy: {:.2f}'.format(test_score))


def cross_val(n_estimators=100, max_depth=None, min_samples_split=2):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print('Cross-validation scores:', scores)
    print('Average accuracy:', scores.mean())

    return scores


def train_model(n_estimators=100, max_depth=None, min_samples_split=2, save=False, show=True):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
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
    for n_estimators in param_grid_rf['n_estimators']:
        for max_depth in param_grid_rf['max_depth']:
            for min_samples_split in param_grid_rf['min_samples_split']:
                a = train_model(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, show=False)
                res[f'n_e={n_estimators}, m_d={max_depth}, m_s_s={min_samples_split}'] = a
                print(f'n_e={n_estimators}, m_d={max_depth}, m_s_s={min_samples_split}', a)

    best = None
    best_a = 0

    for k, v in res.items():
        if v > best_a:
            best_a = v
            best = k

    return best, best_a


if __name__ == '__main__':
    model_path = os.path.join(data_directory, 'rf_c.pkl')
    features_mode = 0
    train_mode = 0

    if train_mode:
        X_train, y_train = load_dataset(synthetic_dataset_directory, mode=features_mode)
        X_test, y_test = load_dataset(dataset_directory, mode=features_mode)
    else:
        X, y = load_dataset(dataset_directory, mode=features_mode)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # grid_search()
    # print(my_grid())
    # train_model()
    train_model(n_estimators=300, max_depth=5, min_samples_split=2)
    # cross_val(n_estimators=300, max_depth=5, min_samples_split=2)

    # use_model_dir(dataset_directory, model_path)
    # use_model_dir(synthetic_dataset_directory)

    # for i in range(6):
    #     X, y = load_dataset(dataset_directory, mode=i)
    #     cross_val()
