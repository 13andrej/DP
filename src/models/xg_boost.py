import os

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV

import xgboost as xgb
from src.utils.constants import data_directory, synthetic_dataset_directory, dataset_directory, param_grid_xgb
from src.utils.file_handler import load_dataset, save_model


best_params = {
    0: {'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 20, 'subsample': 1.0},

}


def grid_search(param_grid=param_grid_xgb):
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric="logloss")
    search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print('Best parameters found: ', search.best_params_)
    print('Best cross-validation accuracy: {:.2f}'.format(search.best_score_))

    test_score = search.score(X_test, y_test)
    print('Test set accuracy: {:.2f}'.format(test_score))


def randomized_search(n=20, param_grid=param_grid_xgb):
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric="logloss")
    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n, cv=5, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)

    print('Best parameters found: ', search.best_params_)
    print('Best cross-validation accuracy: {:.2f}'.format(search.best_score_))

    test_score = search.score(X_test, y_test)
    print('Test set accuracy: {:.2f}'.format(test_score))


def cross_val(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8):
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=n_estimators,
                              learning_rate=learning_rate, max_depth=max_depth, subsample=subsample)
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print('Cross-validation scores:', scores)
    print('Average accuracy:', scores.mean())


def train_model(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, save=False, show=True):
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=n_estimators,
                              learning_rate=learning_rate, max_depth=max_depth, subsample=subsample)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    if show:
        print(accuracy)
        print(report)

        cm = confusion_matrix(y_test, predictions)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.show()

    if save:
        save_model(model, model_path)

    return accuracy


def my_grid():
    res = {}
    for n_estimators in param_grid_xgb['n_estimators']:
        for max_depth in param_grid_xgb['max_depth']:
            for learning_rate in param_grid_xgb['learning_rate']:
                for subsample in param_grid_xgb['subsample']:
                    a = train_model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, show=False)
                    res[f'n_e={n_estimators}, m_d={max_depth}, l_r={learning_rate}, s={subsample}'] = a
                    print(f'n_e={n_estimators}, m_d={max_depth}, l_r={learning_rate}, s={subsample}', a)

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
    train_model(n_estimators=100, learning_rate=0.01, max_depth=10, subsample=0.8)
