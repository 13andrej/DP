

coefficients = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
thresholds = {'a0': (20, 40), 'a1': (-2, 2), 'a2': (-2, 2), 'a3': (-2, 2), 'a4': (-2, 2), 'a5': (-2, 2), 'a6': (-2, 2),
              'a7': (-2, 2), 'a8': (-2, 2), 'b1': (-2, 2), 'b2': (-2, 2), 'b3': (-2, 2), 'b4': (-2, 2), 'b5': (-2, 2),
              'b6': (-2, 2), 'b7': (-2, 2), 'b8': (-2, 2)}
day_seconds = 86400
skip_lines_filter = 1
skip_lines_tracks = 7

glint_probability = {0: 50, 1: 50}
sigma = (13, 40)
width = (4, 15)
no_of_points = 300

data_directory = r'C:\Users\13and\PycharmProjects\DP\data'
dataset_directory = r'C:\Users\13and\PycharmProjects\DP\data\dataset\04'
synthetic_dataset_directory = r'C:\Users\13and\PycharmProjects\DP\data\dataset\03'
fourier_directory = r'C:\Users\13and\PycharmProjects\DP2\data\fourier'
mmt_directory = r'C:\Users\13and\PycharmProjects\DP\data\mmt'
annotation_file = r'C:\Users\13and\PycharmProjects\DP\data\annotation.json'


param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, 25, 30, 35, None],
    'min_samples_split': [2, 5, 10]
}

param_grid_svm = {
    'kernel': ['rbf', 'poly'],
    'C': [0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0],
    'gamma': ['scale', 'auto']
}

param_grid_xgb = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, 20, None],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
