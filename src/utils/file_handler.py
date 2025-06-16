import json
import os
import pickle

import numpy as np
from astropy.time import Time

from src.utils.classes import LC, LC2
from src.utils.constants import mmt_directory, day_seconds
from src.utils.functions import fourier, fourier_coefficients, normalize


def read_tracks_file(tracks_file, filter_file):
    """read light curves of periodic objects from track file"""
    t = Time.now().mjd

    with open(filter_file) as file:
        file.readline()
        starting_points = {}

        for line in file:
            _, track, starting_point = line.strip().split()

            if track not in starting_points:
                starting_points[track] = []
            starting_points[track].append(int(starting_point))

    with open(tracks_file) as file:
        time_array, mag_array = {}, {}

        for line in file:
            if line.startswith('#'):
                continue

            date, time, _, mag, _, _, d, _, _, track = line.strip().split()

            if track not in time_array:
                time_array[track] = []
                mag_array[track] = []

            time_array[track].append(Time(f'{date}T{time}', format='isot'))
            mag_array[track].append(float(mag))

    print('read time: ', (Time.now().mjd - t) * day_seconds)
    res = []

    for track in starting_points:
        starting_points[track].append(9999999)
        for i in range(len(starting_points[track]) - 1):
            a, b = starting_points[track][i], starting_points[track][i + 1]
            res.append(LC(time_array[track][a: b], mag_array[track][a: b], track, str(a)))

    return res


def read_tracks_dir(tracks_dir):
    """read light curves of periodic objects from directory with track files"""
    res = []

    for directory in os.listdir(tracks_dir):
        if not os.path.isdir(os.path.join(tracks_dir, directory)) or 'results' not in directory:
            continue

        tracks_file = os.path.join(tracks_dir, directory, directory.split('_')[-1] + '_tracks.txt')
        filter_file = os.path.join(tracks_dir, directory, 'Filter')
        res += read_tracks_file(tracks_file, filter_file)

    return res


def read_results_file(results_file, *elements):
    res = {element: [] for element in elements}
    res['RMS[%]'] = []
    res['Points'] = np.array([])

    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    for track in data[key].keys():
        for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
            for element in elements:
                res[element].append(data[key][track][starting_point]['Fourier'][element])
            if not np.isnan(data[key][track][starting_point]['RMS[%]']):
                res['RMS[%]'].append(data[key][track][starting_point]['RMS[%]'])

        points = data[key][track]['Points']
        starts = [int(x) for x in data[key][track].keys() if x.isdigit()]
        starts.sort()
        ends = [x for x in starts[1:] + [points]]
        res['Points'] = np.concatenate((res['Points'], np.array(ends) - np.array(starts)))

    return res


def read_results_dir(results_dir, *elements):
    res = {element: [] for element in elements}
    res['RMS[%]'] = []
    res['Points'] = np.array([])

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)) or 'results' not in directory:
            continue

        results_file = os.path.join(results_dir, directory, directory.split('_')[-1] + '.txt')
        res_temp = read_results_file(results_file, *elements)

        for element in elements:
            res[element] += res_temp[element]
        res['RMS[%]'] += res_temp['RMS[%]']
        res['Points'] = np.concatenate((res['Points'], res_temp['Points']))

    return res


def read_specific_lc(obj_number, track_number, starting_point):
    t = Time.now().mjd
    tracks_file = os.path.join(mmt_directory, f'results_{obj_number}', f'{obj_number}_tracks.txt')
    filter_file = os.path.join(mmt_directory, f'results_{obj_number}', 'Filter')
    reading_track, counter, time_array, mag_array, dist_array = 0, 0, [], [], []

    with open(filter_file) as file:
        file.readline()

        for line in file:
            _, track, sp = line.strip().split()

            if track == track_number and sp == starting_point:
                temp = file.readline()
                if temp is None or len(temp) == 0:
                    ending_point = 999999
                else:
                    _, track, sp = temp.strip().split()
                    ending_point = int(sp)
                    ending_point = 999999 if track != track_number else ending_point
                break

    with open(tracks_file) as file:
        for line in file:
            if line.startswith('#'):
                continue

            # if len(line.strip()) == 0:
            #     reading_track = 2
            # else:
            date, time, _, mag, _, _, d, _, _, track = line.strip().split()

            if reading_track == 0:
                if track == track_number:
                    reading_track = 1

            elif reading_track == 1:
                if track != track_number or counter >= ending_point:
                    reading_track = 2
                elif counter >= int(starting_point):
                    time_array.append(Time(f'{date}T{time}', format='isot'))
                    mag_array.append(float(mag))
                    dist_array.append(float(d))
                counter += 1

            elif reading_track == 2:
                print('read time: ', (Time.now().mjd - t) * day_seconds)
                return LC2(time_array, mag_array, dist_array, obj_number, track_number, starting_point)


def read_annotation(annotation_file):
    res = []
    with open(annotation_file) as file:
        data = json.load(file)

    for o in data:
        for t in data[o]:
            for s in data[o][t]:
                res.append(read_specific_lc(o, t, s))
                res[-1].set_values(glints=data[o][t][s]['glint'], period=data[o][t][s]['P[s]'])

    return res


def read_annotation_meta(annotation_file):
    res = []
    with open(annotation_file) as file:
        data = json.load(file)

    for o in data:
        for t in data[o]:
            for s in data[o][t]:
                res.append((o, t, s, data[o][t][s]['glint'], data[o][t][s]['P[s]']))

    return res


def read_filter_meta(filter_file):
    res = []
    with open(filter_file) as file:
        file.readline()

        for line in file:
            o, t, s = line.strip().split()
            res.append((o, t, s, None, None))


def read_filter_meta_dir(mmt_dir):
    res = []
    for directory in os.listdir(mmt_dir):
        if not os.path.isdir(os.path.join(mmt_dir, directory)) or 'results' not in directory:
            continue

        filter_file = os.path.join(mmt_dir, directory, 'Filter')
        res += read_filter_meta(filter_file)


def get_header(tracks_file):
    """get header of track file"""
    skip_lines = 7
    res = []

    with open(tracks_file) as file:
        for i in range(skip_lines):
            res.append(file.readline().strip())

    return res


def read_light_curve(light_curve_path, binary=True):
    """read light curve from file"""
    reading_data = False
    phase = []
    mag = []
    glint = []
    label = None

    with open(light_curve_path, encoding="utf8") as file:
        for line in file:
            if reading_data:
                x, y, z = line.strip().split()
                phase.append(float(x))
                mag.append(float(y))
                glint.append(float(z))
            elif line.startswith('#'):
                reading_data = True
            elif line.startswith('Glint:'):
                label = int(line.strip().split()[1])
                if binary:
                    label = 1 if label > 0 else 0

    return np.array(phase), np.array(mag), np.array(glint), label


def load_dataset(dataset_path, mode=4):
    """load dataset and extract features"""
    if mode == 0:
        return load_dataset2(dataset_path, 18)

    files = os.listdir(dataset_path)
    files = [f for f in files if f.endswith('.txt')]
    X_ = np.zeros((len(files), 300))
    y_ = np.zeros(len(files))

    for index, filename in enumerate(files):
        phase, mag, glint, label = read_light_curve(os.path.join(dataset_path, filename))
        norm_mag = normalize(mag)
        f1 = fourier(phase, mag, 8)
        f2 = fourier(phase, mag, 18)
        der1 = np.gradient(f2, phase[1] - phase[0])
        der2 = np.gradient(der1, phase[1] - phase[0])

        if mode == 1:
            temp = mag
        elif mode == 2:
            temp = norm_mag
        elif mode == 3:
            temp = mag - f1
        elif mode == 4:
            temp = f1 - f2
        elif mode == 5:
            temp = der1

        X_[index] = temp
        y_[index] = label

    p = np.random.permutation(len(X_))
    X_ = X_[p]
    y_ = y_[p]

    return X_, y_


def load_dataset2(dataset_path, n=20):
    """load dataset and extract features as Fourier coefficients"""
    files = os.listdir(dataset_path)
    files = [f for f in files if f.endswith('.txt')]
    X_ = np.zeros((len(files), 2*n))
    y_ = np.zeros(len(files))

    for index, filename in enumerate(files):
        phase, mag, glint, label = read_light_curve(os.path.join(dataset_path, filename))
        fc = fourier_coefficients(phase, mag, n)

        X_[index] = fc[1:]
        y_[index] = label

    p = np.random.permutation(len(X_))
    X_ = X_[p]
    y_ = y_[p]

    return X_, y_


def load_dataset3(dataset_path):
    """load dataset and extract features"""
    files = os.listdir(dataset_path)
    files = [f for f in files if f.endswith('.txt')]
    X_ = np.zeros((len(files), 300))
    y_ = np.zeros((len(files), 300))

    for index, filename in enumerate(files):
        phase, mag, glint, label = read_light_curve(os.path.join(dataset_path, filename))

        X_[index] = normalize(mag)
        y_[index] = glint < -0.01

    p = np.random.permutation(len(X_))
    X_ = X_[p]
    y_ = y_[p]

    return X_, y_


def save_model(model, model_path):
    """save ml model into pickle file"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """load ml model from file"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def use_model(lc_path, model_path):
    """test ml model on given light curve"""
    phase, mag, glint, label = read_light_curve(lc_path)
    f8 = fourier(phase, mag, 8)
    f15 = fourier(phase, mag, 15)
    features = np.abs(f15 - f8)
    features.resize((1, len(features)))

    rf_c = load_model(model_path)
    res = rf_c.predict(features)

    return res[0], label


def use_model_dir(dir_with_lc, model_path):
    """test ml model on all light curves in given directory"""
    wrong0 = 0
    wrong1 = 0
    good = 0
    files = [x for x in os.listdir(dir_with_lc) if x.endswith('.txt')]
    for file in files:
        pred, act = use_model(os.path.join(dir_with_lc, file), model_path)
        if pred - act == 1:
            wrong1 += 1
            print(f'predicted value: {pred}, actual value: {act}   {file}')
        elif pred - act == -1:
            wrong0 += 1
            print(f'predicted value: {pred}, actual value: {act}   {file}')
        elif pred == act:
            good += 1

    print(f'wrong 0: {wrong0}  wrong 1: {wrong1} good: {good}')
