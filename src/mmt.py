import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from functions_v03 import our_fourier8, get_reduced_mag_np


def save_plot(time_array, mag_array, period, fd, save_path):
    plt.clf()
    # x = np.arange(0, 1.00001, 1/len(data))
    x = np.array([(((i - time_array[0]).value * 86400) % period) / period for i in time_array])

    if fd is not None:
        fd = fd['Fourier']
        y = our_fourier8(x, fd['a0'], fd['a1'], fd['a2'], fd['a3'], fd['a4'], fd['a5'], fd['a6'], fd['a7'], fd['a8'],
                         fd['b1'], fd['b2'], fd['b3'], fd['b4'], fd['b5'], fd['b6'], fd['b7'], fd['b8'])
    else:
        return
    plt.plot(x, mag_array)
    plt.plot(x, y)
    plt.savefig(save_path)


def iterate_through_one_track(results_file):
    res = {}

    with open(results_file) as file:
        first_line = True

        for line in file:
            if first_line:
                first_line = False
                continue

            obj, track, start = line.strip().split()

            if obj not in res:
                res[obj] = {}
            res[obj][track] = int(start)


def plot_one_track(results_file, tracks_file, save_plots):
    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    # for track in sorted([int(x) for x in data[key].keys()]):
    #    for starting_point in sorted([x for x in data[key][str(track)].keys() if not x.isdigit()]):
    #        print(starting_point)

    with open(tracks_file) as file:
        skip_lines = 7
        last_track, counter, start_point, start_points, time_array, mag_array, dist_array = None, 0, 0, [], [], [], []

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            split_entry = line.strip().split()
            date, time, mag, d, track = split_entry[0], split_entry[1], split_entry[3], split_entry[6], split_entry[9]

            if last_track != track:
                last_track = track
                counter = 0
                if track in data[key].keys():
                    start_points = sorted([int(x) for x in data[key][track].keys() if x.isdigit()])

            if len(start_points) > 0 and counter == start_points[0]:
                if len(time_array) > 0:
                    period = data[key][track]['P[s]']
                    reduced_mag = get_reduced_mag_np(mag_array, dist_array)
                    reduced_mag[-1] = reduced_mag[-2]
                    m = np.argmin(reduced_mag)
                    reduced_mag = np.concatenate((reduced_mag[m:], reduced_mag[:m]))

                    save_path = os.path.join(save_plots, key + '_' + track + '_' + str(start_point) + '.png')
                    save_plot(time_array, reduced_mag, period, data[key][track].get(str(start_point)), save_path)

                time_array = []
                mag_array = []
                dist_array = []
                start_point = start_points.pop(0)

            time_array.append(Time(f'{date}T{time}', format='isot'))
            mag_array.append(float(mag))
            dist_array.append(float(d))
            counter += 1


# ----------------------------------------------------------------------------------------------------------------------


def load_fourier_data_file(results_file, *elements):
    res = {element: [] for element in elements}

    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    for track in data[key].keys():
        for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
            for element in elements:
                res[element].append(data[key][track][starting_point]['Fourier'][element])

    return [res[element] for element in elements]


def load_fourier_data_dir(results_dir, *elements):
    res = {element: [] for element in elements}

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)):
            continue

        results_file = os.path.join(results_dir, directory, directory.split('_')[-1] + '.txt')

        with open(results_file) as file:
            data = json.load(file)

        key = list(data.keys())[0]

        for track in data[key].keys():
            for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
                for element in elements:
                    res[element].append(data[key][track][starting_point]['Fourier'][element])

    return [res[element] for element in elements]


# ----------------------------------------------------------------------------------------------------------------------


def compute_fourier_element_histogram(results_path, fourier_element, save_dir=None):
    min_threshold, max_threshold = -2, 2
    res = load_fourier_data_dir(results_path, fourier_element)[0] if os.path.isdir(results_path) \
        else load_fourier_data_file(results_path, fourier_element)[0]
    original_length = len(res)
    res = [x for x in res if min_threshold < x < max_threshold]
    print(f'{len(res) / original_length:.0%}')
    n, bins, _ = plt.hist(res, bins=200)

    plt.xlim(min_threshold, max_threshold)
    plt.ylim(0, 1000)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution Histogram {fourier_element}')
    plt.grid(True)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f'{fourier_element}.png'))
        with open(os.path.join(save_dir, f'{fourier_element}.pkl'), 'wb') as f:
            pickle.dump([n, bins], f)

    plt.show()


def compare_two_fourier_elements(results_path, e1, e2):
    min_threshold, max_threshold = -2, 2
    res1, res2 = load_fourier_data_dir(results_path, e1, e2) if os.path.isdir(results_path) \
        else load_fourier_data_file(results_path, e1, e2)

    plt.xlim(min_threshold, max_threshold)
    plt.ylim(min_threshold, max_threshold)
    plt.scatter(res1, res2)
    plt.xlabel(e1)
    plt.ylabel(e2)
    plt.title(f'{e1} compared to {e2}')
    plt.grid(True)
    plt.show()

    print(np.corrcoef(res1, res2))


# ----------------------------------------------------------------------------------------------------------------------


def load_pkl(pkl_dir):
    coefficients = {'a0': 0, 'a1': 0, 'a2': 0, 'a3': 0, 'a4': 0, 'a5': 0, 'a6': 0, 'a7': 0, 'a8': 0,
                    'b1': 0, 'b2': 0, 'b3': 0, 'b4': 0, 'b5': 0, 'b6': 0, 'b7': 0, 'b8': 0}

    for c in coefficients.keys():
        with open(os.path.join(pkl_dir, f'{c}.pkl'), 'rb') as f:
            coefficients[c] = pickle.load(f)

    x = np.arange(0, 1.00001, 1/500)
    fd = {c: np.random.choice(bins[:-1], p=n / n.sum()) for c, (n, bins) in coefficients.items()}
    y = our_fourier8(x, fd['a0'], fd['a1'], fd['a2'], fd['a3'], fd['a4'], fd['a5'], fd['a6'], fd['a7'], fd['a8'],
                     fd['b1'], fd['b2'], fd['b3'], fd['b4'], fd['b5'], fd['b6'], fd['b7'], fd['b8'])
    y = add_noise(x, y)
    # y2 = add_lorentz(x, y)
    y2 = add_delta(x, y)

    plt.ylabel('Magnitude')
    plt.xlabel('Phase')
    # plt.plot(x, y, linewidth=5, alpha=0.8, label='lc')
    plt.plot(x, y2, label='lc with lorentz')
    plt.show()

    # y2 = add_lorentz(x, y)
    #
    # plt.ylabel('Magnitude')
    # plt.xlabel('Phase')
    # plt.plot(x, y, linewidth=5, alpha=0.8, label='lc')
    # plt.plot(x, y2, label='lc with lorentz')
    # plt.show()


def add_lorentz(x, y):
    amplitude = 1.0
    center = 0.5  # np.random.rand()
    width = 0.01

    lorentzian = amplitude / (1 + ((x - center) / width) ** 2)
    return y + lorentzian


def add_delta(x, y):
    sigma = 0.25
    center = np.random.rand()
    width = 0.1
    delta_approx = np.exp(-((x - center) / (sigma * width**2))**2) / (sigma * np.sqrt(np.pi))

    return y + delta_approx


def add_noise(x, y):
    noise_amplitude = 0.25
    noise = np.random.normal(0, noise_amplitude, len(y))

    return y + noise


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # plot_one_track(r'C:\Users\13and\PycharmProjects\diplomovka\data\43.txt',
    #                r'C:\Users\13and\PycharmProjects\diplomovka\data\43_tracks.txt',
    #                r'C:\Users\13and\PycharmProjects\diplomovka\data\mmt2')
    # compute_fourier_element_histogram(r'C:\Users\13and\PycharmProjects\diplomovka\data\43.txt', 'a1')
    # compare_two_fourier_elements(r'C:\Users\13and\PycharmProjects\diplomovka\data\43.txt', 'a6', 'b6')
    load_pkl(r'C:\Users\13and\PycharmProjects\diplomovka\data\fourier')

