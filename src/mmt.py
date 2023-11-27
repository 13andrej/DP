import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from functions_v03 import our_fourier8, get_reduced_mag_np


coefficients = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']


def save_plot(time_array, mag_array, period, fd, save_path):
    plt.clf()
    x = np.array([(((i - time_array[0]).value * 86400) % period) / period for i in time_array])

    if fd is not None:
        fd = fd['Fourier']
        y = our_fourier8(x, *[fd[c] for c in coefficients])
        residuals = mag_array - y
        rms = np.sqrt(np.mean(np.square(residuals)))

    else:
        return

    plt.scatter(x, mag_array, label='Data')
    plt.plot(x, y, 'red', label='Fit')
    plt.title(f'RMS = {rms:.2%}')
    plt.xlabel('phase')
    plt.ylabel('magnitude')
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


def compute_number_of_points_histogram(results_path, save_dir=None):
    res = np.array([])
    with open(results_path) as file:
        d = json.load(file)

    d = d[list(d.keys())[0]]

    for k, v in d.items():
        points = v['Points']
        starts = [int(x) for x in v.keys() if x.isdigit()]
        starts.sort()
        ends = [x for x in starts[1:] + [points]]
        res = np.concatenate((res, np.array(ends) - np.array(starts)))

    res = res[res < 1500]
    n, bins, _ = plt.hist(res, bins=200)
    plt.xlim(0, 1500)
    plt.xlabel('No. Points')
    plt.ylabel('Frequency')
    plt.title(f'Distribution Histogram for number of points')
    plt.grid(True)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'Points.png'))
        with open(os.path.join(save_dir, 'Points.pkl'), 'wb') as f:
            pickle.dump([n, bins], f)

    plt.show()


def compute_fourier_element_histogram(results_path, fourier_element, save_dir=None):
    min_threshold, max_threshold = -2, 2
    res = load_fourier_data_dir(results_path, fourier_element)[0] if os.path.isdir(results_path) \
        else load_fourier_data_file(results_path, fourier_element)[0]
    original_length = len(res)
    res = [x for x in res if min_threshold < x < max_threshold]
    print(f'{len(res) / original_length:.0%} ({len(res)} / {original_length})')
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


def generate_light_curve(pkl_dir, points=300, glint=False, save_dir=None, verbose=False):
    pickles = {x: None for x in coefficients}

    for c in pickles.keys():
        with open(os.path.join(pkl_dir, f'{c}.pkl'), 'rb') as f:
            pickles[c] = pickle.load(f)

    with open(os.path.join(pkl_dir, 'RMS.pkl'), 'rb') as f:
        rms_n, rms_bins = pickle.load(f)

    random_values = {c: np.random.choice(bins[:-1], p=n / n.sum()) for c, (n, bins) in pickles.items()}
    rms = np.random.choice(rms_bins[:-1], p=rms_n / rms_n.sum()) / 100
    x = np.arange(0, 1.00001, 1/(points - 1))
    y = our_fourier8(x, *[random_values[c] for c in coefficients])
    y2 = add_noise(y, rms)
    if glint is True:
        y2, center = add_delta(x, y2)
    else:
        center = None

    plt.ylabel('Magnitude')
    plt.xlabel('Phase')
    plt.scatter(x, y2, label='Data')
    plt.plot(x, y, 'red', label='Fit')
    plt.legend()

    if save_dir is not None:
        name = len(os.listdir(os.path.join(save_dir, 'exports')))
        plt.savefig(os.path.join(save_dir, 'images', f'{name}.png'))
        with open(os.path.join(save_dir, 'exports', f'{name}.txt'), 'w') as file:
            file.write(f'RMS: {rms:.3%}\n')
            file.write(f'Number of points: {points}\n')
            file.write(f'Glint: {glint}\n')
            file.write(f'Glint position: {center:.3}\n')
            for c in coefficients:
                file.write(f'{c}: {random_values[c]:.3}\n')
            file.write('Phase\tMag\tMagErr\n')
            file.write(f'#{65*"="}\n')
            for i in range(len(x)):
                file.write(f'{x[i]}\t{y2[i]}\t{y2[i]-y[i]}\n')

    if verbose:
        plt.show()
        plt.plot(x, y2)
        plt.plot([center, center], [min(y2), max(y2)])
        plt.show()
        print(f'RMS: {rms:.2%}')
        print(f'number of points: {len(y2)}')
        print(f'glint position: {center}')


def add_lorentz(x, y):
    amplitude = 1.0
    center = np.random.rand()
    width = 0.01

    lorentzian = amplitude / (1 + ((x - center) / width) ** 2)
    return y + lorentzian, center


def add_delta(x, y):
    sigma = 0.5
    center = np.random.rand()
    width = 0.01
    delta_approx = np.exp(-((x - center) / (sigma * width))**2) / (sigma * np.sqrt(np.pi))

    return y + delta_approx, center


def add_noise(y, desired_rms):
    noise = np.random.normal(0, 1, len(y))
    rms = np.sqrt(np.mean(np.square(noise)))
    scaled_noise = noise * (desired_rms / rms)

    return y + scaled_noise


# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # plot_one_track(r'C:\Users\13and\PycharmProjects\DP\data\43.txt',
    #                r'C:\Users\13and\PycharmProjects\DP\data\43_tracks.txt',
    #                r'C:\Users\13and\PycharmProjects\DP\data\mmt')
    # compute_fourier_element_histogram(r'C:\Users\13and\PycharmProjects\diplomovka\data\43.txt', 'a1')
    # compare_two_fourier_elements(r'C:\Users\13and\PycharmProjects\diplomovka\data\43.txt', 'a6', 'b6')
    # generate_light_curve(r'C:\Users\13and\PycharmProjects\DP\data\fourier', points=300, glint=True,
    #                      save_dir=r'C:\Users\13and\PycharmProjects\DP\data\dataset', verbose=True)
    generate_light_curve(r'C:\Users\13and\PycharmProjects\DP\data\fourier', points=300, glint=True,
                         save_dir=None, verbose=True)
