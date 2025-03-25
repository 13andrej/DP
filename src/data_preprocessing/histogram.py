import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import coefficients, thresholds


def load_fourier_data_file(results_file, *elements):
    res = {element: [] for element in elements}

    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    for track in data[key].keys():
        for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
            for element in elements:
                res[element].append(data[key][track][starting_point]['Fourier'][element])

    return res


def load_fourier_data_dir(results_dir, *elements):
    res = {element: [] for element in elements}

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)) or 'results' not in directory:
            continue

        results_file = os.path.join(results_dir, directory, directory.split('_')[-1] + '.txt')

        with open(results_file) as file:
            data = json.load(file)

        key = list(data.keys())[0]

        for track in data[key].keys():
            for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
                for element in elements:
                    res[element].append(data[key][track][starting_point]['Fourier'][element])

    return res


def compute_fourier_element_histogram(results_path, fourier_elements, output_directory=None, display=False):
    res = load_fourier_data_dir(results_path, *fourier_elements) if os.path.isdir(results_path) \
        else load_fourier_data_file(results_path, *fourier_elements)

    for fourier_element in fourier_elements:
        min_threshold, max_threshold = thresholds[fourier_element]
        original_length = len(res[fourier_element])
        res_f = [x for x in res[fourier_element] if min_threshold < x < max_threshold]
        print(f'{fourier_element} {len(res_f) / original_length:.0%} ({len(res_f)} / {original_length})')
        plt.clf()
        n, bins, _ = plt.hist(res_f, bins=200)  # , weights=np.ones(len(res_f)) / len(res_f) * 100)

        plt.xlim(min_threshold, max_threshold)
        plt.ylim(0, 50000)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution Histogram {fourier_element}')
        plt.grid(True)

        if output_directory is not None:
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(os.path.join(output_directory, f'{fourier_element}.png'))
            with open(os.path.join(output_directory, f'{fourier_element}.pkl'), 'wb') as f:
                pickle.dump([n, bins], f)

        if display:
            plt.show()


def compute_rms_histogram(results_path, output_directory=None, display=False):
    rms = []
    for directory in os.listdir(results_path):
        if not os.path.isdir(os.path.join(results_path, directory)) or 'results' not in directory:
            continue

        results_file = os.path.join(results_path, directory, directory.split('_')[-1] + '.txt')

        with open(results_file) as file:
            data = json.load(file)
        key = list(data.keys())[0]
        for track in data[key].keys():
            for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
                if not np.isnan(data[key][track][starting_point]['RMS[%]']):
                    rms.append(data[key][track][starting_point]['RMS[%]'])

    original_length = len(rms)
    rms = [x for x in rms if 0 <= x < 100]
    print(f'RMS {len(rms) / original_length: .0%}, ({len(rms)} / {original_length})')
    n, bins, _ = plt.hist(rms, bins=200)  # ,  weights=np.ones(len(rms)) / len(rms) * 100)
    plt.xlabel('Value [%]')
    plt.ylabel('Frequency')
    plt.title(f'Distribution Histogram RMS')
    plt.grid(True)

    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(os.path.join(output_directory, f'RMS.png'))
        with open(os.path.join(output_directory, f'RMS.pkl'), 'wb') as f:
            pickle.dump([n, bins], f)

    if display:
        plt.show()


def compute_number_of_points_histogram(results_path, output_directory=None, display=False):
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

    if output_directory is not None:
        plt.savefig(os.path.join(output_directory, 'Points.png'))
        with open(os.path.join(output_directory, 'Points.pkl'), 'wb') as f:
            pickle.dump([n, bins], f)

    if display:
        plt.show()


def compare_two_fourier_elements(results_path, e1, e2):
    min_threshold, max_threshold = -2, 2
    res1, res2 = load_fourier_data_dir(results_path, e1, e2).values() if os.path.isdir(results_path) \
        else load_fourier_data_file(results_path, e1, e2).values()

    plt.xlim(min_threshold, max_threshold)
    plt.ylim(min_threshold, max_threshold)
    plt.scatter(res1, res2)
    plt.xlabel(e1)
    plt.ylabel(e2)
    plt.title(f'{e1} compared to {e2}')
    plt.grid(True)
    plt.show()

    print(np.corrcoef(res1, res2))


if __name__ == '__main__':
    results_p = r'C:\Users\13and\PycharmProjects\DP\data\mmt'
    out_dir = r'C:\Users\13and\PycharmProjects\DP2\data\fourier'
    compute_fourier_element_histogram(results_p, coefficients, out_dir)
    compute_rms_histogram(results_p, out_dir)
