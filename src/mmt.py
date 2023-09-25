import json
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from functions_v03 import our_fourier8


def plot_results(time_array, mag_array, period, fd, save_path):
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


def main(results_file):
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


def main2(results_file, tracks_file, save_plots):
    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    # for track in sorted([int(x) for x in data[key].keys()]):
    #    for starting_point in sorted([x for x in data[key][str(track)].keys() if not x.isdigit()]):
    #        print(starting_point)

    with open(tracks_file) as file:
        skip_lines = 7
        last_track, counter, start_point, start_points, time_array, mag_array = None, 0, 0, [], [], []

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            split_entry = line.strip().split()
            date, time, mag, track = split_entry[0], split_entry[1], split_entry[3], split_entry[9]

            if last_track != track:
                last_track = track
                counter = 0
                if track in data[key].keys():
                    start_points = sorted([int(x) for x in data[key][track].keys() if x.isdigit()])

            if len(start_points) > 0 and counter == start_points[0]:
                if len(time_array) > 0:
                    period = data[key][track]['P[s]']
                    save_path = os.path.join(save_plots, key + '_' + track + '_' + str(start_point) + '.png')
                    plot_results(time_array, mag_array, period, data[key][track].get(str(start_point)), save_path)

                time_array = []
                mag_array = []
                start_point = start_points.pop(0)

            time_array.append(Time(f'{date}T{time}', format='isot'))
            mag_array.append(float(mag))
            counter += 1


def main3(results_file, fourier_element):
    res = []

    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    for track in data[key].keys():
        for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
            res.append(data[key][track][starting_point]['Fourier'][fourier_element])

    original_length = len(res)
    res = [x for x in res if -30 < x < 30]
    print(len(res) / original_length)
    # n, bins, patches = plt.hist(sorted(res)[55:-80], bins=100)
    n, bins, patches = plt.hist(res, bins=100)
    # print(n)
    # print(bins)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution Histogram {fourier_element}')
    plt.grid(True)
    plt.show()


def main4(results_dir, fourier_element):
    res = []

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)):
            continue

        # content = os.listdir(os.path.join(results_dir, directory))
        results_file = os.path.join(results_dir, directory, directory.split('_')[-1] + '.txt')

        with open(results_file) as file:
            data = json.load(file)

        key = list(data.keys())[0]

        for track in data[key].keys():
            for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
                res.append(data[key][track][starting_point]['Fourier'][fourier_element])

    original_length = len(res)
    res = [x for x in res if -30 < x < 30]
    print(len(res) / original_length)
    n, bins, patches = plt.hist(res, bins=100)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution Histogram {fourier_element}')
    plt.grid(True)
    # plt.savefig('')
    plt.show()


if __name__ == '__main__':
    # main(r'C:\Users\13and\Desktop\idk\Filter')
    # main2(r'C:\Users\13and\PycharmProjects\DP\data\43.txt', r'C:\Users\13and\PycharmProjects\DP\data\43_tracks.txt',
    #      r'C:\Users\13and\PycharmProjects\DP\data\mmt2')
    main3(r'C:\Users\13and\PycharmProjects\DP\data\43.txt', 'a0')
