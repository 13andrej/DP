import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main(results_file, tracks_file):
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
        last_track, counter, start_point, start_points, res = None, 0, 0, [], []

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            split_entry = line.strip().split()
            mag, track = split_entry[3], split_entry[9]

            if last_track != track:
                last_track = track
                counter = 0
                if track in data[key].keys():
                    start_points = sorted([int(x) for x in data[key][track].keys() if x.isdigit()])

            if len(start_points) > 0 and counter == start_points[0]:
                if len(res) > 0:
                    plt.clf()
                    plt.plot(res)
                    plt.savefig(os.path.join(save_plots, key + '_' + track + '_' + str(start_point) + '.png'))
                res = []
                start_point = start_points.pop(0)

            res.append(float(mag))
            counter += 1


def main3(results_file):
    res = []

    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

    for track in data[key].keys():
        for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
            res.append(data[key][track][starting_point]['Fourier']['a1'])

    histogram, bin_edges = np.histogram(res)
    plt.hist(histogram, edgecolor='k')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution Histogram')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # main(r'C:\Users\13and\Desktop\idk\Filter', None)
    # main2(r'C:\Users\13and\Desktop\idk\43.txt', r'C:\Users\13and\Desktop\idk\43_tracks.txt',
    #      r'C:\Users\13and\PycharmProjects\diplomovka\data\mmt')
    main3(r'C:\Users\13and\Desktop\idk\43.txt')


