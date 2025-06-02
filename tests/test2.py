import cmath
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.time import Time

from src.utils.constants import mmt_directory, coefficients
from src.utils.functions import fourier, fourier8_from_coefficients, get_x_y_reduced


def plot_one_track(obj_number, track_number, starting_point):
    results_file = os.path.join(mmt_directory, f'results_{obj_number}', f'{obj_number}.txt')
    tracks_file = os.path.join(mmt_directory, f'results_{obj_number}', f'{obj_number}_tracks.txt')

    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]
    starting_points_list = sorted([int(x) for x in data[key][track_number].keys() if x.isdigit()]) + [999999]
    starting_point = int(starting_point)
    ending_point = starting_points_list[starting_points_list.index(starting_point) + 1]

    with open(tracks_file) as file:
        skip_lines = 7
        reading_track, counter, time_array, mag_array, dist_array = 0, 0, [], [], []

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            split_entry = line.strip().split()
            date, time, mag, d, track = split_entry[0], split_entry[1], split_entry[3], split_entry[6], split_entry[9]

            if reading_track == 0:
                if track == track_number:
                    reading_track = 1

            elif reading_track == 1:
                if track != track_number or counter >= ending_point:
                    reading_track = 2
                elif counter >= starting_point:
                    time_array.append(Time(f'{date}T{time}', format='isot'))
                    mag_array.append(float(mag))
                    dist_array.append(float(d))
                counter += 1

            elif reading_track == 2:
                period1 = data[key][track]['P[s]']
                period2 = (time_array[-1] - time_array[0]).value * 86400
                print(period1, period2)

                # reduced_mag = get_x_y_reduced(time_array, mag_array, dist_array)
                # reduced_mag[-1] = reduced_mag[-2]
                # m = np.argmin(reduced_mag)
                # reduced_mag = np.concatenate((reduced_mag[m:], reduced_mag[:m]))

                plt.clf()
                plt.gca().invert_yaxis()

                x = np.array([(((i - time_array[0]).value * 86400) % period1) / period1 for i in time_array])
                # y = our_fourier8(x, *[data[key][track][str(starting_point)]['Fourier'][c] for c in coefficients])
                y = fourier(x, mag_array, 8)
                y2 = fourier(x, mag_array, 18)
                # y = fourier_test(x, mag_array, 8)

                sorted_indices = np.argsort(x)
                mag_array = np.array(mag_array)
                x = x[sorted_indices]
                y = y[sorted_indices]
                y2 = y2[sorted_indices]

                rms = np.sqrt(np.mean(np.square(mag_array - y)))
                # plt.title(f'RMS = {rms:.2%}')
                plt.plot(x, y, label='Fourier8', color='lime')
                plt.plot(x, y2, label='Fourier18', color='red')
                plt.scatter(x, mag_array, label='Data')
                # plt.plot(x[10:-9], moving_average(mag_array, 20), color='red')
                plt.xlabel('phase')
                plt.ylabel('magnitude')
                plt.legend()
                plt.show()

                # plt.gca().invert_yaxis()
                plt.scatter(x, y - y2)
                plt.show()

                # plt.scatter(x, np.gradient(y2, x[1]-x[0]))
                # plt.show()
                #
                # plt.scatter(x, np.abs(y - y2), label='rms')
                # plt.show()
                return


if __name__ == '__main__':
    with open(r'C:\Users\13and\PycharmProjects\DP\data\annotation.json') as file:
        data = json.load(file)

    oo = list(data.keys())
    random.shuffle(oo)

    for o in oo:
        for t in data[o]:
            for s in data[o][t]:
                if data[o][t][s]['glint'] > 0:
                    plot_one_track(o, t, s)
                    input('Press [enter] to continue.')
            break
