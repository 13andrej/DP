import json
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from src.utils.constants import coefficients
from src.utils.functions import our_fourier8, get_reduced_mag_np


def save_image(time_array, mag_array, period, save_path=None, results_record=None):
    plt.clf()
    plt.gca().invert_yaxis()

    x = np.array([(((i - time_array[0]).value * 86400) % period) / period for i in time_array])

    if results_record is not None:
        y = our_fourier8(x, *[results_record['Fourier'][c] for c in coefficients])
        rms = np.sqrt(np.mean(np.square(mag_array - y)))
        plt.plot(x, y, 'red', label='Fit')
        plt.title(f'RMS = {rms:.2%}')

    plt.scatter(x, mag_array, label='Data')
    plt.xlabel('phase')
    plt.ylabel('magnitude')

    if save_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)


def save_export(time_array, mag_array, period, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    x = np.array([(((i - time_array[0]).value * 86400) % period) / period for i in time_array])

    with open(save_path, 'w') as file:
        file.write(f'RMS: \n')
        file.write(f'Number of points: {len(mag_array)}\n')
        file.write(f'Glint: False\n')
        file.write('Glint position: None\n')
        for c in coefficients:
            file.write(f'{c}: \n')
        file.write('Phase\tMag\tMagErr\n')
        file.write(f'#{65 * "="}\n')
        for i in range(len(x)):
            file.write(f'{x[i]}\t{mag_array[i]}\t0.0\n')


def process_track(results_file, tracks_file, output_directory):
    with open(results_file) as file:
        data = json.load(file)

    key = list(data.keys())[0]

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

                    save_path_image = os.path.join(output_directory, 'images', f'{key}_{track}_{str(start_point)}.png')
                    save_path_export = os.path.join(output_directory, 'exports',  f'{key}_{track}_{str(start_point)}.txt')
                    save_image(time_array, reduced_mag, period, save_path_image, data[key][track].get(str(start_point)))
                    save_export(time_array, reduced_mag, period, save_path_export)

                time_array = []
                mag_array = []
                dist_array = []
                start_point = start_points.pop(0)

            time_array.append(Time(f'{date}T{time}', format='isot'))
            mag_array.append(float(mag))
            dist_array.append(float(d))
            counter += 1


def plot_one_track(results_file, tracks_file, output_directory, track_info, display=False):
    track_number, starting_point = track_info.split('_')

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
                period = data[key][track]['P[s]']
                # period = (time_array[-1] - time_array[0]).value * 86400 + 0.001
                reduced_mag = get_reduced_mag_np(mag_array, dist_array)
                reduced_mag[-1] = reduced_mag[-2]
                m = np.argmin(reduced_mag)
                reduced_mag = np.concatenate((reduced_mag[m:], reduced_mag[:m]))

                save_path_image = os.path.join(output_directory, 'images', f'{key}_{track}_{str(starting_point)}.png')
                save_path_export = os.path.join(output_directory, 'exports', f'{key}_{track}_{str(starting_point)}.txt')
                save_image(time_array, mag_array, period, None, data[key][track].get(str(starting_point)))
                save_export(time_array, reduced_mag, period, save_path_export)
                return


def fold_track(tracks_file, filter_file, track_number):
    with open(filter_file) as file:
        skip_lines = 1
        reading_track, starting_points_array = 0, []

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            split_entry = line.strip().split()
            _, track, starting_point = split_entry

            if reading_track == 0:
                if track == track_number:
                    reading_track = 1
            elif reading_track == 1:
                if track != track_number:
                    reading_track = 2
                else:
                    starting_points_array.append(int(starting_point))
            elif reading_track == 2:
                break

    with open(tracks_file) as file:
        skip_lines = 7
        reading_track, time_array, mag_array, dist_array = 0, [], [], []

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
                if track != track_number:
                    reading_track = 2
                else:
                    time_array.append(Time(f'{date}T{time}', format='isot'))
                    mag_array.append(float(mag))
                    dist_array.append(float(d))
            elif reading_track == 2:
                break

    time_array_seconds = np.array([(x.mjd - time_array[0].mjd) * (24 * 60 * 60) for x in time_array])
    starting_points_array += [len(mag_array)]

    for i in range(len(starting_points_array)-1):
        start, end = starting_points_array[i], starting_points_array[i+1]
        t_temp = time_array_seconds[start: end] - time_array_seconds[start]
        plt.scatter(t_temp, mag_array[start:end])
        print(len(t_temp))
        break

    plt.show()


if __name__ == '__main__':
    res_file = r'C:\Users\13and\PycharmProjects\DP\data\mmt\results_13\13.txt'
    tra_file = r'C:\Users\13and\PycharmProjects\DP\data\mmt\results_13\13_tracks.txt'
    fil_file = r'C:\Users\13and\PycharmProjects\DP\data\mmt\results_13\Filter'
    out_dir = r'C:\Users\13and\PycharmProjects\DP\data\mmt\results_13'

    # process_track(res_file, tra_file, out_dir)
    fold_track(tra_file, fil_file, '1926340')
