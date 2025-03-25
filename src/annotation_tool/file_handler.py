import os.path

from astropy.time import Time
from classes import LC


def process_track(tracks_file, filter_file):
    t = Time.now().mjd

    with open(filter_file) as file:
        skip_lines = 1
        starting_points = {}

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            _, track, starting_point = line.strip().split()

            if track not in starting_points:
                starting_points[track] = []
            starting_points[track].append(int(starting_point))

    with open(tracks_file) as file:
        skip_lines = 7
        time_array, mag_array = {}, {}

        for line in file:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            date, time, _, mag, _, _, d, _, _, track = line.strip().split()

            if track not in time_array:
                time_array[track] = []
                mag_array[track] = []

            time_array[track].append(Time(f'{date}T{time}', format='isot'))
            mag_array[track].append(float(mag))

    print('read time: ', (Time.now().mjd - t)*86400)
    t = Time.now().mjd
    res = []

    for track in starting_points:
        starting_points[track].append(9999999)
        for i in range(len(starting_points[track]) - 1):
            a, b = starting_points[track][i], starting_points[track][i + 1]
            res.append(LC(time_array[track][a: b], mag_array[track][a: b], track, str(a)))

    print('construction time: ', (Time.now().mjd - t)*86400)
    return res


def resolve_dir(dir_name):
    num = os.path.basename(dir_name).split('_')[-1]
    return os.path.join(dir_name, f'{num}_tracks.txt'), os.path.join(dir_name, 'Filter'), num
