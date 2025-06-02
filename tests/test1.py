import json
import os

from src.utils.constants import coefficients
from src.utils.constants import mmt_directory, annotation_file


def count_lc1(results_dir):
    res = 0

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)) or 'results' not in directory:
            continue

        results_file = os.path.join(results_dir, directory, 'Filter')

        c = 0
        with open(results_file) as file:
            for line in file:
                if len(line.split()) == 3:
                    c += 1

        res += c

    return res


def count_lc2(results_dir):
    res1 = 0
    res2 = 0
    res3 = 0

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)) or 'results' not in directory:
            continue

        results_file = os.path.join(results_dir, directory, directory.split('_')[-1] + '.txt')
        res1 += 1

        with open(results_file) as file:
            data = json.load(file)
            key = list(data.keys())[0]

        for track in data[key].keys():
            res2 += 1
            for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
                res3 += 1

    return res1, res2, res3


def count_lc_annotation(annotation_path):
    res1 = 0
    res2 = 0

    with open(annotation_path) as file:
        data = json.load(file)

    for obj in data:
        for track in data[obj]:
            for start_point in data[obj][track]:
                if data[obj][track][start_point]['glint'] == 0:
                    res1 += 1
                else:
                    res2 += 1

    return res1, res2


if __name__ == '__main__':
    lc_count = count_lc1(mmt_directory)
    objects, tracks, lcs = count_lc2(mmt_directory)
    no_glint, glint = count_lc_annotation(annotation_file)

    print(f'light curves: {lc_count}')
    print(f'objects: {objects}, tracks: {tracks}, light curves: {lcs}')
    print(f'no glint: {no_glint}, glint: {glint}')
