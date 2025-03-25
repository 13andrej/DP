import json
import os

from src.data_preprocessing.synthetic_lc import generate_light_curve
from src.data_preprocessing.histogram import compute_fourier_element_histogram, compute_rms_histogram


def count(results_dir):
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

def count2(results_dir):
    res1 = 0
    res2 = 0

    for directory in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, directory)) or 'results' not in directory:
            continue

        results_file = os.path.join(results_dir, directory, directory.split('_')[-1] + '.txt')

        with open(results_file) as file:
            data = json.load(file)

        key = list(data.keys())[0]

        for track in data[key].keys():
            res1 += 1
            for starting_point in [x for x in data[key][track].keys() if x.isdigit()]:
                res2 += 1

    return res1, res2


if __name__ == '__main__':
    # print(count(r'C:\Users\13and\PycharmProjects\DP\data\mmt'))
    # print(count2(r'C:\Users\13and\PycharmProjects\DP\data\mmt'))
    # generate_light_curve(r'C:\Users\13and\PycharmProjects\DP2\data\fourier',
    #                      points=300, glint=True, output_directory=None, verbose=True)
    # compute_fourier_element_histogram(r'C:\Users\13and\PycharmProjects\DP\data\mmt', ['a2'],
    #                                   output_directory=None, display=True)
    compute_rms_histogram(r'C:\Users\13and\PycharmProjects\DP\data\mmt', output_directory=None, display=True)
