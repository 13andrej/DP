import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import coefficients
from src.utils.functions import our_fourier8


def generate_light_curve(pickle_directory, points=300, glint=False, output_directory=None, verbose=False):
    pickles = {x: None for x in coefficients}

    for c in pickles.keys():
        with open(os.path.join(pickle_directory, f'{c}.pkl'), 'rb') as f:
            pickles[c] = pickle.load(f)

    with open(os.path.join(pickle_directory, 'RMS.pkl'), 'rb') as f:
        rms_n, rms_bins = pickle.load(f)

    random_values = {c: np.random.choice(bins[:-1], p=n / n.sum()) for c, (n, bins) in pickles.items()}
    rms = np.random.choice(rms_bins[:-1], p=rms_n / rms_n.sum()) / 100
    center = 0.5
    x = np.arange(0, 1.00001, 1/(points - 1))
    y = our_fourier8(x, *[random_values[c] for c in coefficients])
    y2 = add_noise(y, rms)
    if glint:
        y2, center = add_delta(x, y2)

    plt.clf()
    plt.gca().invert_yaxis()
    plt.title(['light curve without glint', f'light curve with glint ({center:.3})'][int(glint)])
    plt.ylabel('Instrumental Magnitude [mag]')
    plt.xlabel('Phase')
    plt.scatter(x, y2, label='Data')
    plt.plot(x, y, 'red', label='Fit')
    plt.legend()

    if output_directory is not None:
        os.makedirs(os.path.join(output_directory, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_directory, 'exports'), exist_ok=True)
        name = len(os.listdir(os.path.join(output_directory, 'exports')))
        plt.savefig(os.path.join(output_directory, 'images', f'{name}.png'))
        with open(os.path.join(output_directory, 'exports', f'{name}.txt'), 'w') as file:
            file.write(f'RMS: {rms:.3%}\n')
            file.write(f'Number of points: {points}\n')
            file.write(f'Glint: {glint}\n')
            file.write(['Glint position: None\n', f'Glint position: {center:.3}\n'][int(glint)])
            for c in coefficients:
                file.write(f'{c}: {random_values[c]:.3}\n')
            file.write('Phase\tMag\tMagErr\n')
            file.write(f'#{65*"="}\n')
            for i in range(len(x)):
                file.write(f'{x[i]}\t{y2[i]}\t{y2[i]-y[i]}\n')

    if verbose:
        plt.show()
        plt.clf()
        plt.gca().invert_yaxis()
        plt.plot(x, y2)
        # plt.plot([center, center], [min(y2), max(y2)])
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
    width = 0.02
    delta_approx = np.exp(-((x - center) / (sigma * width))**2) / (sigma * np.sqrt(np.pi))

    return y - delta_approx, center


def add_noise(y, desired_rms):
    noise = np.random.normal(0, 1, len(y))
    rms = np.sqrt(np.mean(np.square(noise)))
    scaled_noise = noise * (desired_rms / rms)

    return y + scaled_noise


if __name__ == '__main__':
    number_of_lc = 100
    points_in_lc = 300
    probability_glint = [0.5, 0.5]
    pkl_dir = r'C:\Users\13and\PycharmProjects\DP2\data\fourier'
    out_dir = r'C:\Users\13and\PycharmProjects\DP2\data\dataset\01'

    for _ in range(number_of_lc):
        glint_ = np.random.choice([True, False], p=probability_glint)
        generate_light_curve(pkl_dir, points=points_in_lc, glint=glint_, output_directory=out_dir, verbose=False)
