import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from src.utils.constants import coefficients, fourier_directory
from src.utils.functions import fourier8_from_coefficients


class LightCurveGenerator:
    def __init__(self):
        self.x = None
        self.y_diffuse = None
        self.y_specular = None
        self.y_specular_test = None
        self.y_noise = None
        self.pickles = {x: None for x in coefficients}
        self.lc_info = {}

        for c in self.pickles.keys():
            with open(os.path.join(fourier_directory, f'{c}.pkl'), 'rb') as f:
                self.pickles[c] = pickle.load(f)

        with open(os.path.join(fourier_directory, 'RMS.pkl'), 'rb') as f:
            self.rms_n, self.rms_bins = pickle.load(f)

    def generate_light_curve(self, points=300, glints=0):
        """generate synthetic light curve with given number of points and glints"""
        self.lc_info = {}
        random_values = {c: np.random.choice(bins[:-1], p=n / n.sum()) for c, (n, bins) in self.pickles.items()}
        rms = np.random.choice(self.rms_bins[:-1], p=self.rms_n / self.rms_n.sum()) / 100

        self.x = np.array([x / (points-1) for x in range(points)])
        self.y_diffuse = fourier8_from_coefficients(self.x, *[random_values[c] for c in coefficients])
        self.y_specular = np.zeros(self.x.shape)
        self.y_specular_test = np.zeros(self.x.shape)
        self.y_noise = np.zeros(self.x.shape)
        self.add_gaussian_noise2(rms)

        self.lc_info = {'rms': rms, 'points': len(self.x), 'glints': 0}
        self.lc_info.update({c: random_values[c] for c in coefficients})

        for i in range(glints):
            center = i/glints + (1/glints)*np.random.random()
            sigma = 0.13 + 0.27 * np.random.random()
            width = 0.04 + 0.11 * np.random.random()
            self.add_delta_glint(center, sigma, width)
            self.merge_specular()

    def plot_light_curve(self):
        """construct light curve plot image"""
        plt.clf()
        plt.gca().invert_yaxis()
        plt.title(f'glints: {self.lc_info["glints"]}')
        plt.ylabel('Magnitude [mag]')
        plt.xlabel('Phase')
        plt.scatter(self.x, self.y_diffuse + self.y_noise + self.y_specular, label='Data')
        plt.plot(self.x, self.y_diffuse, 'red', label='Fit')
        # plt.scatter(self.x[self.y_s < -0.01], self.y_s[self.y_s < -0.01] + self.y_d[self.y_s < -0.01], label='Specular part')
        plt.legend()

    def show_light_curve(self):
        """display light curve using matplotlib"""
        self.plot_light_curve()
        plt.show()

    def save_light_curve_img(self, output_directory, name):
        """export light curve into image"""
        self.plot_light_curve()
        plt.savefig(os.path.join(output_directory, f'{name}.png'))

    def save_light_curve_txt(self, output_directory, name):
        """export light curve into txt file"""
        with open(os.path.join(output_directory, f'{name}.txt'), 'w') as file:
            file.write(f'RMS: {self.lc_info["rms"]:.3%}\n')
            file.write(f'Number of points: {self.lc_info["points"]}\n')
            file.write(f'Glint: {self.lc_info["glints"]}\n')
            # file.write(['Glint position: None\n', f'Glint position: {center:.3}\n'][int(glint)])
            for c in coefficients:
                file.write(f'{c}: {self.lc_info[c]:.3}\n')
            file.write('Phase\tMag\tGlint\n')
            file.write(f'#{65*"="}\n')
            for i in range(self.lc_info['points']):
                file.write(f'{self.x[i]}\t{self.y_diffuse[i] + self.y_noise[i] + self.y_specular[i]}\t{self.y_specular[i]}\n')

    def add_gaussian_noise(self, mean=0, std_dev=0.01):
        """add gaussian noise to its assigned array"""
        noise = np.random.normal(mean, std_dev, size=self.x.shape)
        self.y_noise += noise

    def add_gaussian_noise2(self, desired_rms):
        """add gaussian noise to its assigned array"""
        noise = np.random.normal(0, 1, len(self.y_noise))
        rms = np.sqrt(np.mean(np.square(noise)))
        scaled_noise = noise * (desired_rms / rms)

        self.y_noise += scaled_noise

    def add_lorentz_glint(self, center=0.5, amplitude=1.0, width=0.01):
        """add lorentz glint to its assigned array"""
        lorentzian = amplitude / (1 + ((self.x - center) / width) ** 2)
        self.y_specular_test -= lorentzian

    def add_delta_glint(self, center=0.5, sigma=0.2, width=0.05):
        """add delta glint to testing specular array"""
        width /= 4 * sigma
        delta_approx = np.exp(-((self.x - center) / (sigma * width)) ** 2) / (sigma * np.sqrt(np.pi))
        self.y_specular_test -= delta_approx

    def clear_glints(self):
        """clear testing specular array"""
        self.y_specular_test = np.zeros(self.y_specular.shape)

    def merge_specular(self):
        """merge testing specular array into definitive specular array"""
        self.lc_info['glints'] += 1
        self.y_specular += self.y_specular_test
        self.y_specular_test = np.zeros(self.y_specular.shape)


if __name__ == '__main__':
    LCG = LightCurveGenerator()
    LCG.generate_light_curve(300, 2)
    LCG.show_light_curve()
