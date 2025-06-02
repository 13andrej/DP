import os

import numpy as np
from matplotlib import pyplot as plt

from src.utils.constants import day_seconds, coefficients
from src.utils.functions import get_x_y, get_x_y_reduced, interpolate


class LC:
    def __init__(self, time_array, mag_array, track, starting_point):
        self.time_array = time_array
        self.mag_array = mag_array
        self.track = track
        self.starting_point = starting_point

    def print_lc(self):
        # print(self.track, self.starting_point)
        return (f'number of points: {len(self.time_array)} \n'
                f'period [s]: {round(self.get_period_seconds(), 3)}')

    def get_period(self):
        return self.time_array[-1].mjd - self.time_array[0].mjd

    def get_period_seconds(self):
        return (self.time_array[-1].mjd - self.time_array[0].mjd) * day_seconds


class LC2:
    def __init__(self, time_array, mag_array, dist_array, obj_number, track_number, starting_point):
        self.time_array = time_array
        self.mag_array = mag_array
        self.dist_array = dist_array
        self.obj_number = obj_number
        self.track_number = track_number
        self.starting_point = starting_point

        self.glints = None
        self.period = None
        self.rms = None

        self.x = None
        self.y = None

    def set_values(self, glints=None, period=None, rms=None):
        self.glints = glints
        self.period = period
        self.rms = rms

    def get_name(self):
        return f'{self.obj_number}_{self.track_number}_{self.starting_point}'

    def print(self):
        return (f'Object id: {self.obj_number}\n'
                f'Track id: {self.track_number}\n'
                f'Starting point: {self.starting_point}\n'
                f'Glints: {self.glints}\n'
                f'Period [s]: {self.period}\n'
                f'Period [s]: {round(self.get_period_seconds(), 3)}\n'
                f'Number of points: {len(self.time_array)}\n')

    def plot(self):
        plt.clf()
        plt.gca().invert_yaxis()
        plt.title(f'track: {self.track_number}, starting point: {self.starting_point}')
        plt.ylabel('Magnitude [mag]')
        plt.xlabel('Phase')
        plt.scatter(self.x, self.y, label='Data')
        plt.legend()

    def plot_gui(self, ax):
        ax.invert_yaxis()
        ax.set_title(f'track: {self.track_number}, starting point: {self.starting_point}')
        ax.set_ylabel('Magnitude [mag]')
        ax.set_xlabel('Phase')
        ax.scatter(self.x, self.y, label='Data')
        ax.legend()

    def show(self):
        self.plot()
        plt.show()

    def save_img(self, output_directory):
        self.plot()
        plt.savefig(os.path.join(output_directory, f'{self.get_name()}.png'))

    def save_txt(self, output_directory):
        with open(os.path.join(output_directory, f'{self.get_name()}.txt'), 'w') as file:
            file.write(f'RMS: {self.rms:.3%}\n') if self.rms is not None else file.write(f'RMS: None\n')
            file.write(f'Number of points: {len(self.x)}\n')
            file.write(f'Glint: {self.glints}\n')
            file.write(f'Glint position: None\n')
            for c in coefficients:
                file.write(f'{c}: None\n')
            file.write(f'Phase\tMag\tMagFit\n')
            file.write(f'#{65*"="}\n')
            for i in range(len(self.x)):
                file.write(f'{self.x[i]}\t{self.y[i]}\t{None}\n')

    def get_period(self):
        return self.time_array[-1].mjd - self.time_array[0].mjd

    def get_period_seconds(self):
        return (self.time_array[-1].mjd - self.time_array[0].mjd) * day_seconds

    def compute_normal(self):
        self.x, self.y = get_x_y(self.time_array, self.mag_array, self.period)

    def compute_reduced(self):
        self.x, self.y = get_x_y_reduced(self.time_array, self.mag_array, self.dist_array, self.period)

    def compute_interpolated(self, n, kind='linear'):
        if self.x is None:
            self.compute_normal()
        self.x, self.y = interpolate(self.x, self.y, n, kind)
