import os.path
import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.constants import mmt_directory, coefficients, thresholds
from src.utils.file_handler import read_tracks_file, read_tracks_dir, read_results_file, read_results_dir
from src.utils.functions import get_x_y, fourier_coefficients


class HistogramBuilder:
    def __init__(self):
        self.res = {element: [] for element in coefficients}
        self.res['RMS[%]'] = []
        self.res['Points'] = np.array([])

    def clear(self):
        """clear loaded data"""
        self.res = {element: [] for element in coefficients}
        self.res['RMS[%]'] = []
        self.res['Points'] = np.array([])

    def load_coefficients_tracks(self, obj=None, n=8):
        """load data from tracks files and compute Fourier elements"""
        if obj is None:
            lc_array = read_tracks_dir(mmt_directory)
        else:
            tracks_file = os.path.join(mmt_directory, f'results_{obj}', f'{obj}_tracks.txt')
            filter_file = os.path.join(mmt_directory, f'results_{obj}', 'Filter')
            lc_array = read_tracks_file(tracks_file, filter_file)

        for lc in lc_array:
            x, y = get_x_y(lc.time_array, lc.mag_array)
            fc = fourier_coefficients(x, y, n)
            for i, element in enumerate(coefficients):
                self.res[element].append(fc[i])

    def load_coefficients_results(self, obj=None):
        """load Fourier elements from results computed in json files"""
        if obj is None:
            self.res = read_results_dir(mmt_directory, *coefficients)
        else:
            results_file = os.path.join(mmt_directory, f'results_{obj}', f'{obj}.txt')
            res_temp = read_results_file(results_file, *coefficients)

            for element in coefficients:
                self.res[element] += res_temp[element]
            self.res['RMS[%]'] += res_temp['RMS[%]']
            self.res['Points'] = np.concatenate((self.res['Points'], res_temp['Points']))

    def make_histograms(self, elements=coefficients, show=True, save=None):
        for element in elements:
            self.make_histogram(element, show, save)

    def make_histogram(self, element, show=True, save=None):
        """create histogram of particular Fourier element"""
        # min_threshold, max_threshold = thresholds[element]
        # res_f = [x for x in self.res[element] if min_threshold < x < max_threshold]
        res_f = self.res[element]
        print(f'{element} {len(res_f) / len(self.res[element]):.0%} ({len(res_f)} / {len(self.res[element])})')
        plt.clf()
        n, bins, _ = plt.hist(res_f, bins=200)  # , weights=np.ones(len(res_f)) / len(res_f) * 100)
        # plt.xlim(min_threshold, max_threshold)
        # plt.ylim(0, 50000)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution Histogram {element}')
        plt.grid(True)

        if save:
            os.makedirs(save, exist_ok=True)
            plt.savefig(os.path.join(save, f'{element}.png'))
            with open(os.path.join(save, f'{element}.pkl'), 'wb') as f:
                pickle.dump([n, bins], f)

        if show:
            plt.show()

    def make_rms_histogram(self, show=True, save=None):
        """create histogram of rms of light curves and their 8th Fourier fit"""
        original_length = len(self.res['RMS[%]'])
        rms = [x for x in self.res['RMS[%]'] if 0 <= x < 100]
        print(f'RMS {len(rms) / original_length: .0%}, ({len(rms)} / {original_length})')
        n, bins, _ = plt.hist(rms, bins=200)  # ,  weights=np.ones(len(rms)) / len(rms) * 100)
        plt.xlabel('Value [%]')
        plt.ylabel('Frequency')
        plt.title(f'Distribution Histogram RMS')
        plt.grid(True)

        if save:
            os.makedirs(save, exist_ok=True)
            plt.savefig(os.path.join(save, f'RMS.png'))
            with open(os.path.join(save, f'RMS.pkl'), 'wb') as f:
                pickle.dump([n, bins], f)

        if show:
            plt.show()

    def make_no_points_histogram(self, show=True, save=None):
        """create histogram of number of points in light curves"""
        points = [x for x in self.res['Points'] if x < 2000]
        n, bins, _ = plt.hist(points, bins=200)
        plt.xlim(0, 2000)
        plt.xlabel('Number of points in light curves')
        plt.ylabel('Frequency')
        plt.title('Histogram of number of data points in MMT light curves')
        plt.grid(True)

        if save is not None:
            plt.savefig(os.path.join(save, 'Points.png'))
            with open(os.path.join(save, 'Points.pkl'), 'wb') as f:
                pickle.dump([n, bins], f)

        if show:
            plt.show()

    def compare_2_elements(self, e1, e2):
        """compare 2 Fourier coefficients and plot them"""
        min_threshold, max_threshold = -2, 2
        # plt.xlim(min_threshold, max_threshold)
        # plt.ylim(min_threshold, max_threshold)
        plt.scatter(self.res[e1], self.res[e2], s=1)
        plt.xlabel(e1)
        plt.ylabel(e2)
        plt.title(f'{e1} compared to {e2}')
        plt.grid(True)
        plt.show()

        print(np.corrcoef(self.res[e1], self.res[e2]))

    def compare_all_elements(self):
        """compare all Fourier coefficients and plot correlation matrix"""
        min_threshold, max_threshold = -2, 2
        plt.xlim(min_threshold, max_threshold)
        plt.ylim(min_threshold, max_threshold)

        data = np.vstack([self.res[c] for c in coefficients])
        corr_matrix = np.corrcoef(data)
        sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap='coolwarm', cbar=True,
                    xticklabels=coefficients, yticklabels=coefficients)
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()
        plt.show()

        print(corr_matrix)


if __name__ == '__main__':
    HB = HistogramBuilder()
    HB.load_coefficients_results()
    HB.make_no_points_histogram()
    # HB.make_rms_histogram()
    # HB.compare_elements('a1', 'a2')
    # HB.make_no_points_histogram()
