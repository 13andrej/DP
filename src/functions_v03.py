# ----------------------------------------------------------------------- #
# This module contains supporting calculations for light curve processing #
#                                                                         #
# Author: Jiri Silha, Matej Zigo (FMPI, 2020)                             #
# Copyright Comenius University, 2020                                     #
# ----------------------------------------------------------------------- #

from astropy.time import Time
import numpy as np
from numpy import pi
from scipy import optimize
from scipy import stats
import scipy as sp
import math

"""
This method reads the track file data, calculates the time difference between 
the i-th measurement point the the first measureemnt point in seconds and calculates
the average point distance time.

Input:
    String filename - String of the MMT light curve file name

Output:
    list lc - all intensity values in the light curve 
    list lc_t - delta time between the 1st and the i-th measurement
    float apd - average point distance across the time axis
"""


def read_lc(filename):
    d = {'t_obs': list(), 'dt': list(), 'lc_t': [0], 'lc': list(), 'ph_ang': list(), 'range': list()}
    with open(filename, 'r') as f:
        text = f.read().split('\n')
        # text=f.readlines()[10:]
        for line in text:
            if len(line) != 0:
                a = line.split(' ')
                d['t_obs'].append(Time('{}T{}'.format(a[0], a[1])))
                #                d['lc'].append(float(a[2]))
                d['lc'].append(float(a[3]))
                d['ph_ang'].append(float(a[7]))
                d['range'].append(float(a[6]))

        for i in range(1, len(d['t_obs'])):
            d['dt'].append((d['t_obs'][i] - d['t_obs'][i - 1]).sec)
            d['lc_t'].append((d['t_obs'][i] - d['t_obs'][0]).sec)

        apd = np.average(np.array(d['dt'][1:]))
        d['lc_t'] = np.asarray(d['lc_t'])
        d['lc'] = np.asarray(d['lc'])
        f.close()
    return d['lc'], d['lc_t'], apd, d['t_obs'][0].isot, d['ph_ang'], d['range'], d['t_obs']


"""
This method interpolates the measureemnts by using Fourier harmonics 8th degree.
Taken from code of Matej Zigo - test_lc.py

Input:
    float coefficients - all the coefficients are inputs
    
Output:
    float y - float y value

"""


def our_fourier8(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
    y = a0 + a1 * np.cos(x * 2 * pi) + b1 * np.sin(x * 2 * pi) + \
        a2 * np.cos(2 * x * 2 * pi) + b2 * np.sin(2 * x * 2 * pi) + \
        a3 * np.cos(3 * x * 2 * pi) + b3 * np.sin(3 * x * 2 * pi) + \
        a4 * np.cos(4 * x * 2 * pi) + b4 * np.sin(4 * x * 2 * pi) + \
        a5 * np.cos(5 * x * 2 * pi) + b5 * np.sin(5 * x * 2 * pi) + \
        a6 * np.cos(6 * x * 2 * pi) + b6 * np.sin(6 * x * 2 * pi) + \
        a7 * np.cos(7 * x * 2 * pi) + b7 * np.sin(7 * x * 2 * pi) + \
        a8 * np.cos(8 * x * 2 * pi) + b8 * np.sin(8 * x * 2 * pi)
    return y


def our_fourier8_dev(x, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
    y = b1 * 1 * 2 * pi * np.cos(1 * x * 2 * pi) - a1 * 1 * 2 * pi * np.sin(1 * x * 2 * pi) + \
        b2 * 2 * 2 * pi * np.cos(2 * x * 2 * pi) - a2 * 2 * 2 * pi * np.sin(2 * x * 2 * pi) + \
        b3 * 3 * 2 * pi * np.cos(3 * x * 2 * pi) - a3 * 3 * 2 * pi * np.sin(3 * x * 2 * pi) + \
        b4 * 4 * 2 * pi * np.cos(4 * x * 2 * pi) - a4 * 4 * 2 * pi * np.sin(4 * x * 2 * pi) + \
        b5 * 5 * 2 * pi * np.cos(5 * x * 2 * pi) - a5 * 5 * 2 * pi * np.sin(5 * x * 2 * pi) + \
        b6 * 6 * 2 * pi * np.cos(6 * x * 2 * pi) - a6 * 6 * 2 * pi * np.sin(6 * x * 2 * pi) + \
        b7 * 7 * 2 * pi * np.cos(7 * x * 2 * pi) - a7 * 7 * 2 * pi * np.sin(7 * x * 2 * pi) + \
        b8 * 8 * 2 * pi * np.cos(8 * x * 2 * pi) - a8 * 8 * 2 * pi * np.sin(8 * x * 2 * pi)
    return y


"""
Folding function to get the phase function.

Input:
    numpy.array - time
    float - period
Output: 
    numpy.array - phase

"""


def fold(time, period):
    return divmod(time, period)[1] / period


"""
Polynomial function of 2nd order to be fitted with scipy.optimize function

Input:
    float x - x parameter
    float a,b,c - quadratic coefficients

Output:
    
"""


def func_poly2order(x, a, b, c):
    y = a * x * x + b * x + c
    return y


"""
Get the detrended light curve from the original one, thanks to the 2nd 
polynom order
Input:
    
Output:

"""


def detrend(time, ys):
    mmethod = func_poly2order
    params, params_covariance = optimize.curve_fit(mmethod, time, ys)
    return ys - func_poly2order(time, *params)


"""
prepare data - sort intensities according the phase and subtract the median intensity
Input:
    xs - np.array 
        np.floats - folded phase
    ys - np.array 
        np.floats - magnitudes
Output:
    xs - np.array 
        sorted phase from 0 to 1
    ys - np.array
        sorted intensities with subtracted median value
"""


def prepare(time_np_fold, ys_np_cor):
    xs = np.array(time_np_fold).astype(np.float)
    ys = np.array(ys_np_cor).astype(np.float)
    sorted_data = [[], []]
    data = [xs, ys]
    a = np.argsort(data, axis=1)
    for i in range(len(ys)):
        sorted_data[0].append(data[0][a[0][i]])
        sorted_data[1].append(data[1][a[0][i]])

    xs = sorted_data[0]
    ys = sorted_data[1]

    # mediany = np.median(ys)	 - function removed!!!
    # return xs, np.subtract(ys, mediany)- function removed!!!
    return xs, ys


"""
Push the data to its first maximum 
"""


def push_to_max(xs, ys):
    minimal_y_value = np.amin(ys)
    index_minimal = np.where(ys == minimal_y_value)[0][0]

    xxs = list()
    for elem in xs:
        elem -= xs[index_minimal]
        if elem < 0:
            elem += 1
        xxs.append(elem)

    xs = np.array(xxs)
    xs = np.roll(xs, -index_minimal)
    ys = np.roll(ys, -index_minimal)
    return xs, ys


"""
Find the global maximum and minimum of the measured data. Idea is to find the min/max value
and then take to account data in +/-0.0025 phase interval and calculate the median of this interval. 
Input:
    xs - np.array 
        np.floats - folded and sorted phase
    ys - np.array 
        np.floats - magnitudes with subtracted median
Output:
    min_graph_x: np.float,
    min_graph_y: np.float,
    max_graph_x: np.float,
    max_graph_y: np.float
"""


def data_glob_extrems(xs, ys):
    minimal_y_value = np.amin(ys)
    index_minimal = np.where(ys == minimal_y_value)[0][0]
    minimal_x_value = xs[index_minimal]
    maximal_y_value = np.amax(ys)
    index_maximal = np.where(ys == maximal_y_value)[0][0]
    maximal_x_value = xs[index_maximal]

    length = np.ceil(len(xs) * 0.0025)

    first_slice_min = index_minimal - length
    if first_slice_min < 0:
        first_slice_min = 0

    xs_for_min = xs[first_slice_min:int(index_minimal + length)]
    ys_for_min = ys[first_slice_min:int(index_minimal + length)]
    xs_for_max = xs[int(index_maximal - length):int(index_maximal + length)]
    ys_for_max = ys[int(index_maximal - length):int(index_maximal + length)]

    min_graph_x = np.mean(xs_for_min)
    min_graph_y = np.mean(ys_for_min)
    max_graph_x = np.mean(xs_for_max)
    max_graph_y = np.mean(ys_for_max)
    return min_graph_x, min_graph_y, max_graph_x, max_graph_y


"""
Fitting function to obtain fourier coefficient and residuals
Input:
    xs: np.array
    ys: np.array
Output:
    params: np.array
        fourier coef a0,..,a8 and b1,...,b8
    std: np.array
        uncertainties of fourier coef. from covariance matrix
    residuals: np.array
        differences between measured and calculated values
    rms: float
        complete root-mean-square  
"""


def foufit(xs, ys):
    params, params_covariance = optimize.curve_fit(our_fourier8, xs, ys, absolute_sigma=False, method="lm",
                                                   maxfev=10000)
    std = np.sqrt(np.diag(params_covariance))
    residuals = ys - our_fourier8(np.sort(xs), *params)
    rms = np.sqrt(sp.sum(residuals ** 2) / (residuals.size - 2))
    return params, std, residuals, rms


"""
Find the fourier global extrems with the similar strategy as in the data extrems determination
Input:
    xs: np.array
    ys: np.array
    params: np.array
Output:
    minx: float, 
        phase position of the global minimum of the coresponding fourier function 
    miny: float,
        intensity value in the global minimum of the coresponding fourier function
    maxx: float,
        phase position of the global maximum of the coresponding fourier function 
    maxy: float
        intensity value in the global maximum of the coresponding fourier function
"""


def fourier_glob_extrems(xs, ys, params):
    maxy = -0.0
    miny = np.inf
    maxx = 0
    minx = 0
    for i in xs:
        x = i
        temp = our_fourier8(x, *params)
        if temp > maxy:
            maxy = temp
            maxx = x
        if temp < miny:
            miny = temp
            minx = x

    results_max_y = list()
    for tbm in np.arange(maxx - 0.0025, maxx + 0.0025, 0.001):
        yss = our_fourier8(tbm, *params)
        results_max_y.append(yss)
    maxy = np.median(results_max_y)
    results_min_y = list()
    for tbm in np.arange(minx - 0.0025, minx + 0.0025, 0.001):
        yss = our_fourier8(tbm, *params)
        results_min_y.append(yss)
    miny = np.median(results_min_y)
    return minx, miny, maxx, maxy


"""
Use the first derivation to determine how many time the function change from increasing 
to decreasing function and vice versa, so to count the number of local extrems 
Input:
    xs: np.array
    params: np.array
Output:
    mins: int
       number of local minima
    maxs: int
       number of local maxima  
"""


def count_extrems(xs, params):
    y_dev = our_fourier8_dev(np.sort(xs), *params[1:19:1]) / np.abs(our_fourier8_dev(np.sort(xs), *params[1:19:1]))
    y_dev[(len(y_dev) - 1)] = y_dev[0]
    mins = 0
    maxs = 0
    for i in range(1, len(y_dev)):
        if y_dev[i] - y_dev[i - 1] == -2.0:
            mins = mins + 1
        elif y_dev[i] - y_dev[i - 1] == 2.0:
            maxs = maxs + 1
    return mins, maxs


"""
Uncertainty of obtainted light curve characteristics, period, phase, frequency and amplitude.
Input:
    sigma: float
        root-mean-square of the dataset
    period: float
        obtained period value 
    apd: float
        average point distance in time in seconds
    point: int
        number of points in the dataset
    max_y/min_y: float
        intensity value in the global max/min
Output:
    rel_rms: float
        rms/amplitude - [%]
    sigma_amp: float
        uncertainty of the measured amplitude - [mag]
    sigma_phase: float
        uncertainty of the folded phase - [dimensionless]
    sigma_freq: float
        uncertainty of the estimated frequency - [1/s];[Hz]
    sigma_P: float
        uncertainty of the estimated period - [s] 
"""


def error_estimation(sigma, period, apd, points, max_y, min_y):
    amp = np.abs(max_y - min_y)
    rel_rms = np.round((sigma / amp) * 100, 5)
    sigma_amp = np.sqrt(2 / points) * sigma
    sigma_phase = sigma_amp / (amp)
    sigma_freq = np.sqrt(6 / points) * (1 / (pi * points * apd)) * (sigma / (amp))
    sigma_P = np.round(sigma_freq * (period ** 2), 6)
    return rel_rms, sigma_amp, sigma_phase, sigma_freq, sigma_P


"""
Method to calculate mean values.
Input:
    numpy array values_np - numpy array of float values

Output:
    flloat mean - mean value of the numpy array values

"""


def get_mean(values_np):
    return np.mean(values_np)


"""
Method to reduce the magnitude for the change of distance between object and observer.
Formula taken from Buchheim (2010) - Methods and Lessons Learned Determining the H-G Parameters of Asteroid Phase Curves
http://adsabs.harvard.edu/full/2010SASS...29..101B

Input:
    numpy array mag_obs_np - numpy array of float values [mag]
    numpy array distance_np - numpy array of float values [km]

Output:
    numpy array mag_red_np - numpy array of reduced magnitudes

"""


def get_reduced_mag_np(mag_obs_np, distance_np):
    # Numpy array with reduced magnitude
    mag_red_np = np.zeros(len(mag_obs_np))

    # print("len(mag_obs_np: ", len(mag_obs_np))
    # print("len(distance_np: ", len(distance_np))
    for x in range(len(mag_obs_np) - 1):
        # print(x)
        mag_red_np[x] = mag_obs_np[x] - 5 * np.log10(distance_np[x] / 149597871.0 * 1)
    #        mag_red_np[x] = mag_obs_np[x] - 5*np.log10(distance_np[x]*1000)

    return mag_red_np


"""
INVERTED METHOD
Method to reduce the magnitude for the change of distance between object and observer.
Formula taken from Buchheim (2010) - Methods and Lessons Learned Determining the H-G Parameters of Asteroid Phase Curves
http://adsabs.harvard.edu/full/2010SASS...29..101B


Input:
    numpy array mag_obs_np - numpy array of float values [mag]
    numpy array distance_np - numpy array of float values [km]

Output:
    numpy array mag_red_np - numpy array of reduced magnitudes

"""


def get_reduced_mag_np_inv(mag_red_np, distance_np):
    # Numpy array with reduced magnitude
    mag_obs_np = np.zeros(len(mag_red_np))

    for x in range(len(mag_red_np) - 1):
        # print(x)
        mag_obs_np[x] = mag_red_np[x] + 5 * np.log10(distance_np[x] / 149597871.0 * 1)

    return mag_obs_np


"""
Method to clean the 2D array from outliers. As input are X and Y vectors and output is cleaned X and Y. 

Input:

    numpy array x_array - numpy array of float values
    numpy array y_array - numpy array of float values

Output:
    numpy array x_array_clean - numpy array of float values
    numpy array y_array_clean - numpy array of float values

"""


def remove_outliers_2d_np(x_array, y_array, poly_deg):
    # TEST
    # print("TEEEEEST")
    # print(len(x_array), len(y_array))

    # Get the polynomial fit
    fit = np.polyfit(x_array, y_array, poly_deg)
    fit_fn = np.poly1d(fit)

    # Get calculated values
    y_array_calculated = fit_fn(x_array)

    # Get RMS
    rms_array = get_rms_np(y_array, y_array_calculated)
    # print(rms_array)

    # Remove outliers using Z-score
    z_score = np.abs(stats.zscore(rms_array))
    z_thershold = 3
    # print(rms_array)
    # print("Z-score")
    # print(z_score)

    # New arrays
    x_array_out = []
    y_array_out = []

    # Remove outliers from the array
    for x in range(len(z_score)):
        if z_score[x] < z_thershold:
            # print("-------------IN---------------", x)
            x_array_out.append(x_array[x])
            y_array_out.append(y_array[x])
    return x_array_out, y_array_out


"""
Method calculates RMS between two arrays. 

Input:
    numpy array x_array - numpy array of float values
    numpy array y_array - numpy array of float values

Output:
    numpy array rms_array - numpy array of float values

"""


def get_rms_np(x_array, y_array):
    # Get residuals
    rms_array = np.array(len(x_array))
    # for x in range(len(x_array)):
    rms_array = x_array - y_array
    return rms_array


"""
Method to calculate the standard deviation of absolute magnitude using Hejduk Dfiffuse-Specula formular with A*ro and Beta parameters. 

Input:

Output:

"""


def get_stDev_H_Hejduk(aRo, beta, std_aRo, std_beta):
    abs_mag_std = 0

    # Diffuse function F1(0) for phase = 0deg
    f1_0 = 2. / 3. / math.pi
    f2_0 = 1. / 4. / math.pi

    # Get derivatives
    deriv_dAro = -2.5 / (aRo * math.log(10))
    deriv_dBeta = -2.5 / (((beta * f1_0) + (1 - beta) * f2_0) * math.log(10))

    # Standard deviation for abs magnitude
    abs_mag_std = math.sqrt(abs(deriv_dAro) * math.pow(std_aRo, 2) + abs(deriv_dBeta) * math.pow(std_beta, 2))
    # print("============ ABS_MAG_STDEV=======================")
    # print(aRo, beta, std_aRo, std_beta)
    # print("f1_0", f1_0)
    # print("f2_0", f2_0)
    # print("deriv_dAro", deriv_dAro)
    # print("deriv_dBeta", deriv_dBeta)
    # print(abs_mag_std)

    return abs_mag_std
