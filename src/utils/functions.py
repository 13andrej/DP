import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from src.utils.constants import day_seconds


def fourier(x, y, n):
    """fit input signal with Fourier of n-th degree"""
    fft = np.fft.fft(y)
    fft[n+1:-n] = 0
    res = sp.fft.ifft(fft)

    return np.abs(res)


def polyfit(x, y, n):
    """fit input signal with polynom of n-th degree"""
    model = np.poly1d(np.polyfit(x, y, n))

    return model(x)


def fourier_coefficients(x, y, n):
    """fit input signal with Fourier of n-th degree and return coefficients"""
    fft = np.fft.fft(y)
    res = [fft[0].real / len(y)]

    for i in range(1, n+1):
        res.append(fft[i].real)
    for i in range(1, n+1):
        res.append(fft[i].imag)

    return res


def fourier_test(x, y, n):
    N = x.shape[0]
    fft = np.fft.fft(y)
    fft[n+1:-n] = 0

    res = np.zeros((N,))
    freqs = np.fft.fftfreq(N, 1/N)

    for i in range(N):
        # if fft[i].real != 0:
        #     print(i, fft[i].real, fft[i].imag, freqs[i])
        res += 1 / N * (fft[i].real * np.cos(freqs[i] * 2 * np.pi * x) - fft[i].imag * np.sin(freqs[i] * 2 * np.pi * x))

    return res


def fourier8_from_coefficients(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
    """reconstruct signal from Fourier coefficients"""
    y = a0 + a1 * np.cos(x * 2 * np.pi) + b1 * np.sin(x * 2 * np.pi) + \
        a2 * np.cos(2 * x * 2 * np.pi) + b2 * np.sin(2 * x * 2 * np.pi) + \
        a3 * np.cos(3 * x * 2 * np.pi) + b3 * np.sin(3 * x * 2 * np.pi) + \
        a4 * np.cos(4 * x * 2 * np.pi) + b4 * np.sin(4 * x * 2 * np.pi) + \
        a5 * np.cos(5 * x * 2 * np.pi) + b5 * np.sin(5 * x * 2 * np.pi) + \
        a6 * np.cos(6 * x * 2 * np.pi) + b6 * np.sin(6 * x * 2 * np.pi) + \
        a7 * np.cos(7 * x * 2 * np.pi) + b7 * np.sin(7 * x * 2 * np.pi) + \
        a8 * np.cos(8 * x * 2 * np.pi) + b8 * np.sin(8 * x * 2 * np.pi)
    return y


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def max_rolling(a, window, axis=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return np.max(rolling, axis=axis)


def get_x_y(time_array, mag_array, p=None):
    """get light curve as 2 corresponding arrays of phase and magnitude"""
    p = (time_array[-1] - time_array[0]).value if p is None else p/day_seconds
    x = np.array([((i - time_array[0]).value % p) / p for i in time_array])
    y = np.array(mag_array)

    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]

    return x, y


def get_x_y_reduced(time_array, mag_array, dist_array, p=None):
    """get reduced magnitude"""
    x, y = get_x_y(time_array, mag_array, p)
    mag_red_np = np.zeros(len(mag_array))

    for i in range(len(mag_array)):
        mag_red_np[i] = mag_array[i] - 5 * np.log10(dist_array[i] / 149597871.0 * 1)

    return x, mag_red_np


def interpolate(x, y, n, kind='linear'):
    """interpolate light curve to given number of points"""
    x_res = np.linspace(x.min(), x.max(), n)
    interpolator = interp1d(x, y, kind=kind)
    y_res = interpolator(x_res)

    return x_res, y_res


def normalize(y):
    """normalize input signal"""
    min_val = y.min()
    max_val = y.max()

    return (y - min_val) / (max_val - min_val)
