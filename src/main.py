import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def plot(x, y, scatter=True, final=True, invert=True):
    plt.gca().invert_yaxis() if invert else None
    plt.scatter(x, y) if scatter else plt.plot(x, y)
    plt.show() if final else None


def normalize_phase(lc, duration=1.0):
    interp_func = sp.interpolate.interp1d(lc[0], lc[1])
    phase_normalised = np.arange(0, lc[0].values[-1], 0.002)
    mag_normalised = interp_func(phase_normalised)
    phase_normalised *= duration

    return phase_normalised, mag_normalised


def extend(phase, mag, times=2):
    period = np.max(phase)
    x = np.array([])
    y = np.array([])

    for i in range(times):
        x = np.concatenate((x, phase+(i*period)))
        y = np.concatenate((y, mag))

    return x, y


def fourier(x, y):
    N = len(x)
    fs = N / x[-1]
    ft = sp.fft.fft(y) / fs
    freq = sp.fft.fftfreq(N, 1/fs)
    # ft[0] = 0
    # ft[-10:-3] = 0

    # plot(freq[:N//2], ft[:N//2], scatter=False, invert=False)
    plt.semilogy(freq[:N//2], np.abs(ft[:N//2]))
    plt.show()

    ift = sp.fft.ifft(ft)
    return ift


def sft(x, y):
    # f, t, Zxx = sp.signal.stft(y, fs=len(x)/x[-1])
    f, t, Zxx = sp.signal.stft(y, fs=len(x)/x[-1], window='hamming', nperseg=100, noverlap=50)
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='jet')
    plt.colorbar()
    plt.show()
    plt.specgram(y, Fs=len(x) / x[-1])
    plt.show()
    # plot(phase_normalised, mag_normalised, scatter=False)
    print(np.sum(np.abs(Zxx), axis=0))


if __name__ == '__main__':
    # filename = '04011A'
    filename = '01014A_2_3_R_20181212_DATA_fit.txt'
    file_path = os.path.join(data_dir, filename)
    light_curve = pd.read_table(file_path, skiprows=9, header=None, usecols=[0, 1, 2])  # skip 9 if fit 2 if phase data

    # plot(light_curve[0], light_curve[1], scatter=False)
    phase_normalised, mag_normalised = normalize_phase(light_curve, 183.93)
    mag_normalised[200:205] = 5
    plot(phase_normalised, mag_normalised, scatter=False)

    # ift = fourier(phase_normalised, mag_normalised)
    # plot(phase_normalised, ift, scatter=False)

    sft(phase_normalised, mag_normalised)

    # derivation
    # dx = x[1] - x[2]
    # plot(x[:-1], np.diff(y) / dx, scatter=False)
