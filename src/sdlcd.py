import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp


data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def read_light_curve(filename: str, directory: str):
    file_path = os.path.join(directory, filename)

    if filename.endswith('DATA_fit.txt'):
        temp = pd.read_table(file_path, skiprows=9, header=None, usecols=[0, 1])  # skip 9 if fit 2 if phase data
        return temp[0].to_numpy(), temp[1].to_numpy()
    elif filename.endswith('DATA.txt'):
        temp = pd.read_table(file_path, skiprows=9, header=None, usecols=[0, 4])
        return temp[0].to_numpy(), temp[4].to_numpy()
    elif filename.endswith('PHASEdata.txt'):
        temp = pd.read_table(file_path, skiprows=2, header=None, usecols=[0, 1])
        return temp[0].to_numpy(), temp[1].to_numpy()
    elif filename.endswith('RMSdata.txt'):
        temp = pd.read_table(file_path, skiprows=2, header=None, usecols=[0, 1])
        return temp[0].to_numpy(), temp[1].to_numpy()


def interpolate(x, y, count=1000):
    interp_func = sp.interpolate.interp1d(x, y)
    period = x[-1]
    step = period / (count-1)
    x_i = np.arange(0, period, step)
    x_i = np.concatenate((x_i, np.array([period])))
    y_i = interp_func(x_i)

    return x_i, y_i


def extend(x, y, times=2):
    period = np.max(x)
    x_e = np.array([])
    y_e = np.array([])

    for i in range(times):
        x_e = np.concatenate((x_e, x+(i*period)))
        y_e = np.concatenate((y_e, y))

    return x_e, y_e


def fourier(x, y):
    N = x.shape[0]
    fs = N / x[-1]
    fft = np.fft.fft(y)
    frequencies = np.fft.fftfreq(N, 1/fs)

    plt.semilogy(frequencies[:N // 2], np.abs(fft[:N // 2]))
    plt.xlabel('frequency[Hz]')
    plt.show()

    print(np.argsort(np.abs(fft[:N // 2]))[-15:])
    print(np.abs(fft[:N // 2])[:15])
    a = np.argsort(np.abs(fft))[:-14]

    print(len(a), len(a[:-14]))
    # fft[a] = 0
    fft[8:-8] = 0

    res = sp.fft.ifft(fft)
    return np.abs(res)


def fourier2():
    x = np.arange(0, 100, 0.1)
    N = len(x)
    fs = N / x[-1]
    X = np.sin(x)  # + np.sin(2*x)
    fft = np.fft.fft(X)  # / fs
    frequencies = np.fft.fftfreq(N, 1 / fs)

    argmax = np.argmax(np.abs(fft[:N//2]))
    print(x[argmax])
    print(frequencies[argmax])

    plt.plot(x, X)
    plt.show()
    # plt.plot(frequencies[:N // 2], np.abs(fft[:N // 2]))
    plt.semilogy(frequencies[:N // 2], np.abs(fft[:N // 2]))
    plt.show()


def sft(x, y):
    f, t, Zxx = sp.signal.stft(y, fs=len(x)/x[-1], window='hamming')
    plt.pcolormesh(t, f, np.abs(Zxx), cmap='jet')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.show()
    plt.specgram(y, Fs=len(x) / x[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    threshold = 0.5 * np.max(np.abs(Zxx))
    mask = np.abs(Zxx) > threshold
    mask[0:5, :] = False
    mask[-5:, :] = False
    glint_indices = np.where(mask)
    print(f'Found {len(glint_indices[0])} glints')


if __name__ == '__main__':
    # filename = '02003C_C_2_3_4_5_6_7_20180407_DATA_fit.txt'
    # filename = '09033B_1_3_R_20181204_DATA_fit.txt'  # 257.1
    # filename = '01014A_2_3_R_20181212_DATA_fit.txt'
    # filename = '96002C_X_all_20170801_PHASEdata.txt'
    filename = '91087A_2_3_4_R_20181114_PHASEdata.txt'

    x1, y1 = read_light_curve(filename, data_dir)
    x1 *= 7000
    x2, y2 = interpolate(x1, y1, 2000)
    x3, y3 = extend(x2, y2, 1)

    plt.plot(x2, y2)
    plt.xlabel('time[s]')
    plt.ylabel('magnitude')
    plt.gca().invert_yaxis()
    plt.show()

    # fourier2()
    ift = fourier(x2, y2)
    plt.plot(x2, y2)
    plt.plot(x2, ift)
    plt.xlabel('time[s]')
    plt.ylabel('magnitude')
    plt.gca().invert_yaxis()
    plt.show()

    plt.plot(x2, y2 - ift)
    plt.gca().invert_yaxis()
    plt.xlabel('time[s]')
    plt.ylabel('residuals')
    plt.show()

    # sft(x3, y3)
