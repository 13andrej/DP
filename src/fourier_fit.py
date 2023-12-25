from matplotlib import pyplot as plt

from src.random_forest import load_curve
import numpy as np
import scipy as sp
from functions_v03 import foufit, our_fourier8


def our_fourier_n(x, *params):
    pi = np.pi
    n = len(params) // 2
    y = params[0]

    for i in range(1, n + 1):
        y += params[i] * np.cos(i * x * 2 * pi) + params[8 + i] * np.sin(i * x * 2 * pi)

    return y


def fourier(x, y, n):
    N = x.shape[0]
    fs = N / x[-1]
    fft = np.fft.fft(y)
    frequencies = np.fft.fftfreq(N, 1/fs)

    # plt.semilogy(frequencies[:N // 2], np.abs(fft[:N // 2]))
    # plt.xlabel('frequency[Hz]')
    # plt.show()

    # print(np.argsort(np.abs(fft[:N // 2]))[-15:])
    # print(np.abs(fft[:N // 2])[:15])
    a = np.argsort(np.abs(fft))[:-14]

    # print(len(a), len(a[:-14]))
    # fft[a] = 0
    fft[n:-n] = 0
    print(np.abs(fft[:n]))
    print(np.abs(fft[-n:]))

    res = sp.fft.ifft(fft)
    return np.abs(res)


if __name__ == '__main__':
    y, _ = load_curve(r'C:\Users\13and\PycharmProjects\DP\data\dataset\exports\0.txt')
    x = np.arange(0, 1.00001, 1/(len(y) - 1))
    # plt.plot(x, y)

    params, std, residuals, rms = foufit(x, y)
    #print(params)
    # plt.plot(x, our_fourier13(x, *params))

    y8 = fourier(x, y, 8)
    plt.plot(x, y-y8)
    plt.show()
