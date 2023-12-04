from matplotlib import pyplot as plt

from src.random_forest import load_curve
import numpy as np
import scipy as sp


def foufit(xs, ys):
    params, params_covariance = sp.optimize.curve_fit(our_fourier13, xs, ys, absolute_sigma=False, method="lm", maxfev=10000)
    std = np.sqrt(np.diag(params_covariance))
    residuals = ys - our_fourier13(np.sort(xs), *params)
    rms = np.sqrt(np.sum(residuals ** 2) / (residuals.size - 2))
    return params, std, residuals, rms


def our_fourier_n(x, *params):
    pi = np.pi
    n = len(params) // 2
    y = params[0]

    for i in range(1, n + 1):
        y += params[i] * np.cos(i * x * 2 * pi) + params[8 + i] * np.sin(i * x * 2 * pi)

    return y


def our_fourier8(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8):
    pi = np.pi
    y = a0 + a1 * np.cos(x * 2 * pi) + b1 * np.sin(x * 2 * pi) + \
        a2 * np.cos(2 * x * 2 * pi) + b2 * np.sin(2 * x * 2 * pi) + \
        a3 * np.cos(3 * x * 2 * pi) + b3 * np.sin(3 * x * 2 * pi) + \
        a4 * np.cos(4 * x * 2 * pi) + b4 * np.sin(4 * x * 2 * pi) + \
        a5 * np.cos(5 * x * 2 * pi) + b5 * np.sin(5 * x * 2 * pi) + \
        a6 * np.cos(6 * x * 2 * pi) + b6 * np.sin(6 * x * 2 * pi) + \
        a7 * np.cos(7 * x * 2 * pi) + b7 * np.sin(7 * x * 2 * pi) + \
        a8 * np.cos(8 * x * 2 * pi) + b8 * np.sin(8 * x * 2 * pi)
    return y


def our_fourier13(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
                  b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13):
    pi = np.pi
    y = a0 + a1 * np.cos(x * 2 * pi) + b1 * np.sin(x * 2 * pi) + \
        a2 * np.cos(2 * x * 2 * pi) + b2 * np.sin(2 * x * 2 * pi) + \
        a3 * np.cos(3 * x * 2 * pi) + b3 * np.sin(3 * x * 2 * pi) + \
        a4 * np.cos(4 * x * 2 * pi) + b4 * np.sin(4 * x * 2 * pi) + \
        a5 * np.cos(5 * x * 2 * pi) + b5 * np.sin(5 * x * 2 * pi) + \
        a6 * np.cos(6 * x * 2 * pi) + b6 * np.sin(6 * x * 2 * pi) + \
        a7 * np.cos(7 * x * 2 * pi) + b7 * np.sin(7 * x * 2 * pi) + \
        a8 * np.cos(8 * x * 2 * pi) + b8 * np.sin(8 * x * 2 * pi) + \
        a9 * np.cos(9 * x * 2 * pi) + b9 * np.sin(9 * x * 2 * pi) + \
        a10 * np.cos(10 * x * 2 * pi) + b10 * np.sin(10 * x * 2 * pi) + \
        a11 * np.cos(11 * x * 2 * pi) + b11 * np.sin(11 * x * 2 * pi) + \
        a12 * np.cos(12 * x * 2 * pi) + b12 * np.sin(12 * x * 2 * pi) + \
        a13 * np.cos(13 * x * 2 * pi) + b13 * np.sin(13 * x * 2 * pi)
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
