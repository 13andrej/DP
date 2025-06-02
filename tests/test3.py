import json
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, curve_fit

from src.utils.constants import dataset_directory, data_directory
from src.utils.file_handler import read_light_curve, read_specific_lc
from src.utils.functions import fourier, interpolate


def delta(x, center=0.5, sigma=0.2, width=0.05, k=0.0):
    width /= 4 * sigma
    return -np.exp(-((x - center) / (sigma * width)) ** 2) / (sigma * np.sqrt(np.pi)) + k


def lorentz(x, center=0.5, amplitude=1.0, width=0.01, k=0.0):
    return -amplitude / (1 + ((x - center) / width) ** 2) + k


def lorentz_rms(params, x, y):
    center, amplitude, width, k = params
    generated_array = -lorentz(x, center, amplitude, width, k)
    return np.mean((generated_array - y)**2)


def test1():
    lc_path = os.path.join(dataset_directory, '3_1925456_0.txt')
    x, mag, label = read_light_curve(lc_path)
    y8 = fourier(x, mag, 8)
    y15 = fourier(x, mag, 15)

    plt.plot(x, y8, label='Fit8', color='lime')
    plt.plot(x, y15, label='Fit15', color='red')
    plt.scatter(x, mag, label='Data')
    plt.gca().invert_yaxis()
    plt.xlabel('phase')
    plt.ylabel('magnitude')
    plt.legend()
    plt.scatter(x, 10-np.abs(mag - y8))
    plt.show()

    plt.scatter(x, np.abs(y15 - y8))
    plt.scatter(x, y15 - y8)
    plt.show()


def test2():
    lc = read_specific_lc('13', '1926340', '664')
    lc.compute_normal()

    x2 = lc.x[(lc.x < 0.38) | (lc.x > 0.43)][1:]
    y2 = lc.y[(lc.x < 0.38) | (lc.x > 0.43)][1:]

    x3, y3 = interpolate(x2, y2, len(lc.x), 'linear')
    f8 = fourier(x2, y2, 8)
    lor = lorentz(lc.x, 0.39, 4, 0.015)
    #lc.y[(lc.x > 0.38) & (lc.x < 0.43)] += lor[(lc.x > 0.38) & (lc.x < 0.43)]
    f8 = fourier(lc.x, lc.y, 8)

    initial_params = np.array([0.4, 3.0, 0.01, 6.5])
    x_values = lc.x[(lc.x > 0.38) & (lc.x < 0.43)]
    target_array = lc.y[(lc.x > 0.38) & (lc.x < 0.43)]
    result = minimize(lorentz_rms, initial_params, args=(x_values, target_array))
    optimal_params = result.x
    lc.y[(lc.x > 0.38) & (lc.x < 0.43)] += lorentz(x_values, *optimal_params) + 7.5

    print("Optimal parameters:", optimal_params)
    print("Optimization successful:", result.success)
    print("Message:", result.message)

    plt.scatter(x_values, target_array)
    plt.scatter(x_values, -lorentz(x_values, *optimal_params))
    plt.show()

    # plt.gca().invert_yaxis()
    # plt.plot(lc.x, f8, color='red')
    # plt.plot(lc.x, 8 - lor, color='lime')
    # plt.scatter(x2, y2)
    # plt.scatter(lc.x, lc.y)
    # plt.show()

    plt.gca().invert_yaxis()
    plt.scatter(lc.x, lc.y)
    # plt.scatter(x_values, lorentz(x_values, *optimal_params))
    plt.show()


def test3():
    """fit specular"""
    lc = read_specific_lc('13', '1926340', '664')
    lc.compute_normal()

    x2 = lc.x[(lc.x > 0.38) & (lc.x < 0.43)][1:]
    y2 = lc.y[(lc.x > 0.38) & (lc.x < 0.43)][1:]

    initial_guess = [0.4, 3, 0.03, 5]
    initial_guess = [0.4, 0.2, 0.05, 5]
    # params, covariance = curve_fit(lorentz, x2, y2, p0=initial_guess)
    params, covariance = curve_fit(delta, x2, y2, p0=initial_guess)
    # params = initial_guess
    print('Optimal parameters :', params)
    c, a, w, k = params

    y_fitted = lorentz(x2, c, a, w, k)
    y_fitted = delta(x2, c, a, w, k)

    plt.plot(x2, y2, 'o', label='Original Data')
    plt.plot(x2, y_fitted, label='Fitted Curve')
    plt.legend()
    plt.title('Data Fitting using curve_fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def test4(o, t, s):
    """poly fit"""
    lc = read_specific_lc(o, t, s)
    lc.compute_normal()

    x2 = lc.x[(lc.x < 0.38) | (lc.x > 0.43)][1:]
    y2 = lc.y[(lc.x < 0.38) | (lc.x > 0.43)][1:]

    x3 = x2[(x2 < 0.62) | (x2 > 0.68)]
    y3 = y2[(x2 < 0.62) | (x2 > 0.68)]

    x3, y3 = lc.x, lc.y

    for degree in range(14, 15):
        model = np.poly1d(np.polyfit(x3, y3, degree))
        fit = model(x3)
        error = np.mean((fit - y3) ** 2)
        print(f'error for {degree}: {error}')

        plt.scatter(x3, y3)
        plt.plot(x3, fit, color='red')
        plt.title(f'poly {degree}')
        plt.show()

    m1, m2 = np.poly1d(np.polyfit(x3, y3, 10)), np.poly1d(np.polyfit(x3, y3, 15))
    temp = y3 - m2(x3)
    # temp[temp > 0] = 0
    plt.scatter(x3, temp)
    plt.show()


if __name__ == '__main__':
    # test4('13', '1926340', '664')

    # threshold = 0.4
    # mag_healed = mag.copy()
    # residuals = np.abs(mag - y8)
    #
    # for i in range(len(mag_healed)):
    #     if residuals[i] > threshold:
    #         mag_healed[i] = y8[i] + residuals[i]
    #
    # plt.gca().invert_yaxis()
    # plt.scatter(x, mag_healed)
    # plt.show()

    with open(r'C:\Users\13and\PycharmProjects\DP\data\annotation.json') as file:
        data = json.load(file)

    oo = list(data.keys())
    random.shuffle(oo)

    for o in oo:
        for t in data[o]:
            for s in data[o][t]:
                if data[o][t][s]['glint'] > 0:
                    test4(o, t, s)
                    input('Press [enter] to continue.')
            break
