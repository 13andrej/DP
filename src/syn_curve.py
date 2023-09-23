import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Generate time values
    time = np.linspace(0, 10, 1000)

    # Generate Gaussian function
    gaussian_amplitude = 1.0
    gaussian_center = 5.0
    gaussian_sigma = 1.0
    gaussian = gaussian_amplitude * np.exp(-(time - gaussian_center)**2 / (2 * gaussian_sigma**2))

    # Generate Lorentzian function
    lorentzian_amplitude = 1.0
    lorentzian_center = 5.0
    lorentzian_gamma = 1.0
    lorentzian = lorentzian_amplitude / (1 + ((time - lorentzian_center) / lorentzian_gamma)**2)

    # Plot the synthetic light curves
    plt.figure(figsize=(10, 6))
    plt.plot(time, gaussian, label='Gaussian')
    plt.plot(time, lorentzian, label='Lorentzian')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title('Synthetic Light Curves')
    plt.legend()
    plt.show()
