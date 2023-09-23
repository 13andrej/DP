import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import stft

# Generate a test signal with a glint
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
f = 1  # Frequency of main signal
signal = np.sin(2*np.pi*f*t) + np.random.normal(0, 0.1, len(t))  # Add some noise
signal[500:510] = 3  # Add a glint at t=0.5s

# Compute the STFT
f, t, Zxx = stft(signal, fs=fs, window='hamming', nperseg=128, noverlap=64)

# Plot the spectrogram
print(Zxx.shape)
plt.pcolormesh(t, f, np.abs(Zxx), cmap='jet')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.show()
plt.plot(signal)
plt.show()
plt.specgram(signal, Fs=fs)
plt.show()
plt.plot(sp.fft.fft(signal))
plt.show()

# Threshold the spectrogram to detect glints
threshold = 0.5 * np.max(np.abs(Zxx))  # Set threshold to half of max value
mask = np.abs(Zxx) > threshold
mask[0:5, :] = False  # Remove low-frequency noise
mask[-5:, :] = False  # Remove high-frequency noise
glint_indices = np.where(mask)  # Get indices of glints
print(f'Found {len(glint_indices[0])} glints')
