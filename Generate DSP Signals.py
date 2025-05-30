# Importing the needed libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
import os


# Make a file directory for the signals gnenrated
output_dir = "signals_images"
os.makedirs(output_dir, exist_ok=True)


# Types of Noise

def add_noise(signal, noise_level=0.02):
    # adding general noise
    return signal + noise_level * np.random.randn(len(signal))



def add_spike(signal, intensity=1.0, probability=0.01):
    # is a unpredicted noise or unordinary noise
    spike_mask = np.random.rand(len(signal)) < probability
    signal[spike_mask] += intensity * np.random.randn(np.sum(spike_mask))
    return signal



def drift_signal(signal, drift_rate=0.001):
    # the signal is being drifted ovre the time
    drift = np.linspace(0, drift_rate * len(signal), len(signal))
    return signal + drift



def quantize_signal(signal, levels=16):
    # Low efficient instruments measuring or A-D converter
    min_val, max_val = np.min(signal), np.max(signal)
    quantized = np.round((signal - min_val) / (max_val - min_val) * (levels - 1)) / (levels - 1)
    return quantized * (max_val - min_val) + min_val



# Generate different signals types
def generate_signal(signal_type, freq, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if signal_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif signal_type == 'square':
        return square(2 * np.pi * freq * t)
    elif signal_type == 'sawtooth':
        return sawtooth(2 * np.pi * freq * t)
    elif signal_type == 'triangle':
        return sawtooth(2 * np.pi * freq * t, width=0.5)
    elif signal_type == 'am':
        carrier = np.cos(2 * np.pi * freq * t)
        modulator = np.sin(2 * np.pi * 5 * t)
        return (1 + modulator) * carrier
    elif signal_type == 'fm':
        kf = 10
        return np.cos(2 * np.pi * freq * t + kf * np.sin(2 * np.pi * 5 * t))
    else:
        raise ValueError("Unsupported signal type")




# saving fft images at the specified directory
def save_fft_plot(signal, fs, filename):
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/fs)
    plt.figure(figsize=(3, 3))
    plt.plot(fft_freqs[:len(fft_vals)//2], np.abs(fft_vals)[:len(fft_vals)//2])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()




# options for generating signals

signal_types = ['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm']
fs = 1000
duration = 1.0
n_per_type = 200

augmentations = [
    ("clean", lambda x: x),
    ("noise", add_noise),
    ("spike", add_spike),
    ("drift", drift_signal),
    ("quantized", quantize_signal),
]




# Generate and Save Images.
for sig_type in signal_types:
    for aug_name, aug_func in augmentations:
        # Specifing each signal type for every fiel.
        signal_folder = os.path.join(output_dir, sig_type)
        os.makedirs(signal_folder, exist_ok=True)

        for i in range(n_per_type):
            freq = np.random.uniform(1, 50)
            signal = generate_signal(sig_type, freq, fs, duration)
            signal_aug = aug_func(signal.copy())

            # اسم الملف
            filename = os.path.join(signal_folder, f"{sig_type}_{aug_name}_{i}.png")
            save_fft_plot(signal_aug, fs, filename)



# # appending the class signals names to python file
with open(r'd:\AI Projects\Signals detecting and classification system\config.py', "a") as f:
    f.write(f"\nSignals = ['Am', 'Fm', 'sawtooth', 'Sine', 'Square', 'Triangele']\n")

print("✅")