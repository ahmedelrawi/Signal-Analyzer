import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth
import os

# فولدر حفظ الصور
os.makedirs('test_signals_fft', exist_ok=True)

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

def save_fft_image(signal, fs, path):
    fft_vals = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(len(signal), 1/fs)

    plt.figure(figsize=(3, 3))
    plt.plot(fft_freqs[:len(fft_vals)//2], np.abs(fft_vals)[:len(fft_vals)//2])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# إعدادات الإشارة
fs = 1000  # sampling rate
duration = 1.0  # sec
signal_types = ['sine', 'square', 'sawtooth', 'triangle', 'am', 'fm']

# توليد 10 إشارات وحفظ الصور
for i in range(10):
    signal_type = np.random.choice(signal_types)
    freq = np.random.uniform(5, 50)
    signal = generate_signal(signal_type, freq, fs, duration)
    save_fft_image(signal, fs, f"test_signals_fft/{i}_{signal_type}.png")

print("✅ Done: 10 FFT signals saved in test_signals_fft/")
