# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Atur parameter
# fs = 1000          # Sampling rate (Hz)
# f = 5              # Frekuensi gelombang sinus (Hz)
# duration = 1       # Durasi dalam detik
# t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# # 2. Tentukan DELTA FASE (misal: beda 90 derajat atau pi/2 radian)
# delta_phase = np.pi / 2 

# # 3. Buat dua gelombang sinus
# wave1 = np.sin(2 * np.pi * f * t)
# wave2 = np.sin(2 * np.pi * f * t + delta_phase)

# # 4. Plot hasilnya
# plt.plot(t, wave1)
# plt.plot(t, wave2, linestyle="--")
# plt.legend()
# plt.xlabel("Waktu (detik)")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. Define Signal & Sampling Parameters
f_signal = 2           # Frequency of the analog wave (Hz)
t_start, t_end = 0, 1  # Time window (seconds)

# Continuous representation of the analog signal
t_analog = np.linspace(t_start, t_end, 1000)
analog_signal = np.sin(2 * np.pi * f_signal * t_analog)

# 2. Step 1: Sampling (Discretizing Time)
fs = 14                # Sampling frequency (Hz)
t_sampled = np.arange(t_start, t_end, 1/fs)
sampled_signal = np.sin(2 * np.pi * f_signal * t_sampled)

# 3. Step 2: Quantization (Discretizing Amplitude)
bits = 3               # Number of ADC bits (e.g., 3-bit gives 2^3 = 8 levels)
levels = 2**bits
# Map the signal ranging from -1 to 1 into quantization bins
quant_levels = np.linspace(-1, 1, levels)

# Find the closest predefined quantization level for each sample
quantized_signal = np.array([quant_levels[np.argmin(np.abs(level - quant_levels))] for level in sampled_signal])

# 4. Plotting the Results
plt.figure(figsize=(12, 8))

# Subplot 1: Pure Analog Signal
plt.subplot(3, 1, 1)
plt.plot(t_analog, analog_signal, color='blue', label='Analog Signal')
plt.title('1. Continuous Analog Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Subplot 2: Sampled Signal
plt.subplot(3, 1, 2)
plt.plot(t_analog, analog_signal, color='blue', alpha=0.3, linestyle='--', label='Original Analog')
plt.stem(t_sampled, sampled_signal, linefmt='orange', markerfmt='or', basefmt=' ', label='Sampled Discretes')
plt.title(f'2. Sampling (Time Discretization at $f_s$ = {fs} Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Subplot 3: Quantized Digital Signal
plt.subplot(3, 1, 3)
# Draw the threshold lines to show quantization levels
for lvl in quant_levels:
    plt.axhline(y=lvl, color='gray', linestyle=':', alpha=0.5)
plt.step(t_sampled, quantized_signal, where='mid', color='green', linewidth=2, label='Digital Signal')
plt.plot(t_sampled, quantized_signal, 'go', alpha=0.6)
plt.title(f'3. Quantization (Amplitude Discretization using {bits}-Bits / {levels} Levels)')
plt.xlabel('Time (s)')
plt.ylabel('Quantized Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
