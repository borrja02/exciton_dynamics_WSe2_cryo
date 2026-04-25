import os
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.constants as const
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline

# === Physical Constants ===
h = const.Planck  # Planck constant (J·s)
c = const.c       # Speed of light in vacuum (m/s)
e = const.e       # Elementary charge (C)

# === File and Folder Setup ===
folder = r"C:\Users\Borja\Desktop\TFG\data\Grating 600 líneas por mm\Estudio temperatura"
files = [f for f in os.listdir(folder) if f.lower().endswith(".asc")]  # List .asc files

# === Labels for spectral peaks in the lowest-temperature spectrum ===
peak_labels = [r"$X^0$", r"$X^-$", r"$L_1$", r"$L_2$", r"$L_3$", r"$L_4$", r"$L_5$"]
wvs = np.array([705, 721, 729.1, 733.2, 738.5, 747, 755])
wvs = wvs + (712.35 - wvs[0])  # Align labels using X0 as reference

def extract_temperature(filename):
    """Extract temperature in Kelvin from filename (e.g., '77K' → 77)."""
    match = re.search(r"(\d+)[Kk]", filename)
    return int(match.group(1)) if match else 0

def varshni_wavelength(T, E0=1.75, alpha=4.5e-4, beta=170):
    """Compute bandgap wavelength (in nm) using the Varshni equation."""
    E_T = E0 - (alpha * T**2) / (T + beta)  # Bandgap energy in eV
    lambda_nm = (h * c / (E_T * e)) * 1e9   # Convert to wavelength (nm)
    return lambda_nm

# Sort files by extracted temperature
files = sorted(files, key=extract_temperature)

# === Initialize Plot ===
plt.figure(figsize=(6, 6))
offset = .5  # Vertical offset between spectra
colors = plt.cm.winter(np.linspace(0, 1, len(files)))  # Color map by temperature

exciton_peaks = []
exciton_heights = []
temperatures = []
# === Temperatures I wanna label ===
temps_to_label = {4, 50, 100, 150, 200, 250, 300}

# === Process Each Spectrum ===
for i, filename in enumerate(files):
    filepath = os.path.join(folder, filename)

    # Read file content
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse data lines (wavelength, intensity)
    data = []
    for line in lines:
        line = line.replace(',', '.').strip()
        parts = line.split()
        if len(parts) == 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                data.append([x, y])
            except ValueError:
                continue

    data = np.array(data)
    if data.size == 0:
        print(f"[!] Empty or unreadable file: {filename}")
        continue

    x = data[:, 0]
    y = data[:, 1]
    y_norm = y / np.max(y)  # Normalize intensity

    T = extract_temperature(filename)
    temperatures.append(T)

    # Estimate exciton region using Varshni model
    lambda_center = varshni_wavelength(T)
    delta = 6  # Search range in nm
    search_min = lambda_center - delta
    search_max = lambda_center + delta

    # Detect peaks
    peaks, properties = find_peaks(y_norm, height=0.3)
    candidate_peaks = [(x[peak], y_norm[peak]) for peak in peaks if search_min <= x[peak] <= search_max]

    if candidate_peaks:
        peak_wavelength, peak_height = max(candidate_peaks, key=lambda p: p[1])
        exciton_peaks.append(peak_wavelength)
        exciton_heights.append(peak_height)
        print(f"[INFO] T = {T} K, exciton peak = {peak_wavelength:.2f} nm")
    else:
        print(f"[WARN] No exciton peak found between 700–760 nm in {filename}")
        exciton_peaks.append(np.nan)
        exciton_heights.append(np.nan)

    # Plot the spectrum with vertical offset
    plt.plot(x, y_norm + i * offset, color=colors[i])

    # Annotate only selected temperature labels, with bigger text
    if T in temps_to_label:
        label_x = 702
        try:
            label_y = y_norm[x >= label_x][0] + i * offset + 0.03
        except IndexError:
            label_y = np.max(y_norm) + i * offset + 0.03  # fallback

        plt.text(label_x, label_y, f"{T} K",
                color='k', fontsize=13, fontweight='bold',
                ha='left', va='bottom',
                bbox=dict(facecolor=colors[i], alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))


# === Plot Formatting ===
ax = plt.gca()
face_color = 'azure'
border_color = 'slateblue'
ax.set_facecolor(face_color)

ax.set_xlabel("Longitud de onda (nm)", fontsize=15, fontweight='bold')
ax.set_yticklabels([])
ax.tick_params(axis='y', which='both', length=0)
ax.tick_params(axis='x', which='major', labelsize=15, width=2,
               color='k', direction='inout', length=6)

for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('k')

# === Add Secondary X Axis: Energy in eV ===
def nm_to_ev(x_nm):
    return (h * c / (x_nm * 1e-9)) / e

def ev_to_nm(E_eV):
    return (h * c / (E_eV * e)) * 1e9

secax = ax.secondary_xaxis('top', functions=(nm_to_ev, ev_to_nm))
secax.set_xlabel('Energía (eV)', fontsize=15, fontweight='bold')
secax.tick_params(axis='x', which='major', labelsize=15, width=2,
                  color='k', direction='inout', length=6)

# === Plot Exciton Peaks ===
valid_indices = ~np.isnan(exciton_peaks)
valid_peaks = np.array(exciton_peaks)[valid_indices]
valid_heights = np.array(exciton_heights)[valid_indices]
valid_i = np.arange(len(files))[valid_indices]

plt.plot(valid_peaks, valid_heights + valid_i * offset,
         color='slateblue', ls='None', marker='o', markersize=5, zorder=3)

# === Optional: Smooth interpolation ===
if len(valid_peaks) >= 4:
    spline = UnivariateSpline(valid_peaks, valid_heights + valid_i * offset, s=150)
    x_smooth = np.linspace(min(valid_peaks), max(valid_peaks), 600)
    y_smooth = spline(x_smooth)
    plt.plot(x_smooth, y_smooth, color='slateblue', linestyle='--', linewidth=1.5, zorder=2)
else:
    print("[WARN] Smooth interpolation skipped: insufficient valid data points.")

# Set plot x-range
ax.set_xlim([700, 765])

# === Add Peak Labels on the First Spectrum ===
first_filepath = os.path.join(folder, files[0])
with open(first_filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.replace(',', '.').strip()
    parts = line.split()
    if len(parts) == 2:
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
            data.append([x_val, y_val])
        except ValueError:
            continue

data = np.array(data)
x_data = data[:, 0]
y_data = data[:, 1]
y_norm_data = y_data / np.max(y_data)

for wl, label in zip(wvs, peak_labels):
    idx = (np.abs(x_data - wl)).argmin()
    y_val = y_norm_data[idx]
    plt.text(wl, y_val + 0.3, label, fontsize=13, ha='center', va='bottom',
         color='navy', fontweight='bold',
         bbox=dict(facecolor='whitesmoke', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.2'))

# Layout and show
plt.tight_layout()
plt.show()

# === Save Peak Results ===
output_file = os.path.join(folder, "excitonic_peaks.txt")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("Temperature (K)\tWavelength (nm)\n")
    for T, peak in zip(temperatures, exciton_peaks):
        if not np.isnan(peak):
            f.write(f"{T}\t{peak:.2f}\n")
        else:
            f.write(f"{T}\tNaN\n")

print(f"[INFO] Output file saved: {output_file}")
