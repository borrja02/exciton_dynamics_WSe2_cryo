# === Libraries ===
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.constants as const
from scipy.optimize import curve_fit

# === Configuration ===
folder = r"C:\Users\Borja\Desktop\TFG\data\Grating 600 líneas por mm\Estudio potencia"
files = [f for f in os.listdir(folder) if f.endswith(".asc")]

# Peak positions (in nm), corrected to align with 712.35 nm
wvs = np.array([706, 721, 729.1, 733.2, 738.5, 747, 755])
wvs = wvs + (712.35 - wvs[0])  # Reference shift

# Extract power from filename (in µW)
def extract_power(filename):
    match = re.search(r"([\d\.]+)uW", filename)
    return float(match.group(1)) if match else 0

files = sorted(files, key=extract_power)
colors = plt.cm.viridis(np.linspace(0, 1, len(wvs)))
integrated_intensities = {wv: [] for wv in wvs}
powers = []

# nm → eV conversion
def wavelength_to_eV(wavelength_nm):
    return (const.h * const.c) / (wavelength_nm * 1e-9 * const.e)

# Peak labels
peak_labels = [r"$X^0$", r"$X^-$", r"$L_1$", r"$L_2$", r"$L_3$", r"$L_4$", r"$L_5$"]

# === Stacked Plot ===
face_color, edge_color = 'floralwhite', 'wheat'
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.set_facecolor(face_color)
for spine in ax.spines.values():
    spine.set_edgecolor('k')
    spine.set_linewidth(2)

offset = 0.5  # Vertical offset for stacked display

for i, filename in enumerate(files):
    filepath = os.path.join(folder, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

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
        continue

    x = data[:, 0]
    y = data[:, 1]
    power_val = extract_power(filename)
    powers.append(power_val)

    # Plot spectrum with vertical offset
    ax.plot(x, y + i * offset, label=f"{power_val:.1f} µW", c=colors[::-1][i])

    # Integration around each peak (±1.8 nm)
    for wv in wvs:
        mask = (x >= wv - 1.8) & (x <= wv + 1.8)
        area = np.trapz(y[mask], x[mask])
        integrated_intensities[wv].append(area)
        ax.fill_between(x[mask], y[mask] + i * offset, i * offset,
                        alpha=0.6, color='wheat')

# Label peaks on the highest power spectrum (i=0)
for wv, label in zip(wvs, peak_labels):
    ax.text(wv, 1.2e4, label, ha='center', va='bottom',
            fontsize=12, weight='bold', color='saddlebrown')

ax.set_xlabel("Longitud de onda (nm)", fontsize=15, weight='bold')
ax.set_ylabel("Intensidad (un. arb.)", fontsize=15, weight='bold')
ax.legend(title="Potencia de excitación", fontsize=13, loc='upper left',
          edgecolor=edge_color, facecolor=face_color)
ax.tick_params(axis='both', labelsize=15)

# Top axis: Energy in eV
def nm_to_ev_ticks(x): return wavelength_to_eV(x)
def ev_to_nm_ticks(x): return (const.h * const.c) / (x * const.e) * 1e9

secax = ax.secondary_xaxis('top', functions=(nm_to_ev_ticks, ev_to_nm_ticks))
secax.set_xlabel('Energía (eV)', fontsize=15, weight='bold')
secax.tick_params(axis='x', labelsize=15)

plt.tight_layout()
plt.show()

# === Log-Log Fit Excluding First Three Powers ===

powers = np.array(powers)
log_powers = np.log10(powers)

plt.figure(figsize=(6, 6))
ax2 = plt.gca()
ax2.set_facecolor(face_color)
for spine in ax2.spines.values():
    spine.set_edgecolor('k')
    spine.set_linewidth(2)

vertical_offset = 1.2  # Visual y-axis offset

for i, (wv, label) in enumerate(zip(wvs, peak_labels)):
    intensities = np.array(integrated_intensities[wv])
    log_intensities = np.log10(intensities)

    # Exclude first three points (low power region)
    x_fit = log_powers[3:]
    y_fit = log_intensities[3:]

    # Fit model: log I = α * log P + log A
    def linear_model(x, alpha, logA):
        return alpha * x + logA

    popt, pcov = curve_fit(linear_model, x_fit, y_fit)
    alpha, logA = popt
    alpha_err = np.sqrt(np.diag(pcov))[0]

    x_plot = np.linspace(log_powers.min(), log_powers.max(), 100)
    y_plot = linear_model(x_plot, *popt)

    # Offset each curve for better visibility
    y_offset = i * vertical_offset

    # Excluded points (hollow markers)
    ax2.plot(log_powers[:3], log_intensities[:3] + y_offset, 'o',
             markerfacecolor='none', markeredgecolor=colors[i])

    # Fitted points (solid)
    ax2.plot(log_powers[3:], log_intensities[3:] + y_offset, 'o',
             color=colors[i],
             label=f"{label}: α = {alpha:.2f} ± {alpha_err:.2f}")

    # Fitted line
    ax2.plot(x_plot, y_plot + y_offset, '-', color=colors[i])

ax2.set_xlabel("log(Potencia (µW))", fontsize=15, weight='bold')
ax2.set_ylabel("log(Intensidad integrada (un. arb.))", fontsize=15, weight='bold')
ax2.legend(fontsize=12, loc='upper left', edgecolor=edge_color, facecolor=face_color)
ax2.tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()
