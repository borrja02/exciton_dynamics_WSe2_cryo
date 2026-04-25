import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
import scipy.constants as const
import os

# === Physical Constants ===
h = const.Planck   # Planck constant (J·s)
c = const.c        # Speed of light (m/s)
e = const.e        # Elementary charge (C)

def nm_to_ev(wavelength_nm):
    """
    Convert wavelength from nanometers to electronvolts.
    """
    return (h * c / (wavelength_nm * 1e-9)) / e

# === Varshni Model for ODR Fitting ===
def varshni_odr(B, T):
    """
    Varshni equation: describes the temperature dependence of the semiconductor bandgap.
    B: [E0, alpha, beta] are the model parameters.
    T: temperature in Kelvin.
    Returns E(T) in eV.
    """
    E0, alpha, beta = B
    return E0 - (alpha * T**2) / (T + beta)

# === Load Experimental Excitonic Peaks Data ===
folder = r"C:\Users\Borja\Desktop\TFG\data\Grating 600 líneas por mm\Estudio temperatura"
filename = os.path.join(folder, "excitonic_peaks.txt")

temperatures = []
wavelengths = []

# Read two-column data: temperature (K) and excitonic peak wavelength (nm)
with open(filename, 'r', encoding='utf-8') as f:
    next(f)  # Skip header line
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            T = float(parts[0])
            try:
                wl = float(parts[1])
                temperatures.append(T)
                wavelengths.append(wl)
            except ValueError:
                continue  # Skip lines with invalid values

temperatures = np.array(temperatures)
wavelengths = np.array(wavelengths)
energies = nm_to_ev(wavelengths)  # Convert wavelength to energy (eV)

# === Orthogonal Distance Regression (ODR) ===
# Define measurement uncertainties
T_error = 0.5     # Temperature uncertainty in K
y_error = 0.005   # Energy uncertainty in eV

# Prepare data and model for ODR
model = Model(varshni_odr)
data = RealData(temperatures, energies, sx=T_error, sy=y_error)
odr = ODR(data, model, beta0=[1.75, 4e-4, 200])  # Initial parameter guesses
output = odr.run()

# Extract fitted parameters and standard deviations
E0, alpha, beta = output.beta
dE0, dalpha, dbeta = output.sd_beta

# Generate smooth fit curve over temperature range
T_fit = np.linspace(min(temperatures), max(temperatures), 200)
E_fit = varshni_odr([E0, alpha, beta], T_fit)

# === Manual Calculation of Coefficient of Determination (R²) ===
residuals = energies - varshni_odr([E0, alpha, beta], temperatures)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((energies - np.mean(energies))**2)
r_squared = 1 - ss_res / ss_tot

# === Stylized Plot ===
face_color, edge_color = 'thistle', 'plum'
plt.figure(figsize=(6, 5))
ax = plt.gca()
ax.set_facecolor(face_color)
for spine in ax.spines.values():
    spine.set_edgecolor('k')
    spine.set_linewidth(2)

# Plot experimental data with error bars
ax.errorbar(
    temperatures, energies, xerr=T_error, yerr=y_error,
    fmt='o', color='slateblue', ecolor='navy', capsize=3,
    label="Picos excitónicos", zorder=3, mec='k'
)

# Plot the Varshni model fit
ax.plot(
    T_fit, E_fit, '--', color='rebeccapurple', linewidth=2, alpha=0.6,
    label=r"$E(T) = E_0 - \frac{\alpha T^2}{T + \beta}$"
)

# Labels and legend
ax.set_xlabel("Temperatura (K)", fontsize=14, weight='bold')
ax.set_ylabel("Energía de enlace (eV)", fontsize=14, weight='bold')
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=15, loc='best', edgecolor=edge_color, facecolor=face_color)

plt.tight_layout()
plt.show()

# === Display Fit Results ===
print("[ODR FIT RESULTS]")
print(f"E0     = {E0:.4f} ± {dE0:.4f} eV")
print(f"alpha  = {alpha:.2e} ± {dalpha:.2e} eV/K")
print(f"beta   = {beta:.2f} ± {dbeta:.2f} K")
print(f"Coefficient of determination R² = {r_squared:.4f}")
