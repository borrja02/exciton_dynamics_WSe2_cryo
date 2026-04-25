# === Import Required Libraries ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from io import StringIO
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

# === Plotting Options (set True to activate) ===
plot_PL_map = False         # Option to display photoluminescence intensity map
plot_spec = True            # Option to display and fit the selected spectrum

# === File Paths (spectral data and coordinate data) ===
spectra_file = r'C:\Users\Borja\Desktop\TFG\data\Grating 150 lineas por mm\Mapas\plmap_19-6-2025_102720_515nm_50uW_30x30um_r05_02s_plmap_WSe2_ML_jun25_4K_Map.ASC'
coords_file = r'C:\Users\Borja\Desktop\TFG\data\Grating 150 lineas por mm\Mapas\plmap_19-6-2025_102720_515nm_50uW_30x30um_r05_02s_plmap_WSe2_ML_jun25_4K_Map_xy'

# === Helper Functions ===

def get_spectrum_at(x_query, y_query):
    """Return the spectrum at a specific (x, y) coordinate (in µm)."""
    for i, (x, y) in enumerate(coords):
        if np.isclose(x, x_query, atol=1e-6) and np.isclose(y, y_query, atol=1e-6):
            return wavelengths, spectra[:, i]
    print(f"No point found at X={x_query}, Y={y_query}")
    return None, None

def pseudo_voigt(x, amp, cen, sigma, eta):
    """Pseudo-Voigt profile: a linear combination of Gaussian and Lorentzian profiles."""
    gaussian = amp * np.exp(-((x - cen)**2) / (2 * sigma**2))
    lorentzian = amp / (1 + ((x - cen)/sigma)**2)
    return eta * lorentzian + (1 - eta) * gaussian

def fit_peaks(wavelength, intensity, p0_list, masks):
    """Fit multiple pseudo-Voigt peaks to a given spectrum using non-linear least squares."""
    results = []
    for p0, mask in zip(p0_list, masks):
        popt, _ = curve_fit(pseudo_voigt, wavelength[mask], intensity[mask], p0=p0)
        results.append(popt)
    return results

def plot_spectrum(wavelength, intensity, peak_params, masks, labels, title):
    """Plot the original spectrum with its fitted peaks and total convolution."""
    fig, ax = plt.subplots(figsize=(6, 6), clear=True)
    ax.set_facecolor('antiquewhite')
    for spine in ax.spines.values():
        spine.set_edgecolor('k')
        spine.set_linewidth(2)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('Longitud de onda (nm)', fontsize=15, weight='bold')
    ax.set_ylabel('Intensidad normalizada (un. arb.)', fontsize=15, weight='bold')
    ax.set_xlim([650, 850])

    # Plot original spectrum
    ax.plot(wavelength, intensity, color='chocolate', label='Espectro original')

    # Highlight integration regions
    integration_mask = np.any(np.column_stack(masks), axis=1)
    #ax.fill_between(wavelength[integration_mask], intensity[integration_mask], color='sienna', alpha=0.4, label='Rango de integración')

    # Plot individual fitted peaks
    cmap = plt.get_cmap('cool')
    colors = [cmap(0.2 + i * 0.6 / (len(peak_params)-1)) for i in range(len(peak_params))]
    ideal_fit = np.zeros_like(wavelength)
    for popt in peak_params:
        ideal_fit += pseudo_voigt(wavelength, *popt)

    # Convolution to simulate spectral broadening
    sigma_conv_nm = 1.5
    step_nm = np.mean(np.diff(wavelength))
    sigma_pix = sigma_conv_nm / step_nm
    convolved_fit = gaussian_filter1d(ideal_fit, sigma=sigma_pix)

    # Rescale convolution and plot
    combined_mask = np.any(np.column_stack(masks), axis=1) & (wavelength >= 710)
    wl_fit_range = wavelength[combined_mask]
    convolved_fit_range = convolved_fit[combined_mask]
    if np.max(convolved_fit_range) > 0:
        scale_factor = np.max(intensity) / np.max(convolved_fit_range)
        convolved_fit_scaled = convolved_fit_range * scale_factor
        ax.plot(wl_fit_range, convolved_fit_scaled, c='k', lw=2.5, ls='solid', label='Convolución', alpha=0.7)

    # Draw each fitted peak
    for popt, mask, color, label in zip(peak_params, masks, colors, labels):
        peak_cen = popt[1]
        label_with_cen = f"{label}: {peak_cen:.1f} nm"
        ax.plot(wavelength[integration_mask], pseudo_voigt(wavelength[integration_mask], *popt),
               c=color, lw=2, ls='dashed', alpha=0.9, label=label_with_cen)

    # Add energy axis (in eV)
    def wavelength_to_energy(wl_nm):
        h = 4.135667696e-15  # Planck's constant in eV·s
        c = 2.99792458e8     # Speed of light in m/s
        wl_m = wl_nm * 1e-9  # Convert nm to meters
        return (h * c) / wl_m

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks_nm = np.linspace(*ax.get_xlim(), num=6)
    ticks_eV = wavelength_to_energy(ticks_nm)
    ax2.set_xticks(ticks_nm)
    ax2.set_xticklabels([f"{e:.2f}" for e in ticks_eV])
    ax2.set_xlabel('Energía (eV)', fontsize=15, weight='bold')
    ax2.tick_params(labelsize=15)

    # Final layout
    ax.legend(facecolor='antiquewhite', edgecolor='tan', loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()

# === 1. Load Spectral Data (comma → dot correction included) ===
with open(spectra_file, 'r') as f:
    lines = [line.replace(',', '.') for line in f.readlines()]
data = np.loadtxt(StringIO(''.join(lines)))
wavelengths = data[:, 0]     # First column: wavelength (nm)
spectra = data[:, 1:]        # Remaining columns: intensity at each spatial point

# === 2. Load Spatial Coordinates from Map ===
with open(coords_file, 'r') as f:
    coords_raw = f.readlines()

coords = []
for line in coords_raw:
    x_match = re.search(r'X:([-+]?\d*\.\d+|\d+)', line)
    y_match = re.search(r'Y:([-+]?\d*\.\d+|\d+)', line)
    if x_match and y_match:
        coords.append((float(x_match.group(1)), float(y_match.group(1))))
coords = np.array(coords)

# === 3. Integrate Over Selected Spectral Range ===
wl_min, wl_max = 700, 775
integration_mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
intensities = np.trapz(spectra[integration_mask, :], x=wavelengths[integration_mask], axis=0)

# Normalize intensities between 0 and 1
intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

# === 4. Build PL Intensity Map ===
x_vals = np.unique(coords[:, 0])
y_vals = np.unique(coords[:, 1])
map_image = np.zeros((len(y_vals), len(x_vals)))
for i, (x, y) in enumerate(coords):
    xi = np.where(x_vals == x)[0][0]
    yi = np.where(y_vals == y)[0][0]
    map_image[yi, xi] = intensities[i]

# === 5. Plot PL Map (if activated) ===
if plot_PL_map:
    plt.figure(figsize=(6, 5))
    im = plt.imshow(map_image, extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                    origin='lower', aspect='auto', cmap='BrBG')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Intensidad normalizada (un. arb.)', fontsize=14, weight='bold')
    plt.xlabel(r'X ($\mu$m)', fontsize=14, weight='bold')
    plt.ylabel(r'Y ($\mu$m)', fontsize=14, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Add annotation for measured point
    points = {'Measured': (30, 10.2)}
    for label, (x, y) in points.items():
        plt.plot(x, y, 'v', color='gold', markersize=8, label=label)
        plt.text(x, y + .6, label, color='gold', fontsize=12, weight='bold', style='italic')
    plt.show()

# === 6. Spectrum Analysis at Specific Point ===
x_target, y_target = 30, 10.2
wavelength, intensity = get_spectrum_at(x_target, y_target)
intensity_norm = intensity / np.max(intensity)

# === 7. Define Peak Fitting Parameters ===
# Define wavelength ranges for each peak
peak1_mask = (wavelengths >= 704) & (wavelengths <= 710)
peak2_mask = (wavelengths >= 717.5) & (wavelengths <= 720.66)
peak3_mask = (wavelengths >= 724) & (wavelengths <= 730.6)
peak4_mask = (wavelengths >= 730.65) & (wavelengths <= 734.3)
peak5_mask = (wavelengths >= 735.7) & (wavelengths <= 739.5)
peak7_mask = (wavelengths >= 743.7) & (wavelengths <= 749)
peak8_mask = (wavelengths >= 754.49) & (wavelengths <= 780)
all_peak_masks = [peak1_mask, peak2_mask, peak3_mask, peak4_mask, peak5_mask, peak7_mask, peak8_mask]

# Labels for each transition
peak_labels = ['Excitón neutro', 'Trión', 'Estado localizado 1', 'Estado localizado 2',
               'Estado localizado 3', 'Estado localizado 4', 'Estado localizado 5']

# Initial parameter guesses: [amplitude, center, width, eta]
target_p0 = [
    [0.52, 705, 4, 0.5],
    [0.33, 719.9, 1, 0.5],
    [0.99, 728.7, 1, 0],
    [1.00, 732.6, 2, 0.5],
    [0.90, 736.3, 1, 0.5],
    [0.77, 743.7, 1, 0],
    [0.1, 755, 1, 0],
]

# === 8. Fit and Plot Spectrum (if enabled) ===
if plot_spec:
    target_peaks = fit_peaks(wavelength, intensity_norm, target_p0, all_peak_masks)
    plot_spectrum(wavelength, intensity_norm, target_peaks, all_peak_masks, peak_labels, title='Measured Point')
