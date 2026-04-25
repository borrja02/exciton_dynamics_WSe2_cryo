# Excitonic Dynamics in WSe2 Monolayers: A Spectroscopic Investigation

This repository documents the experimental and analytical workflow of my Final Physics Degree Project (TFG), centered on the study of Transition Metal Dichalcogenide (TMD) monolayers. The research focuses on identifying and characterizing excitonic landscapes in Tungsten Diselenide ($WSe_2$) via micro-photoluminescence ($\mu$-PL) spectroscopy conducted at cryogenic temperatures (~4 K).

## Repository Overview

The content is organized to reflect the dual nature of the project: the development of custom instrumentation and the subsequent physical analysis of nanomaterials.

### 1. Data Acquisition and Control (`src/`)
The software suite highlights the automation and hardware-software interfacing developed for the laboratory:
* **`monochromator.py`**: A multi-threaded GUI application developed for the real-time control of a custom-built spectrometer, managing instrument synchronization and data logging via Arduino and serial communication.

### 2. Physical Characterization and Modeling (`src/`)
These scripts were utilized to extract physical parameters from the experimental datasets:
* **`temp_study.py` & `varshni_fit.py`**: Implementation of the Varshni model to characterize the thermal dependence of the semiconductor band-gap, utilizing Orthogonal Distance Regression (ODR) for robust parameter estimation.
* **`power_study.py`**: Investigation of excitonic species through power-law scaling analysis, enabling the differentiation between neutral excitons, charged trions, and defect-bound localized states.
* **`PL_map.py`**: An analytical tool for spatial mapping of photoluminescence intensity, used to determine monolayer quality and identify localized emission regions across the substrate.

### 3. Documentation and Datasets
* **`data/`**: Experimental spectra and integrated intensity datasets obtained during the measurement campaigns. (Note: Large-scale high-resolution maps are archived separately due to storage constraints).
* **`memory.pdf` & `annexes.pdf`**: The complete academic thesis and its corresponding supplementary information, detailing the theoretical framework and the mechanical exfoliation/dry-transfer fabrication protocols.

## Methodology

The research involved the mechanical exfoliation of $WSe_2$ bulk crystals and their subsequent deterministic transfer onto PDMS-based architectures. Optical measurements were performed in a liquid helium cryostat, where the high spectral resolution achieved permitted the observation of fine structural transitions and the validation of environmental modulation on the material's electronic properties.

**Author:** Borja Álvarez Reguera  
**Academic Institution:** University of Oviedo (UniOvi)  
**Date:** July 2025
