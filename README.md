# Θ-Control: Memory as a Control Resource

**Version 2.0 (Revised) - June 2026**

This repository contains the code, data, and LaTeX source for the paper 
**"Θ-Control: Memory as a Control Resource"**.

## 📌 Important: Version 2.0 Update

This is a **revised version** of the work. Key improvements include:

- ✅ Corrected system parameters for exact consistency with `ωn = 2.00 rad/s` and `ζ = 0.600`
- ✅ Enhanced numerical simulations with longer transients
- ✅ Improved figures with clear visual comparisons
- ✅ New spectral analysis for the Duffing oscillator

**The main conclusions remain unchanged and are further strengthened by this revision.**

For the original version, see the  `v1_archive/` directory.

## 📁 Repository Structure

```text
theta-control/
├── paper/             # LaTeX source of the article
│   ├── Theta-Control_v2.tex
│   ├── references.bib
│   └── figures/                # All main figures
├── code/
│   └── python/             
│       ├── README.md
│       ├── theta_control_duffing_poincare.py
│       ├── theta_control_duffing_spectrum.py
│       └── theta_control_duffing_time_phase.py
├── CHANGELOG.md        # NEW: registro de mudanças
├── LICENSE             # MIT License
├── README.md           # This file (updated)
├── requirements.txt
└── v1_archive/ # Original version (retained for reference)
```

## 📄 Compiling the Paper

The paper is written in LaTeX and has been compiled using **TeXLive 2026**.
To reproduce the document:

1. Navigate to the `paper/` directory:
```bash
   cd paper/
```
2. Compile with pdflatex and bibtex:
```bash
   pdflatex Theta-Control_v2.tex
   bibtex references
   pdflatex Theta-Control_v2.tex
   pdflatex Theta-Control_v2.tex
```
3. The output will be Theta-Control_v2.pdf


## 🚀 Run Simulations and Generate Figures

### Install Python Dependencies
```bash
pip install -r requirements.txt
```
### Run the linear system simulation:
```bash
python theta_control_linear.py
```
### Run the Duffing oscillator simulation:
```bash
python theta_control_duffing_time_phase.py
python theta_control_duffing_poincare.py
python theta_control_duffing_spectrum.py
```

## 📄 Citation:
Please cite the **latest version** (v2.0) available on Zenodo:

```bibtex
@misc{madrid2026thetacontrol,
  author = {Madrid, Marconi Kolm},
  title = {$\Theta$-Control: Memory as a Control Resource},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.20739704},
  note = {Version 2.0}
}
```
> Madrid, M. K. (2026). Θ-Control: Memory as a Control Resource. Zenodo. https://doi.org/10.5281/zenodo.20739704

## 📜 License: MIT