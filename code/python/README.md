# Θ-Control: Code Documentation

This folder contains the Python scripts used to generate all numerical results and figures in the paper.

## Files

- `theta_control_linear.py` : Simulates the second-order linear system with PID and Θ-Control.
- `theta_control_duffing.py` : Simulates the forced Duffing oscillator with and without Θ-Control.
- `generate_all_figures.py` : Generates all figures used in the paper.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
Run the linear system simulation:
```bash
python theta_control_linear.py
```

Run the Duffing oscillator simulation:
```bash
python theta_control_duffing_time_phase.py
python theta_control_duffing_poincare.py
python theta_control_duffing_spectrum.py
```

## Comments
The 4 python files mentioned above auto generate all figures. All code is written in English and commented for clarity. The simulations are fully reproducible.