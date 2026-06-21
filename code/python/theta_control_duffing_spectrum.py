"""
THETA-CONTROL: DUFFING OSCILLATOR - COMPLETE
Generates: Time Series, Phase Space, Poincaré Maps, and Frequency Spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. PARAMETERS OF THE DUFFING SYSTEM
# ============================================================

epsilon = 0.2
A_force = 0.3
omega_force = 1.0
K_theta = 2.0
tau = 0.05

dt = 0.001
T_sim = 2000.0
t_transient = 200.0

# ============================================================
# 2. FUNCTIONS FOR SIMULATION
# ============================================================

def simulate_duffing_with_theta(K_theta_val, T_sim, dt_sim, 
                                x0=[0.5, 0.5], tau_val=tau):
    n_steps = int(T_sim / dt_sim)
    t = np.linspace(0, T_sim, n_steps)
    
    x = np.array(x0)
    y = np.zeros((n_steps, 2))
    y[0] = x.copy()
    
    buffer_size = int(tau_val / dt_sim) + 2
    buffer = np.zeros((buffer_size, 2))
    buffer[0] = x.copy()
    idx = 0
    tau_steps = max(1, int(tau_val / dt_sim))
    
    for i in range(1, n_steps):
        past_idx = (idx - tau_steps) % buffer_size
        x_past = buffer[past_idx]
        
        diff = x - x_past
        theta = K_theta_val * diff[0]
        
        force = A_force * np.cos(omega_force * t[i])
        x1, x2 = x
        dx1 = x2
        dx2 = x1 - x1**3 - epsilon * x2 + force + theta
        
        x = x + np.array([dx1, dx2]) * dt_sim
        
        idx = (idx + 1) % buffer_size
        buffer[idx] = x.copy()
        y[i] = x.copy()
    
    return t, y

def simulate_duffing_uncontrolled(T_sim, dt_sim, x0=[0.5, 0.5]):
    n_steps = int(T_sim / dt_sim)
    t = np.linspace(0, T_sim, n_steps)
    x = np.array(x0)
    y = np.zeros((n_steps, 2))
    y[0] = x.copy()
    
    for i in range(1, n_steps):
        x1, x2 = x
        force = A_force * np.cos(omega_force * t[i])
        dx1 = x2
        dx2 = x1 - x1**3 - epsilon * x2 + force
        x = x + np.array([dx1, dx2]) * dt_sim
        y[i] = x.copy()
    
    return t, y

# ============================================================
# 3. POINCARÉ MAP
# ============================================================

def poincare_map_by_cos_zero(y, t, dt, omega_force, t_transient):
    n_steps = len(y)
    points = []
    idx_transient = int(t_transient / dt)
    
    cos_prev = np.cos(omega_force * t[0])
    
    for i in range(1, n_steps):
        cos_curr = np.cos(omega_force * t[i])
        
        if cos_prev < 0 and cos_curr > 0:
            if i > idx_transient:
                t_cross = t[i-1] + (t[i] - t[i-1]) * (-cos_prev) / (cos_curr - cos_prev)
                
                x1_cross = y[i-1, 0] + (y[i, 0] - y[i-1, 0]) * (t_cross - t[i-1]) / (t[i] - t[i-1])
                x2_cross = y[i-1, 1] + (y[i, 1] - y[i-1, 1]) * (t_cross - t[i-1]) / (t[i] - t[i-1])
                
                points.append([x1_cross, x2_cross])
        
        cos_prev = cos_curr
    
    return np.array(points)

# ============================================================
# 4. FREQUENCY SPECTRUM
# ============================================================

def plot_duffing_spectrum_zoomed(y_chaotic, y_controlled, t, dt, t_transient, omega_force,
                                   save_name='fig_duffing_spectrum_zoomed'):
    """
    Frequency spectrum with a limited vertical scale to visualize both regimes
    """
    
    print(f"\n  Generating a zoom spectrum: {save_name}")
    
    idx_transient = int(t_transient / dt)
    
    # Extract steady regimes
    y_chaotic_steady = y_chaotic[idx_transient:, 0]
    y_controlled_steady = y_controlled[idx_transient:, 0]
    
    # Remove trend
    y_chaotic_detrended = y_chaotic_steady - np.mean(y_chaotic_steady)
    y_controlled_detrended = y_controlled_steady - np.mean(y_controlled_steady)
    
    # Hann's window
    window = windows.hann(len(y_chaotic_detrended))
    y_chaotic_windowed = y_chaotic_detrended * window
    y_controlled_windowed = y_controlled_detrended * window
    
    # FFT
    n = len(y_chaotic_windowed)
    freqs = fftfreq(n, dt)
    
    fft_chaotic = fft(y_chaotic_windowed)
    fft_controlled = fft(y_controlled_windowed)
    
    # Normalized magnitude
    mag_chaotic = np.abs(fft_chaotic) * 2 / n
    mag_controlled = np.abs(fft_controlled) * 2 / n
    
    # --- FIGURE 1: Wide-range scale (0–2 Hz) with vertical limit ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Mask for 0–2 Hz
    mask = (freqs >= 0) & (freqs <= 2.0)
    freqs_plot = freqs[mask]
    mag_chaotic_plot = mag_chaotic[mask]
    mag_controlled_plot = mag_controlled[mask]
    
    # Plot with boundaries
    ax.plot(freqs_plot, mag_chaotic_plot, 
            label='Uncontrolled (chaotic)', color='#1f77b4', linewidth=1.8, alpha=0.8)
    
    ax.plot(freqs_plot, mag_controlled_plot, 
            label=r'$\Theta$-Control (periodic)', color='#d62728', linewidth=2.2, alpha=0.9)
    
    # Fundamental frequency
    f_drive = omega_force / (2 * np.pi)
    ax.axvline(x=f_drive, color='gray', linestyle='--', 
               alpha=0.5, linewidth=1.5, label=f'Driving: {f_drive:.3f} Hz')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Magnitude', fontsize=13)
    ax.set_title('Frequency Spectrum of Duffing Oscillator', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, 2.0])
    
    # ADJUSTED VERTICAL LIMIT - clip the peak to see the chaos
    max_val = np.max(mag_controlled_plot)
    if max_val > 0:
        ax.set_ylim([0, max_val * 0.15])  # Shows only 15% of the periodic peak
    
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_name}.png/.pdf (zoom vertical 15%)")
    
    # --- FIGURE 2: Log scale to see both ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Add a small epsilon to avoid log(0)
    eps = 1e-10
    mag_chaotic_log = np.maximum(mag_chaotic_plot, eps)
    mag_controlled_log = np.maximum(mag_controlled_plot, eps)
    
    ax.semilogy(freqs_plot, mag_chaotic_log, 
                label='Uncontrolled (chaotic)', color='#1f77b4', linewidth=1.8, alpha=0.8)
    
    ax.semilogy(freqs_plot, mag_controlled_log, 
                label=r'$\Theta$-Control (periodic)', color='#d62728', linewidth=2.2, alpha=0.9)
    
    ax.axvline(x=f_drive, color='gray', linestyle='--', 
               alpha=0.5, linewidth=1.5, label=f'Driving: {f_drive:.3f} Hz')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Magnitude (log scale)', fontsize=13)
    ax.set_title('Frequency Spectrum (Log Scale)', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2, which='both')
    ax.set_xlim([0, 2.0])
    
    plt.tight_layout()
    plt.savefig(f'{save_name}_log.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_name}_log.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_name}_log.png/.pdf (log scale)")
    
    return freqs_plot, mag_chaotic_plot, mag_controlled_plot

# ============================================================
# 5. RUN SIMULATIONS
# ============================================================

print("="*70)
print("DUFFING OSCILLATOR - COMPLETE SIMULATION")
print("="*70)

print(f"\nTTotal time: {T_sim} s")
print(f"Transient discarded: {t_transient} s")

print("\nRunning simulations...")
t_uncontrolled, y_uncontrolled = simulate_duffing_uncontrolled(T_sim, dt)
t_controlled, y_controlled = simulate_duffing_with_theta(K_theta, T_sim, dt)
print("Simulations completed!")

# ============================================================
# 6. POINCARÉ MAPS
# ============================================================

print("\nCreating Poincaré maps...")
poincare_chaotic = poincare_map_by_cos_zero(y_uncontrolled, t_uncontrolled, dt, omega_force, t_transient)
poincare_periodic = poincare_map_by_cos_zero(y_controlled, t_controlled, dt, omega_force, t_transient)

if len(poincare_periodic) > 0:
    mean_x1 = np.mean(poincare_periodic[:, 0])
    mean_x2 = np.mean(poincare_periodic[:, 1])
    print(f"  Fixed point: ({mean_x1:.4f}, {mean_x2:.4f})")

print(f"  Poincaré (chaotic): {len(poincare_chaotic)} points")
print(f"  Poincaré (periodic): {len(poincare_periodic)} points")

# ============================================================
# 7. GENERATE FREQUENCY SPECTRUM
# ============================================================

print("\n" + "="*70)
print("GENERATING FREQUENCY SPECTRUM...")
print("="*70)

freqs, mag_chaotic, mag_controlled = plot_duffing_spectrum_zoomed(
    y_uncontrolled, y_controlled, 
    t_uncontrolled, dt, t_transient, omega_force,
    save_name='fig_duffing_spectrum_final'
)

# ============================================================
# 8. CHECK THE GENERATED FILES
# ============================================================

print("\n" + "="*70)
print("CHECKING GENERATED FILES...")
print("="*70)

arquivos_esperados = [
    'fig_duffing_spectrum_final.png', 
    'fig_duffing_spectrum_final.pdf',
    'fig_duffing_spectrum_final_log.png',
    'fig_duffing_spectrum_final_log.pdf'
]

for file in arquivos_esperados:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✓ {file} existe (size: {size} bytes)")
    else:
        print(f"  ✗ {file} NOT FOUND!")

print("\n" + "="*70)
print("PROCESS COMPLETED!")
print("="*70)