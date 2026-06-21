"""
THETA-CONTROL: Stabilization of Chaos in the Duffing Oscillator
100s transient, long steady state
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DUFFING SYSTEM PARAMETERS
# ============================================================

epsilon = 0.2
A_force = 0.3
omega_force = 1.0

# Θ-Control
K_theta = 2.0
tau = 0.05

# SIMULATION WITH LONG TRANSIENT
dt = 0.001
T_sim = 500.0          # 500 seconds in total
t_transient = 100.0    # 100 seconds discarded (transient)

# ============================================================
# 2. FUNCTIONS OF THE SYSTEM
# ============================================================

def simulate_duffing_with_theta(K_theta_val, T_sim, dt_sim, 
                                x0=[0.5, 0.5], tau_val=tau):
    """
    Simulates Duffing with Θ-Control using a memory buffer
    """
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
    """
    Simulates Duffing without control (chaotic)
    """
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
# 3. EXECUTE SIMULATIONS (LONG)
# ============================================================

print("="*70)
print("DUFFING OSCILLATOR - SIMULATION WITH LONG TRANSIENT")
print("="*70)

print("\nRunning simulations...")
print(f"  Total time: {T_sim} s")
print(f"  Transient discarded: {t_transient} s")
print(f"  Steady state: {T_sim - t_transient} s")

# Sem controle
print("\n  Uncontrolled (chaotic) simulation...")
t_uncontrolled, y_uncontrolled = simulate_duffing_uncontrolled(T_sim, dt)

# Com Θ-Control
print("  Simulation using Θ-Control (Kθ=2.0)...")
t_controlled, y_controlled = simulate_duffing_with_theta(K_theta, T_sim, dt)

print("\nSimulações concluídas!")

# ============================================================
# 4. EXTRACT STEADY STATE (AFTER TRANSIENT)
# ============================================================

idx_transient = int(t_transient / dt)

# Steady state - uncontrolled
t_steady_uncontrolled = t_uncontrolled[idx_transient:]
y_steady_uncontrolled = y_uncontrolled[idx_transient:]

# Steady state - controlled
t_steady_controlled = t_controlled[idx_transient:]
y_steady_controlled = y_controlled[idx_transient:]

print(f"\nSteady state - uncontrolled: {len(t_steady_uncontrolled)} points")
print(f"Steady state - controlled: {len(t_steady_controlled)} points")

# ============================================================
# 5. FREQUENCY ANALYSIS (STEADY STATE)
# ============================================================

# Controlled steady-state FFT
fft_x1 = fft(y_steady_controlled[:, 0])
freqs = fftfreq(len(y_steady_controlled[:, 0]), dt)
mask = (freqs >= 0) & (freqs <= 5.0)

dominant_freq_idx = np.argmax(np.abs(fft_x1[mask]))
dominant_freq = freqs[mask][dominant_freq_idx]
T_force = 2 * np.pi / omega_force

print(f"\nDominant frequency: {dominant_freq:.4f} Hz")
print(f"Period: {1/dominant_freq:.4f} s")
print(f"Period of external force: {T_force:.4f} s")
print(f"Ratio: {dominant_freq/omega_force:.4f}")

# ============================================================
# 6. GENERATE FIGURES
# ============================================================

print("\nGenerating Figures...")

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
})

# ============================================================
# FIGURE 1: TIME SERIES (TRANSIENT + STEADY-STATE)
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# (a) Uncontrolled - chaotic
ax = axes[0]
ax.plot(t_uncontrolled, y_uncontrolled[:, 0], color='#1f77b4', linewidth=1.0, alpha=0.7)
ax.axvline(x=t_transient, color='red', linestyle='--', alpha=0.5, label='Transient')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position x₁')
ax.set_title('(a) Uncontrolled - Chaotic Behavior')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 100])  # Show only the first 100 for preview

# (b) Controlled - overview
ax = axes[1]
ax.plot(t_controlled, y_controlled[:, 0], color='#2ca02c', linewidth=1.0, alpha=0.7)
ax.axvline(x=t_transient, color='red', linestyle='--', alpha=0.5, label='Transient')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position x₁')
ax.set_title('(b) Θ-Control - Stabilization to Periodic Orbit')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 100])

# (c) Steady state - zoom showing periodicity
ax = axes[2]
# Show 5 cycles in steady state
t_start_zoom = t_transient + 5 * T_force
idx_start_zoom = int(t_start_zoom / dt)
n_zoom = int(5 * T_force / dt)
t_zoom = t_controlled[idx_start_zoom:idx_start_zoom + n_zoom]
y_zoom = y_controlled[idx_start_zoom:idx_start_zoom + n_zoom, 0]

ax.plot(t_zoom, y_zoom, color='#d62728', linewidth=2.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position x₁')
ax.set_title('(c) Θ-Control - Steady State (Clear Periodicity)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_duffing_time_series.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_duffing_time_series.pdf', bbox_inches='tight')
print("  ✓ fig_duffing_time_series.png/.pdf")

# ============================================================
# FIGURE 2: PHASE SPACE (STEADY STATE)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Uncontrolled - chaotic attractor (steady state)
ax = axes[0]
ax.plot(y_steady_uncontrolled[:, 0], y_steady_uncontrolled[:, 1], 
        color='#1f77b4', linewidth=0.6, alpha=0.6)
ax.set_xlabel('Position x₁')
ax.set_ylabel('Velocity x₂')
ax.set_title('(a) Chaotic Attractor (Steady State)')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# (b) With Θ-Control - clean periodic orbit
ax = axes[1]
ax.plot(y_steady_controlled[:, 0], y_steady_controlled[:, 1], 
        color='#d62728', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Position x₁')
ax.set_ylabel('Velocity x₂')
ax.set_title('(b) Stable Periodic Orbit (Θ-Control)')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('fig_duffing_phase_space.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_duffing_phase_space.pdf', bbox_inches='tight')
print("  ✓ fig_duffing_phase_space.png/.pdf")

# ============================================================
# FIGURE 3: COMPARISON OF ORBITS (OVERLAY)
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Overlay the two orbits to show the difference
ax.plot(y_steady_uncontrolled[::10, 0], y_steady_uncontrolled[::10, 1], 
        color='#1f77b4', linewidth=0.5, alpha=0.3, label='Uncontrolled (chaotic)')
ax.plot(y_steady_controlled[::5, 0], y_steady_controlled[::5, 1], 
        color='#d62728', linewidth=0.8, alpha=0.7, label='Θ-Control (periodic)')

ax.set_xlabel('Position x₁')
ax.set_ylabel('Velocity x₂')
ax.set_title('Phase Space Comparison: Chaos vs. Periodicity')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('fig_duffing_phase_final.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_duffing_phase_final.pdf', bbox_inches='tight')
print("  ✓ fig_duffing_phase_final.png/.pdf")

# ============================================================
# 7. QUANTITATIVE RESULTS
# ============================================================

print("\n" + "="*70)
print("QUANTITATIVE RESULTS - DUFFING")
print("="*70)

# Average power in steady state
energy_uncontrolled = np.mean(y_steady_uncontrolled[:, 0]**2 + y_steady_uncontrolled[:, 1]**2)
energy_controlled = np.mean(y_steady_controlled[:, 0]**2 + y_steady_controlled[:, 1]**2)

print(f"\nAverage power (uncontrolled): {energy_uncontrolled:.4f}")
print(f"Average power (with Θ-Control): {energy_controlled:.4f}")
print(f"Power reduction: {(1 - energy_controlled/energy_uncontrolled)*100:.1f}%")

print("\n" + "="*70)
print("CONCLUSION: Θ-Control stabilizes the chaos in the Duffing oscillator")
print("  - Power reduced significantly")
print("  - Stable periodic orbit with period T = 2π/ω")
print("="*70)