"""
THETA-CONTROL: Stabilization of Chaos in the Duffing Oscillator
POINCARÉ MAP
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. PARAMETERS OF THE DUFFING SYSTEM
# ============================================================

epsilon = 0.2
A_force = 0.3
omega_force = 1.0

# Θ-Control
K_theta = 2.0
tau = 0.05

# SIMULATION
dt = 0.001
T_sim = 2000.0
t_transient = 200.0

# ============================================================
# 2. SYSTEM FUNCTIONS
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
# 3. POINCARÉ MAP - CROSSING BY cos(omega*t) = 0 (RISE)
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
# 4. RUN SIMULATIONS
# ============================================================

print("="*70)
print("DUFFING OSCILLATOR - POINCARÉ MAP")
print("="*70)

print("\nRunning simulations...")
t_uncontrolled, y_uncontrolled = simulate_duffing_uncontrolled(T_sim, dt)
t_controlled, y_controlled = simulate_duffing_with_theta(K_theta, T_sim, dt)
print("Simulations completed!")

# ============================================================
# 5. CONSTRUCTING POINCARÉ MAPS
# ============================================================

print("\nConstructing Poincaré maps...")
poincare_uncontrolled = poincare_map_by_cos_zero(y_uncontrolled, t_uncontrolled, dt, omega_force, t_transient)
poincare_controlled = poincare_map_by_cos_zero(y_controlled, t_controlled, dt, omega_force, t_transient)

print(f"  Poincaré (chaotic): {len(poincare_uncontrolled):.0f} points")
print(f"  Poincaré (periodic): {len(poincare_controlled):.0f} points")

# ============================================================
# 6. FIXED-POINT ANALYSIS
# ============================================================

if len(poincare_controlled) > 0:
    mean_x1 = np.mean(poincare_controlled[:, 0])
    mean_x2 = np.mean(poincare_controlled[:, 1])
    std_x1 = np.std(poincare_controlled[:, 0])
    std_x2 = np.std(poincare_controlled[:, 1])
    
    print(f"\nFixed point (period 1):")
    print(f"  x1 = {mean_x1:.6f} ± {std_x1:.6f}")
    print(f"  x2 = {mean_x2:.6f} ± {std_x2:.6f}")

# ============================================================
# 7. CONFIGURATION OF STYLE
# ============================================================

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
})

# ============================================================
# 8. FIGURE: POINCARÉ MAPS
# ============================================================

print("\nGenerating figure...")

# Set the same scale for both charts
x1_lim = 1.5
x2_lim = 2.5

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Chaotic - Smale horseshoe
ax = axes[0]
ax.scatter(poincare_uncontrolled[:, 0], poincare_uncontrolled[:, 1], 
           s=3, c='#1f77b4', alpha=0.25, marker='.')
ax.set_xlabel('Position $x_1$', fontsize=13)
ax.set_ylabel('Velocity $x_2$', fontsize=13)
ax.set_title(f'(a) Poincaré Map - Uncontrolled ({len(poincare_uncontrolled):.0f} points)', fontsize=14)
ax.grid(True, alpha=0.2)
ax.set_aspect('equal')
ax.set_xlim([-x1_lim, x1_lim])
ax.set_ylim([-x2_lim, x2_lim])

# (b) Periodic—same scale (highlighted point)
ax = axes[1]

# Fixed-Point (s=100)
ax.scatter(poincare_controlled[:, 0], poincare_controlled[:, 1], 
           s=100,
           c='#d62728', 
           alpha=0.9, 
           marker='o', 
           edgecolors='black', 
           linewidth=1.5,
           zorder=10)

ax.set_xlabel('Position $x_1$', fontsize=13)
ax.set_ylabel('Velocity $x_2$', fontsize=13)
ax.set_title(f'(b) Poincaré Map - Θ-Control ({len(poincare_controlled):.0f} points)', fontsize=14)
ax.grid(True, alpha=0.2)
ax.set_aspect('equal')
ax.set_xlim([-x1_lim, x1_lim])
ax.set_ylim([-x2_lim, x2_lim])

# Reference Lines
ax.axhline(y=mean_x2, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.axvline(x=mean_x1, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

# --- FIXED POINT LEGEND ---
# Position the legend farther away from the point to avoid overlap
# The point is at (-0.7563, 1.3415)

offset_x1 = 0.5 #-0.25
offset_x2 = -0.45

ax.text(mean_x1 + offset_x1, mean_x2 + offset_x2, 
        f'Fixed point:\n({mean_x1:.4f}, {mean_x2:.4f})', 
        fontsize=11, 
        bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor='gray'),
        ha='center', 
        va='center')

plt.tight_layout()
plt.savefig('fig_duffing_poincare_final.png', dpi=300, bbox_inches='tight')
plt.savefig('fig_duffing_poincare_final.pdf', bbox_inches='tight')
print("  ✓ fig_duffing_poincare_final.png/.pdf")

# ============================================================
# 9. RESULTS
# ============================================================

print("\n" + "="*70)
print("RESULTS - DUFFING OSCILLATOR")
print("="*70)

print(f"\nPoincaré (chaotic):  {len(poincare_uncontrolled):.0f} points")
print(f"Poincaré (periodic): {len(poincare_controlled):.0f} points")

if len(poincare_controlled) > 0:
    print(f"\nFixed point (period 1):")
    print(f"  (x1, x2) = ({mean_x1:.6f}, {mean_x2:.6f})")
    print(f"  Standard deviation: (±{std_x1:.8f}, ±{std_x2:.8f})")

print("\n" + "="*70)
print("GENERATED FIGURE:")
print("  ✓ fig_duffing_poincare_final.png/.pdf")
print("="*70)

plt.show()