"""
reproduce_figures.py - Generate all figures from the paper

This script reproduces all figures for the Theta-Control paper:
1. main_results.png - Step response (PID vs Theta)
2. error_comparison.png - Tracking error
3. theta_signals.png - Theta signal evolution
4. spectra_comparison.png - Frequency spectrum
5. control_signals.png - Control signals comparison
6. step_response_zoom.png - Zoomed step response (0-10s)
7. duffing_timeseries.png - Duffing stabilization
8. duffing_phasespace.png - Duffing phase space
9. duffing_theta.png - Theta signal for Duffing
10. duffing_spectrum.png - Duffing frequency spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# Create figures directory
os.makedirs('../figures', exist_ok=True)

# ============================================
# SYSTEM DEFINITION
# ============================================

A = np.array([[0, 1], [-4, -0.4]])
B = np.array([[0], [1]])
K = np.array([[6, 2]])

Acl = A - B @ K
BK = B @ K

dt = 0.001
T = 20.0
step_time = 5.0
tau = 0.05
buffer_size = int(tau / dt)

# Pre-compensation gain for zero steady-state error
C = np.array([1, 0])
N_ref = -1.0 / (C @ np.linalg.inv(Acl) @ B)[0]


# ============================================
# THETA-CONTROL SIMULATION
# ============================================

def simulate_theta(K_scale):
    """Simulate system with Theta-Control"""
    N = int(T / dt)
    t = np.arange(0, T, dt)
    
    x = np.zeros((N, 2))
    y = np.zeros(N)
    theta_hist = np.zeros(N)
    u_hist = np.zeros(N)
    
    memory = []
    x[0] = [0.0, 0.0]
    
    for i in range(N-1):
        t_i = t[i]
        ref = 1.0 if t_i > step_time else 0.0
        x_curr = x[i]
        
        # Theta calculation
        if len(memory) >= buffer_size:
            x_past = memory[-buffer_size]
            delta = x_curr - x_past
            theta = K_scale * (BK @ delta)[1]
            theta_hist[i] = theta
        else:
            theta = 0.0
        
        memory.append(x_curr.copy())
        if len(memory) > buffer_size * 2:
            memory = memory[-buffer_size:]
        
        # Control law: u = -K*x + N_ref*ref + Theta
        u = -K @ x_curr + N_ref * ref + theta
        u = u[0]
        u_hist[i] = u
        
        dx = A @ x_curr + B.flatten() * u
        x[i+1] = x_curr + dt * dx
        y[i] = x_curr[0]
    
    y[-1] = x[-1, 0]
    ref_signal = np.array([1.0 if ti > step_time else 0.0 for ti in t])
    error = ref_signal - y
    
    return t, y, theta_hist, error, ref_signal, u_hist


# ============================================
# PID SIMULATION
# ============================================

def simulate_pid(Kp, Ki, Kd):
    """Simulate system with PID controller"""
    N = int(T / dt)
    t = np.arange(0, T, dt)
    
    x = np.zeros((N, 2))
    y = np.zeros(N)
    u = np.zeros(N)
    
    integral = 0.0
    prev_error = 0.0
    x[0] = [0.0, 0.0]
    
    for i in range(N-1):
        t_i = t[i]
        ref = 1.0 if t_i > step_time else 0.0
        error = ref - x[i, 0]
        
        P = Kp * error
        integral += error * dt
        I = Ki * integral
        D = Kd * (error - prev_error) / dt
        prev_error = error
        
        u[i] = P + I + D
        
        if u[i] > 50:
            u[i] = 50
        if u[i] < -50:
            u[i] = -50
        
        dx = A @ x[i] + B.flatten() * u[i]
        x[i+1] = x[i] + dt * dx
        y[i] = x[i, 0]
    
    y[-1] = x[-1, 0]
    ref_signal = np.array([1.0 if ti > step_time else 0.0 for ti in t])
    error = ref_signal - y
    
    return t, y, u, error, ref_signal


# ============================================
# DUFFING SIMULATION
# ============================================

class DuffingThetaController:
    """Theta-Control for Duffing oscillator"""
    
    def __init__(self, K1, K2, K_scale, tau, dt):
        self.K1 = K1
        self.K2 = K2
        self.K_scale = K_scale
        self.tau = tau
        self.dt = dt
        self.buffer_size = max(1, int(tau / dt))
        self.memory = []
        
    def compute_theta(self, x_current):
        if len(self.memory) >= self.buffer_size:
            x_past = self.memory[-self.buffer_size]
            theta = self.K_scale * (x_current[0] - x_past[0])
        else:
            theta = 0.0
        return theta
    
    def update_memory(self, x_current):
        self.memory.append(x_current.copy())
        if len(self.memory) > self.buffer_size * 2:
            self.memory = self.memory[-self.buffer_size:]
    
    def compute_control(self, x_current):
        u_fb = -self.K1 * x_current[0] - self.K2 * x_current[1]
        theta = self.compute_theta(x_current)
        return u_fb + theta
    
    def reset(self):
        self.memory = []


def duffing_dynamics(x, t, u=0.0, epsilon=0.2, A_forcing=0.3, omega=1.0):
    x1, x2 = x
    dx1 = x2
    dx2 = x1 - x1**3 - epsilon*x2 + A_forcing * np.cos(omega * t) + u
    return np.array([dx1, dx2])


def simulate_duffing(controller, dt=0.001, T=100.0, x0=None):
    if x0 is None:
        x0 = np.array([0.5, 0.5])
    
    N = int(T / dt)
    t = np.arange(0, T, dt)
    
    x = np.zeros((N, 2))
    u = np.zeros(N)
    theta = np.zeros(N) if controller is not None else None
    
    x[0] = x0
    
    if controller is not None:
        controller.reset()
    
    for i in range(N-1):
        t_i = t[i]
        x_curr = x[i]
        
        if controller is not None:
            u[i] = controller.compute_control(x_curr)
            theta[i] = controller.compute_theta(x_curr)
            controller.update_memory(x_curr)
        
        dx = duffing_dynamics(x_curr, t_i, u[i])
        x[i+1] = x_curr + dt * dx
    
    return t, x, u, theta


# ============================================
# RUN ALL SIMULATIONS
# ============================================

print("="*70)
print("THETA-CONTROL: REPRODUCING PAPER FIGURES")
print("="*70)

print("\n[1/4] Running PID simulation...")
t_pid, y_pid, u_pid, e_pid, ref_pid = simulate_pid(Kp=8.0, Ki=2.0, Kd=0.5)

print("[2/4] Running Theta-Control (K=1.0)...")
t_th1, y_th1, th1, e_th1, ref_th1, u_th1 = simulate_theta(K_scale=1.0)

print("[3/4] Running Theta-Control (K=2.0)...")
t_th2, y_th2, th2, e_th2, ref_th2, u_th2 = simulate_theta(K_scale=2.0)

print("[4/4] Running Duffing simulations...")
controller = DuffingThetaController(K1=0.5, K2=0.2, K_scale=2.0, tau=0.05, dt=0.001)
t_duff, x_duff, u_duff, theta_duff = simulate_duffing(controller)
t_duff_no, x_duff_no, _, _ = simulate_duffing(None)


# ============================================
# METRICS (print to console)
# ============================================

def compute_metrics(y, error, t):
    idx_step = int(step_time / dt)
    ise = np.sum(error[idx_step:]**2) * dt
    iae = np.sum(np.abs(error[idx_step:])) * dt
    
    y_final = np.mean(y[-2000:])
    y_step = y[idx_step:]
    t_step = t[idx_step:]
    
    y_10 = 0.1 * y_final
    y_90 = 0.9 * y_final
    idx_10 = np.where(y_step >= y_10)[0]
    idx_90 = np.where(y_step >= y_90)[0]
    rise = t_step[idx_90[0]] - t_step[idx_10[0]] if len(idx_10) > 0 and len(idx_90) > 0 else np.nan
    
    tolerance = 0.02 * y_final
    idx_settle = np.where(np.abs(y_step - y_final) < tolerance)[0]
    settle = t_step[idx_settle[0]] if len(idx_settle) > 0 else np.nan
    
    max_y = np.max(y_step)
    overshoot = max(0, (max_y - y_final) / y_final * 100)
    
    return ise, iae, rise, settle, overshoot


print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

for name, y, e in [('PID', y_pid, e_pid),
                   ('Theta (K=1.0)', y_th1, e_th1),
                   ('Theta (K=2.0)', y_th2, e_th2)]:
    ise, iae, rise, settle, overshoot = compute_metrics(y, e, t_pid)
    print(f"{name:18} ISE={ise:.4f}, IAE={iae:.4f}, Rise={rise:.2f}s, Overshoot={overshoot:.1f}%")


# ============================================
# GENERATE FIGURES
# ============================================

print("\n" + "="*70)
print("GENERATING FIGURES")
print("="*70)

# Figure 1: Step Response
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(t_pid, ref_pid, 'k--', linewidth=2, label='Reference')
ax1.plot(t_pid, y_pid, 'b-', linewidth=1.5, label='PID')
ax1.plot(t_th1, y_th1, 'g-', linewidth=1.5, label='Θ (K=1.0)')
ax1.plot(t_th2, y_th2, 'r-', linewidth=1.5, label='Θ (K=2.0)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position')
ax1.set_title('Step Response: PID vs Θ-Control')
ax1.legend(loc='lower right')
ax1.grid(True)
ax1.set_xlim(0, 12)
ax1.set_ylim(-0.1, 1.2)
plt.tight_layout()
plt.savefig('../figures/main_results.png', dpi=150)
plt.close()
print("  ✓ main_results.png")

# Figure 2: Error
fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.plot(t_pid, e_pid, 'b-', linewidth=1.0, label='PID')
ax2.plot(t_th1, e_th1, 'g-', linewidth=1.0, label='Θ (K=1.0)')
ax2.plot(t_th2, e_th2, 'r-', linewidth=1.0, label='Θ (K=2.0)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Error')
ax2.set_title('Tracking Error')
ax2.legend()
ax2.grid(True)
ax2.set_xlim(0, 12)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('../figures/error_comparison.png', dpi=150)
plt.close()
print("  ✓ error_comparison.png")

# Figure 3: Theta signals
fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(t_th1, th1, 'g-', linewidth=1.0, label='Θ (K=1.0)')
ax3.plot(t_th2, th2, 'r-', linewidth=1.0, label='Θ (K=2.0)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Θ(t)')
ax3.set_title('Θ Signal = BK[x(t)-x(t-tau)]')
ax3.legend()
ax3.grid(True)
ax3.set_xlim(0, 10)
plt.tight_layout()
plt.savefig('../figures/theta_signals.png', dpi=150)
plt.close()
print("  ✓ theta_signals.png")

# Figure 4: Spectrum
fig4, ax4 = plt.subplots(figsize=(14, 6))
Nfft = len(e_pid)
freqs = fftfreq(Nfft, dt)
mag_pid = np.abs(fft(e_pid * np.hanning(Nfft)))[:Nfft//2]
mag_th2 = np.abs(fft(e_th2 * np.hanning(Nfft)))[:Nfft//2]
freqs_pos = freqs[:Nfft//2]

ax4.plot(freqs_pos[freqs_pos <= 2], mag_pid[freqs_pos <= 2], 'b-', linewidth=1.5, label='PID')
ax4.plot(freqs_pos[freqs_pos <= 2], mag_th2[freqs_pos <= 2], 'r-', linewidth=1.5, label='Θ (K=2.0)')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Magnitude')
ax4.set_title('Error Spectrum')
ax4.legend()
ax4.grid(True)
ax4.set_xlim(0, 1.5)
plt.tight_layout()
plt.savefig('../figures/spectra_comparison.png', dpi=150)
plt.close()
print("  ✓ spectra_comparison.png")

# Figure 5: Control signals
fig5, ax5 = plt.subplots(figsize=(14, 6))
ax5.plot(t_pid, u_pid, 'b-', linewidth=1.0, label='PID', alpha=0.7)
ax5.plot(t_th1, u_th1, 'g-', linewidth=1.0, label='Θ (K=1.0)', alpha=0.7)
ax5.plot(t_th2, u_th2, 'r-', linewidth=1.0, label='Θ (K=2.0)', alpha=0.7)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Control Signal u(t)')
ax5.set_title('Control Signals')
ax5.legend()
ax5.grid(True)
ax5.set_xlim(0, 10)
plt.tight_layout()
plt.savefig('../figures/control_signals.png', dpi=150)
plt.close()
print("  ✓ control_signals.png")

# Figure 6: Zoom
fig6, ax6 = plt.subplots(figsize=(14, 6))
ax6.plot(t_pid, ref_pid, 'k--', linewidth=2, label='Reference')
ax6.plot(t_pid, y_pid, 'b-', linewidth=1.5, label='PID')
ax6.plot(t_th1, y_th1, 'g-', linewidth=1.5, label='Θ (K=1.0)')
ax6.plot(t_th2, y_th2, 'r-', linewidth=1.5, label='Θ (K=2.0)')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Position')
ax6.set_title('Step Response - Zoom (0-10s)')
ax6.legend()
ax6.grid(True)
ax6.set_xlim(0, 10)
ax6.set_ylim(-0.1, 1.2)
plt.tight_layout()
plt.savefig('../figures/step_response_zoom.png', dpi=150)
plt.close()
print("  ✓ step_response_zoom.png")

# Figure 7: Duffing time series
fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(14, 8))
ax7a.plot(t_duff_no, x_duff_no[:, 0], 'b-', linewidth=0.8)
ax7a.set_xlabel('Time (s)')
ax7a.set_ylabel('$x_1$')
ax7a.set_title('Duffing: Uncontrolled (Chaotic)')
ax7a.set_xlim(0, 50)
ax7a.grid(True)

ax7b.plot(t_duff, x_duff[:, 0], 'r-', linewidth=0.8)
ax7b.set_xlabel('Time (s)')
ax7b.set_ylabel('$x_1$')
ax7b.set_title('Duffing: With Θ-Control (Stabilized)')
ax7b.set_xlim(0, 50)
ax7b.grid(True)

plt.tight_layout()
plt.savefig('../figures/duffing_timeseries.png', dpi=150)
plt.close()
print("  ✓ duffing_timeseries.png")

# Figure 8: Duffing phase space
fig8, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(14, 5))
ax8a.plot(x_duff_no[:, 0], x_duff_no[:, 1], 'b-', linewidth=0.5, alpha=0.7)
ax8a.set_xlabel('$x_1$')
ax8a.set_ylabel('$x_2$')
ax8a.set_title('Phase Space: Uncontrolled')
ax8a.grid(True)

ax8b.plot(x_duff[:, 0], x_duff[:, 1], 'r-', linewidth=0.5, alpha=0.7)
ax8b.set_xlabel('$x_1$')
ax8b.set_ylabel('$x_2$')
ax8b.set_title('Phase Space: With Θ-Control')
ax8b.grid(True)

plt.tight_layout()
plt.savefig('../figures/duffing_phasespace.png', dpi=150)
plt.close()
print("  ✓ duffing_phasespace.png")

# Figure 9: Duffing Theta
fig9, ax9 = plt.subplots(figsize=(14, 5))
ax9.plot(t_duff, theta_duff, 'r-', linewidth=0.8)
ax9.set_xlabel('Time (s)')
ax9.set_ylabel('Θ(t)')
ax9.set_title('Theta Signal during Duffing Stabilization')
ax9.set_xlim(0, 50)
ax9.grid(True)
plt.tight_layout()
plt.savefig('../figures/duffing_theta.png', dpi=150)
plt.close()
print("  ✓ duffing_theta.png")

# Figure 10: Duffing spectrum
fig10, ax10 = plt.subplots(figsize=(14, 5))
freqs_d = fftfreq(len(x_duff_no[:, 0]), 0.001)
mag_no = np.abs(fft(x_duff_no[:, 0] * np.hanning(len(x_duff_no[:, 0]))))[:len(freqs_d)//2]
mag_ct = np.abs(fft(x_duff[:, 0] * np.hanning(len(x_duff[:, 0]))))[:len(freqs_d)//2]
freqs_pos_d = freqs_d[:len(freqs_d)//2]

ax10.plot(freqs_pos_d[freqs_pos_d <= 2], mag_no[freqs_pos_d <= 2], 'b-', linewidth=1.2, label='Uncontrolled')
ax10.plot(freqs_pos_d[freqs_pos_d <= 2], mag_ct[freqs_pos_d <= 2], 'r-', linewidth=1.2, label='Controlled')
ax10.set_xlabel('Frequency (Hz)')
ax10.set_ylabel('Magnitude')
ax10.set_title('Duffing: Frequency Spectrum')
ax10.legend()
ax10.grid(True)
ax10.set_xlim(0, 1.5)
plt.tight_layout()
plt.savefig('../figures/duffing_spectrum.png', dpi=150)
plt.close()
print("  ✓ duffing_spectrum.png")

print("\n" + "="*70)
print("ALL 10 FIGURES GENERATED SUCCESSFULLY!")
print("Location: ../figures/")
print("="*70)