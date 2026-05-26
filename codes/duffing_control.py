"""
duffing_control.py - Theta-Control applied to Duffing oscillator

Demonstrates:
1. Chaotic behavior without control
2. Stabilization with Theta-Control
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

os.makedirs('../figures', exist_ok=True)


class DuffingThetaController:
    """
    Theta-Control for Duffing oscillator.
    
    The Duffing system is:
        dx1/dt = x2
        dx2/dt = x1 - x1^3 - epsilon*x2 + A*cos(omega*t) + u
    
    Control law: u = -K1*x1 - K2*x2 + Theta
    where Theta = K_scale * [x1(t) - x1(t-tau)]
    """
    
    def __init__(self, K1, K2, K_scale, tau, dt):
        self.K1 = K1
        self.K2 = K2
        self.K_scale = K_scale
        self.tau = tau
        self.dt = dt
        self.buffer_size = max(1, int(tau / dt))
        self.memory = []
        
    def compute_theta(self, x_current):
        """Theta = K_scale * [x1(t) - x1(t-tau)]"""
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
        """u = -K1*x1 - K2*x2 + Theta"""
        u_fb = -self.K1 * x_current[0] - self.K2 * x_current[1]
        theta = self.compute_theta(x_current)
        return u_fb + theta
    
    def reset(self):
        self.memory = []


def duffing_dynamics(x, t, u=0.0, epsilon=0.2, A_forcing=0.3, omega=1.0):
    """
    Forced Duffing oscillator.
    
    dx1/dt = x2
    dx2/dt = x1 - x1^3 - epsilon*x2 + A*cos(omega*t) + u
    """
    x1, x2 = x
    dx1 = x2
    dx2 = x1 - x1**3 - epsilon*x2 + A_forcing * np.cos(omega * t) + u
    return np.array([dx1, dx2])


def simulate_duffing(controller, dt=0.001, T=100.0, x0=None):
    """
    Simulate Duffing oscillator with given controller.
    
    Parameters
    ----------
    controller : object or None
        If None, no control is applied (chaotic behavior)
    dt : float
        Time step
    T : float
        Total simulation time
    x0 : array
        Initial state [x1, x2]
    
    Returns
    -------
    t : array
        Time vector
    x : array
        State trajectory
    u : array
        Control signal
    theta : array
        Theta signal (if controller exists)
    """
    N = int(T / dt)
    t = np.arange(0, T, dt)
    
    if x0 is None:
        x0 = np.array([0.5, 0.5])
    
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
# MAIN SIMULATION
# ============================================

def run_duffing_demo():
    """Run Duffing oscillator demonstration."""
    
    dt = 0.001
    T = 100.0
    x0 = np.array([0.5, 0.5])
    
    print("="*70)
    print("DUFFING OSCILLATOR: THETA-CONTROL DEMONSTRATION")
    print("="*70)
    
    # Case 1: No control (chaotic)
    print("\n[1/2] Simulating uncontrolled Duffing (chaotic)...")
    t1, x1, _, _ = simulate_duffing(None, dt, T, x0)
    
    # Case 2: With Theta-Control (stabilized)
    print("[2/2] Simulating Duffing with Theta-Control (stabilized)...")
    controller = DuffingThetaController(
        K1=0.5, K2=0.2, K_scale=2.0, tau=0.05, dt=dt
    )
    t2, x2, u2, theta2 = simulate_duffing(controller, dt, T, x0)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Uncontrolled: max|x1| = {np.max(np.abs(x1[:,0])):.2f}")
    print(f"Controlled:   max|x1| = {np.max(np.abs(x2[:,0])):.2f}")
    print(f"Max Theta:    {np.max(np.abs(theta2)):.4f}")
    
    # ========================================
    # FIGURES
    # ========================================
    
    print("\nGenerating figures...")
    
    # Figure 1: Time series comparison
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(t1, x1[:, 0], 'b-', linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('$x_1$')
    ax1.set_title('Duffing Oscillator: Uncontrolled (Chaotic)')
    ax1.set_xlim(0, 50)
    ax1.grid(True)
    
    ax2.plot(t2, x2[:, 0], 'r-', linewidth=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('$x_1$')
    ax2.set_title('Duffing Oscillator: With Theta-Control (Stabilized)')
    ax2.set_xlim(0, 50)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/duffing_timeseries.png', dpi=150)
    plt.close()
    print("  ✓ duffing_timeseries.png")
    
    # Figure 2: Phase space (x1 vs x2)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(x1[:, 0], x1[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Phase Space: Uncontrolled (Chaotic)')
    ax1.grid(True)
    
    ax2.plot(x2[:, 0], x2[:, 1], 'r-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Phase Space: With Theta-Control (Stabilized)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/duffing_phasespace.png', dpi=150)
    plt.close()
    print("  ✓ duffing_phasespace.png")
    
    # Figure 3: Theta signal
    fig3, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t2, theta2, 'r-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Θ(t)')
    ax.set_title('Theta Signal during Duffing Stabilization')
    ax.set_xlim(0, 50)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('../figures/duffing_theta.png', dpi=150)
    plt.close()
    print("  ✓ duffing_theta.png")
    
    # Figure 4: Spectrum comparison
    fig4, ax = plt.subplots(figsize=(14, 5))
    
    Nfft = len(x1[:, 0])
    freqs = fftfreq(Nfft, dt)
    mag_uncontrolled = np.abs(fft(x1[:, 0] * np.hanning(Nfft)))[:Nfft//2]
    mag_controlled = np.abs(fft(x2[:, 0] * np.hanning(Nfft)))[:Nfft//2]
    freqs_pos = freqs[:Nfft//2]
    
    ax.plot(freqs_pos[freqs_pos <= 2], mag_uncontrolled[freqs_pos <= 2], 
            'b-', linewidth=1.2, label='Uncontrolled (Chaotic)')
    ax.plot(freqs_pos[freqs_pos <= 2], mag_controlled[freqs_pos <= 2], 
            'r-', linewidth=1.2, label='Controlled (Stabilized)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Frequency Spectrum: Duffing Oscillator')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1.5)
    plt.tight_layout()
    plt.savefig('../figures/duffing_spectrum.png', dpi=150)
    plt.close()
    print("  ✓ duffing_spectrum.png")
    
    print("\n" + "="*70)
    print("ALL DUFFING FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    
    return t1, x1, t2, x2, u2, theta2


if __name__ == "__main__":
    run_duffing_demo()