"""
theta_controller.py - Core implementation of Theta-Control

This module provides the ThetaController class for memory-based control.
"""

import numpy as np


class ThetaController:
    """
    Theta-Control for linear systems with state feedback.
    
    Control law: u = -K*x + N*ref + Theta
    where Theta = K_scale * (BK @ [x(t) - x(t-tau)])[1]
    
    Parameters
    ----------
    A : np.ndarray (n x n)
        System dynamics matrix
    B : np.ndarray (n x m)
        Input matrix
    K : np.ndarray (m x n)
        State feedback gain
    K_scale : float
        Scaling factor for Theta term (main tuning parameter)
    tau : float
        Memory delay in seconds
    dt : float
        Simulation time step in seconds
    """
    
    def __init__(self, A, B, K, K_scale, tau, dt):
        self.A = A
        self.B = B
        self.K = K
        self.K_scale = K_scale
        self.tau = tau
        self.dt = dt
        
        # Precompute closed-loop matrix
        self.Acl = A - B @ K
        self.BK = B @ K
        
        # Pre-computation gain for zero steady-state error
        C = np.array([1, 0])
        self.N_ref = -1.0 / (C @ np.linalg.inv(self.Acl) @ B)[0]
        
        # Memory buffer
        self.buffer_size = max(1, int(tau / dt))
        self.memory = []
        
    def compute_theta(self, x_current):
        """
        Compute Theta = K_scale * (BK @ [x(t) - x(t-tau)])[1]
        
        Parameters
        ----------
        x_current : np.ndarray (n,)
            Current state vector
            
        Returns
        -------
        theta : float
            Theta term (acts on acceleration channel)
        """
        if len(self.memory) >= self.buffer_size:
            x_past = self.memory[-self.buffer_size]
            delta = x_current - x_past
            theta = self.K_scale * (self.BK @ delta)[1]
        else:
            theta = 0.0
        return theta
    
    def update_memory(self, x_current):
        """Update memory buffer with current state."""
        self.memory.append(x_current.copy())
        if len(self.memory) > self.buffer_size * 2:
            self.memory = self.memory[-self.buffer_size:]
    
    def compute_control(self, x_current, reference):
        """
        Compute complete control signal.
        
        u = -K*x + N_ref*ref + Theta
        
        Parameters
        ----------
        x_current : np.ndarray (n,)
            Current state vector
        reference : float
            Current reference value
            
        Returns
        -------
        u : float
            Control signal
        """
        # State feedback term
        u_fb = -(self.K @ x_current)[0]
        
        # Reference pre-compensation
        u_ref = self.N_ref * reference
        
        # Memory term
        theta = self.compute_theta(x_current)
        
        return u_fb + u_ref + theta
    
    def reset(self):
        """Reset memory buffer for new simulation."""
        self.memory = []


def plant_dynamics(x, u):
    """
    Second-order damped oscillator.
    
    States:
        x[0] = position
        x[1] = velocity
    
    Dynamics:
        dx0/dt = x1
        dx1/dt = -4*x0 - 0.4*x1 + u
    """
    dx0 = x[1]
    dx1 = -4*x[0] - 0.4*x[1] + u
    return np.array([dx0, dx1])


# System matrices (for convenience)
A = np.array([[0, 1], [-4, -0.4]])
B = np.array([[0], [1]])
K = np.array([[6, 2]])