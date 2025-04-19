import numpy as np
import matplotlib.pyplot as plt

def alpha_n(V):
    """Rate constant alpha_n(V) for gating variable n."""
    return (0.01 * (10.0 - (V + 65.0))) / (np.exp((10.0 - (V + 65.0)) / 10.0) - 1.0)

def beta_n(V):
    """Rate constant beta_n(V) for gating variable n."""
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

def alpha_m(V):
    """Rate constant alpha_m(V) for gating variable m."""
    return (0.1 * (25.0 - (V + 65.0))) / (np.exp((25.0 - (V + 65.0)) / 10.0) - 1.0)

def beta_m(V):
    """Rate constant beta_m(V) for gating variable m."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    """Rate constant alpha_h(V) for gating variable h."""
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

def beta_h(V):
    """Rate constant beta_h(V) for gating variable h."""
    return 1.0 / (np.exp((30.0 - (V + 65.0)) / 10.0) + 1.0)

def hodgkin_huxley_sim(t_start=-30.0, t_end=200.0, dt=0.01):
    """Simulate the Hodgkin-Huxley model with multiple external current steps."""
    # Conductances (mS/cm²)
    gNa = 120.0
    gK = 36.0
    gL = 0.3

    # Reversal potentials (mV)
    ENa = 50.0    
    EK = -77.0    
    EL = -54.4    
    
    # Membrane capacitance (µF/cm²)
    Cm = 1.0

    # Time vector
    t = np.arange(t_start, t_end + dt, dt)
    nSteps = len(t)
    
    # Allocate buffers for storing results
    V_trace = np.zeros(nSteps)
    n_trace = np.zeros(nSteps)
    m_trace = np.zeros(nSteps)
    h_trace = np.zeros(nSteps)
    I_ext_trace = np.zeros(nSteps)
    
    # Initial conditions
    V = -65.0   # membrane potential (mV)
    n_gate = alpha_n(V) / (alpha_n(V) + beta_n(V))  # steady-state n
    m_gate = alpha_m(V) / (alpha_m(V) + beta_m(V))  # steady-state m
    h_gate = alpha_h(V) / (alpha_h(V) + beta_h(V))  # steady-state h
    
    # Run the simulation loop
    for i, tt in enumerate(t):
        # Store current state
        V_trace[i] = V
        n_trace[i] = n_gate
        m_trace[i] = m_gate
        h_trace[i] = h_gate
        
        # External current steps:
        if (tt >= 5 and tt < 25):
            I_ext = 10 + 2 * np.random.randn()   # depolarizing pulse with noise
        elif (tt >= 50 and tt < 70):
            I_ext = -5                           # hyperpolarizing pulse
        elif (tt >= 100 and tt < 120):
            I_ext = 7 + np.random.randn()        # depolarizing pulse with noise
        elif (tt >= 150 and tt < 170):
            I_ext = -3                           # hyperpolarizing pulse
        elif (tt >= 170 and tt < 200):
            I_ext = 5                            # long step at 5 µA/cm²
        else:
            I_ext = 0.0
        I_ext_trace[i] = I_ext
        
        # Update gating variables (Euler method)
        dn_dt = alpha_n(V) * (1.0 - n_gate) - beta_n(V) * n_gate
        dm_dt = alpha_m(V) * (1.0 - m_gate) - beta_m(V) * m_gate
        dh_dt = alpha_h(V) * (1.0 - h_gate) - beta_h(V) * h_gate
        
        n_gate += dn_dt * dt
        m_gate += dm_dt * dt
        h_gate += dh_dt * dt
        
        # Compute ionic currents (µA/cm²)
        IK = gK * (n_gate**4) * (V - EK)
        INa = gNa * (m_gate**3) * h_gate * (V - ENa)
        IL = gL * (V - EL)
        I_ion = IK + INa + IL
        
        # Update membrane potential using Euler's method: dV/dt = (I_ext - I_ion)/Cm
        V += dt * (I_ext - I_ion) / Cm
    
    return t, V_trace, n_trace, m_trace, h_trace, I_ext_trace
