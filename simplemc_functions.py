import numpy as np

#### Global Params
K = 1.0
T = 1.0
M = 4.0 # value in the equation
n_asian = 10 # Number of time steps for Asian option

### Drift (mu) and Diffusion (sigma) Functions for S process

def mu_S(S, M):
    """Drift coefficient for S process"""
    return 0.1 * (np.sqrt(np.minimum(M, S)) - 1) * S

def sigma_S(S):
    """Diffusion coefficient for S process"""
    return 0.5 * S

### Drift (mu) and Diffusion (sigma) Functions for X process

def mu_X(X, M):
    """Drift coefficient for X process"""
    return 0.1 * (np.sqrt(np.minimum(M, np.exp(X))) - 1) - 1/8

def sigma_X(X):
    """Diffusion coefficient for X process (constant)"""
    return 0.5


def mu_X_2(X, M):
    """Drift coefficient for X process"""
    return 0.5 * (1 - X)

def sigma_X_2(X):
    """Diffusion coefficient for X process (constant)"""
    return 0.2 * np.sqrt(np.maximum(X, 1e-5))

### Generic Monte Carlo Simulation Functions

def Run_Generic_MC(N_mc, M_mc, initial_value, mu_func, sigma_func, M, T):
    """
    Generic Monte Carlo simulation
    
    Args:
        N_mc: Number of Monte Carlo samples
        M_mc: Number of time steps
        initial_value: Initial value of the process
        mu_func: Drift function mu(state, M)
        sigma_func: Diffusion function sigma(state)
        M: Parameter M
        T: Time horizon
    
    Returns:
        Final state values for all trajectories
    """
    dt = T / M_mc
    DW = np.sqrt(dt) * np.random.randn(N_mc, M_mc)
    
    state = np.ones(N_mc) * initial_value
    
    for i in range(M_mc):
        drift = mu_func(state, M)
        diffusion = sigma_func(state)
        state += drift * dt + diffusion * DW[:, i]
    
    return state

def Run_Generic_MC_Asian(N_mc, M_mc, n_asian, initial_value, mu_func, sigma_func, M, T):
    """
    Generic Monte Carlo simulation for Asian option (discrete average)
    
    Args:
        N_mc: Number of Monte Carlo samples
        M_mc: Number of time steps
        n_asian: Number of observation times
        initial_value: Initial value of the process
        mu_func: Drift function mu(state, M)
        sigma_func: Diffusion function sigma(state)
        M: Parameter M
        T: Time horizon
    
    Returns:
        State observations at n_asian time points (shape: N_mc x n_asian)
    """
    dt = T / M_mc
    DW = np.sqrt(dt) * np.random.randn(N_mc, M_mc)
    
    step = M_mc // n_asian
    
    state = np.ones(N_mc) * initial_value
    state_observations = np.zeros((N_mc, n_asian))
    
    for i in range(M_mc):
        drift = mu_func(state, M)
        diffusion = sigma_func(state)
        state += drift * dt + diffusion * DW[:, i]
        
        if (i + 1) % step == 0:
            obs_idx = (i + 1) // step - 1
            if obs_idx < n_asian:
                state_observations[:, obs_idx] = state
    
    return state_observations

### Convenience Wrappers (backward compatible)

def Run_Simple_MC_S(N_mc, M_mc):
    """Simple MC for S process - European payoff"""
    return Run_Generic_MC(N_mc, M_mc, 1.0, mu_S, sigma_S, M, T)

def Run_Simple_MC_X(N_mc, M_mc):
    """Simple MC for X process - European payoff"""
    return Run_Generic_MC(N_mc, M_mc, 0.0, mu_X, sigma_X, M, T)

def Run_Simple_MC_Asian_S(N_mc, M_mc, n_asian):
    """Simple MC for S process - Asian payoff"""
    return Run_Generic_MC_Asian(N_mc, M_mc, n_asian, 1.0, mu_S, sigma_S, M, T)

def Run_Simple_MC_Asian_X(N_mc, M_mc, n_asian):
    """Simple MC for X process - Asian payoff"""
    return Run_Generic_MC_Asian(N_mc, M_mc, n_asian, 0.0, mu_X, sigma_X, M, T)

def Run_Simple_MC_X_2(N_mc, M_mc):
    """Simple MC for X process - European payoff"""
    return Run_Generic_MC(N_mc, M_mc, 1.0, mu_X_2, sigma_X_2, M, T)

def Run_Simple_MC_Asian_X_2(N_mc, M_mc, n_asian):
    """Simple MC for X process - Asian payoff"""
    return Run_Generic_MC_Asian(N_mc, M_mc, n_asian, 1.0, mu_X_2, sigma_X_2, M, T)