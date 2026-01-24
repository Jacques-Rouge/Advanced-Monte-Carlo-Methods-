import numpy as np

#### Global Params
K = 1.0
T = 1.0
M = 4.0 # value in the equation
sqrt_M = np.sqrt(M)
n_asian = 10 # Number of time steps for Asian option

### Functions for Simple Monte Carlo Simulation

def Run_Simple_MC_Asian_S(N_mc, M_mc, n_asian):
    dt = T / M_mc
    DW = np.sqrt(dt) * np.random.randn(N_mc, M_mc)
    
    step = M_mc // n_asian  # e.g., 1000/10 = 100
    
    S = np.ones(N_mc)
    S_observations = np.zeros((N_mc, n_asian))
    
    for i in range(M_mc):
        # Calculate the drift: 0.1 * sqrt(min(M, S)) - 1) * S
        drift_coeff = 0.1 * (np.sqrt(np.minimum(M, S)) - 1) * S
        # Calculate the diffusion: (1/2) * S
        diffusion_coeff = 0.5 * S
        # Update S: S_next = S_prev + drift*dt + diffusion*dW
        S += drift_coeff * dt + diffusion_coeff * DW[:, i]

        if (i + 1) % step == 0:
                obs_idx = (i + 1) // step - 1
                if obs_idx < n_asian:
                    S_observations[:, obs_idx] = S
    
    return S_observations



def Run_Simple_MC_Asian_X(N_mc, M_mc, n_asian):
    dt = T / M_mc
    DW = np.sqrt(dt) * np.random.randn(N_mc, M_mc)
    
    step = M_mc // n_asian  # e.g., 1000/10 = 100
    
    X = np.zeros(N_mc)
    X_observations = np.zeros((N_mc, n_asian))
    
    for i in range(M_mc):
        # Calculate the drift: 0.1 * (min(sqrt_M, exp(X)) - 1) - 1/8
        drift_coeff = 0.1 * (np.sqrt(np.minimum(M, np.exp(X))) - 1) - 1/8
        # Calculate the diffusion: (1/2)
        diffusion_coeff = 0.5
        # Update X: X_next = X_prev + drift*dt + diffusion*dW
        X += drift_coeff * dt + diffusion_coeff * DW[:, i]
        # Record at indices: 0, step, 2*step, 3*step, ...
        if (i + 1) % step == 0:
                obs_idx = (i + 1) // step - 1
                if obs_idx < n_asian:
                    X_observations[:, obs_idx] = X

    return X_observations


def Run_Simple_MC_S(N_mc, M_mc):
    dt=T/M_mc

    DW=np.sqrt(dt)*np.random.randn(N_mc,M_mc)

    S = np.ones(N_mc)
    for i in range(M_mc):
        # Calculate the drift: 0.1 * sqrt(min(M, S)) - 1) * S
        drift_coeff = 0.1 * (np.sqrt(np.minimum(M, S)) - 1) * S
        # Calculate the diffusion: (1/2) * S
        diffusion_coeff = 0.5 * S
        # Update S: S_next = S_prev + drift*dt + diffusion*dW
        S += drift_coeff * dt + diffusion_coeff * DW[:, i]

    return S

def Run_Simple_MC_X(N_mc, M_mc):
    dt=T/M_mc

    DW=np.sqrt(dt)*np.random.randn(N_mc,M_mc)

    X = np.zeros(N_mc)
    for i in range(M_mc):
        # Calculate the drift: 0.1 * (min(sqrt_M, exp(X)) - 1) - 1/8
        drift_coeff = 0.1 * (np.sqrt(np.minimum(M, np.exp(X))) - 1) - 1/8
        # Calculate the diffusion: (1/2)
        diffusion_coeff = 0.5
        # Update X: X_next = X_prev + drift*dt + diffusion*dW
        X += drift_coeff * dt + diffusion_coeff * DW[:, i]

    return X