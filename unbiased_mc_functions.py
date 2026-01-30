"""
Unbiased Monte Carlo Methods
Implementation of Henry-Labordère, Tan, and Touzi algorithm for unbiased simulation
"""

import numpy as np


# =============================================================================
# SDE Coefficients
# =============================================================================

def mu_X(X, M):
    """Drift coefficient for X process"""
    return 0.1 * (np.sqrt(np.minimum(M, np.exp(X))) - 1) - 1/8


def sigma_X(X):
    """Diffusion coefficient for X process (constant)"""
    return 0.5


# =============================================================================
# Parameter Calculation
# =============================================================================

def compute_gamma(sigma, mu_max, d, T):
    """
    Compute gamma parameter for optimal beta calculation.
    
    Args:
        sigma: Diffusion matrix
        mu_max: Maximum drift bound
        d: Dimension
        T: Time horizon
    """
    term_1 = (1 + mu_max * np.sqrt(T))**2
    term_2 = np.trace(np.linalg.inv(sigma * np.transpose(sigma)))
    term_3 = 2 * (3*d + d*(d-1))
    return term_1 * term_2 * term_3


def compute_beta(gamma, mu_lipschitz, T):
    """
    Compute optimal beta parameter.
    
    Args:
        gamma: Gamma parameter
        mu_lipschitz: Lipschitz constant for drift
        T: Time horizon
    """
    beta_star = np.sqrt(gamma * mu_lipschitz**2 * T + (1/4) * T**2) - 1/(2*T)
    return beta_star


# =============================================================================
# Random Grid Generation
# =============================================================================

def generate_random_grid(N_mc, T, beta, n_max=None):
    """
    Generate random discrete time grids following equation (2.4): T_k := (Σ τ_i) ∧ T
    
    Args:
        N_mc: Number of Monte Carlo trajectories
        T: Final time horizon
        beta: Intensity parameter for exponential arrivals (rate parameter)
        n_max: Maximum number of jumps (default: estimate based on E[N_T])
    
    Returns:
        T_matrix: Arrival times clipped to T (N_mc × n_max)
        dt_matrix: Time increments ΔT_k (N_mc × n_max)
        dW_matrix: Brownian increments ΔW (N_mc × n_max)
        valid_mask: Boolean mask for unique steps (not repeated T)
        n_grids: Number of arrivals before T (N_mc,)
    """
    if n_max is None:
        n_max = int(3 * beta * T) + 12

    # Exponential increments: τ_i ~ Exp(β), so scale = 1/β
    tau_matrix = np.random.exponential(scale=1/beta, size=(N_mc, n_max))

    # Cumulative sum: Σ_{i=1}^k τ_i
    T_cumsum = np.cumsum(tau_matrix, axis=1)
    
    # Equation (2.4): T_k := (Σ τ_i) ∧ T (min with T)
    T_matrix = np.minimum(T_cumsum, T)
    
    # Compute dt_k = T_k - T_{k-1}
    T_matrix_shifted = np.column_stack([np.zeros(N_mc), T_matrix[:, :-1]])
    dt_matrix = T_matrix - T_matrix_shifted
    
    # Valid mask: where T_k != T_{k-1} (i.e., not a repeated T)
    valid_mask = dt_matrix > 1e-14
    
    # N_T = number of steps before reaching T
    n_grids = np.sum(valid_mask, axis=1)
    
    # Brownian increments: ΔW_{T_{k+1}} ~ N(0, ΔT_{k+1})
    dW_matrix = np.random.randn(N_mc, n_max) * np.sqrt(dt_matrix)
    
    # Zero out invalid entries (after reaching T)
    dW_matrix[~valid_mask] = 0
    dt_matrix[~valid_mask] = 0

    return T_matrix, dt_matrix, dW_matrix, valid_mask, n_grids


# =============================================================================
# Unbiased MC for European Options
# =============================================================================

def run_unbiased_mc(N_mc, X_0, beta, mu_func, sigma_0, M, T):
    """
    Unbiased Monte Carlo for European options with random grids.
    
    Implements equations (2.5), (2.6), (2.7) from Henry-Labordère et al.
    
    Args:
        N_mc: Number of Monte Carlo paths
        X_0: Initial value
        beta: Intensity for random grid generation
        mu_func: Drift function
        sigma_0: Diffusion coefficient (constant)
        M: SDE parameter
        T: Time horizon
    
    Returns:
        X_T: Terminal values at T
        X_T_NT: Values at last random arrival T_{N_T}
        N_T: Number of random arrivals
        weights: Unbiased weights
    """
    # Generate random grids
    T_matrix, dt_matrix, dW_matrix, _, n_grids = \
        generate_random_grid(N_mc, T, beta)
    
    N_T = n_grids - 1  # Number of random arrivals before T
    
    # Initialize
    X = np.full(N_mc, X_0)
    X_prev = np.full(N_mc, X_0)
    X_T_NT = np.full(N_mc, X_0)
    w_product = np.ones(N_mc)
    
    # Main loop
    for k in range(T_matrix.shape[1]):
        active = (k < n_grids)
        if not np.any(active):
            break
        
        mu_k = mu_func(X, M)
        
        # Compute weights for k >= 1 and k <= N_T
        if k >= 1:
            compute_w = active & (k <= N_T)
            if np.any(compute_w):
                mu_prev = mu_func(X_prev, M)
                w = (mu_k - mu_prev) * dW_matrix[:, k] / (sigma_0 * dt_matrix[:, k])
                w_product[compute_w] *= w[compute_w]
        
        # Save state at last random arrival
        save_NT = (k == N_T) & (N_T > 0)
        X_T_NT = np.where(save_NT, X, X_T_NT)
        
        # Euler step
        X_next = X + mu_k * dt_matrix[:, k] + sigma_0 * dW_matrix[:, k]
        
        X_prev[active] = X[active]
        X[active] = X_next[active]
    
    # Final weights (equation 2.7)
    weights = np.exp(beta * T) * (beta ** (-N_T.astype(float))) * w_product
    
    return X, X_T_NT, N_T, weights


# =============================================================================
# Unbiased MC for Asian Options
# =============================================================================

def run_unbiased_asian_mc(N_mc, X_0, beta, mu_func, sigma_0, M, n_timesteps, T):
    """
    Unbiased MC for Asian Options following equations (2.13)-(2.16).
    
    Args:
        N_mc: Number of Monte Carlo paths
        X_0: Initial value
        beta: Intensity for random grid generation
        mu_func: Drift function
        sigma_0: Diffusion coefficient (constant)
        M: SDE parameter
        n_timesteps: Number of fixed evaluation points (e.g., 10 for Asian)
        T: Final time horizon
    
    Returns:
        X_paths: Simulated paths at evaluation points (N_mc × (n_timesteps+1))
        X_zero_paths: Zero-arrival paths (N_mc × (n_timesteps+1))
        psi_estimate: Final unbiased estimator ψ̃
    """
    # Fixed evaluation grid: 0 = t_0 < t_1 < ... < t_n = T
    t_grid = np.linspace(0, T, n_timesteps + 1)
    
    # Storage for paths at fixed evaluation points
    X_paths = np.zeros((N_mc, n_timesteps + 1))
    X_zero_paths = np.zeros((N_mc, n_timesteps + 1))
    X_paths[:, 0] = X_0
    X_zero_paths[:, 0] = X_0
    
    # Storage for weights per interval
    weights_k = []
    N_k_list = []
    
    # Step 1: Simulate forward through all intervals k=1,...,n
    for k in range(1, n_timesteps + 1):
        dt_interval = t_grid[k] - t_grid[k-1]
        
        # Generate random grid for interval [t_{k-1}, t_k]
        _, dt_mat, dW_mat, _, n_grids = \
            generate_random_grid(N_mc, dt_interval, beta)
        
        N_k = n_grids - 1  # Number of random arrivals in this interval
        N_k_list.append(N_k)
        
        # Initialize for interval k (equation 2.13)
        X_curr = X_paths[:, k-1].copy()  # X^{k,x}
        X_prev = X_curr.copy()
        X_at_N_k = X_curr.copy()  # X^{k,x,0} at T_{N^k}
        
        w_product = np.ones(N_mc)
        
        # Simulate within interval [t_{k-1}, t_k]
        for j in range(dt_mat.shape[1]):
            active = (j < n_grids)
            if not np.any(active):
                break
            
            mu_j = mu_func(X_curr, M)
            
            # Compute differentiation weights (equation 2.14) for j >= 1
            if j >= 1:
                compute_w = active & (j <= N_k)
                if np.any(compute_w):
                    mu_prev = mu_func(X_prev, M)
                    w = (mu_j - mu_prev) * dW_mat[:, j] / (sigma_0 * dt_mat[:, j])
                    w_product[compute_w] *= w[compute_w]
            
            # Save state at T_{N^k} (last random arrival before t_k)
            # When N_k > 0, save at the last jump; when N_k = 0, keep initial value
            save_mask = (j == N_k) & (N_k > 0)
            X_at_N_k[save_mask] = X_curr[save_mask]
            
            # Euler step (equation 2.13)
            X_next = X_curr + mu_j * dt_mat[:, j] + sigma_0 * dW_mat[:, j]
            X_prev[active] = X_curr[active]
            X_curr[active] = X_next[active]
        
        # Store final values for this interval
        X_paths[:, k] = X_curr  # X^{k,x} at t_k
        
        # For paths with N_k = 0, X^{k,x,0} should be x_{k-1} (no evolution)
        # This is already in X_at_N_k from initialization
        X_zero_paths[:, k] = X_at_N_k  # X^{k,x,0}_{N^k>0}
        
        # Compute weight for this interval (part of equation 2.15)
        w_k = np.exp(beta * dt_interval) * (beta ** (-N_k.astype(float))) * w_product
        weights_k.append(w_k)
    
    # Step 2: Compute final estimator
    # For Asian options, we need to evaluate g(x_1,...,x_n) for each simulated path
    # and apply the weight from the forward pass
    
    def asian_payoff(X_matrix):
        """Compute Asian call payoff from path matrix"""
        S_values = np.exp(X_matrix[:, 1:])  # Skip t_0, use t_1,...,t_n
        return np.maximum(np.mean(S_values, axis=1) - 1.0, 0)
    
    # Compute payoff on the full simulated path
    payoff = asian_payoff(X_paths)
    
    # Combine all weights from all intervals
    total_weight = np.ones(N_mc)
    for k in range(n_timesteps):
        total_weight *= weights_k[k]
    
    # Final unbiased estimator
    psi_estimate = total_weight * payoff
    
    return X_paths, X_zero_paths, psi_estimate


