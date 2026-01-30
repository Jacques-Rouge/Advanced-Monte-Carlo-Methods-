"""
Multilevel Monte Carlo Methods
Following Giles (2015) algorithm for European and Asian option pricing
"""

import numpy as np


# =============================================================================
# SDE Coefficients
# =============================================================================

def mu_S(S, M):
    """Drift coefficient for S process"""
    return 0.1 * (np.sqrt(np.minimum(M, S)) - 1) * S


def sigma_S(S):
    """Diffusion coefficient for S process"""
    return 0.5 * S


# =============================================================================
# Simulation Functions
# =============================================================================

def euler_maruyama(N, M, S0, mu_func, sigma_func, M_param, T, n_monitoring=None):
    """
    Euler-Maruyama scheme for SDE simulation
    
    Args:
        N: Number of paths
        M: Number of time steps
        S0: Initial value
        mu_func: Drift function
        sigma_func: Diffusion function
        M_param: SDE parameter
        T: Time horizon
        n_monitoring: Number of monitoring points for Asian option (if None, return only terminal value)
        
    Returns:
        If n_monitoring is None: Terminal values S(T) for all paths
        If n_monitoring is set: (N, n_monitoring) array of monitored values
    """
    dt = T / M
    DW = np.sqrt(dt) * np.random.randn(N, M)
    S = np.full(N, S0)
    
    # For Asian option, store values at monitoring points
    if n_monitoring is not None:
        monitoring_interval = M // n_monitoring
        S_monitoring = np.zeros((N, n_monitoring))
        monitor_idx = 0
    
    for i in range(M):
        mu = mu_func(S, M_param)
        sigma = sigma_func(S)
        S += mu * dt + sigma * DW[:, i]
        S = np.maximum(S, 1e-10)  # Prevent negative values
        
        # Record at monitoring points
        if n_monitoring is not None and (i + 1) % monitoring_interval == 0:
            S_monitoring[:, monitor_idx] = S
            monitor_idx += 1
    
    if n_monitoring is not None:
        return S_monitoring
    return S


def mlmc_level(N, M, mu_func, sigma_func, M_param, T, K, n_asian=None):
    """
    Simulate single MLMC level (for L=0)
    
    Args:
        n_asian: If set, compute Asian option with this many monitoring points
    
    Returns:
        mean: E[payoff]
        var: Var[payoff]
    """
    if n_asian is not None:
        # Asian option: average over monitoring points
        S_monitoring = euler_maruyama(N, M, 1.0, mu_func, sigma_func, M_param, T, n_monitoring=n_asian)
        S_avg = np.mean(S_monitoring, axis=1)
        payoff = np.maximum(S_avg - K, 0)
    else:
        # European option: terminal value only
        S_T = euler_maruyama(N, M, 1.0, mu_func, sigma_func, M_param, T)
        payoff = np.maximum(S_T - K, 0)
    
    return np.mean(payoff), np.var(payoff)


def mlmc_level_diff(N, M_coarse, M_fine, mu_func, sigma_func, M_param, T, K, n_asian=None):
    """
    Simulate MLMC level difference (for L>0)
    
    Uses same Brownian motion for coarse and fine paths (telescoping)
    
    Args:
        n_asian: If set, compute Asian option with this many monitoring points
    
    Returns:
        mean_diff: E[P_fine - P_coarse]
        var_diff: Var[P_fine - P_coarse]
    """
    dt_fine = T / M_fine
    dt_coarse = T / M_coarse
    ratio = M_fine // M_coarse
    
    # Generate fine Brownian increments
    DW_fine = np.sqrt(dt_fine) * np.random.randn(N, M_fine)
    
    # Aggregate to coarse Brownian (same underlying motion)
    DW_coarse = np.zeros((N, M_coarse))
    for j in range(M_coarse):
        DW_coarse[:, j] = np.sum(DW_fine[:, j*ratio:(j+1)*ratio], axis=1)
    
    # For Asian option, track monitoring points
    if n_asian is not None:
        monitoring_interval_fine = M_fine // n_asian
        monitoring_interval_coarse = M_coarse // n_asian
        S_fine_monitoring = np.zeros((N, n_asian))
        S_coarse_monitoring = np.zeros((N, n_asian))
        monitor_idx_fine = 0
        monitor_idx_coarse = 0
    
    # Simulate fine path
    S_fine = np.ones(N)
    for i in range(M_fine):
        mu = mu_func(S_fine, M_param)
        sigma = sigma_func(S_fine)
        S_fine += mu * dt_fine + sigma * DW_fine[:, i]
        S_fine = np.maximum(S_fine, 1e-10)
        
        if n_asian is not None and (i + 1) % monitoring_interval_fine == 0:
            S_fine_monitoring[:, monitor_idx_fine] = S_fine
            monitor_idx_fine += 1
    
    # Simulate coarse path (same Brownian)
    S_coarse = np.ones(N)
    for i in range(M_coarse):
        mu = mu_func(S_coarse, M_param)
        sigma = sigma_func(S_coarse)
        S_coarse += mu * dt_coarse + sigma * DW_coarse[:, i]
        S_coarse = np.maximum(S_coarse, 1e-10)
        
        if n_asian is not None and (i + 1) % monitoring_interval_coarse == 0:
            S_coarse_monitoring[:, monitor_idx_coarse] = S_coarse
            monitor_idx_coarse += 1
    
    # Compute payoffs
    if n_asian is not None:
        # Asian option: average over monitoring points
        S_fine_avg = np.mean(S_fine_monitoring, axis=1)
        S_coarse_avg = np.mean(S_coarse_monitoring, axis=1)
        P_fine = np.maximum(S_fine_avg - K, 0)
        P_coarse = np.maximum(S_coarse_avg - K, 0)
    else:
        # European option: terminal value
        P_fine = np.maximum(S_fine - K, 0)
        P_coarse = np.maximum(S_coarse - K, 0)
    
    diff = P_fine - P_coarse
    
    return np.mean(diff), np.var(diff)


def compute_optimal_samples(variances, M_levels, T, epsilon):
    """
    Compute optimal sample allocation using Giles' formula
    
    N_l = ceil((2/ε²) × √(V_l × h_l) × Σ√(V_j / h_j))
    
    Args:
        variances: List of variances [V_0, ..., V_L]
        M_levels: List of time steps [M_0, ..., M_L]
        T: Time horizon
        epsilon: Target accuracy
        
    Returns:
        List of optimal sample sizes [N_0, ..., N_L]
    """
    h_levels = [T / M for M in M_levels]
    sum_term = sum(np.sqrt(variances[j] / h_levels[j]) for j in range(len(variances)))
    
    N_optimal = []
    for l in range(len(variances)):
        N_l = int(np.ceil((2 / epsilon**2) * np.sqrt(variances[l] * h_levels[l]) * sum_term))
        N_optimal.append(N_l)
    
    return N_optimal


# =============================================================================
# MLMC Algorithm
# =============================================================================

def run_mlmc(mu_func, sigma_func, T, K, M_param, epsilon, max_levels, N_initial, 
             n_asian=None, M_base=1, verbose=True):
    """
    Run Multilevel Monte Carlo algorithm (Giles 2015)
    
    Args:
        mu_func: Drift function
        sigma_func: Diffusion function
        T: Time horizon
        K: Strike price
        M_param: SDE parameter
        epsilon: Target accuracy
        max_levels: Maximum number of levels
        N_initial: Initial samples per new level
        n_asian: Number of monitoring points for Asian option (None for European)
        M_base: Base number of timesteps for level 0
        verbose: Print progress information
        
    Returns:
        dict with keys:
            - 'estimate': MLMC estimator
            - 'levels': Number of levels
            - 'total_samples': Total number of samples
            - 'N_current': Samples per level
            - 'M_levels': Time steps per level
            - 'variances': Variance per level
    """
    if verbose:
        print("="*70)
        if n_asian is not None:
            print(f"MULTILEVEL MONTE CARLO - ASIAN OPTION (n_asian={n_asian})")
        else:
            print("MULTILEVEL MONTE CARLO - EUROPEAN OPTION")
        print("="*70)
    
    # Initialize
    L = 0
    M_levels = []
    means_sum = []
    variances = []
    N_current = []
    converged = False
    
    while L < max_levels and not converged:
        if verbose:
            print(f"\n{'='*70}")
            print(f"LEVEL L = {L}")
            print(f"{'='*70}")
        
        # Step 2: Estimate variance with initial samples
        if L == 0:
            M = M_base
            M_levels.append(M)
            if verbose:
                print(f"Step 2: Initial sampling (N={N_initial}, M={M})")
            
            mean, var = mlmc_level(N_initial, M, mu_func, sigma_func, M_param, T, K, n_asian)
            means_sum.append(mean * N_initial)
            variances.append(var)
            N_current.append(N_initial)
            
            if verbose:
                print(f"  E[P_0] = {mean:.6f}, Var[P_0] = {var:.6f}")
        else:
            M_coarse = M_base * 4**(L-1)
            M_fine = M_base * 4**L
            M_levels.append(M_fine)
            
            if verbose:
                print(f"Step 2: Initial sampling (N={N_initial}, M_c={M_coarse}, M_f={M_fine})")
            
            mean_diff, var_diff = mlmc_level_diff(N_initial, M_coarse, M_fine, 
                                                   mu_func, sigma_func, M_param, T, K, n_asian)
            means_sum.append(mean_diff * N_initial)
            variances.append(var_diff)
            N_current.append(N_initial)
            
            if verbose:
                print(f"  E[P_{L} - P_{L-1}] = {mean_diff:.6f}, Var = {var_diff:.6f}")
        
        # Step 3: Compute optimal samples for all levels
        if verbose:
            print(f"\nStep 3: Optimal sample allocation")
        N_optimal = compute_optimal_samples(variances, M_levels, T, epsilon)
        
        if verbose:
            for l in range(L+1):
                print(f"  Level {l}: N_opt = {N_optimal[l]:,} (current = {N_current[l]:,})")
        
        # Step 4: Add extra samples as needed
        if verbose:
            print(f"\nStep 4: Adding extra samples")
        for l in range(L+1):
            N_extra = N_optimal[l] - N_current[l]
            
            if N_extra > 0:
                if verbose:
                    print(f"  Level {l}: +{N_extra:,} samples")
                
                if l == 0:
                    mean_extra, _ = mlmc_level(N_extra, M_levels[0], mu_func, sigma_func, M_param, T, K, n_asian)
                    means_sum[l] += mean_extra * N_extra
                else:
                    M_c = M_levels[l-1]
                    M_f = M_levels[l]
                    mean_extra, _ = mlmc_level_diff(N_extra, M_c, M_f, mu_func, sigma_func, M_param, T, K, n_asian)
                    means_sum[l] += mean_extra * N_extra
                
                N_current[l] = N_optimal[l]
        
        # Step 5: Test convergence (L ≥ 2)
        if L >= 2:
            mean_L = means_sum[L] / N_current[L]
            threshold = 0.5 * epsilon / np.sqrt(2)
            
            if verbose:
                print(f"\nStep 5: Convergence test")
                print(f"  |E[P_{L} - P_{L-1}]| = {abs(mean_L):.6e}")
                print(f"  Threshold = {threshold:.6e}")
            
            if abs(mean_L) < threshold:
                converged = True
                if verbose:
                    print(f"  ✓ CONVERGED")
        
        # Step 6: Continue to next level
        if not converged:
            L += 1
    
    # Compute final estimator
    V_mlmc = sum(means_sum[l] / N_current[l] for l in range(len(means_sum)))
    total_samples = sum(N_current)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"\nOption Price: {V_mlmc:.6f}")
        print(f"Levels: {len(means_sum)}")
        print(f"Total samples: {total_samples:,}")
        print(f"Epsilon: {epsilon}")
        print(f"\nSample distribution:")
        for l in range(len(N_current)):
            print(f"  Level {l}: {N_current[l]:,} samples")
        print("="*70)
    
    return {
        'estimate': V_mlmc,
        'levels': len(means_sum),
        'total_samples': total_samples,
        'N_current': N_current,
        'M_levels': M_levels,
        'variances': variances
    }
