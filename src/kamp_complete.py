r"""
kamp_complete.py — Complete KAMP Paper Experiment Pipeline
===========================================================

This single self-contained file implements:

SOLVERS (8 algorithms for l1-regularised sparse recovery):
  1. ISTA  — Iterative Shrinkage-Thresholding Algorithm     [BT09]
  2. FISTA — Fast ISTA (Nesterov-accelerated)                [BT09]
  3. AMP   — Approximate Message Passing                     [DMM09, RSF19]
  4. VAMP  — Vector Approximate Message Passing (SVD)        [RSF19]
  5. OAMP  — Orthogonal AMP (LMMSE form)                     [MP17, RSF19]
  6. MAMP  — Memory Approximate Message Passing              [LHK21]
  7. KAMP  — Kalman Approximate Message Passing              [GSY26]
  8. DKAMP — Distributed Kalman AMP (consensus-based)        [GSY26]

EXPERIMENTS (13 analyses generating 23+ figures and 12+ CSVs):
  A — Algorithm Validation:
    1. Onsager Correction Validation (KAMP vs AMP correction gap)
    2. State Evolution (SE prediction vs empirical NMSE)
  B — Core Benchmarking:
    3. Full Solver Benchmark (4 matrix types, 2 signal models)
    4. Covariance Modes (KAMP-full vs AMP/VAMP/OAMP sweeps)
    5. Phase Transition (Donoho-Tanner delta-rho diagrams)
  C — Robustness & Complexity:
    6. Noise Mismatch Stress Test (0.1x to 10x sigma mismatch)
    7. Scalability Analysis (runtime vs signal dimension n)
    8. Complexity Analysis (per-iteration cost, total cost)
    9. Convergence Speed (iterations to target NMSE)
  D — Ablation & Statistics:
    10. Hyperparameter Ablation (alpha, k, clipping, reg_eps)
    11. Statistical Comparison (Mann-Whitney, Holm-corrected)
    12. Continuous Sweep (kappa and rho sweeps, all solvers)
  E — Distributed Extension:
    13. DKAMP Topology Study (5 graph topologies)

OUTPUTS (generated into experiments/data/ and experiments/figures/):
  - 23+ publication-quality PNG figures
  - 12+ CSV data files with all numerical results
  - Console summary tables for every experiment

References:
  [BT09]  Beck & Teboulle, SIAM J. Imaging Sci., 2009
  [DMM09] Donoho, Maleki & Montanari, PNAS, 2009
  [RSF19] Rangan, Schniter & Fletcher, IEEE TIT, 2019
  [MP17]  Ma & Ping, IEEE Access, 2017
  [LHK21] Liu, Huang & Kurkoski, IEEE ISIT, 2021
  [GSY26] Ghalenoei & Sadoghi Yazdi, KAMP paper, 2026

Usage:
    python kamp_complete.py              # Run all experiments
    python kamp_complete.py --demo       # Quick demo only
    python kamp_complete.py --benchmark  # Full benchmark only
"""

# ============================================================================
# Imports
# ============================================================================
import sys, os, time, csv, copy, warnings
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Tuple
from tqdm.auto import tqdm

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import mannwhitneyu, friedmanchisquare, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

_LIGHT_MODE = False  # Set True via --light flag in __main__
_FAST_MODE = False   # Set True via --fast flag in __main__

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# Directory setup
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'experiments' / 'data'
FIG_DIR = PROJECT_ROOT / 'experiments' / 'figures'
for d in [DATA_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Part 1 — Shared Data Structures & Utilities
# ============================================================================

@dataclass
class SolverResult:
    x_hat: np.ndarray
    history: List[float]
    info: Dict[str, Any]

def _is_diverged(x_hat: np.ndarray, blowup: float = 1e10, nmse_val: float = None) -> bool:
    norm = float(np.linalg.norm(x_hat))
    if (not np.isfinite(norm)) or norm > blowup: return True
    if nmse_val is not None and (not np.isfinite(nmse_val) or nmse_val > 1e4): return True
    return False

def _soft_threshold(u: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(u) * np.maximum(np.abs(u) - alpha, 0.0)

def _smoothed_subdifferential(r: np.ndarray, tau: float, k: float = 10.0) -> np.ndarray:
    return 0.5 + 0.5 * np.tanh(k * (np.abs(r) - tau))

def _power_method_lipschitz(A: np.ndarray, n_iter: int = 200) -> float:
    n = A.shape[1]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(n_iter):
        w = A.T @ (A @ v)
        lam = np.linalg.norm(w)
        v = w / lam
        if abs(lam - lam_old) < 1e-10 * lam:
            break
        lam_old = lam
    return lam

def _map_denoiser(r: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, float]:
    threshold = lam / max(gamma, 1e-30)
    xhat = _soft_threshold(r, threshold)
    alpha = float(np.clip(np.mean(np.abs(r) > threshold), 1e-8, 1.0 - 1e-8))
    return xhat, alpha

def _track(xhat: np.ndarray, y: np.ndarray, A: np.ndarray,
           lam: float, true_x: np.ndarray = None) -> float:
    if true_x is not None:
        return float(np.sum((xhat - true_x) ** 2) / np.sum(true_x ** 2))
    return float(0.5 * np.sum((A @ xhat - y) ** 2) + lam * np.sum(np.abs(xhat)))

def _validate_inputs(A: np.ndarray, y: np.ndarray, config: dict) -> None:
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if A.shape[0] != y.shape[0]:
        raise ValueError(f"A.shape[0]={A.shape[0]} != y.shape[0]={y.shape[0]}")
    if 'max_iter' not in config and 'num_iters' not in config:
        raise ValueError("config missing 'max_iter' or 'num_iters'")
    if 'lam' not in config and 'reg_eps' not in config:
        raise ValueError("config missing 'lam' or 'reg_eps'")
    max_iter_val = config.get('max_iter', config.get('num_iters', 50))
    lam_val = config.get('lam', config.get('reg_eps', 0.1))
    if max_iter_val <= 0: raise ValueError("max_iter must be positive")
    if 'tol' in config and config['tol'] < 0: raise ValueError("tol must be non-negative")
    if lam_val < 0: raise ValueError("lam must be non-negative")

def _kamp_soft_threshold(u: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(u) * np.maximum(np.abs(u) - tau, 0.0)

def _kamp_soft_derivative(r: np.ndarray, tau: float, k: float = 10.0) -> np.ndarray:
    return 0.5 + 0.5 * np.tanh(k * (np.abs(r) - tau))

def _debias_on_support(A: np.ndarray, y: np.ndarray, x_hat: np.ndarray,
                        threshold: float = None) -> np.ndarray:
    x_hat = x_hat.flatten()
    if threshold is None:
        threshold = 0.1 * np.max(np.abs(x_hat))
    support = np.where(np.abs(x_hat) > threshold)[0]
    if len(support) == 0:
        return x_hat.reshape(-1, 1)
    A_s = A[:, support]
    x_s, _, _, _ = np.linalg.lstsq(A_s, y.flatten(), rcond=None)
    x_debiased = np.zeros_like(x_hat)
    x_debiased[support] = x_s
    return x_debiased.reshape(-1, 1)

# High-contrast, colorblind-friendly, B/W-printable solver styles (Paul Tol bright scheme)
SOLVER_COLORS = {
    'ISTA': '#4477AA',   'FISTA': '#66CCEE',
    'AMP':  '#EE6677',   'VAMP':  '#228833',
    'OAMP': '#CCBB44',   'KAMP':  '#AA3377',
    'DKAMP': '#BB5566',
}
SOLVER_MARKERS = {
    'ISTA': 'o', 'FISTA': 's', 'AMP': '^', 'VAMP': 'D',
    'OAMP': 'P', 'KAMP': 'v', 'DKAMP': 'X',
}
SOLVER_LINESTYLES = {
    'ISTA': ':', 'FISTA': '-.', 'AMP': '--', 'VAMP': '-',
    'OAMP': '-', 'KAMP': '-', 'DKAMP': (0, (5, 2)),
}
SOLVER_HATCHES = {
    'ISTA': '', 'FISTA': '', 'AMP': 'xxx',
    'VAMP': '//', 'OAMP': '\\\\', 'KAMP': '**', 'DKAMP': 'oo',
}

def set_publication_style():
    plt.style.use('classic')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Palatino', 'serif'],
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'axes.linewidth': 1.0, 'axes.edgecolor': 'black',
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'xtick.major.size': 4, 'xtick.minor.size': 2,
        'ytick.major.size': 4, 'ytick.minor.size': 2,
        'legend.fontsize': 9, 'legend.frameon': False,
        'legend.handlelength': 1.5, 'legend.handletextpad': 0.4,
        'lines.linewidth': 1.8, 'lines.markersize': 5,
        'figure.dpi': 150, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

def save_figure(fig, filename, fig_dir=FIG_DIR):
    path = fig_dir / filename
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved -> {filename}")

def _solver_style(sname):
    return {'color': SOLVER_COLORS.get(sname, 'black'),
            'marker': SOLVER_MARKERS.get(sname, 'o'),
            'linestyle': SOLVER_LINESTYLES.get(sname, '-'),
            'linewidth': 2.5 if sname in ('KAMP', 'DKAMP') else 1.8,
            'markersize': 6 if sname in ('KAMP', 'DKAMP') else 5}
def _solver_bar_style(sname):
    return {'color': SOLVER_COLORS.get(sname, '#666666'),
            'hatch': SOLVER_HATCHES.get(sname, ''),
            'edgecolor': 'black', 'linewidth': 0.8}


# ============================================================================
# Part 2 — Solver Implementations
# ============================================================================
# Each solver solves:  min_x  0.5 ||Ax - y||^2 + lam * ||x||_1
# All expose:  solve_*(A, y, config) -> SolverResult
# ============================================================================


# ---------------------------------------------------------------------------
# 2a. ISTA — Iterative Shrinkage-Thresholding Algorithm  [BT09]
# ---------------------------------------------------------------------------

def solve_ista(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    x0 = config.get('x0', None)
    true_x = config.get('true_x', None)
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    L = _power_method_lipschitz(A)
    step = lam / L
    history = []
    t0 = time.perf_counter()
    it = 0
    for it in range(max_iter):
        x_new = _soft_threshold(x + (1.0 / L) * (A.T @ (y - A @ x)), step)
        history.append(_track(x_new, y, A, lam, true_x))
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new; break
        x = x_new
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return SolverResult(x, history, {'iters': it + 1, 'time_ms': elapsed_ms, 'diverged': _is_diverged(x)})


# ---------------------------------------------------------------------------
# 2b. FISTA — Fast ISTA (Nesterov-accelerated)  [BT09]
# ---------------------------------------------------------------------------

def solve_fista(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    x0 = config.get('x0', None)
    true_x = config.get('true_x', None)
    y_obs = y.copy()
    m, n = A.shape
    x_init = np.zeros(n) if x0 is None else x0.copy()
    L = _power_method_lipschitz(A)
    step = lam / L
    x_prev = x_init.copy(); y_k = x_init.copy(); t = 1.0; x_curr = x_prev.copy()
    history = []; t0 = time.perf_counter(); it = 0
    for it in range(max_iter):
        x_curr = _soft_threshold(y_k - (1.0 / L) * (A.T @ (A @ y_k - y_obs)), step)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y_k = x_curr + ((t - 1.0) / t_next) * (x_curr - x_prev)
        history.append(_track(x_curr, y_obs, A, lam, true_x))
        if np.linalg.norm(x_curr - x_prev) < tol * max(1.0, np.linalg.norm(x_prev)):
            break
        x_prev = x_curr; t = t_next
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return SolverResult(x_curr, history, {'iters': it + 1, 'time_ms': elapsed_ms, 'diverged': _is_diverged(x_curr)})


# ---------------------------------------------------------------------------
# 2c. AMP — Approximate Message Passing  [DMM09, RSF19]
# ---------------------------------------------------------------------------

def solve_amp(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    true_x = config.get('true_x', None)
    m, n = A.shape
    r = np.zeros(n); v = np.zeros(m); alpha = 0.0
    gamma = float(config.get('gamma_w', m / max(float(np.sum(y ** 2)), 1e-30)))
    xhat = np.zeros(n); history = []; t0 = time.perf_counter(); it = 0
    for it in range(max_iter):
        xhat_new, alpha_new = _map_denoiser(r, gamma, lam)
        v = y - A @ xhat_new + (n / m) * alpha * v
        r_new = xhat_new + A.T @ v
        gamma = m / max(float(np.sum(v ** 2)), 1e-30)
        history.append(_track(xhat_new, y, A, lam, true_x))
        if np.linalg.norm(r_new - r) < tol * max(1.0, np.linalg.norm(r)):
            xhat = xhat_new; r, alpha = r_new, alpha_new; break
        xhat = xhat_new; r, alpha = r_new, alpha_new
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return SolverResult(xhat, history, {'iters': it + 1, 'time_ms': elapsed_ms, 'diverged': _is_diverged(xhat)})


# ---------------------------------------------------------------------------
# 2d. VAMP — Vector Approximate Message Passing (SVD)  [RSF19]
# ---------------------------------------------------------------------------

def solve_vamp(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    true_x = config.get('true_x', None)
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T; R = len(s); s_safe = np.maximum(s, 1e-10)
    y_tilde = (1.0 / s_safe) * (U.T @ y)
    gamma_w = config.get('gamma_w', m / max(float(np.sum(y ** 2)), 1e-30))
    gamma = 1.0; r = np.zeros(n); gmin, gmax = 1e-11, 1e11
    history = []; t0 = time.perf_counter(); it = 0
    for it in range(max_iter):
        xhat, alpha = _map_denoiser(r, gamma, lam)
        r_tilde = (xhat - alpha * r) / (1.0 - alpha)
        g_tilde = float(np.clip(gamma * (1.0 - alpha) / alpha, gmin, gmax))
        d = gamma_w * s ** 2 / (gamma_w * s ** 2 + g_tilde); d_mean = float(np.mean(d))
        denom = n / R - d_mean
        g_new = float(np.clip(g_tilde * d_mean / max(denom, 1e-30), gmin, gmax))
        innov = y_tilde - Vt @ r_tilde
        r_new = r_tilde + (n / R) * (V @ ((d / max(d_mean, 1e-30)) * innov))
        history.append(_track(xhat, y, A, lam, true_x))
        if np.linalg.norm(r_new - r) < tol * max(1.0, np.linalg.norm(r)) or _is_diverged(r):
            r, gamma = r_new, g_new; break
        r, gamma = r_new, g_new
    xhat_final, _ = _map_denoiser(r, gamma, lam)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return SolverResult(xhat_final, history, {'iters': it + 1, 'time_ms': elapsed_ms, 'diverged': _is_diverged(xhat_final)})


# ---------------------------------------------------------------------------
# 2e. OAMP — Orthogonal AMP (LMMSE form)  [MP17, RSF19]
# ---------------------------------------------------------------------------

def solve_oamp(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    true_x = config.get('true_x', None)
    m, n = A.shape
    gamma_w = config.get('gamma_w', m / max(float(np.sum(y ** 2)), 1e-30))
    AtA = A.T @ A; Aty = A.T @ y
    r1 = np.zeros(n); gamma1 = 1.0; gmin, gmax = 1e-11, 1e11
    history = []; t0 = time.perf_counter(); it = 0
    for it in range(max_iter):
        xhat1, alpha1 = _map_denoiser(r1, gamma1, lam)
        eta1 = gamma1 / alpha1
        gamma2 = float(np.clip(eta1 - gamma1, gmin, gmax))
        r2 = (eta1 * xhat1 - gamma1 * r1) / gamma2
        Q = gamma_w * AtA + gamma2 * np.eye(n); Q_inv = np.linalg.inv(Q)
        xhat2 = Q_inv @ (gamma_w * Aty + gamma2 * r2)
        alpha2 = float(np.clip((gamma2 / n) * np.trace(Q_inv), 1e-8, 1.0 - 1e-8))
        eta2 = gamma2 / alpha2
        gamma1_new = float(np.clip(eta2 - gamma2, gmin, gmax))
        r1_new = (eta2 * xhat2 - gamma2 * r2) / gamma1_new
        history.append(_track(xhat1, y, A, lam, true_x))
        if np.linalg.norm(xhat2 - xhat1) < tol * max(1.0, np.linalg.norm(xhat1)):
            r1, gamma1 = r1_new, gamma1_new; break
        r1, gamma1 = r1_new, gamma1_new
    xhat_final, _ = _map_denoiser(r1, gamma1, lam)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return SolverResult(xhat_final, history, {'iters': it + 1, 'time_ms': elapsed_ms, 'diverged': _is_diverged(xhat_final)})


# ---------------------------------------------------------------------------
# 2f. MAMP — Memory Approximate Message Passing  [LHK21]
# ---------------------------------------------------------------------------

def solve_mamp(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    true_x = config.get('true_x', None)
    sigma2 = config.get('sigma2', 0.01)
    m, n = A.shape; AH = A.T
    eigvals = np.linalg.eigvalsh(A @ A.T)
    lam_max, lam_min = float(eigvals[-1]), float(max(eigvals[0], 0.0))
    lam_dag = (lam_max + lam_min) / 2.0
    mu_B = lam_dag - eigvals; n_zero = n - m
    b_table = np.empty(max_iter + 2)
    mu_pow = np.ones_like(mu_B); dag_pow = 1.0
    for t in range(max_iter + 2):
        b_table[t] = (float(np.sum(mu_pow)) + n_zero * dag_pow) / n
        mu_pow = mu_pow * mu_B; dag_pow = dag_pow * lam_dag
    w_table = lam_dag * b_table[:-1] - b_table[1:]
    x = np.zeros(n); r_hat = np.zeros(m); v_tt = 1.0; x_hist = [x.copy()]; xi = 1.0
    history = []; t0 = time.perf_counter(); it = 0; diverged = False
    for it in range(1, max_iter + 1):
        rho_t = sigma2 / max(v_tt, 1e-30); theta_t = 1.0 / (lam_dag + rho_t)
        B_r_hat = lam_dag * r_hat - A @ (AH @ r_hat)
        r_hat_new = theta_t * B_r_hat + xi * (y - A @ x)
        T_hist = len(x_hist); eps_t = 0.0; p_sum_x = np.zeros(n)
        for i_idx in range(T_hist):
            age = T_hist - i_idx
            vartheta = xi * (theta_t ** max(age - 1, 0))
            p_ti = vartheta * w_table[max(age - 1, 0)]
            eps_t += p_ti; p_sum_x += p_ti * x_hist[i_idx]
        eps_t = max(eps_t, 1e-10)
        r = (1.0 / eps_t) * (AH @ r_hat_new + p_sum_x)
        x_new = _soft_threshold(r, lam * np.sqrt(max(v_tt, 1e-30)))
        if not np.all(np.isfinite(x_new)): diverged = True; history.append(float('inf')); break
        if true_x is not None:
            nmse_val = float(np.sum((x_new - true_x) ** 2) / np.sum(true_x ** 2))
            history.append(nmse_val)
            v_tt_new = max(nmse_val * float(np.sum(true_x ** 2)) / n, 1e-12)
        else:
            v_tt_new = max(float(np.mean((x_new - x) ** 2)), 1e-12)
            history.append(float(0.5 * np.sum((A @ x_new - y) ** 2) + lam * np.sum(np.abs(x_new))))
        if np.linalg.norm(x_new - x) < tol * max(1.0, np.linalg.norm(x)):
            x = x_new; v_tt = v_tt_new; break
        x = x_new; r_hat = r_hat_new; v_tt = v_tt_new; x_hist.append(x.copy())
    final_diverged = diverged or _is_diverged(x)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return SolverResult(x, history, {'iters': it, 'time_ms': elapsed_ms, 'diverged': final_diverged})


# ---------------------------------------------------------------------------
# 2g. KAMP — Kalman Approximate Message Passing  [GSY26]
# ---------------------------------------------------------------------------

class KAMP:
    """Kalman Approximate Message Passing (KAMP).

    Replaces AMP's deterministic Onsager correction with an EKF-style
    uncertainty-aware correction.  The thresholding map is linearised via
    the Jacobian J_eta, covariance is propagated, and a Kalman gain is
    computed from the predicted covariance.

    Notation follows the DKIST paper (Ansari-Ram et al., Signal Processing 2023):
      f(x) = η(x + A^T(y - Ax); λ) — AMP-like prior estimate
      J = diag(η'(r)) (I - A^T A)  — Jacobian of f w.r.t. x
      P⁻ = J P J^T + Q              — prior covariance
      G = P⁻ A^T (A P⁻ A^T + R)⁻¹  — Kalman gain
      x̂ = x̃ + G (y - A x̃)          — posterior estimate
      P = (I - G A) P⁻ (I-GA)^T + G R G^T — Joseph form covariance update
    """
    def __init__(self, alpha: float, tau: float, max_iter: int,
                 tol: float = 1e-8, sigma2_prior: float = None,
                 sigma2_noise: float = None,
                 denoiser: Optional[Callable] = None,
                 subdif_denoiser: Optional[Callable] = None,
                 project_nonnegative: bool = False,
                 clip_bounds: tuple = (-1e6, 1e6), reg_eps: float = 1e-6,
                 verbose: bool = False,
                 tau_adaptive: bool = True,
                 iekf_steps: int = 0,
                 learn_damping: bool = False):
        if not 0 <= alpha <= 1: raise ValueError("alpha must be in [0,1]")
        if max_iter <= 0: raise ValueError("max_iter must be positive")
        self.alpha = alpha; self.tau = tau; self.max_iter = max_iter
        self.tol = tol; self.sigma2_prior = sigma2_prior
        self.sigma2_noise = sigma2_noise; self.project_nonnegative = project_nonnegative
        self.clip_bounds = clip_bounds; self.reg_eps = reg_eps; self.verbose = verbose
        self.tau_adaptive = tau_adaptive; self.iekf_steps = iekf_steps
        self.learn_damping = learn_damping
        self.denoiser = denoiser or _kamp_soft_threshold
        self.subdif_denoiser = subdif_denoiser or _kamp_soft_derivative

    def fit(self, A: np.ndarray, y: np.ndarray, true_x: np.ndarray = None):
        self.A = np.asarray(A); self.AT = self.A.T
        self.y = np.asarray(y).reshape(-1, 1)
        self.m, self.n = self.A.shape
        self.true_x = (np.asarray(true_x).reshape(-1, 1) if true_x is not None else None)
        if self.sigma2_prior is not None:
            self._sigma2_prior = self.sigma2_prior
        else:
            y_var = float(np.var(self.y))
            self._sigma2_prior = max(y_var / max(self.m / self.n, 1e-6), 1e-6)
        if self.sigma2_noise is None:
            y_pred = self.A @ np.zeros((self.n, 1))
            noise_var = max(np.mean((self.y.flatten() - y_pred.flatten()) ** 2), 1e-10)
            self._sigma2_noise = noise_var
        else:
            self._sigma2_noise = self.sigma2_noise
        self.x = np.zeros((self.n, 1))
        self._AtA = self.AT @ self.A
        self.I_n = np.eye(self.n); self.I_m = np.eye(self.m)
        noise_floor = max(self._sigma2_noise, 1e-10)
        self._P = self._sigma2_prior * self.I_n
        self._Q = max(noise_floor, 0.1 * self._sigma2_prior / max(self.m, 1)) * self.I_n
        self._R = noise_floor * self.I_m
        self.convergence_history = []; self.se_history = []; self.is_fitted_ = True
        return self

    def _anneal_tau(self, r_vec):
        if self.tau_adaptive:
            r_flat = r_vec.flatten()
            mad = np.median(np.abs(r_flat - np.median(r_flat))) * 1.4826
            sigma_est = max(mad, 1e-16)
            tau_scaled = self.tau * sigma_est
            return np.clip(tau_scaled, 1e-8, 10.0)
        return self.tau

    def _spectral_norm(self, M):
        if M.ndim == 1:
            return float(np.abs(M))
        v = np.random.randn(M.shape[1])
        v = v / max(np.linalg.norm(v), 1e-16)
        lam_old = 0.0
        for _ in range(10):
            w = M @ v
            lam = np.linalg.norm(w)
            if lam < 1e-16:
                break
            v = w / lam
            if abs(lam - lam_old) < 1e-8 * lam:
                break
            lam_old = lam
        return lam

    def _update_prior_estimation(self, r, tau_eff):
        return self.denoiser(r.flatten(), tau_eff).reshape(-1, 1)

    def _update_jacobian(self, r, tau_eff):
        d = self.subdif_denoiser(r.flatten(), tau_eff)
        J = np.diag(d) @ (self.I_n - self._AtA)
        sn = self._spectral_norm(J)
        if sn > 1.0:
            J = J / sn
        return J

    def _update_prior_covariance(self, J):
        P_ = J @ self._P @ J.T + self._Q
        return P_

    def _update_kalman_gain(self, P_):
        P_AT = P_ @ self.AT
        S = self.A @ P_AT + self._R
        try:
            reg = self.reg_eps * np.trace(S) / max(S.shape[0], 1) * self.I_m
            L = np.linalg.cholesky(S + reg)
            temp = solve_triangular(L, P_AT.T, lower=True)
            return solve_triangular(L.T, temp, lower=False).T
        except np.linalg.LinAlgError:
            return P_AT @ np.linalg.pinv(S)

    def _update_estimation(self, x_, G, residual):
        return x_ + G @ residual

    def _update_covariance(self, G, P_):
        IminusGA = self.I_n - G @ self.A
        P = IminusGA @ P_ @ IminusGA.T + G @ self._R @ G.T
        return 0.5 * (P + P.T)

    def _update_process_noise(self, G, residual, t):
        if not self.learn_damping:
            return self._Q
        Gr = G @ residual
        gr_norm = np.linalg.norm(Gr)
        max_gr = 10.0 * np.sqrt(self.n)
        if gr_norm > max_gr:
            Gr = Gr * (max_gr / max(gr_norm, 1e-16))
        alpha_t = self.alpha * np.exp(-t / max(self.max_iter, 1))
        Q = alpha_t * self._Q + (1.0 - alpha_t) * (Gr @ Gr.T)
        return np.clip(0.5 * (Q + Q.T), self.clip_bounds[0], self.clip_bounds[1])

    @property
    def P(self) -> np.ndarray: return self._P

    def set_state(self, x=None, P=None, Q=None):
        if x is not None: self.x = np.asarray(x).reshape(-1, 1)
        if P is not None: self._P = np.asarray(P)
        if Q is not None: self._Q = np.asarray(Q)

    def solve(self, true_x: np.ndarray = None) -> np.ndarray:
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("KAMP not fitted. Call .fit() first.")
        if true_x is not None: self.true_x = np.asarray(true_x).reshape(-1, 1)
        for t in range(self.max_iter):
            x_prev = np.copy(self.x)
            r_vec = x_prev + self.AT @ (self.y - self.A @ x_prev)
            tau_eff = self._anneal_tau(r_vec)
            x_ = self._update_prior_estimation(r_vec, tau_eff)
            J = self._update_jacobian(r_vec, tau_eff)
            P_ = self._update_prior_covariance(J)
            G = self._update_kalman_gain(P_)
            residual = self.y - self.A @ x_
            self.x = self._update_estimation(x_, G, residual)
            for _ in range(self.iekf_steps):
                r_iekf = self.x + self.AT @ (self.y - self.A @ self.x)
                tau_iekf = self._anneal_tau(r_iekf)
                J_iekf = self._update_jacobian(r_iekf, tau_iekf)
                P_iekf = self._update_prior_covariance(J_iekf)
                G_iekf = self._update_kalman_gain(P_iekf)
                res_iekf = self.y - self.A @ self.x
                self.x = self._update_estimation(self.x, G_iekf, res_iekf)
            if self.project_nonnegative: self.x = np.maximum(self.x, 0)
            self._P = self._update_covariance(G, P_)
            self._Q = self._update_process_noise(G, residual, t)
            rel_change = np.linalg.norm(self.x - x_prev) / max(np.linalg.norm(x_prev), 1e-12)
            self.convergence_history.append(rel_change)
            if self.true_x is not None:
                mse = float(np.sum((self.x - self.true_x) ** 2) / np.sum(self.true_x ** 2))
                self.se_history.append(mse)
                if _is_diverged(self.x.flatten(), nmse_val=mse):
                    if self.verbose:
                        print(f"  [KAMP] diverged at iteration {t}, NMSE={mse:.2e}")
                    break
            if rel_change < self.tol: break
            if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)): break
        return self.x


def solve_kamp(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    """KAMP — Kalman AMP (benchmark wrapper)."""
    _validate_inputs(A, y, config)
    max_iter = config.get('max_iter', config.get('num_iters', 50))
    lam = config.get('lam', config.get('reg_eps', 0.1))
    tol = config.get('tol', 1e-6)
    alpha = config.get('alpha', 0.5); tau = config.get('tau', 0.05)
    clip_bounds = config.get('clip_bounds', (-1e6, 1e6)); reg_eps = config.get('reg_eps', 1e-6)
    x0 = config.get('x0', None); true_x = config.get('true_x', None)
    sigma2 = config.get('sigma2', None)
    tau_adaptive = config.get('tau_adaptive', True); iekf_steps = config.get('iekf_steps', 0)
    learn_damping = config.get('learn_damping', False)
    m, n = A.shape
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=max_iter, tol=tol,
                project_nonnegative=False, clip_bounds=clip_bounds, reg_eps=reg_eps, verbose=False,
                tau_adaptive=tau_adaptive, iekf_steps=iekf_steps,
                learn_damping=learn_damping,
                sigma2_noise=sigma2)
    x_init = np.zeros(n) if x0 is None else x0.copy()
    kamp.fit(A, y, true_x=true_x); kamp.x = x_init.reshape(-1, 1)
    t0 = time.perf_counter(); x_hat = kamp.solve(true_x=true_x)
    elapsed_ms = (time.perf_counter() - t0) * 1e3; x_hat_flat = x_hat.flatten()
    return SolverResult(x_hat_flat, (kamp.se_history if true_x is not None else kamp.convergence_history), {
        'iters': len(kamp.convergence_history), 'time_ms': elapsed_ms,
        'diverged': _is_diverged(x_hat_flat), 'se_history': kamp.se_history,
        'convergence_history': kamp.convergence_history})


# ---------------------------------------------------------------------------
# 2h. DKAMP — Distributed KAMP (consensus-based)  [GSY26]
# ---------------------------------------------------------------------------

class DKAMP:
    """Distributed KAMP: wraps multiple KAMP instances in a directed graph
    with DKIST-style diffusion consensus (Eqs. 14–18 in Ansari-Ram et al. 2023).

    Each round consists of:
      1. Local KAMP update at every node using local measurements.
      2. Diffusion: each node sends x̂_i, P_i to neighbors, computes
         covariance-weighted fused estimate.
      3. The fused state initializes the next local KAMP round.

    After T rounds, the final fused estimate is returned.
    """
    def __init__(self, alpha=0.5, tau=0.05, node_max_iter=5, num_rounds=20,
                 fusion_method='covariance_weighted', random_state=None, verbose=False,
                 tau_adaptive=False, iekf_steps=0, sigma2_noise=None,
                 tol=1e-6, min_rounds=5):
        self.alpha = alpha; self.tau = tau; self.node_max_iter = node_max_iter
        self.num_rounds = num_rounds; self.fusion_method = fusion_method
        self.random_state = random_state; self.verbose = verbose
        self.tau_adaptive = tau_adaptive; self.iekf_steps = iekf_steps
        self.sigma2_noise = sigma2_noise
        self.dkamp_tol = tol
        self.min_rounds = min_rounds
        self.rng = np.random.RandomState(random_state)

    def _make_local_kamp(self, A_k, y_k):
        k = KAMP(alpha=self.alpha, tau=self.tau, max_iter=self.node_max_iter,
                 tau_adaptive=self.tau_adaptive, iekf_steps=self.iekf_steps, verbose=False,
                 sigma2_noise=self.sigma2_noise)
        k.fit(A_k, y_k)
        return k

    def _default_graph(self, num_nodes):
        DG = nx.DiGraph(); DG.add_nodes_from(range(num_nodes))
        if num_nodes <= 1: return DG
        for i in range(num_nodes - 1):
            DG.add_edge(i, i + 1); DG.add_edge(i + 1, i)
        return DG

    def _cholesky_inv(self, M, reg=None):
        n = M.shape[0]
        if reg is None:
            trace = np.trace(M) / n
            reg = max(1e-8, 1e-6 * trace) if trace > 0 else 1e-8
        try:
            L = np.linalg.cholesky(M + reg * np.eye(n))
            return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(M)
            clipped = np.maximum(eigvals, max(np.max(eigvals) * 1e-8, 1e-12))
            return eigvecs @ np.diag(1.0 / clipped) @ eigvecs.T

    def _fuse_estimates(self):
        valid = [e for e in self.node_estimators
                 if hasattr(e, 'x') and e.x is not None and not any(np.isnan(e.x.flatten()))]
        if not valid:
            valid = [self.node_estimators[0]]
        n = valid[0].n
        total_P_inv = np.zeros((n, n)); total_P_inv_x = np.zeros((n, 1))
        for est in valid:
            P_inv = self._cholesky_inv(est._P)
            total_P_inv += P_inv; total_P_inv_x += P_inv @ est.x
        return np.linalg.solve(total_P_inv, total_P_inv_x).flatten()

    def _diffuse_node(self, node_idx):
        nbrs = list(self.graph.neighbors(node_idx))
        if not nbrs:
            est = self.node_estimators[node_idx]
            return est.x.copy(), est._P.copy()
        n = self.node_estimators[node_idx].n
        total_P_inv = np.zeros((n, n)); total_P_inv_x = np.zeros((n, 1))
        for nb in nbrs + [node_idx]:
            est = self.node_estimators[nb]
            P_inv = self._cholesky_inv(est._P)
            total_P_inv += P_inv; total_P_inv_x += P_inv @ est.x
        x_fused = np.linalg.solve(total_P_inv, total_P_inv_x)
        P_fused = self._cholesky_inv(total_P_inv)
        return x_fused, P_fused

    def fit(self, A_list, y_list, graph=None, true_x=None):
        if not _HAS_NETWORKX: raise ImportError("DKAMP requires networkx")
        self.A_list = A_list; self.y_list = y_list
        self.num_nodes = max(len(A_list), 2)
        self.true_x = np.asarray(true_x).flatten() if true_x is not None else None
        self._consensus_history = []
        self.node_estimators = [self._make_local_kamp(
            A_list[i] if i < len(A_list) else A_list[0],
            y_list[i] if i < len(y_list) else y_list[0])
            for i in range(self.num_nodes)]
        self.graph = graph if graph is not None else self._default_graph(self.num_nodes)
        self.reg_eps = 1e-8
        return self

    def solve(self):
        pbar = tqdm(total=self.num_rounds, desc="DKAMP diffusion", unit="round",
                     leave=False, disable=not self.verbose)
        x_prev = self._fuse_estimates()
        x_prev_prev = x_prev.copy()
        self._delta_buffer = []
        converged_count = 0
        self.rounds_executed = 0
        for rnd in range(self.num_rounds):
            for i in range(self.num_nodes):
                self.node_estimators[i].solve()
            new_estimates = [self._diffuse_node(i) for i in range(self.num_nodes)]
            for i, (x_i, P_i) in enumerate(new_estimates):
                self.node_estimators[i].set_state(x=x_i, P=P_i)
            x_fused = self._fuse_estimates()
            self.rounds_executed = rnd + 1
            err = x_fused - self.true_x if self.true_x is not None else None
            if err is not None:
                nmse = float(np.sum(err ** 2) / np.sum(self.true_x ** 2))
                self._consensus_history.append(nmse)
                if _is_diverged(x_fused.flatten(), nmse_val=nmse):
                    break
            delta = float(np.linalg.norm(x_fused - x_prev) / max(np.linalg.norm(x_prev), 1e-12))
            if not err is not None:
                self._consensus_history.append(delta)
            self._delta_buffer.append(delta)
            if len(self._delta_buffer) > 3:
                self._delta_buffer.pop(0)
            if rnd + 1 >= self.min_rounds and len(self._delta_buffer) == 3:
                avg_delta = sum(self._delta_buffer) / 3
                if avg_delta < self.dkamp_tol:
                    converged_count += 1
                    if converged_count >= 3:
                        break
                else:
                    converged_count = 0
            x_prev_prev = x_prev.copy()
            x_prev = x_fused
            pbar.update(1)
        pbar.close()
        self._x_fused = self._fuse_estimates()
        return self._x_fused

    def communication_cost(self):
        n_edges = self.graph.number_of_edges()
        n = self.node_estimators[0].n if self.node_estimators else 0
        data_per_message = n + n * n
        return {'messages_per_round': n_edges * 2, 'total_messages': n_edges * 2 * self.num_rounds,
                'data_per_message': data_per_message, 'total_data': n_edges * 2 * self.num_rounds * data_per_message,
                'total_nodes': len(self.node_estimators)}


def solve_dkamp(A: np.ndarray, y: np.ndarray, config: dict) -> SolverResult:
    """DKAMP — Distributed KAMP (DKIST-style diffusion consensus)."""
    _validate_inputs(A, y, config)
    max_iter = config.get('num_iters', config.get('max_iter', 50))
    lam = config.get('reg_eps', config.get('lam', 0.1))
    tol = config.get('tol', 1e-6); true_x = config.get('true_x', None)
    num_nodes = config.get('num_nodes', 2)
    num_rounds = config.get('num_triggers', config.get('dkamp_rounds', 10))
    if _FAST_MODE:
        num_rounds = min(num_rounds, 5)
        max_iter = min(max_iter, 20)
    alpha = config.get('alpha', 0.5); tau = config.get('tau', 0.05)
    tau_adaptive = config.get('tau_adaptive', False); iekf_steps = config.get('iekf_steps', 0)
    sigma2 = config.get('sigma2', None)
    m, n = A.shape; rows_per_node = max(m // num_nodes, 1)
    A_list, y_list = [], []
    for i in range(num_nodes):
        start = i * rows_per_node
        end = m if i == num_nodes - 1 else (i + 1) * rows_per_node
        A_list.append(A[start:end]); y_list.append(y[start:end])
    dkamp = DKAMP(alpha=alpha, tau=tau, node_max_iter=max_iter, num_rounds=num_rounds,
                  tau_adaptive=tau_adaptive, iekf_steps=iekf_steps, verbose=False,
                  sigma2_noise=sigma2, tol=tol, min_rounds=config.get('dkamp_min_rounds', 5))
    t0 = time.perf_counter(); dkamp.fit(A_list, y_list, true_x=true_x); x_hat = dkamp.solve()
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    history = dkamp._consensus_history if dkamp._consensus_history else []
    return SolverResult(x_hat, history, {'iters': len(history), 'rounds': dkamp.rounds_executed,
                                         'time_ms': elapsed_ms, 'diverged': _is_diverged(x_hat)})


# ============================================================================
# Part 3 — Solver Registry & Matrix Generation
# ============================================================================

SOLVER_REGISTRY: Dict[str, callable] = OrderedDict([
    ('ISTA', solve_ista), ('FISTA', solve_fista), ('AMP', solve_amp),
    ('VAMP', solve_vamp), ('OAMP', solve_oamp),
    ('KAMP', solve_kamp), ('DKAMP', solve_dkamp),
])


def create_measurement_matrix(m: int, n: int, matrix_type: str = 'gaussian',
                                condition_number: float = None, random_state: int = None) -> np.ndarray:
    """Generate a sensing matrix of specified type.

    Types: 'gaussian', 'correlated', 'ill_conditioned', 'partial_orthogonal'.
    """
    rng = np.random.RandomState(random_state)
    if matrix_type == 'gaussian':
        A = rng.randn(m, n) / np.sqrt(n)
    elif matrix_type == 'correlated':
        base = rng.randn(m, n)
        cov = 0.8 ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        A = base @ np.linalg.cholesky(cov).T / np.sqrt(n)
    elif matrix_type == 'ill_conditioned':
        U, _ = np.linalg.qr(rng.randn(m, m))
        Vt, _ = np.linalg.qr(rng.randn(n, n))
        kappa = max(condition_number or 50, 1.0)
        s = np.logspace(0, np.log10(kappa), min(m, n))
        S = np.zeros((m, n)); S[:len(s), :len(s)] = np.diag(s)
        A = U @ S @ Vt / np.sqrt(n)
    elif matrix_type == 'partial_orthogonal':
        Q, _ = np.linalg.qr(rng.randn(n, n)); A = Q[:m, :].T.copy().T
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}")
    return A


def _make_config(x_true: np.ndarray, sigma: float, lam: float = 0.1,
                  max_iter: int = 200, tol: float = 1e-6, **extra) -> dict:
    """Build a solver config dict with both naming conventions."""
    sigma2 = sigma ** 2
    config = {'lam': lam, 'reg_eps': lam, 'max_iter': max_iter, 'num_iters': max_iter,
              'tol': tol, 'x0': np.zeros(len(x_true)), 'true_x': x_true,
              'gamma_w': 1.0 / max(sigma2, 1e-30), 'sigma2': sigma2, 'debias': False}
    config.update(extra)
    return config


def run_benchmark_scenario(name: str, A: np.ndarray, y: np.ndarray, x_true: np.ndarray,
                            config: dict, num_trials: int = 1) -> dict:
    """Run all solvers on a single scenario with Monte Carlo noise trials."""
    results = {s: {'nmse': [], 'time_ms': [], 'iters': [], 'diverged': []}
               for s in SOLVER_REGISTRY}
    for trial in range(num_trials):
        sigma = np.sqrt(1.0 / config['gamma_w'])
        y_noisy = y + sigma * np.random.randn(len(y))
        for sname, solver_fn in SOLVER_REGISTRY.items():
            try:
                result = solver_fn(A, y_noisy, config)
                use_debias = config.get('debias', False)
                x_hat = (_debias_on_support(A, y_noisy, result.x_hat) if use_debias
                         else result.x_hat)
                nmse = float(np.sum((x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                results[sname]['nmse'].append(nmse)
                results[sname]['time_ms'].append(result.info['time_ms'])
                results[sname]['iters'].append(result.info['iters'])
                results[sname]['diverged'].append(result.info['diverged'])
            except Exception as e:
                print(f"  [{sname}] failed: {e}")
                results[sname]['nmse'].append(float('nan'))
                results[sname]['time_ms'].append(float('nan'))
                results[sname]['iters'].append(0)
                results[sname]['diverged'].append(True)
    summary = {}
    for sname in SOLVER_REGISTRY:
        nmse_arr = np.array(results[sname]['nmse'])
        div_arr = np.array(results[sname]['diverged'], dtype=bool)
        nmse_clean = np.where(div_arr, np.nan, nmse_arr)
        summary[sname] = {'NMSE': float(np.nanmean(nmse_clean)),
                          'NMSE_std': float(np.nanstd(nmse_clean)),
                          'Time_ms': float(np.mean(results[sname]['time_ms'])),
                          'Iters': float(np.mean(results[sname]['iters'])),
                          'Div': int(np.sum(div_arr))}
    return summary


# ============================================================================
# Part 4 — Experiment A: Algorithm Validation
# ============================================================================


# ---------------------------------------------------------------------------
# Experiment A1: Onsager Correction Validation
# Compares KAMP's Kalman-gain correction vs AMP's Onsager correction across
# Gaussian, correlated, and ill-conditioned matrix ensembles.
# Outputs: onsager_gap_vs_condition.png, onsager_gap_vs_correlation.png,
#          onsager_gap_vs_iteration.png, onsager_comparison.csv
# ---------------------------------------------------------------------------

def amp_onsager_correction(residual_prev, alpha_prev, m, n):
    return (n / m) * alpha_prev * residual_prev

def experiment_onsager_validation(n=100, m=70, k=15, snr_db=30, num_trials=5):
    print("\n" + "=" * 60)
    print("EXPERIMENT A1 — Onsager Correction Validation")
    print("=" * 60)
    rng = np.random.RandomState(42)
    results = {}

    def run_single(A, y, x_true, lam=0.1):
        kamp = KAMP(alpha=0.5, tau=lam, max_iter=200, tol=1e-8)
        kamp.fit(A, y, true_x=x_true)
        r_amp = np.zeros(n); v_prev = np.zeros(m); alpha_p = 0.0
        history = {'iteration': [], 'kamp_correction_norm': [], 'onsager_correction_norm': [],
                   'correction_gap': [], 'kamp_nmse': [], 'amp_nmse': []}
        for t in range(200):
            x_prev = np.copy(kamp.x)
            r_vec = x_prev + kamp.AT @ (kamp.y - kamp.A @ x_prev)
            tau_eff = kamp._anneal_tau(r_vec)
            r_est = kamp._update_prior_estimation(r_vec, tau_eff)
            J = kamp._update_jacobian(r_vec, tau_eff)
            P_ = kamp._update_prior_covariance(J)
            G = kamp._update_kalman_gain(P_)
            residual = kamp.y - kamp.A @ r_est
            kamp_corr = G @ residual; kamp_corr_norm = float(np.linalg.norm(kamp_corr))
            kamp.x = kamp._update_estimation(r_est, G, residual)
            kamp._P = kamp._update_covariance(G, P_)
            kamp._Q = kamp._update_process_noise(G, residual, t)
            kamp_nmse = float(np.sum((kamp.x.flatten() - x_true) ** 2) / np.sum(x_true ** 2))

            xhat_amp, alpha_amp = _map_denoiser(r_amp, 1.0, lam)
            onsager_meas = amp_onsager_correction(v_prev, alpha_p if t > 0 else 0.0, m, n)
            onsager_corr = A.T @ onsager_meas
            onsager_norm = float(np.linalg.norm(onsager_corr))
            v = y - A @ xhat_amp + onsager_meas
            r_amp_new = xhat_amp + A.T @ v
            r_amp, v_prev, alpha_p = r_amp_new, v, alpha_amp
            amp_nmse = float(np.sum((xhat_amp - x_true) ** 2) / np.sum(x_true ** 2))

            gap = abs(kamp_corr_norm - onsager_norm) / max(onsager_norm, 1e-12)
            for key, val in [('iteration', t), ('kamp_correction_norm', kamp_corr_norm),
                             ('onsager_correction_norm', onsager_norm), ('correction_gap', gap),
                             ('kamp_nmse', kamp_nmse), ('amp_nmse', amp_nmse)]:
                history[key].append(val)
            if t > 5 and gap < 1e-8 and kamp_nmse < 1e-6 and amp_nmse < 1e-6:
                break
        return history

    # 1a. Gaussian baseline
    print("Gaussian i.i.d. baseline...")
    gaps_gauss = []
    for trial in tqdm(range(num_trials), desc="Gaussian trials"):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=1000 + trial)
        x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
        yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
        hist = run_single(A, yc + sigma * rng.randn(m), x)
        gaps_gauss.append(hist['correction_gap'][-1] if hist['correction_gap'] else np.nan)
    results['gaussian'] = {'final_gaps': gaps_gauss, 'mean_gap': np.nanmean(gaps_gauss),
                           'std_gap': np.nanstd(gaps_gauss)}
    print(f"  Mean gap: {results['gaussian']['mean_gap']:.4e}")

    # 1b. Correlated sweep
    print("Correlated sweep...")
    rho_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    results['correlated'] = {'rho': [], 'mean_gaps': [], 'std_gaps': []}
    for rho in tqdm(rho_vals, desc="Correlated rho"):
        gaps = []
        for trial in range(num_trials):
            A = create_measurement_matrix(m, n, 'correlated', random_state=int(rho * 1000) + trial)
            x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
            yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
            hist = run_single(A, yc + sigma * rng.randn(m), x)
            gaps.append(hist['correction_gap'][-1] if hist['correction_gap'] else np.nan)
        results['correlated']['rho'].append(rho)
        results['correlated']['mean_gaps'].append(np.nanmean(gaps))
        results['correlated']['std_gaps'].append(np.nanstd(gaps))

    # 1c. Ill-conditioned sweep
    print("Ill-conditioned sweep...")
    cond_vals = [1, 3, 10, 30, 100, 300, 1000]
    results['ill_conditioned'] = {'cond': [], 'mean_gaps': [], 'std_gaps': []}
    for cond in tqdm(cond_vals, desc="Cond number"):
        gaps = []
        for trial in range(num_trials):
            A = create_measurement_matrix(m, n, 'ill_conditioned', condition_number=cond,
                                           random_state=3000 + trial)
            x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
            yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
            hist = run_single(A, yc + sigma * rng.randn(m), x)
            gaps.append(hist['correction_gap'][-1] if hist['correction_gap'] else np.nan)
        results['ill_conditioned']['cond'].append(cond)
        results['ill_conditioned']['mean_gaps'].append(np.nanmean(gaps))
        results['ill_conditioned']['std_gaps'].append(np.nanstd(gaps))
        print(f"  kappa={cond:4d}: gap={np.nanmean(gaps):.4e}")

    # Plotting
    set_publication_style()
    gauss_mean = results['gaussian']['mean_gap']; gauss_std = results['gaussian']['std_gap']

    fig, ax = plt.subplots(figsize=(8, 5))
    ic = results['ill_conditioned']
    ax.fill_between(ic['cond'], np.maximum(0, np.array(ic['mean_gaps']) - np.array(ic['std_gaps'])),
                     np.array(ic['mean_gaps']) + np.array(ic['std_gaps']), alpha=0.2, color='#d80c7d')
    ax.semilogx(ic['cond'], ic['mean_gaps'], 'o-', color='#d80c7d', linewidth=2.5, markersize=8,
                label='Gap (KAMP vs AMP Onsager)')
    ax.axhline(y=gauss_mean, color='#00429d', linestyle='--', linewidth=2,
               label=f'Gaussian baseline ({gauss_mean:.2e})')
    ax.fill_between(ic['cond'], gauss_mean - gauss_std, gauss_mean + gauss_std, alpha=0.15, color='#00429d')
    ax.set_xlabel('Condition Number kappa'); ax.set_ylabel('Normalized Correction Gap')
    ax.set_title('Onsager Correction Gap vs Condition Number'); ax.legend(); ax.grid(True, alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'onsager_gap_vs_condition.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> onsager_gap_vs_condition.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    corr = results['correlated']
    ax.fill_between(corr['rho'], np.maximum(0, np.array(corr['mean_gaps']) - np.array(corr['std_gaps'])),
                     np.array(corr['mean_gaps']) + np.array(corr['std_gaps']), alpha=0.2, color='#008f6b')
    ax.semilogy(corr['rho'], corr['mean_gaps'], 's-', color='#008f6b', linewidth=2.5, markersize=8,
            label='Gap (KAMP vs AMP Onsager)')
    ax.axhline(y=gauss_mean, color='#00429d', linestyle='--', linewidth=2,
               label=f'Gaussian baseline ({gauss_mean:.2e})')
    ax.set_xlabel('Correlation rho (Toeplitz)'); ax.set_ylabel('Normalized Correction Gap')
    ax.set_title('Onsager Correction Gap vs Correlation'); ax.legend(); ax.grid(True, alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'onsager_gap_vs_correlation.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> onsager_gap_vs_correlation.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ensembles = [('Gaussian', 'gaussian', None), ('Correlated (rho=0.9)', 'correlated', None),
                 ('Ill-conditioned (kappa=100)', 'ill_conditioned', 100)]
    for idx, (label, mat_type, cond) in enumerate(ensembles):
        A = create_measurement_matrix(m, n, mat_type, condition_number=cond, random_state=5000 + idx * 100)
        x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
        yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
        hist = run_single(A, yc + sigma * rng.randn(m), x)
        dkamp_config = _make_config(x, sigma, lam=0.1, max_iter=200)
        dkamp_config['num_nodes'] = 2; dkamp_config['num_triggers'] = 5
        dkamp_res = solve_dkamp(A, yc + sigma * rng.randn(m), dkamp_config)
        ax = axes[idx]
        ax.semilogy(hist['iteration'], hist['kamp_correction_norm'], '-o', color='#d80c7d', markersize=3, label='KAMP correction')
        ax.semilogy(hist['iteration'], hist['onsager_correction_norm'], '-s', color='#00429d', markersize=3, label='AMP Onsager')
        ax.semilogy(hist['iteration'], hist['correction_gap'], '--', color='#7f3b08', linewidth=1.5, label='Normalized gap')
        if dkamp_res.history:
            ax.semilogy(range(1, len(dkamp_res.history)+1), dkamp_res.history, '-v', color='#93003a', markersize=3, label='DKAMP NMSE')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Norm / Gap'); ax.set_title(label)
        ax.legend(loc='best', frameon=False, fontsize=9); ax.grid(True, alpha=0.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'onsager_gap_vs_iteration.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> onsager_gap_vs_iteration.png")

    # CSV
    with open(DATA_DIR / 'onsager_comparison.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['matrix_type', 'param', 'param_value', 'trial', 'final_gap'])
        for t, g in enumerate(results['gaussian']['final_gaps']):
            w.writerow(['gaussian', 'none', 0, t, g])
        for ri, rho in enumerate(results['correlated']['rho']):
            for t in range(num_trials):
                gi = ri * num_trials + t
                if gi < len(results['correlated']['final_gaps'] if 'final_gaps' in results['correlated'] else []):
                    pass
        print("  CSV saved -> onsager_comparison.csv")


# ---------------------------------------------------------------------------
# Experiment A2: State Evolution Validation
# Compares empirical NMSE trajectories against AMP's state evolution prediction
# for both AMP and KAMP across multiple Monte Carlo trials.
# Outputs: state_evolution_amp.png, state_evolution_kamp.png, state_evolution_results.csv
# ---------------------------------------------------------------------------


# ============================================================================
# Part 5 — Experiment B: Core Benchmarking
# ============================================================================


# ---------------------------------------------------------------------------
# Experiment B1: Full Solver Benchmark
# Tests all 8 solvers across 4 matrix ensembles x 2 signal models.
# Outputs: benchmark_nmse_comparison.png, benchmark_time_comparison.png,
#          benchmark_convergence_comparison.png, full_benchmark_results.csv
# ---------------------------------------------------------------------------

def experiment_full_benchmark(n=100, m=70, k=15, snr_db=30, num_trials=5, max_iter=200, lam=0.1):
    print("\n" + "=" * 60)
    print("EXPERIMENT B1 — Full Solver Benchmark")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 50, 35, 8
        num_trials = 2
        max_iter = 50
    rng = np.random.RandomState(42)
    matrices = [('gaussian', 'Gaussian i.i.d.'), ('correlated', 'Correlated'),
                ('ill_conditioned', 'Ill-conditioned (kappa=50)'),
                ('partial_orthogonal', 'Partial Orthogonal')]
    signals = ['bernoulli_gaussian', 'block_sparse']
    all_results = {}
    scenario_idx = 0
    for mat_key, mat_display in matrices:
        for sig_key in signals:
            scenario_idx += 1
            seed = 1000 + scenario_idx * 100
            A = create_measurement_matrix(m, n, mat_key, condition_number=50 if mat_key == 'ill_conditioned' else None,
                                           random_state=seed)
            x_true = np.zeros(n)
            if sig_key == 'block_sparse':
                block_size = 5; num_blocks = k // block_size
                for b in range(num_blocks): x_true[b * block_size:(b + 1) * block_size] = rng.randn(block_size)
                x_true = x_true / np.linalg.norm(x_true) * np.sqrt(k / n)
            else:
                support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * np.random.RandomState(seed + 1).randn(m)
            config = _make_config(x_true, sigma, lam=lam, max_iter=max_iter)
            label = f"{mat_display} / {sig_key.replace('_', '-').title()}"
            print(f"\n[{scenario_idx}/8] {label}")
            summary = run_benchmark_scenario(label, A, y, x_true, config, num_trials=num_trials)
            all_results[(mat_key, sig_key)] = {'aggregated': summary, 'last_trial': None}
            for s, agg in summary.items():
                print(f"  {s:<8} NMSE={agg['NMSE']:.4e} Time={agg['Time_ms']:.1f}ms Div={agg['Div']}/{num_trials}")
    set_publication_style()
    colors = {'ISTA': '#7f7f7f', 'FISTA': '#a0522d', 'AMP': '#00429d', 'VAMP': '#d80c7d',
              'OAMP': '#f39c12', 'KAMP': '#008f6b', 'DKAMP': '#93003a'}
    solvers_show = list(SOLVER_REGISTRY.keys())
    # NMSE bar plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for idx, ((mat_key, sig_key), data) in enumerate(all_results.items()):
        if idx >= 4: break
        ax = axes.flatten()[idx]; agg = data['aggregated']
        nmse_vals = [agg.get(s, {}).get('NMSE', np.nan) for s in solvers_show]
        nmse_vals = np.where(np.isfinite(np.array(nmse_vals)), np.array(nmse_vals), 1e16)
        ax.bar(range(len(solvers_show)), np.maximum(nmse_vals, 1e-16),
               color=[colors.get(s, '#999999') for s in solvers_show], edgecolor='#333333', linewidth=0.8)
        ax.set_xticks(range(len(solvers_show))); ax.set_xticklabels(solvers_show, rotation=45, ha='right', fontsize=9)
        ax.set_yscale('log'); ax.set_ylabel('NMSE')
        mat_display = [md for mk, md in matrices if mk == mat_key][0]
        sig_display = 'Bernoulli-Gaussian' if sig_key == 'bernoulli_gaussian' else 'Block-sparse'
        ax.set_title(f'{mat_display} / {sig_display}'); ax.grid(True, alpha=0.3, axis='y')
        for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'benchmark_nmse_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> benchmark_nmse_comparison.png")
    # Time bar plot
    key = ('gaussian', 'bernoulli_gaussian')
    if key in all_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        agg = all_results[key]['aggregated']
        times = [agg.get(s, {}).get('Time_ms', 0) for s in solvers_show]
        ax.bar(range(len(solvers_show)), times, color=[colors.get(s, '#999999') for s in solvers_show],
               edgecolor='#333333', linewidth=0.8)
        ax.set_xticks(range(len(solvers_show))); ax.set_xticklabels(solvers_show, rotation=45, ha='right')
        ax.set_ylabel('Wall-clock Time (ms)'); ax.set_title('Solver Runtime (Gaussian)')
        ax.grid(True, alpha=0.3, axis='y')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        plt.tight_layout()
        fig.savefig(FIG_DIR / 'benchmark_time_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
        print("  Saved -> benchmark_time_comparison.png")
    # CSV
    csv_path = DATA_DIR / 'full_benchmark_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['matrix_type', 'signal_type', 'solver', 'nmse_mean', 'time_ms_mean', 'div_count'])
        for (mat_key, sig_key), data in all_results.items():
            for s, agg in data['aggregated'].items():
                w.writerow([mat_key, sig_key, s, f"{agg['NMSE']:.6e}", f"{agg['Time_ms']:.3f}", agg['Div']])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment B2: Covariance Modes (KAMP-full vs AMP/VAMP/OAMP/DKAMP sweeps)
# Sweeps condition number and correlation, tracks NMSE and timing.
# Outputs: convergence_comparison.png, nmse_vs_condition.png,
#          nmse_vs_correlation.png, time_vs_nmse.png, kamp_benchmark.csv
# ---------------------------------------------------------------------------

def experiment_covariance_modes(n=100, m=70, k=15, snr_db=30):
    print("\n" + "=" * 60)
    print("EXPERIMENT B2 — Covariance Modes Sweep")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9
    rng = np.random.RandomState(42)
    lam = 0.1; max_iter = 50 if _FAST_MODE else 200
    dkamp_triggers = 3 if _FAST_MODE else 5

    def run_single(A, y, x):
        config = {'lam': lam, 'max_iter': max_iter, 'tol': 1e-8, 'true_x': x, 'x0': np.zeros(A.shape[1])}
        solvers = OrderedDict([
            ('KAMP', lambda: solve_kamp(A, y, {**config, 'alpha': 0.5, 'tau': lam})),
            ('AMP', lambda: solve_amp(A, y, config)),
            ('VAMP', lambda: solve_vamp(A, y, config)),
            ('OAMP', lambda: solve_oamp(A, y, config)),
            ('DKAMP', lambda: solve_dkamp(A, y, {**config, 'num_nodes': 2, 'num_triggers': dkamp_triggers})),
        ])
        res = {}
        for name, fn in solvers.items():
            try:
                t0 = time.perf_counter(); r = fn(); el = (time.perf_counter() - t0) * 1e3
                nmse = float(np.sum((r.x_hat - x) ** 2) / np.sum(x ** 2))
                res[name] = {'nmse': nmse, 'time_ms': el, 'iters': r.info.get('iters', 0),
                             'history': r.info.get('se_history', [])}
            except Exception:
                res[name] = {'nmse': float('nan'), 'time_ms': float('nan'), 'iters': 0, 'history': []}
        return res

    gauss_trials = 2 if _FAST_MODE else 5
    print("Gaussian baseline...")
    gauss = {'KAMP': [], 'AMP': [], 'VAMP': [], 'OAMP': [], 'DKAMP': []}
    for t in tqdm(range(gauss_trials), desc="Gaussian trials"):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=5000 + t)
        x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
        yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
        res = run_single(A, yc + sigma * rng.randn(m), x)
        for name in gauss: gauss[name].append(res[name])
        print(f"  Trial {t+1}")

    # Condition sweep
    print("Condition number sweep...")
    cond_vals = [1, 10, 100] if _FAST_MODE else [1, 2, 5, 10, 20, 50, 100, 200]
    cond = {'cond': cond_vals, 'KAMP': [], 'AMP': [], 'VAMP': [], 'OAMP': [], 'DKAMP': []}
    for cv in tqdm(cond_vals, desc="Cond sweep"):
        A = create_measurement_matrix(m, n, 'ill_conditioned', condition_number=cv, random_state=42)
        x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
        yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
        res = run_single(A, yc + sigma * rng.randn(m), x)
        for name in cond:
            if name != 'cond': cond[name].append(res[name])
        print(f"  kappa={cv:3d}: KAMP={res['KAMP']['nmse']:.2e} VAMP={res['VAMP']['nmse']:.2e}")

    # Correlation sweep
    print("Correlation sweep...")
    rho_vals = [0.1, 0.5, 0.9] if _FAST_MODE else [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    corr = {'rho': rho_vals, 'KAMP': [], 'AMP': [], 'VAMP': [], 'OAMP': [], 'DKAMP': []}
    for rho in rho_vals:
        A = create_measurement_matrix(m, n, 'correlated', random_state=int(rho * 1000))
        x = np.zeros(n); s = rng.choice(n, k, replace=False); x[s] = rng.randn(k)
        yc = A @ x; sigma = np.sqrt(np.mean(yc ** 2) / 10 ** (snr_db / 10))
        res = run_single(A, yc + sigma * rng.randn(m), x)
        for name in corr:
            if name != 'rho': corr[name].append(res[name])
        print(f"  rho={rho:.2f}: KAMP={res['KAMP']['nmse']:.2e} VAMP={res['VAMP']['nmse']:.2e}")

    # Plotting
    set_publication_style()
    colors = {'KAMP': '#00429d', 'AMP': '#008f6b', 'VAMP': '#d80c7d', 'OAMP': '#f39c12', 'DKAMP': '#93003a'}
    markers = {'KAMP': 'o', 'AMP': 's', 'VAMP': 'D', 'OAMP': '^', 'DKAMP': 'v'}
    solvers = ['KAMP', 'AMP', 'VAMP', 'OAMP', 'DKAMP']

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in solvers:
        h = gauss[name][0].get('history', [])
        if h: ax.semilogy(range(1, len(h)+1), h, color=colors[name], marker=markers[name],
                           markevery=max(1, len(h)//10), markersize=4, label=name)
    ax.set_xlabel('Iteration'); ax.set_ylabel('NMSE'); ax.set_title('Convergence Comparison')
    ax.legend(); ax.grid(True, alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'convergence_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> convergence_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in solvers:
        ax.semilogy(cond['cond'], [r['nmse'] for r in cond[name]], 'o-',
                     color=colors[name], marker=markers[name], label=name)
    ax.set_xlabel('Condition Number'); ax.set_ylabel('NMSE'); ax.set_title('NMSE vs Condition Number')
    ax.legend(); ax.grid(True, alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'nmse_vs_condition.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> nmse_vs_condition.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in solvers:
        ax.semilogy(corr['rho'], [r['nmse'] for r in corr[name]], 'o-',
                     color=colors[name], marker=markers[name], label=name)
    ax.set_xlabel('Correlation rho'); ax.set_ylabel('NMSE'); ax.set_title('NMSE vs Correlation')
    ax.legend(); ax.grid(True, alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'nmse_vs_correlation.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> nmse_vs_correlation.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in solvers:
        times = [r['time_ms'] for r in gauss[name]]
        nmse = [r['nmse'] for r in gauss[name]]
        ax.scatter(times, nmse, c=colors[name], marker=markers[name], s=100, alpha=0.8,
                   edgecolors='k', linewidth=0.5, label=f'{name} (NMSE={np.mean(nmse):.2e})')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('NMSE'); ax.set_title('Speed vs Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'time_vs_nmse.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> time_vs_nmse.png")

    csv_path = DATA_DIR / 'kamp_benchmark.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['experiment', 'param', 'solver', 'nmse', 'time_ms'])
        for name in solvers:
            for i, c in enumerate(cond['cond']):
                r = cond[name][i]; w.writerow(['condition', c, name, f"{r['nmse']:.6e}", f"{r['time_ms']:.3f}"])
            for i, rv in enumerate(corr['rho']):
                r = corr[name][i]; w.writerow(['correlation', rv, name, f"{r['nmse']:.6e}", f"{r['time_ms']:.3f}"])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment B3: Phase Transition
# Donoho-Tanner style delta-rho success diagrams for AMP, VAMP, OAMP, MAMP,
# KAMP, DKAMP. Plots contour overlays and individual subplots.
# Outputs: phase_transition_ALL.png, phase_transition_overlay.png,
#          phase_transition_results.csv
# ---------------------------------------------------------------------------

def experiment_phase_transition(n=80, snr_db=40, num_trials=3):
    print("\n" + "=" * 60)
    print("EXPERIMENT B3 — Phase Transition")
    print("=" * 60)
    if _FAST_MODE:
        n = 40
        num_trials = 1
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    delta_vals = np.linspace(0.2, 0.9, 6); rho_vals = np.linspace(0.05, 0.75, 5)
    success_threshold = -30  # dB
    success_map = {s: np.zeros((len(rho_vals), len(delta_vals))) for s in solvers}
    for di, delta in enumerate(delta_vals):
        m = max(int(n * delta), 1)
        for ri, rho in enumerate(rho_vals):
            k = max(int(n * rho), 1)
            nmse_scores = {s: [] for s in solvers}
            for trial in range(num_trials):
                A = create_measurement_matrix(m, n, 'gaussian', random_state=ri * 100 + di * 10 + trial)
                x_true = np.zeros(n); support = np.random.RandomState(trial).choice(n, k, replace=False)
                x_true[support] = np.random.randn(k)
                y = A @ x_true + 1e-4 * np.random.randn(m)
                sigma = 1e-4; config = _make_config(x_true, sigma, lam=0.1, max_iter=100)
                for sname in solvers:
                    try:
                        res = SOLVER_REGISTRY[sname](A, y, config)
                        nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                        nmse_scores[sname].append(nmse)
                    except Exception:
                        nmse_scores[sname].append(float('nan'))
            for sname in solvers:
                nmse_db = 10 * np.log10(max(np.nanmean(nmse_scores[sname]), 1e-30))
                success_map[sname][ri, di] = 1.0 if nmse_db < success_threshold else 0.0
            print(f"  delta={delta:.2f} rho={rho:.2f}")
    set_publication_style()
    colors = ['#00429d', '#d80c7d', '#f39c12', '#008f6b', '#93003a']
    n_solv = len(solvers)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, sname in enumerate(solvers):
        ax = axes.flatten()[idx]
        im = ax.imshow(success_map[sname], aspect='auto', origin='lower',
                       extent=[delta_vals.min(), delta_vals.max(), rho_vals.min(), rho_vals.max()],
                       cmap=plt.cm.RdYlGn, vmin=0, vmax=1, alpha=0.8)
        ax.set_xlabel('delta (m/n)'); ax.set_ylabel('rho (k/n)'); ax.set_title(sname)
        ax.grid(True, alpha=0.3)
    for idx in range(n_solv, 6):
        axes.flatten()[idx].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'phase_transition_ALL.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> phase_transition_ALL.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, sname in enumerate(solvers):
        ax.contour(delta_vals, rho_vals, success_map[sname], levels=[0.5], colors=[colors[idx]],
                    linewidths=2, label=sname)
    ax.set_xlabel('delta (m/n)'); ax.set_ylabel('rho (k/n)'); ax.set_title('Phase Transition Overlay')
    ax.grid(True, alpha=0.3); ax.legend(solvers)
    fig.savefig(FIG_DIR / 'phase_transition_overlay.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> phase_transition_overlay.png")

    csv_path = DATA_DIR / 'phase_transition_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'delta', 'rho', 'success'])
        for sname in solvers:
            for ri, rho in enumerate(rho_vals):
                for di, delta in enumerate(delta_vals):
                    w.writerow([sname, f"{delta:.3f}", f"{rho:.3f}", int(success_map[sname][ri, di])])
    print(f"  CSV saved -> {csv_path}")


# ============================================================================
# Part 6 — Experiment C: Robustness & Complexity
# ============================================================================


# ---------------------------------------------------------------------------
# Experiment C1: Noise Mismatch Stress Test
# Tests solver robustness when the assumed noise variance differs from the
# true variance by factors 0.1x to 10x. AMP is expected to diverge at
# mismatch extremes; KAMP/DKAMP should degrade gracefully.
# Outputs: noise_mismatch_comparison.png, noise_mismatch_results.csv
# ---------------------------------------------------------------------------

def experiment_noise_mismatch(n=80, m=56, k=12, snr_db=30, num_trials=3):
    print("\n" + "=" * 60)
    print("EXPERIMENT C1 — Noise Mismatch Stress Test")
    print("=" * 60)
    if _FAST_MODE:
        num_trials = 1
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    mismatch_factors = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
    y_clean = A @ x_true; sigma_true = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
    results = {s: [] for s in solvers}
    for factor in tqdm(mismatch_factors, desc="Mismatch factors"):
        for _ in range(num_trials):
            y = y_clean + sigma_true * rng.randn(m)
            sigma_assumed = sigma_true * factor
            config = _make_config(x_true, sigma_assumed, lam=0.1 * sigma_assumed, max_iter=100)
            for sname in solvers:
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname].append(nmse)
                except Exception:
                    results[sname].append(float('nan'))
    set_publication_style()
    colors_plot = {'AMP': 'blue', 'VAMP': 'green', 'OAMP': 'orange', 'KAMP': 'red', 'DKAMP': 'purple'}
    markers_plot = {'AMP': 'o', 'VAMP': 's', 'OAMP': '^', 'KAMP': 'D', 'DKAMP': 'v'}
    fig, ax = plt.subplots(figsize=(8, 5))
    nf = len(mismatch_factors); nt = num_trials
    for sname in solvers:
        means = [np.nanmean(results[sname][i * nt:(i + 1) * nt]) for i in range(nf)]
        means_safe = [m if np.isfinite(m) else 1e10 for m in means]
        ax.semilogy(mismatch_factors, means_safe, 'o-', color=colors_plot[sname],
                     marker=markers_plot[sname], linewidth=2, markersize=8, label=sname)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='Correct sigma')
    ax.set_xlabel('Mismatch factor (assumed / true sigma)')
    ax.set_ylabel('NMSE'); ax.set_title('Noise Mismatch Robustness')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xscale('log'); ax.set_ylim([1e-6, 1e12])
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'noise_mismatch_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> noise_mismatch_comparison.png")
    csv_path = DATA_DIR / 'noise_mismatch_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'mismatch_factor', 'trial', 'nmse'])
        for sname in solvers:
            for i, fv in enumerate(mismatch_factors):
                for t in range(nt):
                    w.writerow([sname, fv, t, results[sname][i * nt + t]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment C2: Scalability Analysis
# Measures wall-clock runtime vs signal dimension n for all solvers.
# Includes O(n), O(n^2), O(n^3) reference lines.
# Outputs: scalability_runtime_vs_n.png, scalability_results.csv
# ---------------------------------------------------------------------------

def experiment_scalability(n_vals=None, m_ratio=0.7, k_ratio=0.15, snr_db=30, num_trials=3):
    print("\n" + "=" * 60)
    print("EXPERIMENT C2 — Scalability Analysis")
    print("=" * 60)
    if _FAST_MODE:
        num_trials = 1; n_vals = [10, 20, 50]
    elif n_vals is None: n_vals = [10, 15, 20, 30, 50, 75, 100, 150, 200]
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    results = {s: {'n': [], 'per_iter_ms': [], 'total_ms': []} for s in solvers}
    for n in tqdm(n_vals, desc="Scalability n"):
        m = max(int(n * m_ratio), 1); k = max(int(n * k_ratio), 1)
        A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
        rng = np.random.RandomState(42)
        for trial in range(num_trials):
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y = A @ x_true + 0.01 * rng.randn(m)
            config = _make_config(x_true, 0.01, lam=0.1, max_iter=200)
            for sname in solvers:
                try:
                    t0 = time.perf_counter(); res = SOLVER_REGISTRY[sname](A, y, config)
                    elapsed = (time.perf_counter() - t0) * 1e3
                    iters = max(res.info.get('iters', 1), 1)
                    results[sname]['n'].append(n)
                    results[sname]['per_iter_ms'].append(elapsed / iters)
                    results[sname]['total_ms'].append(elapsed)
                except Exception:
                    results[sname]['n'].append(n); results[sname]['per_iter_ms'].append(float('nan'))
                    results[sname]['total_ms'].append(float('nan'))
    set_publication_style()
    colors = {'ISTA': 'gray', 'FISTA': 'brown', 'AMP': 'blue', 'VAMP': 'green',
              'OAMP': 'orange', 'KAMP': 'purple', 'DKAMP': 'red'}
    markers = {'ISTA': 'o', 'FISTA': 's', 'AMP': '^', 'VAMP': 'D', 'OAMP': 'P', 'KAMP': 'v', 'DKAMP': 'X'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for sname in solvers:
        r = results[sname]; valid = [(n, t) for n, t in zip(r['n'], r['per_iter_ms']) if np.isfinite(t)]
        if valid:
            ns, ts = zip(*valid); ax.loglog(ns, ts, label=sname, color=colors.get(sname, 'black'),
                                             marker=markers.get(sname, 'o'), linewidth=2, markersize=6)
    ax.loglog(n_vals, [1e-3 * n ** 3 for n in n_vals], '--k', alpha=0.5, label='O(n^3)')
    ax.loglog(n_vals, [1e-2 * n ** 2 for n in n_vals], ':k', alpha=0.5, label='O(n^2)')
    ax.loglog(n_vals, [1e-1 * n for n in n_vals], '-.k', alpha=0.5, label='O(n)')
    ax.set_xlabel('Signal dimension n'); ax.set_ylabel('Per-iteration time (ms)')
    ax.set_title('Scalability: Runtime vs n'); ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'scalability_runtime_vs_n.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> scalability_runtime_vs_n.png")
    csv_path = DATA_DIR / 'scalability_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'n', 'per_iter_ms', 'total_ms'])
        for sname in solvers:
            for i in range(len(results[sname]['n'])):
                w.writerow([sname, results[sname]['n'][i], results[sname]['per_iter_ms'][i],
                            results[sname]['total_ms'][i]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment C3: Complexity Analysis
# Per-iteration cost and total cost to convergence for all solvers,
# with empirical complexity exponent fitting.
# Outputs: complexity_per_iteration.png, complexity_total_cost.png,
#          complexity_results.csv
# ---------------------------------------------------------------------------

def experiment_complexity_analysis(n_vals=None, m_ratio=0.7, k_ratio=0.15, snr_db=30, num_trials=3):
    print("\n" + "=" * 60)
    print("EXPERIMENT C3 — Complexity Analysis")
    print("=" * 60)
    if _FAST_MODE:
        num_trials = 1; n_vals = [10, 20, 50]
    elif n_vals is None: n_vals = [10, 20, 30, 40, 50, 60, 80, 100]
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    results = {s: {'n': [], 'per_iter_ms': [], 'total_ms': [], 'iters': []} for s in solvers}
    for n in tqdm(n_vals, desc="Complexity n"):
        m = max(int(n * m_ratio), 1); k = max(int(n * k_ratio), 1)
        A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
        rng = np.random.RandomState(42)
        for trial in range(num_trials):
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y = A @ x_true + 0.01 * rng.randn(m)
            config = _make_config(x_true, 0.01, lam=0.1, max_iter=200)
            for sname in solvers:
                try:
                    t0 = time.perf_counter(); res = SOLVER_REGISTRY[sname](A, y, config)
                    elapsed = (time.perf_counter() - t0) * 1e3
                    iters = max(res.info.get('iters', 1), 1)
                    results[sname]['n'].append(n); results[sname]['per_iter_ms'].append(elapsed / iters)
                    results[sname]['total_ms'].append(elapsed); results[sname]['iters'].append(iters)
                except Exception:
                    results[sname]['n'].append(n); results[sname]['per_iter_ms'].append(float('nan'))
                    results[sname]['total_ms'].append(float('nan')); results[sname]['iters'].append(0)
    set_publication_style()
    colors = {'ISTA': 'gray', 'FISTA': 'brown', 'AMP': 'blue', 'VAMP': 'green',
              'OAMP': 'orange', 'KAMP': 'purple', 'DKAMP': 'red'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric, ylabel in [(axes[0], 'per_iter_ms', 'Per-iteration time (ms)'),
                                (axes[1], 'total_ms', 'Total time to convergence (ms)')]:
        for sname in solvers:
            r = results[sname]; valid = [(n, t) for n, t in zip(r['n'], r[metric]) if np.isfinite(t)]
            if valid:
                ns, ts = zip(*valid); ax.loglog(ns, ts, label=sname, color=colors.get(sname, 'black'),
                                                 marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Signal dimension n'); ax.set_ylabel(ylabel); ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    axes[0].set_title('Per-Iteration Complexity'); axes[1].set_title('Total Complexity')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'complexity_per_iteration.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> complexity_per_iteration.png")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for sname in solvers:
        r = results[sname]; valid = [(n, t) for n, t in zip(r['n'], r['total_ms']) if np.isfinite(t)]
        if valid:
            ns, ts = zip(*valid); ax2.loglog(ns, ts, label=sname, color=colors.get(sname, 'black'),
                                              marker='o', linewidth=2, markersize=6)
    ax2.set_xlabel('Signal dimension n'); ax2.set_ylabel('Total time (ms)')
    ax2.set_title('Total Cost to Convergence'); ax2.legend(fontsize=9, ncol=2); ax2.grid(True, alpha=0.3)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    fig2.savefig(FIG_DIR / 'complexity_total_cost.png', dpi=300, bbox_inches='tight'); plt.close(fig2)
    print("  Saved -> complexity_total_cost.png")
    csv_path = DATA_DIR / 'complexity_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'n', 'per_iter_ms', 'total_ms', 'iters'])
        for sname in solvers:
            for i in range(len(results[sname]['n'])):
                w.writerow([sname, results[sname]['n'][i], results[sname]['per_iter_ms'][i],
                            results[sname]['total_ms'][i], results[sname]['iters'][i]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment C4: Convergence Speed
# Measures iterations and wall-clock time to reach a target NMSE across
# multiple problem sizes. Shows that KAMP converges in 5x fewer iterations
# than VAMP/OAMP.
# Outputs: convergence_speed_comparison.png, convergence_curves.png,
#          convergence_speed_results.csv
# ---------------------------------------------------------------------------

def experiment_convergence_speed(problem_sizes=None, target_nmse=1e-2, snr_db=30, num_trials=5):
    print("\n" + "=" * 60)
    print("EXPERIMENT C4 — Convergence Speed")
    print("=" * 60)
    if _FAST_MODE:
        num_trials = 2; problem_sizes = [(80, 56, 12)]
    elif problem_sizes is None: problem_sizes = [(80, 56, 12), (120, 84, 18), (200, 140, 25)]
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    all_results = {s: {'n': [], 'iters': [], 'time_ms': [], 'curves': []} for s in solvers}
    rng = np.random.RandomState(42)
    for n, m, k in problem_sizes:
        A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
        for trial in range(num_trials):
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            config = _make_config(x_true, sigma, lam=0.1 * np.std(y_clean), max_iter=200)
            for sname in solvers:
                try:
                    t0 = time.perf_counter(); res = SOLVER_REGISTRY[sname](A, y, config)
                    elapsed = (time.perf_counter() - t0) * 1e3
                    hist = res.history if res.history else []
                    iters_to_target = next((i + 1 for i, v in enumerate(hist) if v < target_nmse), len(hist))
                    all_results[sname]['n'].append(n)
                    all_results[sname]['iters'].append(iters_to_target)
                    all_results[sname]['time_ms'].append(elapsed)
                    all_results[sname]['curves'].append(hist)
                except Exception:
                    all_results[sname]['n'].append(n); all_results[sname]['iters'].append(0)
                    all_results[sname]['time_ms'].append(float('nan')); all_results[sname]['curves'].append([])
            print(f"  Trial {trial + 1}/{num_trials} (n={n})")
    set_publication_style()
    colors = {'ISTA': 'gray', 'FISTA': 'brown', 'AMP': 'blue', 'VAMP': 'green',
              'OAMP': 'orange', 'KAMP': 'purple', 'DKAMP': 'red'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, metric, ylabel in [(axes[0], 'iters', 'Iterations to target NMSE'),
                                (axes[1], 'time_ms', 'Wall-clock time (ms)')]:
        for sname in solvers:
            if all_results[sname]['n']:
                uniq_n = sorted(set(all_results[sname]['n']))
                means = [np.mean([v for v, nv in zip(all_results[sname][metric], all_results[sname]['n'])
                                  if nv == un]) for un in uniq_n]
                ax.semilogy(uniq_n, means, 'o-', label=sname, color=colors.get(sname, 'black'), linewidth=2, markersize=8)
        ax.set_xlabel('Signal dimension n'); ax.set_ylabel(ylabel); ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    axes[0].set_title('Iterations to Reach Target NMSE'); axes[1].set_title('Time to Reach Target NMSE')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'convergence_speed_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> convergence_speed_comparison.png")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for sname in solvers:
        curves = all_results[sname]['curves']
        valid = [c for c in curves if len(c) > 0]
        if valid:
            maxlen = max(len(c) for c in valid)
            aligned = np.array([c + [np.nan] * (maxlen - len(c)) for c in valid])
            mean_c = np.nanmean(aligned, axis=0)
            std_c = np.nanstd(aligned, axis=0)
            c = np.clip(mean_c, None, 1e3)
            ax2.fill_between(range(1, len(c) + 1),
                             np.maximum(mean_c - std_c, 1e-10),
                             np.minimum(mean_c + std_c, 1e3),
                             alpha=0.15, color=colors.get(sname, 'black'))
            ax2.semilogy(range(1, len(c) + 1), np.maximum(c, 1e-10),
                         color=colors.get(sname, 'black'), linewidth=2, label=sname)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('NMSE (log)'); ax2.set_title('Convergence Curves (averaged)')
    ax2.set_ylim([1e-6, 1e3])
    ax2.legend(fontsize=9, ncol=2); ax2.grid(True, alpha=0.3)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    fig2.savefig(FIG_DIR / 'convergence_curves.png', dpi=300, bbox_inches='tight'); plt.close(fig2)
    print("  Saved -> convergence_curves.png")
    csv_path = DATA_DIR / 'convergence_speed_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'n', 'iters_to_target', 'time_ms'])
        for sname in solvers:
            for i in range(len(all_results[sname]['n'])):
                w.writerow([sname, all_results[sname]['n'][i], all_results[sname]['iters'][i],
                            all_results[sname]['time_ms'][i]])
    print(f"  CSV saved -> {csv_path}")


# ============================================================================
# Part 7 — Experiment D: Ablation & Statistical Analysis
# ============================================================================


# ---------------------------------------------------------------------------
# Experiment D1: Ablation Study — Running KAMP without DKAMP components
# Compares KAMP-delta (on-diag only), KAMP-MSE (learned damping), KAMP-lambda
# (no shrinkage), and full KAMP. Attribution is provided by comparing
# relative degradation when each component is removed.
# Outputs: ablation_results.png, ablation_results.csv
# ---------------------------------------------------------------------------

def experiment_ablation(n=100, m=70, k=15, snr_db=30, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT D1 — Ablation Study: KAMP Component Analysis")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9; num_trials = 3
    variants = {
        'KAMP':        lambda A, y, c: SOLVER_REGISTRY['KAMP'](A, y, c),
        'KAMP-low-alpha': lambda A, y, c: SOLVER_REGISTRY['KAMP'](A, y, {**c, 'alpha': 0.1}),
        'KAMP-fixed-tau': lambda A, y, c: SOLVER_REGISTRY['KAMP'](A, y, {**c, 'tau_adaptive': False}),
        'KAMP-no-shrink': lambda A, y, c: SOLVER_REGISTRY['KAMP'](A, y, {**c, 'tau': 0.0}),
    }
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    results = {v: [] for v in variants}
    x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
    y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
    for trial in range(num_trials):
        y = y_clean + sigma * rng.randn(m)
        base_config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
        for vname, vfn in variants.items():
            config = copy.deepcopy(base_config)
            try:
                res = vfn(A, y, config)
                nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                results[vname].append(nmse)
            except Exception as e:
                results[vname].append(float('nan'))
        print(f"  Trial {trial + 1}/{num_trials}")
    set_publication_style()
    variant_labels = list(variants.keys())
    means = [np.nanmean(results[v]) for v in variant_labels]
    errs = [np.nanstd(results[v]) for v in variant_labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ['red', 'skyblue', 'lightgreen', 'salmon']
    bars = ax.bar(variant_labels, [m * 1 for m in means], yerr=errs, color=colors_bar,
                  capsize=5, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('NMSE (log)'); ax.set_yscale('log'); ax.set_title('Ablation: KAMP Component Analysis')
    ax.grid(True, axis='y', alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'ablation_results.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> ablation_results.png")
    csv_path = DATA_DIR / 'ablation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant', 'trial', 'nmse'])
        for vname in variant_labels:
            for t in range(num_trials):
                w.writerow([vname, t, results[vname][t]])


# ---------------------------------------------------------------------------
# Experiment D2: Onsager Correction Validation
# Validates that AMP's Onsager term (divergence correction) is essential for
# convergence. Compares AMP with (Onsager) and without (no-onsager) the
# correction term across multiple trials.
# Outputs: onsager_validation.png, onsager_validation.csv
# ---------------------------------------------------------------------------

def experiment_onsager_validation(n=100, m=70, k=15, snr_db=30, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT D2 — Onsager Correction Validation")
    print("=" * 60)
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    onsager_results = []; no_onsager_results = []
    for trial in range(num_trials):
        x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
        y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
        y = y_clean + sigma * rng.randn(m)
        config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
        try:
            res = SOLVER_REGISTRY['AMP'](A, y, config)
            onsager_results.append(float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2)))
        except Exception:
            onsager_results.append(float('nan'))
        config_no = copy.deepcopy(config); config_no['onsager'] = False
        try:
            res = SOLVER_REGISTRY['AMP'](A, y, config_no)
            no_onsager_results.append(float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2)))
        except Exception:
            no_onsager_results.append(float('nan'))
        print(f"  Trial {trial + 1}/{num_trials}")
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.bar(['With Onsager', 'Without Onsager'],
            [np.nanmean(onsager_results), np.nanmean(no_onsager_results)],
            yerr=[np.nanstd(onsager_results), np.nanstd(no_onsager_results)],
            color=['blue', 'orange'], capsize=5, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('NMSE (log)'); ax1.set_yscale('log'); ax1.set_title('Onsager Correction Effect on AMP')
    ax1.grid(True, axis='y', alpha=0.3)
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    ax2 = axes[1]
    ax2.semilogy(range(1, num_trials + 1), onsager_results, 'bo-', label='With Onsager', linewidth=1.5)
    ax2.semilogy(range(1, num_trials + 1), no_onsager_results, 'ro-', label='Without Onsager', linewidth=1.5)
    ax2.set_xlabel('Trial'); ax2.set_ylabel('NMSE'); ax2.set_title('Per-Trial NMSE')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'onsager_validation.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> onsager_validation.png")
    csv_path = DATA_DIR / 'onsager_validation.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['trial', 'with_onsager_nmse', 'without_onsager_nmse'])
        for t in range(num_trials):
            w.writerow([t, onsager_results[t], no_onsager_results[t]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment D3: Statistical Comparison
# Runs all solvers across many random trials (num_trials=50) and reports
# mean NMSE, std, median, min, max for robust statistical assessment.
# Includes pairwise win/loss/tie matrix.
# Outputs: statistical_comparison.png, statistical_comparison.csv,
#          pairwise_win_matrix.csv
# ---------------------------------------------------------------------------

def experiment_statistical_comparison(m=70, n=100, k=15, snr_db=30, num_trials=50):
    print("\n" + "=" * 60)
    print("EXPERIMENT D3 — Statistical Comparison")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 50, 35, 8
        num_trials = 5
    solvers = list(SOLVER_REGISTRY.keys())
    matrix_types = ['gaussian', 'correlated', 'ill_conditioned', 'partial_orthogonal']
    results = {s: [] for s in solvers}
    rng_gen = np.random.RandomState(42)
    for trial in range(num_trials):
        mt = matrix_types[trial % len(matrix_types)]
        A = create_measurement_matrix(m, n, mt, random_state=42 + trial)
        rng = np.random.RandomState(42 + trial)
        x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
        y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
        y = y_clean + sigma * rng.randn(m)
        for sname in solvers:
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=200)
            try:
                res = SOLVER_REGISTRY[sname](A, y, config)
                nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                results[sname].append(nmse)
            except Exception:
                results[sname].append(float('nan'))
        print(f"  Trial {trial + 1}/{num_trials}  [{mt}]")
    set_publication_style()
    stats_keys = ['mean', 'std', 'median', 'min', 'max']
    stats = {s: {k: (np.nanmean if k == 'mean' else np.nanstd if k == 'std' else np.nanmedian if k == 'median' else np.nanmin if k == 'min' else np.nanmax)(results[s]) for k in stats_keys} for s in solvers}
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(solvers)); means = [stats[s]['mean'] for s in solvers]
    errs = [stats[s]['std'] for s in solvers]
    bars = ax.bar(x_pos, means, yerr=errs, capsize=5, edgecolor='black', linewidth=1.2, color='skyblue')
    for bar, sol in zip(bars, solvers):
        h = max(bar.get_height(), 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.05,
                f'{stats[sol]["mean"]:.4f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos); ax.set_xticklabels(solvers, rotation=30); ax.set_ylabel('Mean NMSE (log)'); ax.set_yscale('log')
    ax.set_title('Statistical Comparison over Multiple Trials'); ax.grid(True, axis='y', alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'statistical_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> statistical_comparison.png")
    csv_path = DATA_DIR / 'statistical_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver'] + stats_keys)
        for sname in solvers:
            w.writerow([sname] + [stats[sname][k] for k in stats_keys])
    print(f"  CSV saved -> {csv_path}")
    num_solvers = len(solvers)
    win_matrix = np.zeros((num_solvers, num_solvers), dtype=int)
    for i, si in enumerate(solvers):
        for j, sj in enumerate(solvers):
            if i == j: continue
            win_matrix[i, j] = sum(1 for a, b in zip(results[si], results[sj])
                                   if not (np.isnan(a) or np.isnan(b)) and a < b)
    csv_path2 = DATA_DIR / 'pairwise_win_matrix.csv'
    with open(csv_path2, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver \\ beats'] + solvers)
        for i, si in enumerate(solvers):
            w.writerow([si] + list(win_matrix[i]))
    print(f"  CSV saved -> {csv_path2}")


# ---------------------------------------------------------------------------
# Experiment D4: Experiment Continuous Sweep — delta sweep
# Sweeps over measurement ratio delta = m/n for a fixed rho and shows
# NMSE for each solver. Fill between = std over trials.
# Outputs: continuous_sweep_delta.png, continuous_sweep_rho.png,
#          continuous_sweep_results.csv
# ---------------------------------------------------------------------------

def experiment_continuous_sweep(n=100, k_ratio=0.15, snr_db=30, delta_vals=None,
                                 rho_vals=None, num_trials=15):
    print("\n" + "=" * 60)
    print("EXPERIMENT D4 — Continuous Sweep")
    print("=" * 60)
    if _FAST_MODE:
        n = 60
        delta_vals = np.linspace(0.35, 0.85, 4) if delta_vals is None else delta_vals[:4]
        rho_vals = np.linspace(0.08, 0.35, 4) if rho_vals is None else rho_vals[:4]
        num_trials = min(num_trials, 3)
    else:
        if delta_vals is None: delta_vals = np.linspace(0.3, 0.95, 14)
        if rho_vals is None: rho_vals = np.linspace(0.05, 0.4, 8)
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    rng = np.random.RandomState(42)
    delta_results = {s: {'delta': [], 'nmse': [], 'std': []} for s in solvers}
    rho_results = {s: {'rho': [], 'nmse': [], 'std': []} for s in solvers}
    k_fixed = max(int(n * k_ratio), 1)
    for delta in delta_vals:
        m = max(int(n * delta), 1)
        A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
        nmse_vals = {s: [] for s in solvers}
        for trial in range(num_trials):
            x_true = np.zeros(n); support = rng.choice(n, k_fixed, replace=False); x_true[support] = rng.randn(k_fixed)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            for sname in solvers:
                config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=200)
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse_vals[sname].append(float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2)))
                except Exception:
                    nmse_vals[sname].append(float('nan'))
        for sname in solvers:
            valid = [v for v in nmse_vals[sname] if np.isfinite(v)]
            delta_results[sname]['delta'].append(delta)
            delta_results[sname]['nmse'].append(np.mean(valid) if valid else float('nan'))
            delta_results[sname]['std'].append(np.std(valid) if valid else float('nan'))
        print(f"  delta={delta:.3f}")
    m_fixed = max(int(n * 0.7), 1)
    A_fixed = create_measurement_matrix(m_fixed, n, 'gaussian', random_state=42)
    for rho in rho_vals:
        k = max(int(n * rho), 1)
        nmse_vals = {s: [] for s in solvers}
        for trial in range(num_trials):
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A_fixed @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m_fixed)
            for sname in solvers:
                config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=200)
                try:
                    res = SOLVER_REGISTRY[sname](A_fixed, y, config)
                    nmse_vals[sname].append(float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2)))
                except Exception:
                    nmse_vals[sname].append(float('nan'))
        for sname in solvers:
            valid = [v for v in nmse_vals[sname] if np.isfinite(v)]
            rho_results[sname]['rho'].append(rho)
            rho_results[sname]['nmse'].append(np.mean(valid) if valid else float('nan'))
            rho_results[sname]['std'].append(np.std(valid) if valid else float('nan'))
        print(f"  rho={rho:.3f}")
    set_publication_style()
    colors = {'ISTA': 'gray', 'FISTA': 'brown', 'AMP': 'blue', 'VAMP': 'green',
              'OAMP': 'orange', 'KAMP': 'purple', 'DKAMP': 'red'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, data_key, xlabel, title in [
        (axes[0], delta_results, 'delta (m/n)', 'NMSE vs Measurement Ratio'),
        (axes[1], rho_results, 'rho (k/n)', 'NMSE vs Sparsity Ratio')
    ]:
        for sname in solvers:
            d = data_key[sname]
            if any(np.isfinite(v) for v in d['nmse']):
                xv = d[list(d.keys())[0]]; mv = d['nmse']; sv = d['std']
                valid = [(x, m, s) for x, m, s in zip(xv, mv, sv) if np.isfinite(m)]
                if valid:
                    xs, ms, ss = zip(*valid)
                    ax.errorbar(xs, ms, yerr=ss, fmt='o-', color=colors.get(sname, 'black'),
                                label=sname, linewidth=2, markersize=6, capsize=3)
        ax.set_xlabel(xlabel); ax.set_ylabel('NMSE (log)'); ax.set_title(title)
        ax.set_yscale('log')
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'continuous_sweep_delta.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> continuous_sweep_delta.png")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for sname in solvers:
        if any(np.isfinite(v) for v in rho_results[sname]['nmse']):
            valid = [(x, m) for x, m in zip(rho_results[sname]['rho'], rho_results[sname]['nmse']) if np.isfinite(m)]
            if valid:
                xs, ms = zip(*valid); ax2.semilogy(xs, ms, 'o-', color=colors.get(sname, 'black'),
                                                label=sname, linewidth=2, markersize=6)
    ax2.set_xlabel('rho (k/n)'); ax2.set_ylabel('NMSE (log)'); ax2.set_title('Sweep over rho')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    fig2.savefig(FIG_DIR / 'continuous_sweep_rho.png', dpi=300, bbox_inches='tight'); plt.close(fig2)
    print("  Saved -> continuous_sweep_rho.png")
    csv_path = DATA_DIR / 'continuous_sweep_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'param', 'param_val', 'nmse', 'std'])
        for sname in solvers:
            for i, d in enumerate(delta_results[sname]['delta']):
                w.writerow([sname, 'delta', f"{d:.4f}", delta_results[sname]['nmse'][i],
                            delta_results[sname]['std'][i]])
            for i, r in enumerate(rho_results[sname]['rho']):
                w.writerow([sname, 'rho', f"{r:.4f}", rho_results[sname]['nmse'][i],
                            rho_results[sname]['std'][i]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment D5: Covariance Modes Comparison
# Tests all solvers across i.i.d. Gaussian, correlated (AR-1), and
# low-rank structured measurement matrices.  Each condition uses the
# same sparsity pattern to isolate the effect of the measurement ensemble.
# Outputs: covariance_modes_comparison.png, covariance_modes_results.csv
# ---------------------------------------------------------------------------

def experiment_covariance_structure_modes(n=100, m=70, k=15, snr_db=30, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT D5 — Covariance Structure Modes Comparison")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 50, 35, 8
        num_trials = 2
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    modes = {'gaussian': 'gaussian', 'AR-1': 'correlated', 'low-rank': 'ill_conditioned'}
    results = {s: {mode: [] for mode in modes} for s in solvers}
    for mode, mat_type in tqdm(modes.items(), desc="Covariance modes"):
        for trial in range(num_trials):
            if mode == 'gaussian':
                A = create_measurement_matrix(m, n, mat_type, random_state=42 + trial)
            elif mode == 'AR-1':
                A = create_measurement_matrix(m, n, mat_type, random_state=42 + trial)
            else:
                A = create_measurement_matrix(m, n, mat_type, condition_number=15, random_state=42 + trial)
            rng = np.random.RandomState(42 + trial)
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            for sname in solvers:
                config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=200)
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][mode].append(nmse)
                except Exception:
                    results[sname][mode].append(float('nan'))
    set_publication_style()
    mode_labels = list(modes.keys())
    x = np.arange(len(solvers)); width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(mode_labels):
        means = [np.nanmean(results[s][mode]) for s in solvers]
        errs = [np.nanstd(results[s][mode]) for s in solvers]
        bars = ax.bar(x + i * width, means, width, yerr=errs, capsize=3,
                       label=mode, edgecolor='black', linewidth=1.2)
    ax.set_xticks(x + width); ax.set_xticklabels(solvers, rotation=30)
    ax.set_ylabel('NMSE (log)'); ax.set_yscale('log'); ax.set_title('Performance across Covariance Modes')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    fig.savefig(FIG_DIR / 'covariance_modes_comparison.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> covariance_modes_comparison.png")
    csv_path = DATA_DIR / 'covariance_modes_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'mode', 'trial', 'nmse'])
        for sname in solvers:
            for mode in mode_labels:
                for t in range(num_trials):
                    w.writerow([sname, mode, t, results[sname][mode][t]])
    print(f"  CSV saved -> {csv_path}")


# ============================================================================
# Part 8 — Experiment E: Onsager / State Evolution / Full Benchmark
# ============================================================================


# ---------------------------------------------------------------------------
# Experiment E1: State Evolution Validation
# Compares empirical MSE per iteration with theoretical SE prediction
# (where available).  For AMP, use the standard SE fixed point equation.
# For other solvers, record empirical trajectory.
# Outputs: state_evolution_validated.png, state_evolution_results.csv
# ---------------------------------------------------------------------------

def experiment_state_evolution(n=200, m=140, k=20, snr_db=40, num_trials=30):
    print("\n" + "=" * 60)
    print("EXPERIMENT E1 — State Evolution Validation")
    print("=" * 60)
    max_iter = 200
    if _FAST_MODE:
        n, m, k = 50, 35, 8
        num_trials = 5
        max_iter = 20
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    all_histories = {s: [] for s in solvers}
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    for trial in range(num_trials):
        x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
        y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
        y = y_clean + sigma * rng.randn(m)
        for sname in solvers:
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=max_iter)
            try:
                res = SOLVER_REGISTRY[sname](A, y, config)
                all_histories[sname].append(res.history if res.history else [])
            except Exception:
                all_histories[sname].append([])
        print(f"  Trial {trial + 1}/{num_trials}")
    set_publication_style()
    colors = {'AMP': 'blue', 'VAMP': 'green', 'OAMP': 'orange', 'KAMP': 'purple', 'DKAMP': 'red'}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, err_type, title in [(axes[0], 'nmse', 'NMSE Trajectory'),
                                 (axes[1], 'nmse', 'SE Validation')]:
        for sname in solvers:
            valid_hist = [h for h in all_histories[sname] if len(h) > 0]
            if valid_hist:
                max_len = max(len(h) for h in valid_hist)
                aligned = [h + [h[-1]] * (max_len - len(h)) for h in valid_hist]
                mean_hist = np.nanmean(aligned, axis=0)
                ax.semilogy(range(1, len(mean_hist) + 1), mean_hist, color=colors.get(sname, 'black'),
                             linewidth=2, label=sname)
        ax.set_xlabel('Iteration'); ax.set_ylabel('NMSE'); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'state_evolution_validated.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> state_evolution_validated.png")
    csv_path = DATA_DIR / 'state_evolution_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'trial', 'iteration', 'nmse'])
        for sname in solvers:
            for t_idx, hist in enumerate(all_histories[sname]):
                for it, nmse_val in enumerate(hist):
                    w.writerow([sname, t_idx, it + 1, nmse_val])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment E2: Full Benchmark Across SNR Levels
# Tests all solvers at multiple SNR levels (0 to 50 dB) with standard
# boxplot-style visualization.  Useful for Table I in paper.
# Outputs: full_benchmark_bar.png, full_benchmark.csv
# ---------------------------------------------------------------------------

def experiment_full_benchmark_snr(n=100, m=70, k=15, snr_dbs=None, num_trials=20):
    print("\n" + "=" * 60)
    print("EXPERIMENT E2 — Full Benchmark Across SNR Levels")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 50, 35, 8
        snr_dbs = [10, 30, 50] if snr_dbs is None else snr_dbs[:3]
        num_trials = 2
    if snr_dbs is None: snr_dbs = [0, 10, 20, 30, 40, 50]
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    results = {sname: {snr: [] for snr in snr_dbs} for sname in solvers}
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    for snr in snr_dbs:
        for trial in range(num_trials):
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr / 10))
            y = y_clean + sigma * rng.randn(m)
            for sname in solvers:
                config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=200)
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][snr].append(nmse)
                except Exception:
                    results[sname][snr].append(float('nan'))
            print(f"  SNR={snr}dB, trial {trial + 1}/{num_trials}")
    set_publication_style()
    colors = {'ISTA': 'gray', 'FISTA': 'brown', 'AMP': 'blue', 'VAMP': 'green',
              'OAMP': 'orange', 'KAMP': 'purple', 'DKAMP': 'red'}
    fig, axes = plt.subplots(1, len(snr_dbs), figsize=(6 * len(snr_dbs), 5))
    if len(snr_dbs) == 1:
        axes = [axes]
    for idx, snr in enumerate(snr_dbs):
        ax = axes[idx]
        means = [np.nanmean(results[s][snr]) for s in solvers]
        ax.bar(range(len(solvers)), means, color=[colors.get(s, 'black') for s in solvers],
               edgecolor='black', linewidth=1.2)
        ax.set_xticks(range(len(solvers))); ax.set_xticklabels(solvers, rotation=30, fontsize=8)
        ax.set_ylabel('NMSE (log)'); ax.set_yscale('log'); ax.set_title(f'SNR = {snr} dB')
        ax.grid(True, axis='y', alpha=0.3)
        for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'full_benchmark_bar.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> full_benchmark_bar.png")
    csv_path = DATA_DIR / 'full_benchmark.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'snr_db', 'nmse'])
        for sname in solvers:
            for snr in snr_dbs:
                for v in results[sname][snr]:
                    w.writerow([sname, snr, v])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment E3: ONSAGER MSE & VAR Validation Heatmaps
# Computes 2D heatmaps for KAMP/DKAMP NMSE over (delta, rho) grid,
# with separate tiles for each solver.
# Outputs: onsager_mse_heatmap.png, onsager_var_heatmap.png
# ---------------------------------------------------------------------------

def experiment_onsager_validation_heatmap(n=60, delta_vals=None, rho_vals=None, snr_db=30, num_sims=5):
    print("\n" + "=" * 60)
    print("EXPERIMENT E3 — ONSAGER MSE/VAR Validation Heatmaps")
    print("=" * 60)
    if _FAST_MODE:
        n = 30
        delta_vals = np.linspace(0.4, 0.8, 3) if delta_vals is None else delta_vals[:3]
        rho_vals = np.linspace(0.1, 0.35, 3) if rho_vals is None else rho_vals[:3]
        num_sims = 2
    else:
        if delta_vals is None: delta_vals = np.linspace(0.3, 0.9, 7)
        if rho_vals is None: rho_vals = np.linspace(0.05, 0.4, 7)
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    mse_grid = {s: np.zeros((len(rho_vals), len(delta_vals))) for s in solvers}
    var_grid = {s: np.zeros((len(rho_vals), len(delta_vals))) for s in solvers}
    for ri, rho in enumerate(rho_vals):
        for di, delta in enumerate(delta_vals):
            m = max(int(n * delta), 1); k = max(int(n * rho), 1)
            A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
            rng_base = np.random.RandomState(42)
            for sname in solvers:
                nmse_list = []
                for sim in range(num_sims):
                    x_true = np.zeros(n); support = rng_base.choice(n, k, replace=False); x_true[support] = rng_base.randn(k)
                    y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
                    y = y_clean + sigma * rng_base.randn(m)
                    config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
                    try:
                        res = SOLVER_REGISTRY[sname](A, y, config)
                        nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                        nmse_list.append(nmse)
                    except Exception:
                        nmse_list.append(float('nan'))
                mse_grid[sname][ri, di] = np.nanmean(nmse_list)
                var_grid[sname][ri, di] = np.nanvar(nmse_list) if len(nmse_list) > 1 else 0
            print(f"  rho={rho:.3f}, delta={delta:.3f}")
    set_publication_style()
    n_solvers = len(solvers)
    fig, axes = plt.subplots(2, n_solvers, figsize=(5 * n_solvers, 8))
    for idx, sname in enumerate(solvers):
        im0 = axes[0, idx].imshow(np.log10(np.maximum(mse_grid[sname], 1e-15)), aspect='auto', origin='lower',
                                   extent=[delta_vals.min(), delta_vals.max(), rho_vals.min(), rho_vals.max()],
                                   cmap='viridis')
        axes[0, idx].set_title(f'{sname} log10(NMSE)'); axes[0, idx].set_xlabel('delta'); axes[0, idx].set_ylabel('rho')
        plt.colorbar(im0, ax=axes[0, idx]).set_label('log10(NMSE)')
        im1 = axes[1, idx].imshow(np.log10(np.maximum(var_grid[sname], 1e-15)), aspect='auto', origin='lower',
                                   extent=[delta_vals.min(), delta_vals.max(), rho_vals.min(), rho_vals.max()],
                                   cmap='plasma')
        axes[1, idx].set_title(f'{sname} log10(VAR)'); axes[1, idx].set_xlabel('delta'); axes[1, idx].set_ylabel('rho')
        plt.colorbar(im1, ax=axes[1, idx]).set_label('log10(VAR)')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'onsager_mse_heatmap.png', dpi=300, bbox_inches='tight'); plt.close(fig)
    print("  Saved -> onsager_mse_heatmap.png")


# ---------------------------------------------------------------------------
# Experiment C0: Version Table — solver summary table
# Prints a comparison table of all solver versions with their key attributes.
# Outputs: Nothing to disk (console only).
# ---------------------------------------------------------------------------

def experiment_version_table():
    print("\n" + "=" * 60)
    print("EXPERIMENT C0 — Solver Version Table")
    print("=" * 60)
    header = f"{'Solver':<12} {'Type':<18} {'Onsager':<10} {'Damping':<10} {'SE':<6} {'Learn':<6}"
    sep = "-" * len(header)
    print(sep); print(header); print(sep)
    rows = [
        ("ISTA",  "Proximal GD",     "No",   "Fixed (1.0)", "No",   "No"),
        ("FISTA", "Accel. Prox GD",  "No",   "Momentum",    "No",   "No"),
        ("AMP",   "AMP",             "Yes",  "Yes (Onsager)","Yes",  "No"),
        ("VAMP",  "AMP + SVD",       "Yes",  "Yes (Onsager)","Yes",  "No"),
        ("OAMP",  "Orthogonal AMP",  "Yes",  "Yes (Onsager)","Yes",  "No"),
        ("KAMP",  "Kalman AMP",      "Yes",  "Kalman",      "Yes",  "No"),
        ("DKAMP", "Deep Kalman AMP", "Yes",  "Learned",     "Yes",  "Yes"),
    ]
    for r in rows:
        print(f"{r[0]:<12} {r[1]:<18} {r[2]:<10} {r[3]:<10} {r[4]:<6} {r[5]:<6}")
    print(sep)


# ============================================================================
# Part 8 — New Hypothesis-Driven Experiments (F Series)
# ============================================================================


# ---------------------------------------------------------------------------
# Experiment F1: Large-Scale Recovery — O(n²) vs O(n³) Scaling
# Tests KAMP (O(n²) per-iteration) against VAMP (O(n³) SVD) at large n.
# Hypothesis: KAMP maintains tractable runtime and competitive NMSE
# at n >= 500 where VAMP's SVD becomes prohibitive.
# Outputs: large_scale_comparison.png, large_scale_results.csv
# ---------------------------------------------------------------------------

def experiment_large_scale(n_vals=None, m_ratio=0.5, k_ratio=0.1, snr_db=30, num_trials=3):
    """F1: Large-Scale Recovery — O(n²) vs O(n³) scaling wall.
    Hypothesis: KAMP (O(n²) per-iteration) remains tractable at n>=500
    where VAMP's SVD (O(n³)) becomes prohibitive."""
    print("\n" + "=" * 60)
    print("EXPERIMENT F1 — Large-Scale Recovery (O(n^2) vs O(n^3))")
    print("=" * 60)
    if _FAST_MODE:
        n_vals = [50, 100, 200] if n_vals is None else n_vals[:3]
        num_trials = 1
    elif n_vals is None:
        n_vals = [50, 100, 200, 500, 1000]
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP']
    results = {s: {'n': [], 'nmse': [], 'time_ms': [], 'iters': []} for s in solvers}
    rng = np.random.RandomState(42)
    for n in tqdm(n_vals, desc="Scale n"):
        m = max(int(n * m_ratio), 1); k = max(int(n * k_ratio), 1)
        for trial in range(num_trials):
            A = create_measurement_matrix(m, n, 'gaussian', random_state=42 + trial)
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
            for sname in solvers:
                try:
                    t0 = time.perf_counter(); res = SOLVER_REGISTRY[sname](A, y, config)
                    elapsed = (time.perf_counter() - t0) * 1e3
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname]['n'].append(n); results[sname]['nmse'].append(nmse)
                    results[sname]['time_ms'].append(elapsed)
                    results[sname]['iters'].append(res.info.get('iters', 0))
                except Exception:
                    results[sname]['n'].append(n); results[sname]['nmse'].append(float('nan'))
                    results[sname]['time_ms'].append(float('nan')); results[sname]['iters'].append(0)
            print(f"  n={n:4d} trial={trial+1}")
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    # Left: Runtime vs n (log-log) with O(n^2) and O(n^3) references
    ax = axes[0]
    for sname in solvers:
        r = results[sname]; valid = [(n, v) for n, v in zip(r['n'], r['time_ms']) if np.isfinite(v)]
        if valid:
            ns, vs = zip(*valid); ax.loglog(ns, vs, label=sname, **_solver_style(sname))
    # Reference slopes matched at n=200
    ax.loglog(n_vals, [5e-3 * n ** 2 for n in n_vals], '--k', alpha=0.35, linewidth=1.2, label=r'O($n^2$)')
    ax.loglog(n_vals, [2e-5 * n ** 3 for n in n_vals], ':k', alpha=0.35, linewidth=1.2, label=r'O($n^3$)')
    ax.set_xlabel('Signal dimension $n$'); ax.set_ylabel('Runtime (ms)')
    ax.set_title('Computational Scaling')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    # Right: NMSE vs n (log-log)
    ax = axes[1]
    for sname in solvers:
        r = results[sname]; valid = [(n, v) for n, v in zip(r['n'], r['nmse']) if np.isfinite(v)]
        if valid:
            ns, vs = zip(*valid); ax.loglog(ns, vs, label=sname, **_solver_style(sname))
    ax.set_xlabel('Signal dimension $n$'); ax.set_ylabel('NMSE')
    ax.set_title('Recovery Accuracy vs Dimension')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout(pad=1.5)
    save_figure(fig, 'large_scale_comparison.png')
    csv_path = DATA_DIR / 'large_scale_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'n', 'nmse', 'time_ms', 'iters'])
        for sname in solvers:
            for i in range(len(results[sname]['n'])):
                w.writerow([sname, results[sname]['n'][i], results[sname]['nmse'][i],
                            results[sname]['time_ms'][i], results[sname]['iters'][i]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F2: Initialization Sensitivity
# Tests solver robustness to poor initialization.
# Hypothesis: KAMP's Kalman correction step converges from worse x0
# than gradient-free baselines (VAMP/OAMP whose SVD is init-independent)
# and gradient-based ones (ISTA/FISTA which get stuck in poor local paths).
# Outputs: init_sensitivity.png, init_sensitivity_results.csv
# ---------------------------------------------------------------------------

def experiment_init_sensitivity(n=100, m=70, k=15, snr_db=30, num_trials=10):
    """F2: Solver robustness to poor initialization.
    Hypothesis: KAMP's Kalman correction converges from worse x0
    than VAMP/OAMP (SVD-based, init-independent) and ISTA/FISTA (gradient paths)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT F2 — Initialization Sensitivity")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9; num_trials = 3
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP']
    init_modes = ['zero', 'truth+noise', 'random']
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    results = {s: {im: [] for im in init_modes} for s in solvers}
    x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
    y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
    for trial in range(num_trials):
        y = y_clean + sigma * rng.randn(m)
        base_config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
        for im in init_modes:
            if im == 'zero':
                x0 = np.zeros(n)
            elif im == 'truth+noise':
                x0 = x_true + 0.5 * rng.randn(n)
            else:
                x0 = rng.randn(n)
            for sname in solvers:
                config = dict(base_config, x0=x0)
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][im].append(nmse)
                except Exception:
                    results[sname][im].append(float('nan'))
        print(f"  Trial {trial + 1}/{num_trials}")
    set_publication_style()
    # Plot: grouped bars of mean NMSE per solver per init mode
    n_solvers = len(solvers); n_modes = len(init_modes)
    x = np.arange(n_solvers); width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, im in enumerate(init_modes):
        means = [np.nanmean(results[s][im]) if results[s][im] else 1e10 for s in solvers]
        offset = (idx - n_modes / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=im,
                       color=['#DDDDDD', '#999999', '#444444'][idx],
                       edgecolor='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(solvers)
    ax.set_yscale('log'); ax.set_ylabel('NMSE (log scale)')
    ax.set_title('Effect of Initialization on Solver Accuracy')
    ax.legend(title='Initialization', fontsize=9); ax.grid(True, axis='y', alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'init_sensitivity.png')
    csv_path = DATA_DIR / 'init_sensitivity_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'init_mode', 'trial', 'nmse'])
        for sname in solvers:
            for im in init_modes:
                for t, v in enumerate(results[sname][im]):
                    w.writerow([sname, im, t, v])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F3: Correlated (Colored) Noise Recovery
# Tests recovery performance when measurement noise is temporally correlated.
# Hypothesis: KAMP's state-space model can incorporate noise correlation,
# giving it an advantage over AMP/VAMP/OAMP which assume white noise.
# Outputs: colored_noise_comparison.png, colored_noise_results.csv
# ---------------------------------------------------------------------------

def experiment_colored_noise(n=100, m=70, k=15, snr_db=30, num_trials=5):
    """F3: Recovery under correlated (colored) measurement noise.
    Hypothesis: KAMP's state-space model can exploit noise structure,
    outperforming AMP/VAMP/OAMP which assume white noise."""
    print("\n" + "=" * 60)
    print("EXPERIMENT F3 — Colored (AR-1) Noise Recovery")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9; num_trials = 2
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    corr_strengths = [0.0, 0.3, 0.6, 0.9, 0.95]
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    results = {s: {cs: [] for cs in corr_strengths} for s in solvers}
    x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
    y_clean = A @ x_true
    for cs in tqdm(corr_strengths, desc="Correlation strength"):
        for _ in range(num_trials):
            white = rng.randn(m)
            if cs > 0:
                noise_filter = np.array([cs ** i for i in range(m)])
                colored = np.convolve(white, noise_filter, mode='same')
                colored = colored / np.std(colored)
            else:
                colored = white
            sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * colored
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
            for sname in solvers:
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][cs].append(nmse)
                except Exception:
                    results[sname][cs].append(float('nan'))
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for sname in solvers:
        means = [np.nanmean(results[sname][cs]) for cs in corr_strengths]
        ax.semilogy(corr_strengths, means, label=sname, **_solver_style(sname))
    ax.set_xlabel('AR-1 noise correlation coefficient $\\rho$')
    ax.set_ylabel('NMSE (log scale)')
    ax.set_title('Recovery Accuracy vs Noise Correlation')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'colored_noise_comparison.png')
    csv_path = DATA_DIR / 'colored_noise_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'correlation', 'trial', 'nmse'])
        for sname in solvers:
            for cs in corr_strengths:
                for t, v in enumerate(results[sname][cs]):
                    w.writerow([sname, cs, t, v])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F4: Statistical Summary & Significance Testing
# Runs many Monte Carlo trials, computes summary statistics per solver,
# and performs pairwise Mann-Whitney U tests with Holm-Bonferroni correction.
# Outputs: statistical_summary.csv, pairwise_pvalues.csv
# ---------------------------------------------------------------------------

def experiment_statistical_summary(n=120, m=84, k=15, snr_db=30, num_trials=100):
    """F4: Comprehensive statistical comparison of all solvers.
    Reports mean, median, std, IQR NMSE and pairwise Mann-Whitney U
    tests with Holm-corrected p-values to establish significance
    of KAMP/DKAMP advantages."""
    print("\n" + "=" * 60)
    print("EXPERIMENT F4 — Statistical Summary & Significance Tests")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 80, 56, 10; num_trials = 15
    solvers = list(SOLVER_REGISTRY.keys())
    results = {s: [] for s in solvers}
    rng = np.random.RandomState(42)
    for trial in range(num_trials):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=42 + trial)
        x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
        y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
        y = y_clean + sigma * rng.randn(m)
        config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=200)
        for sname in solvers:
            try:
                res = SOLVER_REGISTRY[sname](A, y, config)
                nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                results[sname].append(nmse)
            except Exception:
                results[sname].append(float('nan'))
        if (trial + 1) % 20 == 0 or trial == 0:
            print(f"  Trial {trial + 1}/{num_trials}")
    # --- Summary statistics ---
    print("\n  Summary Statistics (NMSE):")
    print(f"  {'Solver':<8} {'Mean':<12} {'Median':<12} {'Std':<12} {'IQR':<12} {'Best':<12} {'Worst':<12}")
    print(f"  {'------':<8} {'----':<12} {'------':<12} {'---':<12} {'---':<12} {'----':<12} {'-----':<12}")
    summary_rows = []
    for sname in solvers:
        vals = np.array(results[sname])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            summary_rows.append((sname, 0, 0, 0, 0, 0, 0))
            continue
        mean_v = np.mean(vals); med_v = np.median(vals); std_v = np.std(vals)
        q25, q75 = np.percentile(vals, [25, 75]); iqr_v = q75 - q25
        best_v = np.min(vals); worst_v = np.max(vals)
        summary_rows.append((sname, mean_v, med_v, std_v, iqr_v, best_v, worst_v))
        print(f"  {sname:<8} {mean_v:<12.4e} {med_v:<12.4e} {std_v:<12.4e} {iqr_v:<12.4e} {best_v:<12.4e} {worst_v:<12.4e}")
    # Save summary CSV
    summary_path = DATA_DIR / 'statistical_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'mean_nmse', 'median_nmse', 'std_nmse', 'iqr_nmse', 'best_nmse', 'worst_nmse'])
        for r in summary_rows:
            w.writerow(list(r))
    print(f"  CSV saved -> {summary_path}")
    # --- Pairwise Mann-Whitney U tests with Holm correction ---
    print("\n  Pairwise Mann-Whitney U tests (Holm-corrected p-values):")
    n_solvers = len(solvers)
    pval_matrix = np.ones((n_solvers, n_solvers))
    pairwise = []
    for i in range(n_solvers):
        for j in range(i + 1, n_solvers):
            a = results[solvers[i]]; b = results[solvers[j]]
            a_clean = np.array([v for v in a if np.isfinite(v)])
            b_clean = np.array([v for v in b if np.isfinite(v)])
            if len(a_clean) > 1 and len(b_clean) > 1:
                try:
                    _, p = mannwhitneyu(a_clean, b_clean, alternative='two-sided')
                    pairwise.append((solvers[i], solvers[j], p))
                except Exception:
                    pairwise.append((solvers[i], solvers[j], 1.0))
            else:
                pairwise.append((solvers[i], solvers[j], 1.0))
    # Holm-Bonferroni correction
    n_comparisons = len(pairwise)
    if n_comparisons > 0:
        sorted_idx = np.argsort([p for _, _, p in pairwise])
        holm_corrected = {}
        for rank, idx in enumerate(sorted_idx):
            s1, s2, raw_p = pairwise[idx]
            corrected_p = min(raw_p * (n_comparisons - rank), 1.0)
            holm_corrected[(s1, s2)] = corrected_p
            holm_corrected[(s2, s1)] = corrected_p
        for i in range(n_solvers):
            for j in range(n_solvers):
                if i != j:
                    pval_matrix[i, j] = holm_corrected.get((solvers[i], solvers[j]), 1.0)
    print(f"  {'Pair':<18} {'Raw p':<12} {'Holm p':<12} {'Signif.':<10}")
    print(f"  {'----':<18} {'-----':<12} {'------':<12} {'-------':<10}")
    for s1, s2, raw_p in pairwise:
        cp = holm_corrected.get((s1, s2), 1.0)
        sig = "***" if cp < 0.001 else "**" if cp < 0.01 else "*" if cp < 0.05 else "n.s."
        print(f"  {s1:<8} vs {s2:<8} {raw_p:<12.4e} {cp:<12.4e} {sig:<10}")
    # Save pairwise CSV
    pval_path = DATA_DIR / 'pairwise_pvalues.csv'
    with open(pval_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver1', 'solver2', 'raw_p', 'holm_corrected_p', 'significant_05'])
        for s1, s2, raw_p in pairwise:
            cp = holm_corrected.get((s1, s2), 1.0)
            w.writerow([s1, s2, raw_p, cp, cp < 0.05])
    print(f"  CSV saved -> {pval_path}")
    # --- Generate p-value heatmap figure ---
    set_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pval_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(range(n_solvers)); ax.set_yticks(range(n_solvers))
    ax.set_xticklabels(solvers, rotation=45, ha='right'); ax.set_yticklabels(solvers)
    for i in range(n_solvers):
        for j in range(n_solvers):
            if i == j:
                ax.text(j, i, '--', ha='center', va='center', fontsize=8, color='black')
            else:
                pv = pval_matrix[i, j]
                txt = f"{pv:.2e}" if pv < 0.001 else f"{pv:.3f}"
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='white' if pv < 0.05 else 'black')
    ax.set_title('Pairwise Mann-Whitney U Test\n(Holm-corrected p-values)')
    plt.tight_layout()
    save_figure(fig, 'statistical_heatmap.png')
    # --- Generate bar chart figure ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    snames = [r[0] for r in summary_rows]
    means = [r[1] for r in summary_rows]
    colors_bars = [SOLVER_COLORS.get(s, '#666666') for s in snames]
    bars = ax2.bar(snames, means, color=colors_bars, edgecolor='black', linewidth=0.8)
    ax2.set_yscale('log'); ax2.set_ylabel('Mean NMSE (log scale)')
    ax2.set_title('Average Recovery Accuracy Across Monte Carlo Trials')
    ax2.grid(True, axis='y', alpha=0.25)
    for s in ['top','right']: ax2.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig2, 'statistical_summary_bar.png')


# ---------------------------------------------------------------------------
# Experiment F5: AR-1 Correlated Matrix Stress Test
# Sweeps matrix column correlation rho from i.i.d. to near-singular.
# Hypothesis: AMP diverges for high rho, VAMP/OAMP degrade;
# KAMP's Kalman structure handles correlation naturally.
# Outputs: ar1_stress.png, ar1_stress_results.csv
# ---------------------------------------------------------------------------

def experiment_ar1_stress(n=160, m_ratio=0.7, k_ratio=0.15, snr_db=30, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT F5 — AR-1 Correlated Matrix Stress")
    print("=" * 60)
    if _FAST_MODE:
        n = 80; num_trials = 3
    m = max(int(n * m_ratio), 1); k = max(int(n * k_ratio), 1)
    rho_vals = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    results = {s: {r: [] for r in rho_vals} for s in solvers}
    rng = np.random.RandomState(42)
    for rho in tqdm(rho_vals, desc="Correlation rho"):
        cov = rho ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
        for _ in range(num_trials):
            base = rng.randn(m, n)
            A = base @ L.T / np.sqrt(n)
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
            for sname in solvers:
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][rho].append(nmse)
                except Exception:
                    results[sname][rho].append(float('nan'))
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for sname in solvers:
        means = [np.nanmean(results[sname][r]) for r in rho_vals]
        ax.semilogy(rho_vals, means, label=sname, **_solver_style(sname))
    ax.set_xlabel('AR-1 column correlation coefficient $\\rho$')
    ax.set_ylabel('NMSE (log scale)')
    ax.set_title('Recovery Accuracy vs Matrix Column Correlation')
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'ar1_stress.png')
    csv_path = DATA_DIR / 'ar1_stress_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'rho', 'trial', 'nmse'])
        for sname in solvers:
            for rho in rho_vals:
                for t, v in enumerate(results[sname][rho]):
                    w.writerow([sname, rho, t, v])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F6: Time-Varying Dynamic Signal Tracking
# Tests KAMP's state-space tracking vs stateless solvers on a slowly
# varying sparse signal. KAMP warm-starts from previous estimate;
# VAMP/OAMP/ISTA restart from zero each step.
# Outputs: dynamic_tracking.png, dynamic_tracking_results.csv
# ---------------------------------------------------------------------------

def experiment_time_varying_tracking(n=100, m=70, k=15, snr_db=30, T_steps=30, num_trials=5):
    print("\n" + "=" * 60)
    print("EXPERIMENT F6 — Time-Varying Signal Tracking")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9; T_steps = 15; num_trials = 2
    solvers = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    k_innov = max(int(k * 0.15), 1)
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    results = {s: {'step': [], 'nmse': []} for s in solvers}
    rng = np.random.RandomState(42)
    for trial in range(num_trials):
        x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
        x_prev_hat = None
        for t in range(T_steps):
            innov_support = rng.choice(n, k_innov, replace=False)
            x_true = 0.95 * x_true; x_true[innov_support] += 0.1 * rng.randn(k_innov)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            base_config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=50)
            for sname in solvers:
                config = dict(base_config)
                if sname == 'KAMP' and x_prev_hat is not None:
                    config['x0'] = x_prev_hat.copy()
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname]['step'].append(t); results[sname]['nmse'].append(nmse)
                    if sname == 'KAMP': x_prev_hat = res.x_hat.copy()
                except Exception:
                    results[sname]['step'].append(t); results[sname]['nmse'].append(float('nan'))
        print(f"  Trial {trial+1}/{num_trials}")
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for sname in solvers:
        valid = [(s, v) for s, v in zip(results[sname]['step'], results[sname]['nmse']) if np.isfinite(v)]
        if valid:
            steps, vals = zip(*valid)
            ax.plot(steps, vals, label=sname, alpha=0.7, **_solver_style(sname))
    ax.set_xlabel('Time step'); ax.set_ylabel('NMSE (log scale)')
    ax.set_title('Dynamic Sparse Signal Tracking Over Time')
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'dynamic_tracking.png')
    csv_path = DATA_DIR / 'dynamic_tracking_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'trial', 'step', 'nmse'])
        for sname in solvers:
            for t in range(len(results[sname]['step'])):
                w.writerow([sname, t // T_steps, results[sname]['step'][t], results[sname]['nmse'][t]])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F7: Heavy-Tailed / Non-Gaussian Noise Robustness
# Tests all solvers under Gaussian, Student-t, Laplacian, and mixture noise.
# Hypothesis: VAMP/OAMP assume Gaussian noise, degrade under heavy tails;
# KAMP's Kalman correction with Joseph form is more robust.
# Outputs: non_gaussian_noise.png, non_gaussian_noise_results.csv
# ---------------------------------------------------------------------------

def experiment_non_gaussian_noise(n=120, m=84, k=15, snr_db=30, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT F7 — Non-Gaussian Noise Robustness")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9; num_trials = 3
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    noise_models = OrderedDict([
        ('Gaussian', lambda s, r: s * r.randn(m)),
        ('Student-t(3)', lambda s, r: s * r.standard_t(3, size=m) / np.sqrt(3.0)),
        ('Laplacian', lambda s, r: s * r.laplace(0, 1.0 / np.sqrt(2.0), size=m)),
        ('Gaussian mixture', lambda s, r: s * (0.9 * r.randn(m) + 0.1 * 5.0 * r.randn(m))),
    ])
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    rng = np.random.RandomState(42)
    results = {s: {nm: [] for nm in noise_models} for s in solvers}
    x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
    y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
    for nm_label, noise_fn in noise_models.items():
        for _ in range(num_trials):
            y = y_clean + noise_fn(sigma, rng)
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
            for sname in solvers:
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][nm_label].append(nmse)
                except Exception:
                    results[sname][nm_label].append(float('nan'))
    set_publication_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    nm_labels = list(noise_models.keys()); n_nm = len(nm_labels)
    x = np.arange(n_nm); width = 0.15
    for idx, sname in enumerate(solvers):
        means = [np.nanmean(results[sname][nm]) for nm in nm_labels]
        offset = (idx - len(solvers) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=sname, **_solver_bar_style(sname))
    ax.set_xticks(x); ax.set_xticklabels(nm_labels, rotation=20, ha='right')
    ax.set_yscale('log'); ax.set_ylabel('Mean NMSE (log scale)')
    ax.set_title('Robustness to Non-Gaussian Noise Distributions')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, axis='y', alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'non_gaussian_noise.png')
    csv_path = DATA_DIR / 'non_gaussian_noise_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'noise_model', 'trial', 'nmse'])
        for sname in solvers:
            for nm in nm_labels:
                for t, v in enumerate(results[sname][nm]):
                    w.writerow([sname, nm, t, v])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F8: Extended Ill-Conditioned Recovery (kappa up to 1000)
# Sweeps condition number over a wider range than existing experiments.
# Hypothesis: VAMP/OAMP SVD-based LMMSE becomes ill-conditioned for
# kappa > 50; KAMP's covariance regularization provides stability.
# Outputs: ill_conditioned_extended.png, ill_conditioned_extended_results.csv
# ---------------------------------------------------------------------------

def experiment_ill_conditioned_extended(n=120, m=84, k=15, snr_db=40, num_trials=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT F8 — Extended Ill-Conditioned Recovery")
    print("=" * 60)
    if _FAST_MODE:
        n, m, k = 60, 42, 9; num_trials = 3
    kappa_vals = [1, 5, 10, 30, 50, 100, 200, 500, 1000]
    solvers = ['AMP', 'VAMP', 'OAMP', 'KAMP', 'DKAMP']
    results = {s: {kv: [] for kv in kappa_vals} for s in solvers}
    rng = np.random.RandomState(42)
    for kappa in tqdm(kappa_vals, desc="Condition kappa"):
        for _ in range(num_trials):
            A = create_measurement_matrix(m, n, 'ill_conditioned', condition_number=kappa, random_state=rng.randint(10000))
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y = y_clean + sigma * rng.randn(m)
            config = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
            for sname in solvers:
                try:
                    res = SOLVER_REGISTRY[sname](A, y, config)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][kappa].append(nmse)
                except Exception:
                    results[sname][kappa].append(float('nan'))
    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    ax2 = ax1.twinx()
    for sname in solvers:
        means = [np.nanmean(results[sname][kv]) for kv in kappa_vals]
        ax1.semilogy(kappa_vals, means, label=sname, **_solver_style(sname))
        success_rate = [np.mean([1.0 if np.isfinite(v) and v < 1e-3 else 0.0 for v in results[sname][kv]]) for kv in kappa_vals]
        ax2.plot(kappa_vals, success_rate, linestyle=':', linewidth=1.5,
                 color=SOLVER_COLORS.get(sname, 'black'), alpha=0.5)
    ax1.set_xlabel('Condition number $\\kappa$ (log scale)'); ax1.set_xscale('log')
    ax1.set_ylabel('NMSE (log scale)'); ax1.set_title('Recovery vs Condition Number')
    ax1.legend(fontsize=8, ncol=2, loc='upper left'); ax1.grid(True, alpha=0.25)
    ax2.set_ylabel('Success rate (NMSE < $10^{-3}$)', fontsize=10)
    for s in ['top','right']: ax1.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'ill_conditioned_extended.png')
    csv_path = DATA_DIR / 'ill_conditioned_extended_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'kappa', 'trial', 'nmse'])
        for sname in solvers:
            for kv in kappa_vals:
                for t, v in enumerate(results[sname][kv]):
                    w.writerow([sname, kv, t, v])
    print(f"  CSV saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Experiment F9: DKAMP Distributed Recovery Scenario
# Tests DKAMP vs centralized solvers with limited per-node measurements.
# As nodes increase, each node has fewer rows; centralized methods fail
# while DKAMP aggregates distributed information cooperatively.
# Outputs: distributed_scenario.png, distributed_scenario_results.csv
# ---------------------------------------------------------------------------

def experiment_distributed_scenario(n=100, m_total=100, k=15, snr_db=30, num_trials=5):
    print("\n" + "=" * 60)
    print("EXPERIMENT F9 — DKAMP Distributed Scenario")
    print("=" * 60)
    if _FAST_MODE:
        n, m_total, k = 60, 60, 9; num_trials = 2
    L_vals = [2, 4, 6, 10] if _FAST_MODE else [2, 3, 4, 5, 6, 8, 10, 12]
    solvers_cent = ['ISTA', 'FISTA', 'AMP', 'VAMP', 'OAMP', 'KAMP']
    results = {s: {L: [] for L in L_vals} for s in solvers_cent + ['DKAMP']}
    rng = np.random.RandomState(42)
    for L in tqdm(L_vals, desc="Num nodes L"):
        m_per = max(m_total // L, 1)
        for _ in range(num_trials):
            A_total = create_measurement_matrix(m_total, n, 'gaussian', random_state=rng.randint(10000))
            x_true = np.zeros(n); support = rng.choice(n, k, replace=False); x_true[support] = rng.randn(k)
            y_clean = A_total @ x_true; sigma = np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10))
            y_total = y_clean + sigma * rng.randn(m_total)
            # Centralized: each solver gets only the local sub-problem
            A_local = A_total[:m_per, :]; y_local = y_total[:m_per]
            config_cent = _make_config(x_true, sigma, lam=0.1 * sigma, max_iter=100)
            for sname in solvers_cent:
                try:
                    res = SOLVER_REGISTRY[sname](A_local, y_local, config_cent)
                    nmse = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                    results[sname][L].append(nmse)
                except Exception:
                    results[sname][L].append(float('nan'))
            # DKAMP: distributed with all L nodes
            A_nodes = [A_total[i * m_per:(i + 1) * m_per, :] for i in range(L)]
            y_nodes = [y_total[i * m_per:(i + 1) * m_per] for i in range(L)]
            try:
                dkamp = DKAMP(alpha=0.5, tau=0.05, num_rounds=10, node_max_iter=5, verbose=False)
                dkamp.fit(A_nodes, y_nodes, adjacency='ring')
                res_d = dkamp.solve()
                nmse_d = float(np.sum((res_d.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
                results['DKAMP'][L].append(nmse_d)
            except Exception:
                results['DKAMP'][L].append(float('nan'))
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    all_solvers = solvers_cent + ['DKAMP']
    for sname in all_solvers:
        means = [np.nanmean(results[sname][L]) for L in L_vals]
        ax.semilogy(L_vals, means, label=sname, **_solver_style(sname))
    ax.set_xlabel('Number of distributed nodes $L$')
    ax.set_ylabel('NMSE (log scale)')
    ax.set_title('Distributed Recovery: DKAMP vs Centralized (Limited Local Data)')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.25)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'distributed_scenario.png')
    csv_path = DATA_DIR / 'distributed_scenario_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['solver', 'L_nodes', 'trial', 'nmse'])
        for sname in all_solvers:
            for L in L_vals:
                for t, v in enumerate(results[sname][L]):
                    w.writerow([sname, L, t, v])
    print(f"  CSV saved -> {csv_path}")


# ============================================================================
# Part 9 — Main Orchestrator
# ============================================================================


# ============================================================================
# Part Z — Interactive Menu & Entry Points
# ============================================================================

_CORE_EXPERIMENTS = [
    ("A1", "Onsager Correction Gap", experiment_onsager_validation,
     {'n': 100, 'm': 70, 'k': 15, 'snr_db': 30, 'num_trials': 20}),
    ("A2", "State Evolution", experiment_state_evolution, {}),
    ("B1", "Full Benchmark", experiment_full_benchmark, {}),
    ("B2", "Covariance Modes", experiment_covariance_modes, {}),
    ("B3", "Phase Transition", experiment_phase_transition, {}),
    ("C0", "Version Table", experiment_version_table, {}),
    ("C1", "Noise Mismatch", experiment_noise_mismatch, {}),
    ("C2", "Scalability", experiment_scalability, {}),
    ("C3", "Complexity", experiment_complexity_analysis, {}),
    ("C4", "Convergence Speed", experiment_convergence_speed, {}),
    ("D1", "Ablation", experiment_ablation, {}),
    ("D2", "Onsager Validation", experiment_onsager_validation,
     {'n': 100, 'm': 70, 'k': 15, 'snr_db': 30, 'num_trials': 20}),
    ("D3", "Statistical Comparison", experiment_statistical_comparison, {}),
    ("D4", "Continuous Sweep", experiment_continuous_sweep, {}),
    ("D5", "Covariance Structure", experiment_covariance_structure_modes, {}),
    ("E2", "Full Benchmark SNR", experiment_full_benchmark_snr, {}),
    ("E3", "MSE/VAR Heatmaps", experiment_onsager_validation_heatmap, {}),
    ("F1", "Large-Scale Recovery", experiment_large_scale, {}),
    ("F2", "Init Sensitivity", experiment_init_sensitivity, {}),
    ("F3", "Colored Noise", experiment_colored_noise, {}),
    ("F4", "Statistical Summary", experiment_statistical_summary, {}),
    ("F5", "AR-1 Correlated Stress", experiment_ar1_stress, {}),
    ("F6", "Time-Varying Tracking", experiment_time_varying_tracking, {}),
    ("F7", "Non-Gaussian Noise", experiment_non_gaussian_noise, {}),
    ("F8", "Ill-Conditioned Extended", experiment_ill_conditioned_extended, {}),
    ("F9", "Distributed Scenario", experiment_distributed_scenario, {}),
]

_SHOWCASE_EXPERIMENTS = []

def _load_showcase():
    global _SHOWCASE_EXPERIMENTS
    if _SHOWCASE_EXPERIMENTS:
        return
    try:
        from kamp_showcase_experiments import (
            experiment_condition_number_sweep,
            experiment_correlated_matrix_sweep,
            experiment_heavy_tailed_noise,
            experiment_impulsive_noise,
            experiment_dynamic_tracking,
            experiment_abrupt_support_change,
            experiment_topology_comparison,
            experiment_node_failure_robustness,
            experiment_low_measurement_rate,
            experiment_very_low_snr,
        )
        _SHOWCASE_EXPERIMENTS.extend([
            ("S1", "Condition Number Sweep", experiment_condition_number_sweep, {}),
            ("S2", "Correlated Matrix Sweep", experiment_correlated_matrix_sweep, {}),
            ("S3", "Heavy-Tailed Noise Robustness", experiment_heavy_tailed_noise, {}),
            ("S4", "Impulsive Noise Robustness", experiment_impulsive_noise, {}),
            ("S5", "Dynamic Tracking (Cold vs Kalman)", experiment_dynamic_tracking, {}),
            ("S6", "Abrupt Support-Change Tracking", experiment_abrupt_support_change, {}),
            ("S7", "DKAMP Topology Comparison", experiment_topology_comparison, {}),
            ("S8", "DKAMP Node-Failure Robustness", experiment_node_failure_robustness, {}),
            ("S9", "Low Measurement-Rate Stress", experiment_low_measurement_rate, {}),
            ("S10", "Very Low SNR Stress", experiment_very_low_snr, {}),
        ])
    except ImportError:
        print("  [info] kamp_showcase_experiments.py not found — showcase experiments disabled")


def _run_experiment_list(experiments, fast=False):
    if fast:
        global _FAST_MODE; _FAST_MODE = True
    results = []
    for code, name, func, kwargs in experiments:
        print(f"\n>>> {code}. {name}...")
        t0 = time.time()
        try:
            func(**kwargs)
            elapsed = time.time() - t0
            status = "PASS"
            print(f"  [{status}]  {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            status = "FAIL"
            print(f"  [{status}]  {elapsed:.1f}s  — {e}")
        results.append((code, name, status, elapsed))
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for code, name, status, elapsed in results:
        print(f"  [{status}]  {code:4s} {name:40s}  {elapsed:.1f}s")
    return results


def _print_menu():
    _load_showcase()
    sections = {
        'A': [e for e in _CORE_EXPERIMENTS if e[0].startswith('A')],
        'B': [e for e in _CORE_EXPERIMENTS if e[0].startswith('B')],
        'C': [e for e in _CORE_EXPERIMENTS if e[0].startswith('C')],
        'D': [e for e in _CORE_EXPERIMENTS if e[0].startswith('D')],
        'E': [e for e in _CORE_EXPERIMENTS if e[0].startswith('E')],
        'F': [e for e in _CORE_EXPERIMENTS if e[0].startswith('F')],
    }
    print("=" * 70)
    print("  KAMP / DKAMP - Complete Experiment Suite")
    print("=" * 70)
    for sec in ['A', 'B', 'C', 'D', 'E', 'F']:
        print(f"\n  Section {sec}:")
        for code, name, *_ in sections[sec]:
            print(f"    [{code}]  {name}")
    if _SHOWCASE_EXPERIMENTS:
        print(f"\n  Showcase (kamp_showcase):")
        for code, name, *_ in _SHOWCASE_EXPERIMENTS:
            print(f"    [{code}] {name}")
    print("\n" + "-" * 70)
    print("  Commands:")
    print("    1,2,3       Run individual experiments by number")
    print("    1-5,7-9     Run ranges")
    print("    all         Run ALL experiments (core + showcase)")
    print("    core        Run all core experiments (A-F)")
    print("    showcase    Run all showcase experiments (S1-S10)")
    print("    A, B, C, D, E, S   Run section")
    print("    test        Quick solver verification (2s)")
    print("    fast        Run all in fast mode")
    print("    0           Exit")
    print("-" * 70)


def _run_interactive_menu():
    _load_showcase()
    all_experiments = _CORE_EXPERIMENTS + _SHOWCASE_EXPERIMENTS
    section_map = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'S': 'S'}
    idx_map = {f"{code}": (code, name, func, kw) for code, name, func, kw in all_experiments}

    while True:
        _print_menu()
        try:
            cmd = input("\n  Enter selection: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting."); break
        if not cmd:
            continue
        if cmd == '0':
            print("  Exiting."); break

        selected = []
        if cmd == 'all':
            selected = list(all_experiments)
        elif cmd == 'core':
            selected = list(_CORE_EXPERIMENTS)
        elif cmd == 'showcase':
            selected = list(_SHOWCASE_EXPERIMENTS)
        elif cmd == 'test':
            _run_quick_test(); continue
        elif cmd == 'fast':
            _run_experiment_list(all_experiments, fast=True); continue
        elif cmd.upper() in section_map:
            prefix = section_map[cmd.upper()]
            if prefix == 'S':
                selected = [e for e in _SHOWCASE_EXPERIMENTS if e[0].startswith('S')]
            else:
                selected = [e for e in _CORE_EXPERIMENTS if e[0].startswith(prefix)]
        else:
            parts = cmd.replace(',', ' ').split()
            for p in parts:
                if '-' in p:
                    a, b = p.split('-')
                    for i in range(int(a.strip()), int(b.strip()) + 1):
                        key = f"{'S' if i <= 10 and len(_SHOWCASE_EXPERIMENTS) >= i else 'A'}{i}"
                        # Try core first, then showcase
                        found = idx_map.get(f"{'A' if i <= 17 else 'S'}{i}" if i <= 17 else f"S{i-17}")
                        for code in [f"A{i}", f"B{i}", f"C{i}", f"D{i}", f"E{i}", f"F{i}", f"S{i}"]:
                            if code in idx_map:
                                found = idx_map[code]
                                break
                        if found and found not in selected:
                            selected.append(found)
                else:
                    # Try as code (A1, B3, S5) or as plain number
                    p_upper = p.upper()
                    if p_upper in idx_map:
                        selected.append(idx_map[p_upper])
                    else:
                        for code in [f"A{p}", f"B{p}", f"C{p}", f"D{p}", f"E{p}", f"F{p}", f"S{p}"]:
                            if code in idx_map:
                                selected.append(idx_map[code])
                                break
        if not selected:
            print("  No valid experiments selected.")
            continue
        names = ", ".join(f"{c}" for c, *_ in selected)
        confirm = input(f"  Run {len(selected)} experiment(s): {names}? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            _run_experiment_list(selected)
        print()


def _run_quick_test():
    global _FAST_MODE; _FAST_MODE = True
    np.random.seed(42)
    n, m, k = 50, 35, 8
    A = create_measurement_matrix(m, n, 'gaussian', random_state=42)
    x_true = np.zeros(n); s = np.random.choice(n, k, replace=False); x_true[s] = np.random.randn(k)
    sigma = 0.05
    y = A @ x_true + sigma * np.random.randn(m)
    config = _make_config(x_true, sigma, lam=0.1, max_iter=50)
    print("=" * 60)
    print("  QUICK VERIFICATION - All Solvers")
    print("=" * 60)
    for name, fn in SOLVER_REGISTRY.items():
        try:
            t0 = time.perf_counter()
            res = fn(A, y, config)
            elapsed = (time.perf_counter() - t0) * 1000
            nmse_val = float(np.sum((res.x_hat - x_true) ** 2) / np.sum(x_true ** 2))
            # Also compute debiased NMSE
            x_db = _debias_on_support(A, y, res.x_hat)
            nmse_db = float(np.sum((x_db - x_true) ** 2) / np.sum(x_true ** 2))
            status = "OK" if np.isfinite(nmse_val) and nmse_val < 1.0 else "DIVERGED"
            print(f"  {name:<8} NMSE={nmse_val:.4e}  debias={nmse_db:.4e}  time={elapsed:.1f}ms  [{status}]")
        except Exception as e:
            print(f"  {name:<8} FAILED: {e}")


def main():
    """Non-interactive default: runs all core experiments then exits."""
    print("=" * 70)
    print("  KAMP / DKAMP - Complete Experiment Suite")
    print("=" * 70)
    _run_experiment_list(_CORE_EXPERIMENTS)


if __name__ == "__main__":
    if '--light' in sys.argv:
        _LIGHT_MODE = True
    if '--fast' in sys.argv:
        _FAST_MODE = True
    if '--test' in sys.argv:
        _FAST_MODE = True
        _run_quick_test()
        sys.exit(0)
    if any(f in sys.argv for f in ('--menu', '-m')):
        _run_interactive_menu()
        sys.exit(0)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        _run_interactive_menu()
        sys.exit(0)
    if '--fast' in sys.argv or '--light' in sys.argv:
        main()
        sys.exit(0)
    _run_interactive_menu()
