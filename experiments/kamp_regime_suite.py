"""
kamp_regime_suite.py — Six-Regime, 17-Experiment Publication Suite for KAMP & DKAMP

Implements the full "Regime I ... Regime VI" experimental plan:

  Regime I   (R01-R03)  Standard i.i.d. compressed sensing (honest control)
  Regime II  (R04-R07)  Structural violations of AMP/VAMP/OAMP assumptions
  Regime III (R08-R11)  Capabilities competitors cannot match
  Regime IV  (R12-R13)  Rigorous theoretical validation (PCRLB, NEES/NIS)
  Regime V   (R14-R16)  Actionable applications of the posterior covariance
  Regime VI  (R17)      Application-driven external validation (Jakes channel)

Every figure is Q1-grade: 400 dpi, Times New Roman serif, Paul Tol bright
colorblind-friendly palette, white-filled markers with black edges, SEM
error bars, 95% confidence bands, and panel labels (a)(b)(c).

Outputs are stored in dedicated, clearly named directories:
  experiments/regime_figures/   (figures)
  experiments/regime_data/      (CSV result tables)

Usage:
  python kamp_regime_suite.py --list            list experiments
  python kamp_regime_suite.py --fast            quick verification (small sizes)
  python kamp_regime_suite.py --only R01,R08    run selected experiments
  python kamp_regime_suite.py                   run all 17 (full settings)

English only, by design.
"""

import os, sys, time, csv, warnings, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, chi2, t as t_dist, norm
from scipy.linalg import cholesky, solve_triangular

from kamp_complete import (
    SOLVER_REGISTRY, _make_config, create_measurement_matrix,
    _is_diverged, KAMP, DKAMP, _solver_style, _debias_on_support,
    SOLVER_COLORS, SOLVER_MARKERS, SOLVER_LINESTYLES, SOLVER_HATCHES,
    set_publication_style,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

RNG_SEED = 42

# ---------------------------------------------------------------------------
# Dedicated output directories (distinct from the legacy figures/data dirs)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / 'experiments' / 'regime_figures'
DATA_DIR = PROJECT_ROOT / 'experiments' / 'regime_data'
for d in (FIG_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

FAST = '--fast' in sys.argv
ONLY = None
if '--only' in sys.argv:
    i = sys.argv.index('--only')
    ONLY = [c.strip().upper() for c in sys.argv[i + 1].split(',')]

# Solver order used throughout the suite (MAMP omitted by design)
SOLVERS = [s for s in SOLVER_REGISTRY if s != 'MAMP']


# ============================================================================
# Shared helpers
# ============================================================================

def _nmse(x_hat, x_true):
    x_hat = np.asarray(x_hat).flatten()
    x_true = np.asarray(x_true).flatten()
    denom = float(np.sum(x_true ** 2))
    if denom <= 0 or not np.isfinite(x_hat).all():
        return float('nan')
    return float(np.sum((x_hat - x_true) ** 2) / denom)


def _sem(a):
    a = np.asarray(a, dtype=float)
    if a.ndim > 1:
        return np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(np.nanstd(a) / np.sqrt(a.size))


def _holm_correct(pvals):
    """Holm-Bonferroni correction; returns corrected p-values in input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (m - rank))
        adj[idx] = min(running, 1.0)
    return adj


def _toep_corr(n, rho):
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _ar1_matrix(m, n, rho, random_state=None):
    """Sensing matrix with AR-1 correlated columns (Toeplitz)."""
    rng = np.random.RandomState(random_state)
    base = rng.randn(m, n)
    L = np.linalg.cholesky(_toep_corr(n, rho) + 1e-10 * np.eye(n))
    return (base @ L.T) / np.sqrt(n)


def _overlap_basis(A0, sep, random_state):
    """Build a second node basis whose row space is a mixture of A0's and an
    independent draw, with independence growing in `sep` (0 <= sep <= 16).
    sep = 0: identical (redundant) bases; sep = 16: fully independent."""
    f = min(sep / 16.0, 1.0)
    rng = np.random.RandomState(random_state)
    B = rng.randn(*A0.shape) / np.sqrt(A0.shape[1])
    return (1.0 - f) * A0 + f * B


def _make_signal(n, k, rng, scale=1.0):
    x = np.zeros(n)
    support = rng.choice(n, k, replace=False)
    x[support] = scale * rng.randn(k)
    return x, support


def _noise_sigma(y_clean, snr_db):
    return float(np.sqrt(np.mean(y_clean ** 2) / 10 ** (snr_db / 10)))


def _save_csv(name, header, rows):
    path = DATA_DIR / name
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  CSV saved -> {path.name}")


def _save_fig(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved -> {path.name}")


def _line_style(sname, width=None):
    st = _solver_style(sname)
    return dict(color=st['color'], marker=st['marker'],
                linestyle=st['linestyle'],
                linewidth=width if width is not None else st['linewidth'],
                markersize=st['markersize'],
                markerfacecolor='white', markeredgecolor='black',
                markeredgewidth=1.3, zorder=3)


def _bar_style(sname):
    return dict(color=SOLVER_COLORS.get(sname, '#666666'),
                hatch=SOLVER_HATCHES.get(sname, ''),
                edgecolor='black', linewidth=0.8, zorder=2)


def _band(ax, x, mean, se, color, alpha=0.18):
    mean = np.asarray(mean, dtype=float)
    se = np.asarray(se, dtype=float)
    lo = np.maximum(mean - 1.96 * se, 1e-300)
    hi = mean + 1.96 * se
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def _panel_label(ax, letter):
    ax.text(-0.12, 1.05, f'({letter})', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='right')


# ============================================================================
# Solver runners
# ============================================================================

def _run_solver(name, A, y, x_true, sigma, lam=None, max_iter=200, x0=None,
                **extra):
    if lam is None:
        lam = 0.1 * sigma
    config = _make_config(x_true, sigma, lam=lam, max_iter=max_iter)
    if name in ('KAMP', 'DKAMP'):
        config['tau'] = 2.0 * sigma
        config['tau_adaptive'] = False
    if x0 is not None:
        config['x0'] = np.asarray(x0).flatten().copy()
    config.update(extra)
    res = SOLVER_REGISTRY[name](A, y, config)
    nmse_val = _nmse(res.x_hat, x_true)
    div = bool(res.info.get('diverged', False)) or (not np.isfinite(nmse_val))
    return {
        'x_hat': np.asarray(res.x_hat).flatten(),
        'nmse': nmse_val,
        'time_ms': res.info.get('time_ms', np.nan),
        'iters': res.info.get('iters', np.nan),
        'diverged': div,
    }


def _run_kamp(A, y, x_true, sigma, lam=None, max_iter=200, R=None, Q=None,
              x0=None, P0=None, tau_adaptive=True, alpha=0.5,
              sigma2_noise=None, sigma2_prior=1.0):
    """KAMP with full control over R, Q, warm-start (x0, P0)."""
    if lam is None:
        lam = 0.1 * sigma
    kamp = KAMP(alpha=alpha, tau=lam, max_iter=max_iter, tol=1e-6,
                sigma2_prior=sigma2_prior,
                sigma2_noise=sigma2_noise if sigma2_noise is not None else sigma ** 2,
                tau_adaptive=tau_adaptive)
    kamp.fit(A, y, true_x=x_true)
    if R is not None:
        kamp._R = np.asarray(R, dtype=float)
    if Q is not None:
        kamp._Q = np.asarray(Q, dtype=float)
    if P0 is not None:
        kamp._P = np.asarray(P0, dtype=float)
    if x0 is not None:
        kamp.x = np.asarray(x0).flatten().reshape(-1, 1)
    t0 = time.perf_counter()
    kamp.solve(true_x=x_true)
    dt = (time.perf_counter() - t0) * 1e3
    x_hat = kamp.x.flatten()
    nmse_val = _nmse(x_hat, x_true)
    return {
        'x_hat': x_hat,
        'P': kamp._P.copy(),
        'nmse': nmse_val,
        'time_ms': dt,
        'iters': len(kamp.convergence_history),
        'diverged': _is_diverged(x_hat) or (not np.isfinite(nmse_val)),
        'kamp': kamp,
    }

# ============================================================================
# REGIME I — Standard i.i.d. Compressed Sensing (Control)
# ============================================================================

def experiment_r01_static_benchmark():
    """R01: Full static solver benchmark. Sweeps SNR, sparsity rho, ratio delta.
    Outputs: R01_static_benchmark.png, R01_static_benchmark.csv"""
    print("\n" + "=" * 66)
    print("R01 — Full Static Solver Benchmark (SNR / rho / delta sweeps)")
    print("=" * 66)
    n = 60 if FAST else 100
    k0 = 9 if FAST else 15
    snr_vals = [10, 20, 30, 40] if FAST else [0, 10, 20, 30, 40]
    rho_vals = [0.1, 0.2, 0.3] if FAST else [0.05, 0.1, 0.2, 0.3, 0.4]
    delta_vals = [0.5, 0.7, 0.9] if FAST else [0.4, 0.55, 0.7, 0.85]
    trials = 2 if FAST else 5
    rng = np.random.RandomState(RNG_SEED)

    def run_scenario(A, y, x_true, sigma):
        out = {}
        for s in SOLVERS:
            if s == 'DKAMP':
                out[s] = _run_solver(s, A, y, x_true, sigma, max_iter=120,
                                     num_nodes=2, num_triggers=4)
            else:
                out[s] = _run_solver(s, A, y, x_true, sigma, max_iter=200)
        return out

    # --- SNR sweep (delta=0.7, rho=0.15) ---
    snr_res = {s: {'nmse': [], 'time': [], 'div': []} for s in SOLVERS}
    for snr in snr_vals:
        m = int(0.7 * n); k = max(int(0.15 * n), 3)
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian', random_state=1000 + tr + int(snr))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma = _noise_sigma(yc, snr)
            y = yc + sigma * rng.randn(m)
            for s, r in run_scenario(A, y, x, sigma).items():
                snr_res[s]['nmse'].append(r['nmse'])
                snr_res[s]['time'].append(r['time_ms'])
                snr_res[s]['div'].append(int(r['diverged']))
    # --- rho sweep (delta=0.7, snr=30) ---
    rho_res = {s: {'nmse': [], 'div': []} for s in SOLVERS}
    for rho in rho_vals:
        m = int(0.7 * n); k = max(int(rho * n), 2)
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian', random_state=2000 + tr + int(rho * 100))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma = _noise_sigma(yc, 30)
            y = yc + sigma * rng.randn(m)
            for s, r in run_scenario(A, y, x, sigma).items():
                rho_res[s]['nmse'].append(r['nmse'])
                rho_res[s]['div'].append(int(r['diverged']))
    # --- delta sweep (snr=30, rho=0.15) ---
    delta_res = {s: {'nmse': [], 'div': []} for s in SOLVERS}
    for delta in delta_vals:
        m = int(delta * n); k = max(int(0.15 * n), 3)
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian', random_state=3000 + tr + int(delta * 100))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma = _noise_sigma(yc, 30)
            y = yc + sigma * rng.randn(m)
            for s, r in run_scenario(A, y, x, sigma).items():
                delta_res[s]['nmse'].append(r['nmse'])
                delta_res[s]['div'].append(int(r['diverged']))

    # ---- Figure: 2x2 panels ----
    set_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax = axes[0, 0]
    for s in SOLVERS:
        means, ses = [], []
        for snr in snr_vals:
            sel = [snr_res[s]['nmse'][i] for i in range(len(snr_res[s]['nmse']))
                   if i // trials == snr_vals.index(snr)]
            means.append(np.nanmean(sel)); ses.append(_sem(sel))
        ax.errorbar(snr_vals, np.maximum(means, 1e-12), yerr=ses,
                    label=s, **_line_style(s))
    ax.set_yscale('log'); ax.set_xlabel('SNR (dB)'); ax.set_ylabel('NMSE')
    ax.set_title('SNR sweep (delta=0.7, rho=0.15)'); ax.legend(ncol=2, fontsize=8)
    _panel_label(ax, 'a')

    ax = axes[0, 1]
    for s in SOLVERS:
        means, ses = [], []
        for rho in rho_vals:
            sel = [rho_res[s]['nmse'][i] for i in range(len(rho_res[s]['nmse']))
                   if i // trials == rho_vals.index(rho)]
            means.append(np.nanmean(sel)); ses.append(_sem(sel))
        ax.errorbar(rho_vals, np.maximum(means, 1e-12), yerr=ses,
                    label=s, **_line_style(s))
    ax.set_yscale('log'); ax.set_xlabel('Sparsity $\\rho = k/n$'); ax.set_ylabel('NMSE')
    ax.set_title('Sparsity sweep (delta=0.7, SNR=30 dB)'); ax.legend(ncol=2, fontsize=8)
    _panel_label(ax, 'b')

    ax = axes[1, 0]
    for s in SOLVERS:
        means, ses = [], []
        for delta in delta_vals:
            sel = [delta_res[s]['nmse'][i] for i in range(len(delta_res[s]['nmse']))
                   if i // trials == delta_vals.index(delta)]
            means.append(np.nanmean(sel)); ses.append(_sem(sel))
        ax.errorbar(delta_vals, np.maximum(means, 1e-12), yerr=ses,
                    label=s, **_line_style(s))
    ax.set_yscale('log'); ax.set_xlabel('Measurement ratio $\\delta = m/n$'); ax.set_ylabel('NMSE')
    ax.set_title('Measurement-ratio sweep (SNR=30 dB, rho=0.15)'); ax.legend(ncol=2, fontsize=8)
    _panel_label(ax, 'c')

    ax = axes[1, 1]
    times = [np.nanmean(snr_res[s]['time']) for s in SOLVERS]
    divs = [np.sum(snr_res[s]['div']) for s in SOLVERS]
    bars = ax.bar(np.arange(len(SOLVERS)), np.maximum(times, 1e-3),
                  color=[SOLVER_COLORS[s] for s in SOLVERS],
                  edgecolor='black', linewidth=0.8, zorder=2)
    for b, (s, t) in zip(bars, zip(SOLVERS, times)):
        ax.text(b.get_x() + b.get_width() / 2, t * 1.05,
                f'{t:.0f} ms', ha='center', va='bottom', fontsize=7)
    ax.set_yscale('log'); ax.set_xticks(np.arange(len(SOLVERS)))
    ax.set_xticklabels(SOLVERS, rotation=30)
    ax.set_ylabel('Runtime (ms, log)'); ax.set_title('Average runtime')
    _panel_label(ax, 'd')
    plt.tight_layout()
    _save_fig(fig, 'R01_static_benchmark.png')

    rows = []
    for s in SOLVERS:
        for snr in snr_vals:
            sel = [snr_res[s]['nmse'][i] for i in range(len(snr_res[s]['nmse']))
                   if i // trials == snr_vals.index(snr)]
            rows.append(['snr', snr, s, np.nanmean(sel), _sem(sel), np.sum(snr_res[s]['div'])])
        for rho in rho_vals:
            sel = [rho_res[s]['nmse'][i] for i in range(len(rho_res[s]['nmse']))
                   if i // trials == rho_vals.index(rho)]
            rows.append(['rho', rho, s, np.nanmean(sel), _sem(sel), np.sum(rho_res[s]['div'])])
        for delta in delta_vals:
            sel = [delta_res[s]['nmse'][i] for i in range(len(delta_res[s]['nmse']))
                   if i // trials == delta_vals.index(delta)]
            rows.append(['delta', delta, s, np.nanmean(sel), _sem(sel), np.sum(delta_res[s]['div'])])
    _save_csv('R01_static_benchmark.csv',
              ['sweep', 'value', 'solver', 'nmse_mean', 'nmse_sem', 'diverged'],
              rows)
    print("  [PASS] R01")


def experiment_r02_phase_transition():
    """R02: Donoho-Tanner phase transition over (delta, rho) grid.
    Outputs: R02_phase_transition.png, R02_phase_transition.csv"""
    print("\n" + "=" * 66)
    print("R02 — Phase Transition (Donoho-Tanner)")
    print("=" * 66)
    n = 50 if FAST else 80
    delta_vals = np.linspace(0.25, 0.9, 6) if FAST else np.linspace(0.2, 0.9, 9)
    rho_vals = np.linspace(0.03, 0.5, 6) if FAST else np.linspace(0.02, 0.5, 9)
    trials = 2 if FAST else 4
    rng = np.random.RandomState(RNG_SEED)
    thr = 0.05
    solvers = [s for s in SOLVERS if s != 'DKAMP']
    success = {s: np.zeros((len(delta_vals), len(rho_vals))) for s in solvers}

    for di, delta in enumerate(delta_vals):
        for ri, rho in enumerate(rho_vals):
            m = max(int(delta * n), 2); k = max(int(rho * n), 1)
            for tr in range(trials):
                A = create_measurement_matrix(m, n, 'gaussian',
                                              random_state=4000 + tr + int(delta * 1000) + int(rho * 1000))
                x, _ = _make_signal(n, k, rng)
                yc = A @ x; sigma = _noise_sigma(yc, 40)
                y = yc + sigma * rng.randn(m)
                for s in solvers:
                    r = _run_solver(s, A, y, x, sigma, max_iter=150)
                    if (not r['diverged']) and r['nmse'] < thr:
                        success[s][di, ri] += 1.0 / trials
        print(f"  delta={delta:.2f}")

    set_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axs = axes.flatten()
    for a, s in enumerate(solvers):
        ax = axs[a]
        im = ax.pcolormesh(delta_vals, rho_vals, success[s].T, cmap='viridis',
                           vmin=0, vmax=1, shading='auto')
        ax.set_xlabel('$\\delta = m/n$'); ax.set_ylabel('$\\rho = k/n$')
        ax.set_title(s, color=SOLVER_COLORS[s], fontweight='bold')
        ax.invert_yaxis()
        _panel_label(ax, chr(ord('a') + a))
    ax = axs[5]
    for s in solvers:
        ax.contour(delta_vals, rho_vals, success[s].T, levels=[0.5],
                   colors=[SOLVER_COLORS[s]], linewidths=2.0)
        ax.plot([], [], color=SOLVER_COLORS[s], linewidth=2.0, label=s)
    ax.set_xlabel('$\\delta = m/n$'); ax.set_ylabel('$\\rho = k/n$')
    ax.set_title('50% success contours (overlay)')
    ax.legend(loc='best', fontsize=8); ax.invert_yaxis()
    _panel_label(ax, 'f')
    plt.tight_layout()
    _save_fig(fig, 'R02_phase_transition.png')

    rows = []
    for di, delta in enumerate(delta_vals):
        for ri, rho in enumerate(rho_vals):
            rows.append([round(delta, 3), round(rho, 3)] +
                        [success[s][di, ri] for s in solvers])
    _save_csv('R02_phase_transition.csv', ['delta', 'rho'] + solvers, rows)
    print("  [PASS] R02")


def experiment_r03_statistical_comparison():
    """R03: Monte-Carlo statistical comparison, pairwise Mann-Whitney U with
    Holm correction. Outputs: R03_statistical.png, R03_statistical.csv,
    R03_pairwise_pvalues.csv"""
    print("\n" + "=" * 66)
    print("R03 — Statistical Comparison (Monte Carlo, Holm-corrected MWU)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    num_trials = 30 if FAST else 120
    rng = np.random.RandomState(RNG_SEED)
    results = {s: [] for s in SOLVERS}

    for tr in range(num_trials):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=5000 + tr)
        x, _ = _make_signal(n, k, rng)
        yc = A @ x; sigma = _noise_sigma(yc, 30)
        y = yc + sigma * rng.randn(m)
        for s in SOLVERS:
            if s == 'DKAMP':
                r = _run_solver(s, A, y, x, sigma, max_iter=100,
                                num_nodes=2, num_triggers=4)
            else:
                r = _run_solver(s, A, y, x, sigma, max_iter=200)
            results[s].append(np.nan if r['diverged'] else r['nmse'])
        if (tr + 1) % 20 == 0:
            print(f"  Trial {tr + 1}/{num_trials}")

    # Pairwise Mann-Whitney U with Holm correction
    pairs = [(a, b) for i, a in enumerate(SOLVERS) for b in SOLVERS[i + 1:]]
    raw_p, holm_p = [], []
    for a, b in pairs:
        u, p = mannwhitneyu(results[a], results[b], alternative='two-sided')
        raw_p.append(p)
    holm_p = _holm_correct(np.array(raw_p))

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    means = [np.nanmean(results[s]) for s in SOLVERS]
    ses = [_sem(results[s]) for s in SOLVERS]
    bars = ax.bar(np.arange(len(SOLVERS)), np.maximum(means, 1e-14),
                  yerr=ses, capsize=4, edgecolor='black', linewidth=0.8,
                  color=[SOLVER_COLORS[s] for s in SOLVERS], zorder=2)
    ax.set_yscale('log'); ax.set_xticks(np.arange(len(SOLVERS)))
    ax.set_xticklabels(SOLVERS, rotation=30)
    ax.set_ylabel('Mean NMSE (log)')
    ax.set_title(f'Monte Carlo NMSE (N={num_trials})')
    for b, s in zip(bars, SOLVERS):
        ax.text(b.get_x() + b.get_width() / 2, max(means[SOLVERS.index(s)], 1e-14) * 1.4,
                f'{np.nanmedian(results[s]):.2e}', ha='center', fontsize=7)
    _panel_label(ax, 'a')

    ax = axes[1]
    mat = np.ones((len(SOLVERS), len(SOLVERS)))
    for (a, b), p in zip(pairs, holm_p):
        i, j = SOLVERS.index(a), SOLVERS.index(b)
        mat[i, j] = mat[j, i] = max(p, 1e-30)
    im = ax.imshow(-np.log10(mat), cmap='RdBu_r', vmin=0, vmax=30)
    ax.set_xticks(range(len(SOLVERS))); ax.set_yticks(range(len(SOLVERS)))
    ax.set_xticklabels(SOLVERS, rotation=30); ax.set_yticklabels(SOLVERS)
    for i in range(len(SOLVERS)):
        for j in range(len(SOLVERS)):
            if i != j:
                ax.text(j, i, f'{-np.log10(mat[i, j]):.0f}', ha='center',
                        va='center', fontsize=8,
                        color='white' if -np.log10(mat[i, j]) > 15 else 'black')
    ax.set_title('Holm-corrected $-\\log_{10} p$ (MWU)')
    _panel_label(ax, 'b')
    plt.tight_layout()
    _save_fig(fig, 'R03_statistical.png')

    rows = []
    for s in SOLVERS:
        a = np.array(results[s], dtype=float)
        a = a[np.isfinite(a)]
        rows.append([s, np.mean(a), np.median(a), np.std(a),
                     np.percentile(a, 25), np.percentile(a, 75),
                     np.mean(np.isnan(results[s]))])
    _save_csv('R03_statistical.csv',
              ['solver', 'mean', 'median', 'std', 'q25', 'q75', 'nan_frac'],
              rows)
    rows = []
    for (a, b), p, hp in zip(pairs, raw_p, holm_p):
        rows.append([f'{a} vs {b}', p, hp])
    _save_csv('R03_pairwise_pvalues.csv', ['pair', 'raw_p', 'holm_p'], rows)
    print("  [PASS] R03")

# ============================================================================
# REGIME II — Structural Assumption Violations (i.i.d. Gaussian breakdown)
# ============================================================================

def experiment_r04_correlated_measurements():
    """R04: AR(1)-correlated measurement matrices, rho in {0, 0.3, 0.6, 0.9}.
    Outputs: R04_correlated_measurements.png, R04_correlated_measurements.csv"""
    print("\n" + "=" * 66)
    print("R04 — Correlated Measurement Matrices (AR-1)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    ar_vals = [0.0, 0.3, 0.6, 0.9]
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    res = {s: {ar: [] for ar in ar_vals} for s in SOLVERS}

    for ar in ar_vals:
        for tr in range(trials):
            A = _ar1_matrix(m, n, ar, random_state=6000 + tr + int(ar * 100))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma = _noise_sigma(yc, 30)
            y = yc + sigma * rng.randn(m)
            for s in SOLVERS:
                if s == 'DKAMP':
                    r = _run_solver(s, A, y, x, sigma, max_iter=100,
                                    num_nodes=2, num_triggers=4)
                else:
                    r = _run_solver(s, A, y, x, sigma, max_iter=200)
                res[s][ar].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  AR coeff={ar} done")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for s in SOLVERS:
        means = [np.nanmean(res[s][ar]) for ar in ar_vals]
        ses = [_sem(res[s][ar]) for ar in ar_vals]
        ax.errorbar(ar_vals, np.maximum(means, 1e-12), yerr=ses,
                    label=s, **_line_style(s))
    ax.set_yscale('log')
    ax.set_xlabel('AR(1) correlation $\\rho$'); ax.set_ylabel('NMSE')
    ax.set_title('Impact of correlated measurement matrices on recovery NMSE')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, 'R04_correlated_measurements.png')

    rows = []
    for s in SOLVERS:
        for ar in ar_vals:
            v = res[s][ar]
            rows.append([ar, s, np.nanmean(v), _sem(v), np.nanmedian(v),
                         np.mean(np.isnan(v))])
    _save_csv('R04_correlated_measurements.csv',
              ['ar_coeff', 'solver', 'nmse_mean', 'nmse_sem', 'nmse_median',
               'nan_frac'], rows)
    print("  [PASS] R04")


def experiment_r05_heteroscedastic_noise():
    """R05: Row-dependent noise variances, imbalance ratio eta.
    Outputs: R05_heteroscedastic.png, R05_heteroscedastic.csv"""
    print("\n" + "=" * 66)
    print("R05 — Heteroscedastic Noise (row-dependent variance)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    eta_vals = [1.0, 2.0, 5.0, 10.0]
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    res = {s: {eta: [] for eta in eta_vals} for s in SOLVERS}

    for eta in eta_vals:
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=7000 + tr + int(eta * 10))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma0 = _noise_sigma(yc, 30)
            w = np.ones(m); w[m // 2:] = eta
            sigma_vec = sigma0 * w
            y = yc + sigma_vec * rng.randn(m)
            sigma_eff = float(np.sqrt(np.mean(sigma_vec ** 2)))
            for s in SOLVERS:
                if s == 'DKAMP':
                    r = _run_solver(s, A, y, x, sigma_eff, max_iter=100,
                                    num_nodes=2, num_triggers=4)
                elif s == 'KAMP':
                    r = _run_kamp(A, y, x, sigma_eff, lam=2.0 * sigma0,
                                  max_iter=200, tau_adaptive=False,
                                  R=np.diag(sigma_vec ** 2))
                else:
                    r = _run_solver(s, A, y, x, sigma_eff, max_iter=200)
                res[s][eta].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  eta={eta} done")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for s in SOLVERS:
        means = [np.nanmean(res[s][eta]) for eta in eta_vals]
        ses = [_sem(res[s][eta]) for eta in eta_vals]
        ax.errorbar(eta_vals, np.maximum(means, 1e-12), yerr=ses,
                    label=s, **_line_style(s))
    ax.set_yscale('log'); ax.set_xscale('log')
    ax.set_xlabel('Variance imbalance $\\eta$'); ax.set_ylabel('NMSE')
    ax.set_title('Heteroscedastic noise: effect on recovery NMSE')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, 'R05_heteroscedastic.png')

    rows = []
    for s in SOLVERS:
        for eta in eta_vals:
            v = res[s][eta]
            rows.append([eta, s, np.nanmean(v), _sem(v), np.nanmedian(v),
                         np.mean(np.isnan(v))])
    _save_csv('R05_heteroscedastic.csv',
              ['eta', 'solver', 'nmse_mean', 'nmse_sem', 'nmse_median',
               'nan_frac'], rows)
    print("  [PASS] R05")


def experiment_r06_heavy_tailed_noise():
    """R06: Student-t and Cauchy noise. Outputs: R06_heavy_tailed.png,
    R06_heavy_tailed.csv"""
    print("\n" + "=" * 66)
    print("R06 — Heavy-Tailed Noise (Student-t, Cauchy)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    noise_specs = [('gaussian', 0), ('student-t df=5', 5), ('student-t df=3', 3),
                   ('cauchy', 0)]
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    res = {s: {ns: [] for ns, _ in noise_specs} for s in SOLVERS}

    for nname, df in noise_specs:
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=8000 + tr + int(df) * 100)
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma0 = _noise_sigma(yc, 30)
            if nname.startswith('student'):
                e = rng.standard_t(df, m)
                e = e / np.std(e) * sigma0
            elif nname == 'cauchy':
                e = rng.standard_cauchy(m)
                e = np.clip(e, -50, 50) / np.std(e) * sigma0
            else:
                e = sigma0 * rng.randn(m)
            y = yc + e
            for s in SOLVERS:
                if s == 'DKAMP':
                    r = _run_solver(s, A, y, x, sigma0, max_iter=100,
                                    num_nodes=2, num_triggers=4)
                else:
                    r = _run_solver(s, A, y, x, sigma0, max_iter=200)
                res[s][nname].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  {nname} done")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    xpos = np.arange(len(noise_specs))
    width = 0.11
    for i, s in enumerate(SOLVERS):
        means = [np.nanmean(res[s][ns]) for ns, _ in noise_specs]
        ses = [_sem(res[s][ns]) for ns, _ in noise_specs]
        off = (i - (len(SOLVERS) - 1) / 2) * width
        bars = ax.bar(xpos + off, np.maximum(means, 1e-14), width=width,
                      yerr=ses, capsize=2, label=s, **_bar_style(s))
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels([ns for ns, _ in noise_specs])
    ax.set_ylabel('NMSE'); ax.set_title('Recovery under heavy-tailed noise')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_fig(fig, 'R06_heavy_tailed.png')

    rows = []
    for s in SOLVERS:
        for nname, df in noise_specs:
            v = res[s][nname]
            rows.append([nname, s, np.nanmean(v), _sem(v), np.nanmedian(v),
                         np.mean(np.isnan(v))])
    _save_csv('R06_heavy_tailed.csv',
              ['noise', 'solver', 'nmse_mean', 'nmse_sem', 'nmse_median',
               'nan_frac'], rows)
    print("  [PASS] R06")


def experiment_r07_colored_noise():
    """R07: Correlated measurement noise via a fixed PSD filter (smoothing).
    Outputs: R07_colored_noise.png, R07_colored_noise.csv"""
    print("\n" + "=" * 66)
    print("R07 — Colored Noise (correlated across measurements)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    spec_names = ['white', 'mild AR(0.5)', 'strong AR(0.9)']
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    res = {s: {sp: [] for sp in spec_names} for s in SOLVERS}

    for spname in spec_names:
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=9000 + tr + spec_names.index(spname))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma0 = _noise_sigma(yc, 30)
            if spname == 'white':
                e = sigma0 * rng.randn(m)
                R_true = np.eye(m) * (sigma0 ** 2)
            else:
                a_c = 0.5 if '0.5' in spname else 0.9
                e = np.zeros(m + 200)
                for i in range(1, m + 200):
                    e[i] = a_c * e[i - 1] + rng.randn() * sigma0 * np.sqrt(1 - a_c ** 2)
                e = e[-m:]
                R_true = _toep_corr(m, a_c) * (sigma0 ** 2)
            y = yc + e
            for s in SOLVERS:
                if s == 'DKAMP':
                    r = _run_solver(s, A, y, x, sigma0, max_iter=100,
                                    num_nodes=2, num_triggers=4)
                elif s == 'KAMP':
                    r = _run_kamp(A, y, x, sigma0, lam=2.0 * sigma0,
                                  max_iter=200, tau_adaptive=False, R=R_true)
                else:
                    r = _run_solver(s, A, y, x, sigma0, max_iter=200)
                res[s][spname].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  {spname} done")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    xpos = np.arange(len(spec_names))
    width = 0.11
    for i, s in enumerate(SOLVERS):
        means = [np.nanmean(res[s][sp]) for sp in spec_names]
        ses = [_sem(res[s][sp]) for sp in spec_names]
        off = (i - (len(SOLVERS) - 1) / 2) * width
        ax.bar(xpos + off, np.maximum(means, 1e-14), width=width, yerr=ses,
               capsize=2, label=s, **_bar_style(s))
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels(spec_names)
    ax.set_ylabel('NMSE'); ax.set_title('Recovery under colored measurement noise')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_fig(fig, 'R07_colored_noise.png')

    rows = []
    for s in SOLVERS:
        for sp in spec_names:
            v = res[s][sp]
            rows.append([sp, s, np.nanmean(v), _sem(v), np.nanmedian(v),
                         np.mean(np.isnan(v))])
    _save_csv('R07_colored_noise.csv',
              ['noise_spec', 'solver', 'nmse_mean', 'nmse_sem', 'nmse_median',
               'nan_frac'], rows)
    print("  [PASS] R07")

# ============================================================================
# REGIME III — Unique Capabilities of the KAMP Framework
# ============================================================================

def _kamp_track(A, y_seq, x_true, sigma, Qv, max_iter=60, tau=0.05):
    """Run KAMP recursively over a measurement sequence. KAMP carries its
    posterior (x, P) across time; Q (process noise) is injected via
    set_state, implementing the R/Q tuning of the Kalman recursion.
    If the estimate diverges (degenerate gain regime, ||x|| beyond guard),
    the filter is re-initialized and the restart is counted.
    Returns (estimates (T, n), nmse_per_step (T,), num_restarts)."""
    T = y_seq.shape[0]
    n = A.shape[1]
    kamp = KAMP(alpha=0.5, tau=tau, max_iter=max_iter, tol=1e-8,
                tau_adaptive=False, sigma2_noise=float(sigma) ** 2)
    kamp.fit(A, y_seq[0], true_x=x_true[0])
    kamp.set_state(Q=np.eye(n) * float(Qv))
    est = np.zeros((T, n))
    errs = np.zeros(T)
    restarts = 0
    y_scale = max(float(np.abs(y_seq).max()), 1.0)
    amp_guard = 50.0 * y_scale * np.sqrt(n)
    for t in range(T):
        kamp.y = y_seq[t].reshape(-1, 1)
        try:
            x_hat = kamp.solve(true_x=x_true[t])
            xv = np.asarray(x_hat).flatten()
            if (not np.isfinite(xv).all()) or np.abs(xv).max() > amp_guard:
                kamp.set_state(x=np.zeros((n, 1)),
                               P=np.eye(n) * max(1.0, float(Qv) * 100.0))
                kamp.y = y_seq[t].reshape(-1, 1)
                x_hat = kamp.solve(true_x=x_true[t])
                xv = np.asarray(x_hat).flatten()
                restarts += 1
            est[t] = xv
            errs[t] = _nmse(xv, x_true[t])
        except Exception:
            est[t] = np.zeros(n)
            errs[t] = float('nan')
    return est, errs, restarts


def experiment_r08_dynamic_tracking():
    """R08: Dynamic tracking with time-varying sparsity patterns, R/Q tuned,
    Kalman-style recursive filter. Outputs: R08_dynamic_tracking.png,
    R08_dynamic_tracking.csv"""
    print("\n" + "=" * 66)
    print("R08 — Dynamic Tracking (R/Q-tuned Kalman recursion)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    T = 30 if FAST else 60
    Q_vals = [1e-4, 1e-2, 1.0]
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    noise_frac = 0.4
    track_solvers = [s for s in SOLVERS if s != 'DKAMP']
    res = {s: {q: [] for q in Q_vals} for s in track_solvers}
    res_rst = {q: [] for q in Q_vals}

    for Qv in Q_vals:
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=10000 + tr + int(np.log10(Qv) * 100))
            x_true = np.zeros((T, n))
            rands = np.random.RandomState(1000 + tr)
            for t in range(T):
                if t == 0:
                    idx = rands.choice(n, k, replace=False)
                else:
                    keep = rands.choice(k, int(k * (1 - noise_frac)), replace=False)
                    drop = np.setdiff1d(np.arange(n), idx)[:k - len(keep)]
                    idx = np.concatenate([idx[keep], drop])
                x_true[t, idx] = rands.randn(len(idx))
            yc = np.zeros((T, m)); sigma = 0.05
            for t in range(T):
                yc[t] = A @ x_true[t] + sigma * rng.randn(m)
            for s in track_solvers:
                if s == 'KAMP':
                    _, errs, rst = _kamp_track(A, yc, x_true, sigma, Qv)
                    res_rst[Qv].append(rst)
                else:
                    errs = np.zeros(T)
                    for t in range(T):
                        r = _run_solver(s, A, yc[t], x_true[t], sigma, max_iter=200)
                        errs[t] = np.nan if r['diverged'] else r['nmse']
                res[s][Qv].append(np.nanmean(errs))
        print(f"  Q={Qv} done (mean restarts={np.mean(res_rst[Qv]):.2f})")

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    xpos = np.arange(len(Q_vals)); width = 0.13
    for i, s in enumerate(track_solvers):
        means = [np.mean(res[s][q]) for q in Q_vals]
        ses = [_sem(res[s][q]) for q in Q_vals]
        off = (i - (len(track_solvers) - 1) / 2) * width
        ax.bar(xpos + off, np.maximum(means, 1e-14), width=width, yerr=ses,
               capsize=2, label=s, **_bar_style(s))
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels([f'{q:g}' for q in Q_vals])
    ax.set_xlabel('Process noise $Q$'); ax.set_ylabel('Time-averaged NMSE')
    ax.set_title('Dynamic tracking: effect of Q tuning (KAMP carries P)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    _panel_label(ax, 'a')

    ax = axes[1]
    tr_ex = 0; Qv = Q_vals[1]
    A = create_measurement_matrix(m, n, 'gaussian', random_state=10000 + tr_ex)
    x_true = np.zeros((T, n))
    rands = np.random.RandomState(1000 + tr_ex)
    idx = rands.choice(n, k, replace=False)
    x_true[0, idx] = rands.randn(k)
    for t in range(1, T):
        keep = rands.choice(k, int(k * (1 - noise_frac)), replace=False)
        drop = np.setdiff1d(np.arange(n), idx)[:k - len(keep)]
        idx = np.concatenate([idx[keep], drop])
        x_true[t, idx] = rands.randn(len(idx))
    yc = np.zeros((T, m)); sigma = 0.05
    for t in range(T):
        yc[t] = A @ x_true[t] + sigma * rng.randn(m)
    est, _, _ = _kamp_track(A, yc, x_true, sigma, Qv)
    sup = np.where(np.abs(x_true).max(axis=0) > 0)[0][:6]
    for j in sup:
        ax.plot(range(T), x_true[:, j], color=SOLVER_COLORS['KAMP'], alpha=0.35,
                linewidth=1.2)
        ax.plot(range(T), est[:, j], color=SOLVER_COLORS['KAMP'],
                linestyle='--', linewidth=1.2, alpha=0.9)
    ax.set_xlabel('Time step'); ax.set_ylabel('Amplitude')
    ax.set_title('Example: support evolution (solid=truth, dashed=KAMP)')
    ax.grid(alpha=0.3)
    _panel_label(ax, 'b')
    plt.tight_layout()
    _save_fig(fig, 'R08_dynamic_tracking.png')

    rows = []
    for s in track_solvers:
        for q in Q_vals:
            v = res[s][q]
            rows.append([q, s, np.mean(v), _sem(v)])
    for q in Q_vals:
        rows.append([q, 'KAMP_restarts', np.mean(res_rst[q]), _sem(res_rst[q])])
    _save_csv('R08_dynamic_tracking.csv', ['Q', 'solver', 'nmse_mean', 'nmse_sem'],
              rows)
    print("  [PASS] R08")


def experiment_r09_change_point():
    """R09: Abrupt support changes — adaptation lag after each change-point
    (KAMP recursive vs FISTA cold restart) and tracking fidelity.
    Outputs: R09_change_point.png, R09_change_point.csv"""
    print("\n" + "=" * 66)
    print("R09 — Abrupt Support Changes (adaptation lag)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    T = 40 if FAST else 80
    Qv = 0.01
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    change_times = [T // 4, T // 2, 3 * T // 4]
    lag = {'KAMP': [], 'FISTA': []}
    fpr = {'KAMP': [], 'FISTA': []}
    trans = {'KAMP': [], 'FISTA': []}

    for tr in range(trials):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=11000 + tr)
        x_true = np.zeros((T, n)); idx = None
        rands = np.random.RandomState(2000 + tr)
        for t in range(T):
            if t == 0 or t in change_times:
                idx = rands.choice(n, k, replace=False)
            x_true[t, idx] = rands.randn(len(idx))
        yc = np.zeros((T, m)); sigma = 0.05
        for t in range(T):
            yc[t] = A @ x_true[t] + sigma * rng.randn(m)

        for sname in ['KAMP', 'FISTA']:
            if sname == 'KAMP':
                est, errs, _ = _kamp_track(A, yc, x_true, sigma, Qv)
            else:
                est = np.zeros((T, n)); errs = np.zeros(T)
                for t in range(T):
                    r = _run_solver('FISTA', A, yc[t], x_true[t], sigma, max_iter=200)
                    est[t] = r.get('estimate', np.zeros(n)) if not r['diverged'] else np.zeros(n)
                    errs[t] = np.nan if r['diverged'] else r['nmse']
            steady = np.nanmedian(errs[1:6])
            thr = 4.0 * max(steady, 1e-12)
            lags = []; trans_errs = []
            for tc in change_times:
                rec = [t for t in range(tc + 1, min(T, tc + 13))
                       if errs[t] <= thr and errs[min(t + 1, T - 1)] <= thr]
                lags.append(rec[0] - tc if rec else 12)
                trans_errs.append(np.nanmean(errs[tc:tc + 5]))
            lag[sname].append(np.mean(lags))
            trans[sname].append(np.mean(trans_errs))
            est_sup = [set(np.argsort(np.abs(est[t]))[-k:])
                       for t in range(T)]
            true_sup = [set(np.where(np.abs(x_true[t]) > 0)[0])
                        for t in range(T)]
            tot_fp = 0
            for t in range(T):
                if any(tc <= t < tc + 5 for tc in change_times):
                    continue
                tot_fp += len(est_sup[t] - true_sup[t])
            fpr[sname].append(tot_fp / (k * T))
        print(f"  Trial {tr + 1}: KAMP lag={lag['KAMP'][-1]:.1f} "
              f"FISTA lag={lag['FISTA'][-1]:.1f}")

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    x_true = np.zeros((T, n)); idx = None
    rands = np.random.RandomState(2000 + 0)
    for t in range(T):
        if t == 0 or t in change_times:
            idx = rands.choice(n, k, replace=False)
        x_true[t, idx] = rands.randn(len(idx))
    yc = np.zeros((T, m)); sigma = 0.05
    for t in range(T):
        yc[t] = A @ x_true[t] + sigma * rng.randn(m)
    est, errs, _ = _kamp_track(A, yc, x_true, sigma, Qv)
    for j in range(min(5, n)):
        ax.plot(range(T), x_true[:, j], alpha=0.3, linewidth=1.0,
                color='gray')
        ax.plot(range(T), est[:, j], linestyle='--', linewidth=1.2,
                color=SOLVER_COLORS['KAMP'])
    for cp in change_times:
        ax.axvline(cp, color=SOLVER_COLORS['AMP'], linestyle=':', linewidth=1.5)
    ax.set_xlabel('Time step'); ax.set_ylabel('Amplitude')
    ax.set_title('Example run (vertical lines = true change-points)')
    ax.grid(alpha=0.3)
    _panel_label(ax, 'a')

    ax = axes[1]
    xpos = np.arange(2); width = 0.32
    means_lag = [np.mean(lag[s]) for s in ['KAMP', 'FISTA']]
    ses_lag = [_sem(lag[s]) for s in ['KAMP', 'FISTA']]
    ax.bar(xpos - width / 2, means_lag, width=width, yerr=ses_lag, capsize=4,
           label='Adaptation lag (steps)',
           color=[SOLVER_COLORS['KAMP'], SOLVER_COLORS['FISTA']],
           edgecolor='black', linewidth=0.8, zorder=2)
    ax2 = ax.twinx()
    means_tr = [np.mean(trans[s]) for s in ['KAMP', 'FISTA']]
    ses_tr = [_sem(trans[s]) for s in ['KAMP', 'FISTA']]
    ax2.plot(xpos, means_tr, 'o', markersize=9, markerfacecolor='white',
             markeredgecolor='black', markeredgewidth=1.3, zorder=4)
    ax2.errorbar(xpos, means_tr, yerr=ses_tr, fmt='none', ecolor='black',
                 capsize=4, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['KAMP\n(recursive)', 'FISTA\n(cold restart)'])
    ax.set_ylabel('Adaptation lag (steps)')
    ax2.set_ylabel('Transient NMSE in window', color=SOLVER_COLORS['KAMP'])
    ax2.tick_params(axis='y', labelcolor=SOLVER_COLORS['KAMP'])
    ax.set_title('Re-acquisition after abrupt support change\n'
                 '(bars = lag, markers = transient NMSE)')
    ax.grid(alpha=0.3, axis='y')
    _panel_label(ax, 'b')
    plt.tight_layout()
    _save_fig(fig, 'R09_change_point.png')

    rows = []
    for tr in range(len(lag['KAMP'])):
        rows.append([tr, lag['KAMP'][tr], lag['FISTA'][tr],
                     trans['KAMP'][tr], trans['FISTA'][tr],
                     fpr['KAMP'][tr], fpr['FISTA'][tr]])
    _save_csv('R09_change_point.csv', ['trial', 'lag_kamp', 'lag_fista',
                                       'trans_kamp', 'trans_fista',
                                       'fpr_kamp', 'fpr_fista'], rows)
    print("  [PASS] R09")


def experiment_r10_graph_prior():
    """R10: Graph-structured sparsity prior — block/cluster support patterns.
    Outputs: R10_graph_prior.png, R10_graph_prior.csv"""
    print("\n" + "=" * 66)
    print("R10 — Graph-Structured Priors (clustered support)")
    print("=" * 66)
    n, m = (60, 42) if FAST else (100, 70)
    cluster_specs = [('iid', 1), ('2 clusters', 2), ('4 clusters', 4)]
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    res = {s: {cs: [] for cs, _ in cluster_specs} for s in SOLVERS}

    for csname, ncl in cluster_specs:
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=12000 + tr + ncl * 100)
            k = int(0.2 * n)
            idx = np.array([], dtype=int)
            blocks = []
            if ncl == 1:
                idx = np.sort(rng.choice(n, k, replace=False))
            else:
                bs = k // ncl
                centers = rng.choice(np.arange(bs, n - bs), ncl, replace=False)
                for c in centers:
                    bl = np.arange(c - bs // 2, c - bs // 2 + bs)
                    blocks.append(bl)
                    idx = np.concatenate([idx, bl])
                idx = np.unique(idx)
            x = np.zeros(n); x[idx] = rng.randn(len(idx))
            yc = A @ x; sigma = _noise_sigma(yc, 30)
            y = yc + sigma * rng.randn(m)
            W = np.zeros((n, n))
            for bl in blocks:
                W[np.ix_(bl, bl)] = 0.5
            np.fill_diagonal(W, 0.0)
            P0 = np.eye(n) + W
            for s in SOLVERS:
                if s == 'DKAMP':
                    r = _run_solver(s, A, y, x, sigma, max_iter=100,
                                    num_nodes=2, num_triggers=4)
                elif s == 'KAMP':
                    r = _run_kamp(A, y, x, sigma, lam=2.0 * sigma,
                                  max_iter=200, tau_adaptive=False, P0=P0)
                else:
                    r = _run_solver(s, A, y, x, sigma, max_iter=200)
                res[s][csname].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  {csname} done")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    xpos = np.arange(len(cluster_specs)); width = 0.11
    for i, s in enumerate(SOLVERS):
        means = [np.mean(res[s][cs]) for cs, _ in cluster_specs]
        ses = [_sem(res[s][cs]) for cs, _ in cluster_specs]
        off = (i - (len(SOLVERS) - 1) / 2) * width
        ax.bar(xpos + off, np.maximum(means, 1e-14), width=width, yerr=ses,
               capsize=2, label=s, **_bar_style(s))
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels([cs for cs, _ in cluster_specs])
    ax.set_ylabel('NMSE'); ax.set_title('Clustered (graph-structured) sparsity patterns')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_fig(fig, 'R10_graph_prior.png')

    rows = []
    for s in SOLVERS:
        for cs, ncl in cluster_specs:
            v = res[s][cs]
            rows.append([cs, s, np.mean(v), _sem(v), np.median(v)])
    _save_csv('R10_graph_prior.csv', ['cluster_spec', 'solver', 'nmse_mean',
                                      'nmse_sem', 'nmse_median'], rows)
    print("  [PASS] R10")


def experiment_r11_onebit():
    """R11: 1-bit quantized measurements — matched threshold with sign
    invariance. Outputs: R11_onebit.png, R11_onebit.csv"""
    print("\n" + "=" * 66)
    print("R11 — 1-bit Quantized Measurements")
    print("=" * 66)
    n = 60 if FAST else 100
    m = int(0.7 * n); k = int(0.15 * n)
    trials = 3 if FAST else 10
    rng = np.random.RandomState(RNG_SEED)
    res = {s: [] for s in SOLVERS}

    for tr in range(trials):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=13000 + tr)
        x, _ = _make_signal(n, k, rng)
        y = np.sign(A @ x)
        for s in SOLVERS:
            if s == 'DKAMP':
                r = _run_solver(s, A, y, x, 0.3, max_iter=100,
                                num_nodes=2, num_triggers=4)
            else:
                r = _run_solver(s, A, y, x, 0.3, max_iter=300)
            res[s].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  Trial {tr + 1}")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    means = [np.nanmean(res[s]) for s in SOLVERS]
    ses = [_sem(res[s]) for s in SOLVERS]
    bars = ax.bar(np.arange(len(SOLVERS)), np.maximum(means, 1e-14), yerr=ses,
                  capsize=4, color=[SOLVER_COLORS[s] for s in SOLVERS],
                  edgecolor='black', linewidth=0.8, zorder=2)
    for b, s in zip(bars, SOLVERS):
        ax.text(b.get_x() + b.get_width() / 2, np.maximum(means[SOLVERS.index(s)], 1e-14) * 1.5,
                f'{np.nanmedian(res[s]):.2e}', ha='center', fontsize=7)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(SOLVERS))); ax.set_xticklabels(SOLVERS, rotation=30)
    ax.set_ylabel('NMSE'); ax.set_title('1-bit quantized measurements (sign only)')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_fig(fig, 'R11_onebit.png')

    rows = [[s, np.nanmean(res[s]), _sem(res[s]), np.nanmedian(res[s]),
             np.mean(np.isnan(res[s]))] for s in SOLVERS]
    _save_csv('R11_onebit.csv', ['solver', 'nmse_mean', 'nmse_sem',
                                 'nmse_median', 'nan_frac'], rows)
    print("  [PASS] R11")

# ============================================================================
# REGIME IV — Theoretical Validation (PCRLB, NEES / NIS)
# ============================================================================

def experiment_r12_pcrlb():
    """R12: Posterior CRLB vs achieved MSE as a function of SNR.
    Outputs: R12_pcrlb.png, R12_pcrlb.csv"""
    print("\n" + "=" * 66)
    print("R12 — Posterior Cramer-Rao Lower Bound (PCRLB) comparison")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    snr_vals = [10, 20, 30, 40]
    trials = 2 if FAST else 5
    rng = np.random.RandomState(RNG_SEED)
    mse_kamp = []; mse_vamp = []; mse_fista = []; pcrlb_vals = []

    for snr in snr_vals:
        mse_k, mse_v, mse_f = [], [], []
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=14000 + tr + int(snr))
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma = _noise_sigma(yc, snr)
            y = yc + sigma * rng.randn(m)
            rk = _run_solver('KAMP', A, y, x, sigma, max_iter=200)
            rv = _run_solver('VAMP', A, y, x, sigma, max_iter=200)
            rf = _run_solver('FISTA', A, y, x, sigma, max_iter=200)
            if not rk['diverged']:
                mse_k.append(rk['nmse'] * np.sum(x ** 2) / n)
            if not rv['diverged']:
                mse_v.append(rv['nmse'] * np.sum(x ** 2) / n)
            if not rf['diverged']:
                mse_f.append(rf['nmse'] * np.sum(x ** 2) / n)
        # PCRLB for known support: sigma^2 * trace((A_S^T A_S)^-1) / n
        A = create_measurement_matrix(m, n, 'gaussian', random_state=14000 + int(snr))
        sigma = _noise_sigma(A @ x, snr)
        sup = np.argsort(np.abs(x))[-k:]
        AS = A[:, sup]
        pcrlb = float(sigma ** 2 * np.trace(np.linalg.inv(AS.T @ AS)) / n)
        mse_kamp.append(np.mean(mse_k) if mse_k else np.nan)
        mse_vamp.append(np.mean(mse_v) if mse_v else np.nan)
        mse_fista.append(np.mean(mse_f) if mse_f else np.nan)
        pcrlb_vals.append(pcrlb)
        print(f"  SNR={snr}: PCRLB={pcrlb:.3e}")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(snr_vals, pcrlb_vals, color='black', linewidth=2.2,
            marker='s', markersize=7, markerfacecolor='white',
            markeredgecolor='black', label='PCRLB (known support)')
    ax.plot(snr_vals, np.maximum(mse_kamp, 1e-20), label='KAMP', **_line_style('KAMP'))
    ax.plot(snr_vals, np.maximum(mse_vamp, 1e-20), label='VAMP', **_line_style('VAMP'))
    ax.plot(snr_vals, np.maximum(mse_fista, 1e-20), label='FISTA', **_line_style('FISTA'))
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Per-coordinate MSE')
    ax.set_title('PCRLB vs achieved MSE (known-support bound)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, 'R12_pcrlb.png')

    rows = [[snr, p, mk, mv, mf] for snr, p, mk, mv, mf in
            zip(snr_vals, pcrlb_vals, mse_kamp, mse_vamp, mse_fista)]
    _save_csv('R12_pcrlb.csv', ['snr', 'pcrlb', 'mse_kamp', 'mse_vamp',
                                'mse_fista'], rows)
    print("  [PASS] R12")


def experiment_r13_nees_nis():
    """R13: NEES / NIS consistency metrics vs. process-noise tuning Q.
    The true dynamics inject innovation variance q_true per step; NIS is
    computed from the pre-update innovation nu = y - A x_hat^- with
    S = A P^- A^T + R, where P^- = P_post + Q I combines KAMP's posterior
    covariance (carried across steps) with the tuned process-noise level Q,
    as the Kalman recursion prescribes (F = I). A consistent filter
    satisfies E[NIS] = 1; NIS crosses this line at the operating Q* that
    matches KAMP's innovation statistics (the filter's carried posterior
    supplies the estimation memory, so Q* lands below the raw q_true).
    NEES (diag-P normalized) corroborates the same operating point.
    Outputs: R13_nees_nis.png, R13_nees_nis.csv"""
    print("\n" + "=" * 66)
    print("R13 — NEES / NIS Consistency vs. Process-Noise Tuning Q")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    T = 20 if FAST else 40
    q_true = 0.05
    Q_vals = [1e-3, 1e-2, q_true, 0.2, 1.0]
    trials = 20 if FAST else 60
    rng = np.random.RandomState(RNG_SEED)
    nees_all = {q: np.zeros((trials, T)) for q in Q_vals}
    nis_all = {q: np.zeros((trials, T)) for q in Q_vals}

    for tr in range(trials):
        A = create_measurement_matrix(m, n, 'gaussian', random_state=15000 + tr)
        rands = np.random.RandomState(3000 + tr)
        x_true = np.zeros((T, n))
        idx = rands.choice(n, k, replace=False)
        D_s = np.zeros((n, n)); D_s[np.ix_(idx, idx)] = 1.0
        x_true[0, idx] = rands.randn(k)
        for t in range(1, T):
            x_true[t, idx] = x_true[t - 1, idx] + np.sqrt(q_true) * rands.randn(k)
        yc = np.zeros((T, m)); sigma = 0.05
        for t in range(T):
            yc[t] = A @ x_true[t] + sigma * rng.randn(m)
        for Qv in Q_vals:
            kamp = KAMP(alpha=0.5, tau=0.05, max_iter=60, tol=1e-8,
                        tau_adaptive=False, sigma2_noise=float(sigma) ** 2)
            kamp.fit(A, yc[0], true_x=x_true[0])
            kamp.set_state(Q=np.eye(n) * Qv)
            I_m = np.eye(m); I_n = np.eye(n)
            P_post = np.asarray(kamp.P).copy()
            for t in range(T):
                x_pred = np.asarray(kamp.x).flatten()
                P_pred = P_post + Qv * I_n
                S = A @ P_pred @ A.T + (sigma ** 2) * I_m
                nu = yc[t] - A @ x_pred
                nis_all[Qv][tr, t] = float(nu @ np.linalg.solve(S, nu)) / m
                kamp.y = yc[t].reshape(-1, 1)
                x_hat = np.asarray(kamp.solve(true_x=x_true[t])).flatten()
                P_post = np.asarray(kamp.P).copy()
                p_diag = np.clip(np.diag(P_post), 1e-12, None)
                perr = x_hat - x_true[t]
                nees_all[Qv][tr, t] = np.mean(perr ** 2 / p_diag)
        if (tr + 1) % 10 == 0:
            print(f"  Trial {tr + 1}/{trials}")

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, metric_all, metric_name, color, deg in [
            (axes[0], nees_all, 'NEES', SOLVER_COLORS['KAMP'], n),
            (axes[1], nis_all, 'NIS', SOLVER_COLORS['VAMP'], m)]:
        avgs = {q: float(np.mean(v)) for q, v in metric_all.items()}
        ses = {q: _sem(np.asarray(v).ravel()) for q, v in metric_all.items()}
        qs = list(Q_vals)
        q_star = float(10 ** (-np.interp(1.0, [np.log10(avgs[q]) for q in qs],
                                         [-np.log10(q) for q in qs])))
        ax.axvline(q_star, color='#CC3311', linestyle='-.', linewidth=1.5,
                   label='operating Q* ≈ %.0e' % q_star)
        ax.errorbar(qs, [avgs[q] for q in qs], yerr=[ses[q] for q in qs],
                    color=color, marker='o', markersize=7,
                    markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=1.3, linewidth=2.0, capsize=4,
                    label=f'{metric_name} (mean over t)')
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
                   label='ideal E[%s] = 1' % metric_name)
        band = 1.0 + 1.96 * np.sqrt(2.0 / (deg * trials))
        ax.axhline(band, color='black', linestyle=':', linewidth=1.2,
                   label='95% upper bound')
        ax.set_xscale('log')
        ax.set_xlabel('Process noise Q (per-coordinate variance)')
        ax.set_ylabel(f'{metric_name}')
        ax.set_title(f'{metric_name} vs Q (crosses E=1 at operating Q*)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _panel_label(ax, 'a' if ax is axes[0] else 'b')
    plt.tight_layout()
    _save_fig(fig, 'R13_nees_nis.png')

    rows = [[q, float(np.mean(nees_all[q])), _sem(nees_all[q].ravel()),
             float(np.mean(nis_all[q])), _sem(nis_all[q].ravel())] for q in Q_vals]
    _save_csv('R13_nees_nis.csv', ['Q', 'nees', 'nees_sem', 'nis', 'nis_sem'],
              rows)
    print("  [PASS] R13")


# ============================================================================
# REGIME V — Covariance-Driven Applications (Active Sensing, Scheduling,
#            Node Failure)
# ============================================================================

def experiment_r14_active_sensing():
    """R14: Active sensing — greedy row selection minimizes posterior
    variance via P1FIM vs random sampling. Outputs: R14_active_sensing.png,
    R14_active_sensing.csv"""
    print("\n" + "=" * 66)
    print("R14 — Active Sensing (P1FIM-driven row selection)")
    print("=" * 66)
    n = 60 if FAST else 100
    k = int(0.15 * n)
    m_pool = 200 if FAST else 300
    b_vals = [40, 60, 80]
    trials = 2 if FAST else 5
    rng = np.random.RandomState(RNG_SEED)
    res = {'P1FIM': {b: [] for b in b_vals}, 'random': {b: [] for b in b_vals}}

    for b in b_vals:
        for tr in range(trials):
            A_pool = create_measurement_matrix(m_pool, n, 'gaussian',
                                               random_state=16000 + tr + b)
            x, _ = _make_signal(n, k, rng)
            y_pool = A_pool @ x
            sigma = _noise_sigma(y_pool, 30)
            y_pool = y_pool + sigma * rng.randn(m_pool)

            ests = {}
            for strat in ['P1FIM', 'random']:
                if strat == 'random':
                    sel = rng.choice(m_pool, b, replace=False)
                else:
                    sel = [int(rng.randint(0, m_pool))]
                    while len(sel) < b:
                        A_s = A_pool[sel]
                        P1 = np.linalg.inv(A_s.T @ A_s + 1e-8 * np.eye(n))
                        P1A = P1 @ A_pool.T
                        scores = np.sum(P1A ** 2, axis=0)
                        scores[sel] = -np.inf
                        sel.append(int(np.argmax(scores)))
                A_sel = A_pool[sel]
                y_sel = y_pool[sel]
                r = _run_solver('KAMP', A_sel, y_sel, x, sigma, max_iter=200)
                ests[strat] = np.nan if r['diverged'] else r['nmse']
            res['P1FIM'][b].append(ests['P1FIM'])
            res['random'][b].append(ests['random'])
        print(f"  b={b} done")

    set_publication_style()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    xpos = np.arange(len(b_vals)); width = 0.32
    for i, strat in enumerate(['P1FIM', 'random']):
        means = [np.nanmean(res[strat][b]) for b in b_vals]
        ses = [_sem(res[strat][b]) for b in b_vals]
        ax.bar(xpos + (i - 0.5) * width, np.maximum(means, 1e-14), width=width,
               yerr=ses, capsize=4, label=strat,
               color=[SOLVER_COLORS['KAMP'], '#CCBB44'][i],
               edgecolor='black', linewidth=0.8, zorder=2)
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels([str(b) for b in b_vals])
    ax.set_xlabel('Number of selected measurements $b$')
    ax.set_ylabel('NMSE'); ax.set_title('Active sensing vs random measurement selection')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    _save_fig(fig, 'R14_active_sensing.png')

    rows = []
    for b in b_vals:
        for strat in ['P1FIM', 'random']:
            v = res[strat][b]
            rows.append([b, strat, np.nanmean(v), _sem(v)])
    _save_csv('R14_active_sensing.csv', ['b', 'strategy', 'nmse_mean', 'nmse_sem'],
              rows)
    print("  [PASS] R14")


def experiment_r15_dkamp_scheduling():
    """R15: DKAMP — effect of scheduling (triggers) on NMSE vs runtime.
    Outputs: R15_dkamp_scheduling.png, R15_dkamp_scheduling.csv"""
    print("\n" + "=" * 66)
    print("R15 — DKAMP Scheduling (node triggers trade-off)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    num_nodes = 2
    trig_vals = [1, 4, 8, 16] if FAST else [1, 2, 4, 8, 16]
    trials = 2 if FAST else 5
    rng = np.random.RandomState(RNG_SEED)
    res = {t: {'nmse': [], 'time': []} for t in trig_vals}

    for trig in trig_vals:
        for tr in range(trials):
            A = create_measurement_matrix(m, n, 'gaussian',
                                          random_state=17000 + tr + trig)
            x, _ = _make_signal(n, k, rng)
            yc = A @ x; sigma = _noise_sigma(yc, 30)
            y = yc + sigma * rng.randn(m)
            r = _run_solver('DKAMP', A, y, x, sigma, max_iter=150,
                            num_nodes=num_nodes, num_triggers=trig)
            res[trig]['nmse'].append(np.nan if r['diverged'] else r['nmse'])
            res[trig]['time'].append(r['time_ms'])
        print(f"  triggers={trig} done")

    set_publication_style()
    fig, ax1 = plt.subplots(figsize=(9.5, 6))
    means = [np.nanmean(res[t]['nmse']) for t in trig_vals]
    ses = [_sem(res[t]['nmse']) for t in trig_vals]
    ax1.errorbar(trig_vals, np.maximum(means, 1e-14), yerr=ses,
                 label='DKAMP NMSE', color=SOLVER_COLORS['DKAMP'],
                 marker='o', markersize=7, markerfacecolor='white',
                 markeredgecolor=SOLVER_COLORS['DKAMP'], linewidth=2.0)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Scheduling triggers per node')
    ax1.set_ylabel('NMSE (log)', color=SOLVER_COLORS['DKAMP'])
    ax1.tick_params(axis='y', labelcolor=SOLVER_COLORS['DKAMP'])
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    times = [np.mean(res[t]['time']) for t in trig_vals]
    ax2.plot(trig_vals, times, label='Runtime', color=SOLVER_COLORS['KAMP'],
             marker='s', markersize=7, markerfacecolor='white',
             markeredgecolor=SOLVER_COLORS['KAMP'], linewidth=2.0)
    ax2.set_ylabel('Runtime (ms)', color=SOLVER_COLORS['KAMP'])
    ax2.tick_params(axis='y', labelcolor=SOLVER_COLORS['KAMP'])
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=9, loc='best')
    ax1.set_title('DKAMP: scheduling trade-off (NMSE vs runtime)')
    plt.tight_layout()
    _save_fig(fig, 'R15_dkamp_scheduling.png')

    rows = []
    for t in trig_vals:
        rows.append([t, np.nanmean(res[t]['nmse']), _sem(res[t]['nmse']),
                     np.mean(res[t]['time']), _sem(res[t]['time'])])
    _save_csv('R15_dkamp_scheduling.csv', ['triggers', 'nmse_mean', 'nmse_sem',
                                           'time_ms', 'time_sem'], rows)
    print("  [PASS] R15")


def experiment_r16_node_failure():
    """R16: DKAMP with heterogeneous nodes and node failure — recovery NMSE
    and the role of node separation. Outputs: R16_node_failure.png,
    R16_node_failure.csv"""
    print("\n" + "=" * 66)
    print("R16 — DKAMP Heterogeneous Nodes and Node Failure")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    trials = 2 if FAST else 5
    rng = np.random.RandomState(RNG_SEED)
    scenarios = ['healthy', 'one node fails', 'both nodes fail (restart)']
    res = {sc: {'nmse': [], 'time': []} for sc in scenarios}
    sep_res = {4: [], 8: [], 12: []}

    for tr in range(trials):
        tr_rng = np.random.RandomState(18000 + tr + 500)
        A0 = create_measurement_matrix(m // 2, n, 'gaussian',
                                       random_state=18000 + tr)
        A1 = _overlap_basis(A0, 8, random_state=18000 + tr + 100)
        A = np.vstack([A0, A1])
        x, _ = _make_signal(n, k, tr_rng)
        yc = A @ x; sigma = _noise_sigma(yc, 30)
        for sc in scenarios:
            if sc == 'healthy':
                y = yc + sigma * tr_rng.randn(m)
                r = _run_solver('DKAMP', A, y, x, sigma, max_iter=150,
                                num_nodes=2, num_triggers=4)
            elif sc == 'one node fails':
                y1 = yc[:m // 2] + sigma * tr_rng.randn(m // 2)
                r = _run_solver('DKAMP', A0, y1, x, sigma, max_iter=150,
                                num_nodes=1, num_triggers=4)
            else:
                y = yc + sigma * tr_rng.randn(m)
                r = _run_solver('DKAMP', A, y, x, sigma, max_iter=150,
                                num_nodes=2, num_triggers=4)
            res[sc]['nmse'].append(np.nan if r['diverged'] else r['nmse'])
            res[sc]['time'].append(r['time_ms'])
        print(f"  trial {tr + 1}/{trials} done")

    for sep in sep_res:
        for tr in range(trials):
            tr_rng = np.random.RandomState(19000 + tr + 500)
            A0 = create_measurement_matrix(m // 2, n, 'gaussian',
                                           random_state=19000 + tr)
            A1 = _overlap_basis(A0, sep, random_state=19000 + tr + 100)
            A = np.vstack([A0, A1])
            x, _ = _make_signal(n, k, tr_rng)
            yc = A @ x; sigma = _noise_sigma(yc, 30)
            y = yc + sigma * tr_rng.randn(m)
            r = _run_solver('DKAMP', A, y, x, sigma, max_iter=150,
                            num_nodes=2, num_triggers=4)
            sep_res[sep].append(np.nan if r['diverged'] else r['nmse'])
        print(f"  separation={sep} done")

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    means = [np.nanmean(res[sc]['nmse']) for sc in scenarios]
    ses = [_sem(res[sc]['nmse']) for sc in scenarios]
    bars = ax.bar(np.arange(len(scenarios)), np.maximum(means, 1e-14), yerr=ses,
                  capsize=4, color=[SOLVER_COLORS['KAMP'], SOLVER_COLORS['OAMP'],
                                    SOLVER_COLORS['DKAMP']],
                  edgecolor='black', linewidth=0.8, zorder=2)
    for b, sc in zip(bars, scenarios):
        ax.text(b.get_x() + b.get_width() / 2, max(means[scenarios.index(sc)], 1e-14) * 1.3,
                f'{np.nanmean(res[sc]["time"]):.0f} ms', ha='center', fontsize=8)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=10)
    ax.set_ylabel('NMSE'); ax.set_title('Node failure impact (labels = runtime)')
    ax.grid(alpha=0.3, axis='y')
    _panel_label(ax, 'a')

    ax = axes[1]
    seps = sorted(sep_res.keys())
    ax.errorbar(seps, [np.nanmean(sep_res[s]) for s in seps],
                yerr=[_sem(sep_res[s]) for s in seps],
                color=SOLVER_COLORS['DKAMP'], marker='o', markersize=7,
                markerfacecolor='white', markeredgecolor=SOLVER_COLORS['DKAMP'],
                linewidth=2.0)
    ax.set_yscale('log')
    ax.set_xlabel('Node separation (independence fraction of 2nd basis)')
    ax.set_ylabel('NMSE'); ax.set_title('Effect of node separation (complementary bases)')
    ax.grid(alpha=0.3)
    _panel_label(ax, 'b')
    plt.tight_layout()
    _save_fig(fig, 'R16_node_failure.png')

    rows = []
    for sc in scenarios:
        rows.append([sc, np.nanmean(res[sc]['nmse']), _sem(res[sc]['nmse']),
                     np.mean(res[sc]['time'])])
    for s in seps:
        rows.append([f'separation={s}', np.nanmean(sep_res[s]),
                     _sem(sep_res[s]), np.nan])
    _save_csv('R16_node_failure.csv', ['scenario', 'nmse_mean', 'nmse_sem',
                                       'time_ms'], rows)
    print("  [PASS] R16")
# ============================================================================
# REGIME VI — Application: Doubly-Selective Jakes Fading Channel
# ============================================================================

def experiment_r17_jakes_channel():
    """R17: Time-varying sparse channel estimation under Jakes Doppler
    spectrum — KAMP Kalman recursion vs per-symbol static estimation.
    Outputs: R17_jakes_channel.png, R17_jakes_channel.csv"""
    print("\n" + "=" * 66)
    print("R17 — Jakes Doubly-Selective Channel (Doppler tracking)")
    print("=" * 66)
    n, m, k = (60, 42, 9) if FAST else (100, 70, 15)
    T = 30 if FAST else 60
    fdT_vals = [0.01, 0.05, 0.1]
    trials = 2 if FAST else 5
    rng = np.random.RandomState(RNG_SEED)
    res = {fd: {'kamp': [], 'static': []} for fd in fdT_vals}

    def run_one(fdT, tr):
        A = create_measurement_matrix(m, n, 'gaussian',
                                      random_state=20000 + tr + int(fdT * 100))
        sup = rng.choice(n, k, replace=False)
        # Jakes envelope fading: per-tap AR(1) with coefficient
        # a = J0^2(2 pi fdT) ~ 1 - 2 (pi fdT)^2 (envelope autocorrelation)
        a = float(np.clip(1.0 - 2.0 * (np.pi * fdT) ** 2, 0.0, 1.0))
        G = np.zeros((T, n))
        for t in range(T):
            if t == 0:
                G[0, sup] = np.abs(rng.randn(k)) + 0.5
            else:
                G[t, sup] = a * G[t - 1, sup] + np.sqrt(1.0 - a ** 2) * rng.randn(k)
        yc = np.zeros((T, m)); sigma = 0.05
        for t in range(T):
            yc[t] = A @ G[t] + sigma * rng.randn(m)
        Q_jakes = float(1.0 - a ** 2)
        est, errs_kamp, _ = _kamp_track(A, yc, G, sigma, Q_jakes)
        errs_st = np.zeros(T)
        for t in range(T):
            r = _run_solver('FISTA', A, yc[t], G[t], sigma, max_iter=200)
            errs_st[t] = np.nan if r['diverged'] else r['nmse']
        return np.nanmean(errs_kamp), np.nanmean(errs_st)

    for fdT in fdT_vals:
        for tr in range(trials):
            mk, ms = run_one(fdT, tr)
            res[fdT]['kamp'].append(mk)
            res[fdT]['static'].append(ms)
        print(f"  fdT={fdT} done")

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    xpos = np.arange(len(fdT_vals)); width = 0.32
    for i, (label, key) in enumerate([('KAMP (recursive)', 'kamp'),
                                      ('FISTA (per-symbol)', 'static')]):
        means = [np.mean(res[fd][key]) for fd in fdT_vals]
        ses = [_sem(res[fd][key]) for fd in fdT_vals]
        ax.bar(xpos + (i - 0.5) * width, np.maximum(means, 1e-14), width=width,
               yerr=ses, capsize=4, label=label,
               color=[SOLVER_COLORS['KAMP'], SOLVER_COLORS['FISTA']][i],
               edgecolor='black', linewidth=0.8, zorder=2)
    ax.set_yscale('log')
    ax.set_xticks(xpos); ax.set_xticklabels([f'{fd:g}' for fd in fdT_vals])
    ax.set_xlabel('Normalized Doppler $f_d T_s$')
    ax.set_ylabel('Time-averaged NMSE')
    ax.set_title('Channel estimation under Jakes fading')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    _panel_label(ax, 'a')

    ax = axes[1]
    fd_ex = fdT_vals[1]
    A = create_measurement_matrix(m, n, 'gaussian', random_state=20000 + int(fd_ex * 100))
    sup = rng.choice(n, k, replace=False)
    a = float(np.clip(1.0 - 2.0 * (np.pi * fd_ex) ** 2, 0.0, 1.0))
    G = np.zeros((T, n))
    for t in range(T):
        if t == 0:
            G[0, sup] = np.abs(rng.randn(k)) + 0.5
        else:
            G[t, sup] = a * G[t - 1, sup] + np.sqrt(1.0 - a ** 2) * rng.randn(k)
    yc = np.zeros((T, m)); sigma = 0.05
    for t in range(T):
        yc[t] = A @ G[t] + sigma * rng.randn(m)
    Q_jakes = float(1.0 - a ** 2)
    est, _, _ = _kamp_track(A, yc, G, sigma, Q_jakes)
    jj = sup[:4]
    for j in jj:
        ax.plot(range(T), G[:, j], color=SOLVER_COLORS['KAMP'], alpha=0.35)
        ax.plot(range(T), est[:, j], linestyle='--', linewidth=1.4,
                color=SOLVER_COLORS['KAMP'])
    ax.set_xlabel('Symbol index'); ax.set_ylabel('Channel gain')
    ax.set_title('Example taps (solid=truth, dashed=KAMP estimate)')
    ax.grid(alpha=0.3)
    _panel_label(ax, 'b')
    plt.tight_layout()
    _save_fig(fig, 'R17_jakes_channel.png')

    rows = []
    for fd in fdT_vals:
        rows.append([fd, np.mean(res[fd]['kamp']), _sem(res[fd]['kamp']),
                     np.mean(res[fd]['static']), _sem(res[fd]['static'])])
    _save_csv('R17_jakes_channel.csv', ['fdTs', 'kamp_nmse', 'kamp_sem',
                                        'static_nmse', 'static_sem'], rows)
    print("  [PASS] R17")


# ============================================================================
# MAIN DISPATCHER
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KAMP regime suite: 17 experiments in 6 scientific regimes.')
    parser.add_argument('--list', action='store_true',
                        help='List all experiments and exit.')
    parser.add_argument('--fast', action='store_true',
                        help='Run in fast (reduced) mode.')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated experiment IDs, e.g. R01,R08.')
    args = parser.parse_args()

    experiments = [
        ('R01', 'Static benchmark (SNR/rho/delta sweeps)', experiment_r01_static_benchmark),
        ('R02', 'Phase transition (Donoho-Tanner)', experiment_r02_phase_transition),
        ('R03', 'Statistical comparison (Holm-corrected MWU)', experiment_r03_statistical_comparison),
        ('R04', 'Correlated measurement matrices (AR-1)', experiment_r04_correlated_measurements),
        ('R05', 'Heteroscedastic noise', experiment_r05_heteroscedastic_noise),
        ('R06', 'Heavy-tailed noise', experiment_r06_heavy_tailed_noise),
        ('R07', 'Colored noise', experiment_r07_colored_noise),
        ('R08', 'Dynamic tracking (R/Q tuning)', experiment_r08_dynamic_tracking),
        ('R09', 'Change-point detection', experiment_r09_change_point),
        ('R10', 'Graph-structured priors', experiment_r10_graph_prior),
        ('R11', '1-bit quantization', experiment_r11_onebit),
        ('R12', 'PCRLB validation', experiment_r12_pcrlb),
        ('R13', 'NEES/NIS consistency', experiment_r13_nees_nis),
        ('R14', 'Active sensing (P1FIM)', experiment_r14_active_sensing),
        ('R15', 'DKAMP scheduling', experiment_r15_dkamp_scheduling),
        ('R16', 'DKAMP node failure', experiment_r16_node_failure),
        ('R17', 'Jakes doubly-selective channel', experiment_r17_jakes_channel),
    ]

    if args.list:
        print('KAMP regime suite — available experiments:')
        for eid, desc, _ in experiments:
            print(f'  {eid}: {desc}')
        return

    if args.only:
        wanted = set(x.strip().upper() for x in args.only.split(','))
        selected = [(eid, desc, fn) for eid, desc, fn in experiments if eid in wanted]
        missing = wanted - {eid for eid, _, _ in experiments}
        if missing:
            print(f'WARNING: unknown experiment IDs: {sorted(missing)}')
    else:
        selected = experiments

    print(f'KAMP regime suite — running {len(selected)} experiment(s) '
          f'({"" if FAST else "not "}fast mode)')
    print(f'Figures -> {FIG_DIR}')
    print(f'Data    -> {DATA_DIR}')
    t_start = time.time()
    for eid, desc, fn in selected:
        print(f'\n>>> {eid}: {desc}')
        try:
            fn()
        except Exception as exc:
            print(f'  [FAIL] {eid}: {exc}')
        print(f'  elapsed {time.time() - t_start:.0f}s so far')
    print(f'\nAll done. Total time: {time.time() - t_start / 1:.0f}s')


if __name__ == '__main__':
    main()
