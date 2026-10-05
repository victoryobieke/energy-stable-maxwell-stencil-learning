"""
Energy-Conserving Stencil Learning for 1D Maxwell Equations
============================================================

This script is designed for the paper focus:

    Learn compact Maxwell derivative stencils from data while enforcing
    skew-adjointness, then test the learned stencil under many noise levels,
    hidden operators, initial conditions, stencil radii, training sizes, and
    constraint choices.

The code generates:
  1. classical finite-difference recovery check,
  2. hidden-operator learning from clean data,
  3. many-noise-level robustness study,
  4. many-initial-condition generalization study,
  5. many-hidden-operator study,
  6. learned radius study,
  7. training sample size study,
  8. regularization study,
  9. constraint ablation study,
 10. long-time propagation and energy study.

Outputs are saved in the folder:
    results_stencil_learning/

Requirements:
    numpy, scipy, matplotlib, pandas

Author-style note:
    Keep this script as the experiment engine. The paper can then report the
    CSV tables and figures generated here.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, bmat, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply


# ================================================================
# Global configuration
# ================================================================

SEED = 7
RESULT_DIR = "results_stencil_learning"
os.makedirs(RESULT_DIR, exist_ok=True)

# Main spatial grid. The hidden stencil below is scaled naturally for N=128.
L = 1.0
N = 128
dx = L / N

# Evolution final time used in most tests.
# Matrix-exponential evolution isolates spatial-operator error.
T_FINAL = 0.02

# Default training setup.
DEFAULT_NTRAIN = 200
DEFAULT_MMAX = 18
DEFAULT_LAMBDA = 1.0e-8
DEFAULT_RHO = 10.0
DEFAULT_BOX = 250.0
DEFAULT_ADMM_ITERS = 300
DEFAULT_TOL = 1.0e-10

# Use the same training-data/noise realization across all main experiments.
GLOBAL_SEED_OFFSET = 111


# ================================================================
# Basic stencil and operator tools
# ================================================================

def offsets_from_radius(R):
    return np.arange(-R, R + 1, dtype=int)


def fd_stencil(R, dx):
    """
    Return the standard 2R-th order centered first-derivative stencil
    on offsets -R,...,R.

    The stencil w satisfies
        sum_l l w_l = 1/dx,
        sum_l l^q w_l = 0, q=3,5,...,2R-1,
    with skew symmetry w_-l = -w_l.
    """
    js = np.arange(1, R + 1, dtype=float)
    qs = np.arange(1, 2 * R, 2, dtype=int)
    A = np.zeros((R, R), dtype=float)
    rhs = np.zeros(R, dtype=float)
    rhs[0] = 1.0 / dx
    for row, q in enumerate(qs):
        A[row, :] = 2.0 * js**q
    a = np.linalg.solve(A, rhs)  # positive-side coefficients w_j

    w = np.zeros(2 * R + 1, dtype=float)
    center = R
    for j in range(1, R + 1):
        w[center + j] = a[j - 1]
        w[center - j] = -a[j - 1]
    return w


def embed_stencil(w, R_new):
    """Embed a stencil of radius R_old into radius R_new by zero padding."""
    R_old = (len(w) - 1) // 2
    if R_new < R_old:
        raise ValueError("R_new must be at least the old radius.")
    out = np.zeros(2 * R_new + 1, dtype=float)
    out[R_new - R_old:R_new + R_old + 1] = w
    return out


def truncate_or_pad_stencil(w, R_new):
    """Truncate or pad a stencil to radius R_new."""
    R_old = (len(w) - 1) // 2
    out = np.zeros(2 * R_new + 1, dtype=float)
    keep = min(R_old, R_new)
    old_center = R_old
    new_center = R_new
    out[new_center - keep:new_center + keep + 1] = w[old_center - keep:old_center + keep + 1]
    return out

def safe_log_values(y, floor=1e-18):
    y = np.asarray(y, dtype=float)
    return np.maximum(y, floor)
def periodic_convolution_matrix(w, N):
    """
    Build the periodic convolution matrix D_w such that
        (D_w u)_i = sum_l w_l u_{i+l mod N}.
    """
    R = (len(w) - 1) // 2
    offsets = offsets_from_radius(R)
    rows = []
    cols = []
    data = []
    for i in range(N):
        for coeff, ell in zip(w, offsets):
            rows.append(i)
            cols.append((i + ell) % N)
            data.append(coeff)
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


def maxwell_block_operator(w, N):
    """
    Block operator for U=[H;E]:
        H_t = D E,
        E_t = D H.
    """
    D = periodic_convolution_matrix(w, N)
    Z = csr_matrix((N, N))
    return bmat([[Z, D], [D, Z]], format="csr")


def discrete_energy(H, E, dx):
    return 0.5 * dx * (np.sum(H**2) + np.sum(E**2))


def rel_l2(a, b, dx):
    num = np.sqrt(dx * np.sum((a - b) ** 2))
    den = np.sqrt(dx * np.sum(b**2))
    return num / max(den, 1.0e-15)


def rel_linf(a, b):
    num = np.max(np.abs(a - b))
    den = np.max(np.abs(b))
    return num / max(den, 1.0e-15)


def skew_violation(w):
    R = (len(w) - 1) // 2
    center = R
    val = abs(w[center])
    for j in range(1, R + 1):
        val += abs(w[center - j] + w[center + j])
    return val


# ================================================================
# Constraint matrices
# ================================================================

def skew_constraint_matrix(R):
    """Cw=0 enforces w_0=0 and w_-j + w_j = 0."""
    C = np.zeros((R + 1, 2 * R + 1), dtype=float)
    center = R
    C[0, center] = 1.0
    for j in range(1, R + 1):
        C[j, center - j] = 1.0
        C[j, center + j] = 1.0
    d = np.zeros(R + 1, dtype=float)
    return C, d


def moment_constraint_matrix(R, dx):
    """
    B w = r enforces centered moment conditions for a 2R-th order
    first-derivative stencil.
    """
    offsets = offsets_from_radius(R).astype(float)
    qs = np.arange(1, 2 * R, 2, dtype=int)
    B = np.zeros((R, 2 * R + 1), dtype=float)
    r = np.zeros(R, dtype=float)
    r[0] = 1.0 / dx
    for row, q in enumerate(qs):
        B[row, :] = offsets**q
    return B, r


def combine_constraints(R, dx, use_skew=True, use_moments=False):
    blocks = []
    rhs = []
    if use_skew:
        C, d = skew_constraint_matrix(R)
        blocks.append(C)
        rhs.append(d)
    if use_moments:
        B, r = moment_constraint_matrix(R, dx)
        blocks.append(B)
        rhs.append(r)
    if not blocks:
        return None, None
    return np.vstack(blocks), np.concatenate(rhs)


# ================================================================
# Training data generation
# ================================================================

def random_fourier_fields(n_samples, N, L, mmax=18, rng=None):
    """
    Generate smooth periodic random Maxwell states E,H.
    Amplitudes decay with mode number to keep fields smooth.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    x = np.linspace(0.0, L, N, endpoint=False)
    E_all = np.zeros((n_samples, N), dtype=float)
    H_all = np.zeros((n_samples, N), dtype=float)

    for s in range(n_samples):
        E = np.zeros(N, dtype=float)
        H = np.zeros(N, dtype=float)
        for m in range(1, mmax + 1):
            decay = 1.0 / (m ** 1.5)
            a1 = rng.normal(scale=decay)
            a2 = rng.normal(scale=decay)
            b1 = rng.normal(scale=decay)
            b2 = rng.normal(scale=decay)
            kx = 2.0 * np.pi * m * x / L
            E += a1 * np.sin(kx) + a2 * np.cos(kx)
            H += b1 * np.sin(kx) + b2 * np.cos(kx)
        E_all[s, :] = E
        H_all[s, :] = H
    return E_all, H_all


def build_design_matrix(E_all, H_all, R):
    """
    Build A so that A w approximates stacked targets:
        D_w H approx E_t,
        D_w E approx H_t.
    """
    n_samples, Nloc = E_all.shape
    offsets = offsets_from_radius(R)
    n_rows = 2 * n_samples * Nloc
    n_cols = 2 * R + 1
    A = np.zeros((n_rows, n_cols), dtype=float)

    row = 0
    for s in range(n_samples):
        E = E_all[s]
        H = H_all[s]
        # Rows for E_t = D H
        for col, ell in enumerate(offsets):
            A[row:row + Nloc, col] = np.roll(H, -ell)
        row += Nloc
        # Rows for H_t = D E
        for col, ell in enumerate(offsets):
            A[row:row + Nloc, col] = np.roll(E, -ell)
        row += Nloc
    return A


def build_targets(E_all, H_all, w_true, noise_level=0.0, rng=None):
    """Generate derivative targets from a hidden true stencil, then add noise."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_samples, Nloc = E_all.shape
    Dtrue = periodic_convolution_matrix(w_true, Nloc)
    b = np.zeros(2 * n_samples * Nloc, dtype=float)
    row = 0
    for s in range(n_samples):
        E = E_all[s]
        H = H_all[s]
        Et = Dtrue @ H
        Ht = Dtrue @ E
        b[row:row + Nloc] = Et
        row += Nloc
        b[row:row + Nloc] = Ht
        row += Nloc

    if noise_level > 0:
        sigma = noise_level * np.std(b)
        b = b + sigma * rng.normal(size=b.shape)
    return b


# ================================================================
# Constrained stencil learning
# ================================================================

def solve_equality_constrained_ridge(A, b, G=None, h=None, lam=1.0e-8):
    """Direct KKT solve for ridge least squares with optional equality constraints."""
    n = A.shape[1]
    K = A.T @ A + lam * np.eye(n)
    rhs = A.T @ b
    if G is None or G.size == 0:
        return np.linalg.solve(K, rhs)
    zeros = np.zeros((G.shape[0], G.shape[0]), dtype=float)
    KKT = np.block([[K, G.T], [G, zeros]])
    rhs2 = np.concatenate([rhs, h])
    sol = np.linalg.solve(KKT, rhs2)
    return sol[:n]


def learn_stencil_admm(
    A,
    b,
    R,
    dx,
    lam=1.0e-8,
    rho=10.0,
    box=250.0,
    max_iter=300,
    tol=1.0e-10,
    use_skew=True,
    use_moments=False,
    use_box=True,
    verbose=False,
    return_info=False,
):
    """
    Learn a stencil by constrained ridge regression.

    If use_box=False, solve the equality-constrained problem directly.
    If use_box=True, use ADMM for equality constraints plus box bounds.
    """
    t0 = time.time()
    G, h = combine_constraints(R, dx, use_skew=use_skew, use_moments=use_moments)

    if not use_box:
        w_out = solve_equality_constrained_ridge(A, b, G=G, h=h, lam=lam)
        info = {
            "iterations": 1,
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "equality_residual": 0.0 if G is None else np.linalg.norm(G @ w_out - h),
            "runtime_seconds": time.time() - t0,
        }
        return (w_out, info) if return_info else w_out

    n = A.shape[1]
    ATA = A.T @ A
    ATb = A.T @ b
    K = ATA + (lam + rho) * np.eye(n)

    if G is not None:
        zeros = np.zeros((G.shape[0], G.shape[0]), dtype=float)
        KKT = np.block([[K, G.T], [G, zeros]])
    else:
        KKT = None

    z = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)

    for k in range(max_iter):
        rhs_w = ATb + rho * (z - u)
        if G is None:
            w = np.linalg.solve(K, rhs_w)
        else:
            rhs = np.concatenate([rhs_w, h])
            sol = np.linalg.solve(KKT, rhs)
            w = sol[:n]

        z_old = z.copy()
        z = np.clip(w + u, -box, box)
        u = u + w - z

        r_primal = np.linalg.norm(w - z)
        r_dual = rho * np.linalg.norm(z - z_old)
        if verbose and (k % 50 == 0 or k == max_iter - 1):
            eq = 0.0 if G is None else np.linalg.norm(G @ w - h)
            print(f"  ADMM iter {k:4d}: primal={r_primal:.3e}, dual={r_dual:.3e}, eq={eq:.3e}")
        if r_primal < tol and r_dual < tol:
            break

    w_out = z.copy()
    info = {
        "iterations": k + 1,
        "primal_residual": r_primal,
        "dual_residual": r_dual,
        "equality_residual": 0.0 if G is None else np.linalg.norm(G @ w_out - h),
        "runtime_seconds": time.time() - t0,
    }
    return (w_out, info) if return_info else w_out




# ================================================================
# Structure-preserving ADMM
# ================================================================

def skew_map_matrix(R):
    """
    Build the matrix S such that w = S a is automatically skew-symmetric.

    The reduced vector is
        a = [w_1, w_2, ..., w_R]^T.

    The full stencil is
        w = [-a_R, ..., -a_2, -a_1, 0, a_1, a_2, ..., a_R]^T.

    Therefore every stencil returned by this parameterization satisfies
        w_0 = 0,   w_{-j} = -w_j,
    so the corresponding periodic convolution matrix is skew-adjoint.
    """
    S = np.zeros((2 * R + 1, R), dtype=float)
    center = R
    for j in range(1, R + 1):
        S[center + j, j - 1] = 1.0
        S[center - j, j - 1] = -1.0
    return S


def learn_stencil_structure_preserving_admm(
    A,
    b,
    R,
    lam=1.0e-8,
    rho=10.0,
    box=250.0,
    max_iter=300,
    tol=1.0e-10,
    alpha=1.5,
    adaptive_rho=True,
    use_box=True,
    verbose=False,
    return_info=False,
):
    """
    Skew-Parameterized Structure-Preserving ADMM.

    Instead of learning the full stencil w and imposing skew-adjointness as a
    constraint, this method writes

        w = S a,

    where a contains only the positive-side stencil coefficients. Therefore
    skew-adjointness is satisfied by construction at every iteration.

    The reduced optimization problem is

        min_a 0.5 ||A S a - b||_2^2 + lam ||a||_2^2
        s.t.  -box <= a_j <= box.

    This is compared against the original ADMM solver in the experiments.
    """
    t0 = time.time()

    S = skew_map_matrix(R)
    AS = A @ S

    n = R
    ATS_AS = AS.T @ AS
    ATS_b = AS.T @ b

    # If there is  no box constraint, solve the reduced ridge problem directly.
    if not use_box:
        a = np.linalg.solve(ATS_AS + 2.0 * lam * np.eye(n), ATS_b)
        w = S @ a
        info = {
            "iterations": 1,
            "rho_final": rho,
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "runtime_seconds": time.time() - t0,
        }
        return (w, info) if return_info else w

    z = np.zeros(n, dtype=float)
    u = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=float)

    rho_k = float(rho)
    r_norm = np.inf
    s_norm = np.inf

    for k in range(max_iter):
        z_old = z.copy()

        # a-update: reduced ridge solve in only R unknowns.
        K = ATS_AS + (2.0 * lam + rho_k) * np.eye(n)
        rhs = ATS_b + rho_k * (z - u)
        a = np.linalg.solve(K, rhs)

        # Over-relaxation.
        a_hat = alpha * a + (1.0 - alpha) * z_old

        # z-update: projection onto the box.
        z = np.clip(a_hat + u, -box, box)

        # Dual update.
        u = u + a_hat - z

        # Residuals.
        r_norm = np.linalg.norm(a - z)
        s_norm = rho_k * np.linalg.norm(z - z_old)

        # Adaptive penalty update by residual balancing.
        if adaptive_rho:
            mu = 10.0
            tau = 2.0
            if r_norm > mu * s_norm and s_norm > 0.0:
                rho_k *= tau
                u /= tau
            elif s_norm > mu * r_norm and r_norm > 0.0:
                rho_k /= tau
                u *= tau

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(
                f"  SP-ADMM iter {k:4d}: "
                f"primal={r_norm:.3e}, dual={s_norm:.3e}, rho={rho_k:.3e}"
            )

        if r_norm < tol and s_norm < tol:
            break

    w = S @ z
    info = {
        "iterations": k + 1,
        "rho_final": rho_k,
        "primal_residual": r_norm,
        "dual_residual": s_norm,
        "runtime_seconds": time.time() - t0,
    }
    return (w, info) if return_info else w


def train_sp_admm_default(
    w_true,
    R_learn=3,
    noise_level=0.2,
    ntrain=DEFAULT_NTRAIN,
    lam=DEFAULT_LAMBDA,
    seed_offset=0,
    return_info=False,
):
    """
    Train the structure-preserving ADMM stencil using the same data-generation
    setup as train_default.
    """
    A, b = make_training_system(w_true, R_learn, ntrain, noise_level, seed_offset=seed_offset)
    return learn_stencil_structure_preserving_admm(
        A,
        b,
        R_learn,
        lam=lam,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        alpha=1.5,
        adaptive_rho=True,
        use_box=True,
        verbose=False,
        return_info=return_info,
    )


def train_admm_and_spadmm_same_data(
    w_true,
    R_learn=3,
    noise_level=0.2,
    ntrain=DEFAULT_NTRAIN,
    lam=DEFAULT_LAMBDA,
    seed_offset=0,
):
    """
    Train original ADMM and structure-preserving ADMM on the exact same
    design matrix and noisy target vector. This makes the comparison fair.
    """
    A, b = make_training_system(w_true, R_learn, ntrain, noise_level, seed_offset=seed_offset)

    t0 = time.time()
    w_admm = learn_stencil_admm(
        A, b, R_learn, dx,
        lam=lam,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        use_skew=True,
        use_moments=False,
        use_box=True,
        verbose=False,
    )
    info_admm = {
        "runtime_seconds": time.time() - t0,
        "skew_violation": skew_violation(w_admm),
    }

    w_spadmm, info_sp = learn_stencil_structure_preserving_admm(
        A,
        b,
        R_learn,
        lam=lam,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        alpha=1.5,
        adaptive_rho=True,
        use_box=True,
        verbose=False,
        return_info=True,
    )
    info_sp["skew_violation"] = skew_violation(w_spadmm)

    return w_admm, info_admm, w_spadmm, info_sp


# ================================================================
# Initial conditions and evolution tests
# ================================================================

def periodic_distance(x, x0, L):
    return ((x - x0 + 0.5 * L) % L) - 0.5 * L


def initial_condition(name, N, L):
    """Deterministic Maxwell initial conditions used for testing."""
    x = np.linspace(0.0, L, N, endpoint=False)

    if name == "single_mode":
        E = np.sin(2 * np.pi * x / L)
        H = np.cos(2 * np.pi * x / L)

    elif name == "two_modes":
        E = np.sin(2 * np.pi * x / L) + 0.35 * np.cos(6 * np.pi * x / L + 0.2)
        H = 0.8 * np.cos(2 * np.pi * x / L + 0.4) - 0.25 * np.sin(8 * np.pi * x / L)

    elif name == "five_modes":
        E = np.zeros(N)
        H = np.zeros(N)
        for m in range(1, 6):
            E += (1.0 / m) * np.sin(2 * np.pi * m * x / L + 0.17 * m)
            H += (0.8 / m) * np.cos(2 * np.pi * m * x / L + 0.31 * m)

    elif name == "broadband_fourier":
        E = np.zeros(N)
        H = np.zeros(N)
        for m in range(1, 18):
            E += (1.0 / m**0.9) * np.sin(2 * np.pi * m * x / L + 0.11 * m)
            H += (0.7 / m**0.9) * np.cos(2 * np.pi * m * x / L + 0.23 * m)

    elif name == "gaussian_pulse":
        d = periodic_distance(x, 0.35 * L, L)
        E = np.exp(-(d / 0.065) ** 2)
        H = -0.8 * np.exp(-(d / 0.075) ** 2)

    elif name == "modulated_gaussian":
        d = periodic_distance(x, 0.42 * L, L)
        carrier = np.cos(24 * np.pi * x / L)
        E = np.exp(-(d / 0.08) ** 2) * carrier
        H = -0.75 * np.exp(-(d / 0.08) ** 2) * np.sin(24 * np.pi * x / L + 0.4)

    elif name == "two_gaussian_pulses":
        d1 = periodic_distance(x, 0.25 * L, L)
        d2 = periodic_distance(x, 0.72 * L, L)
        E = np.exp(-(d1 / 0.055) ** 2) - 0.65 * np.exp(-(d2 / 0.075) ** 2)
        H = -0.9 * np.exp(-(d1 / 0.060) ** 2) - 0.45 * np.exp(-(d2 / 0.070) ** 2)

    elif name == "two_wave_exact":
        s = x
        x1 = 0.30 * L
        x2 = 0.70 * L
        kappa1 = 20.0
        kappa2 = 16.0
        F = np.exp(kappa1 * (np.cos(2 * np.pi * (s - x1) / L) - 1.0)) * np.cos(12 * np.pi * s / L)
        F += 0.25 * np.sin(4 * np.pi * s / L + 0.3)
        G = 0.35 * np.exp(kappa2 * (np.cos(2 * np.pi * (s - x2) / L) - 1.0)) * np.sin(8 * np.pi * s / L + 0.5)
        E = F + G
        H = -F + G

    elif name == "high_frequency_packet":
        d = periodic_distance(x, 0.50 * L, L)
        E = np.exp(-(d / 0.06) ** 2) * np.cos(40 * np.pi * x / L)
        H = -np.exp(-(d / 0.06) ** 2) * np.sin(40 * np.pi * x / L)

    elif name == "rough_smooth_profile":
        E = np.zeros(N)
        H = np.zeros(N)
        for m in range(1, 28):
            E += (1.0 / m**0.65) * np.sin(2 * np.pi * m * x / L + 0.09 * m**2)
            H += (0.8 / m**0.65) * np.cos(2 * np.pi * m * x / L + 0.13 * m**2)
        E /= np.max(np.abs(E))
        H /= np.max(np.abs(H))

    else:
        raise ValueError(f"Unknown initial condition: {name}")

    return H.astype(float), E.astype(float)


TEST_ICS = [
    "single_mode",
    "two_modes",
    "five_modes",
    "broadband_fourier",
    "gaussian_pulse",
    "modulated_gaussian",
    "two_gaussian_pulses",
    "two_wave_exact",
    "high_frequency_packet",
    "rough_smooth_profile",
]


def evolve_maxwell(w, H0, E0, T):
    Nloc = len(H0)
    Aop = maxwell_block_operator(w, Nloc)
    U0 = np.concatenate([H0, E0])
    UT = expm_multiply(T * Aop, U0)
    return UT[:Nloc], UT[Nloc:]


def evaluate_model(w_model, w_true, ic_name, T=T_FINAL, N=N, L=L, dx=dx):
    H0, E0 = initial_condition(ic_name, N, L)
    H_true, E_true = evolve_maxwell(w_true, H0, E0, T)
    H_pred, E_pred = evolve_maxwell(w_model, H0, E0, T)
    E0_energy = discrete_energy(H0, E0, dx)
    ET_energy = discrete_energy(H_pred, E_pred, dx)
    return {
        "ic": ic_name,
        "T": T,
        "rel_E_L2": rel_l2(E_pred, E_true, dx),
        "rel_H_L2": rel_l2(H_pred, H_true, dx),
        "rel_E_Linf": rel_linf(E_pred, E_true),
        "rel_H_Linf": rel_linf(H_pred, H_true),
        "energy_drift": abs(ET_energy - E0_energy) / max(abs(E0_energy), 1.0e-15),
    }


def coefficient_error(w_learned, w_true):
    Rl = (len(w_learned) - 1) // 2
    Rt = (len(w_true) - 1) // 2
    Rmax = max(Rl, Rt)
    wl = embed_stencil(w_learned, Rmax) if Rl < Rmax else w_learned
    wt = embed_stencil(w_true, Rmax) if Rt < Rmax else w_true
    return np.linalg.norm(wl - wt) / max(np.linalg.norm(wt), 1.0e-15)


# ================================================================
# Hidden operators
# ================================================================

def hidden_operator_dictionary(dx):
    """
    Several radius-3 hidden skew-adjoint operators.

    Operator A is the one in the current manuscript draft.
    The others are nonstandard skew-adjoint stencils used to make the
    experiment section less cherry-picked.
    """
    fd3 = fd_stencil(3, dx)

    opA = np.array([-6.2222, 19.5556, -84.4444, 0.0, 84.4444, -19.5556, 6.2222])

    # Slightly lower main coefficient and stronger long-range coefficient.
    opB = np.array([-7.5, 23.0, -82.0, 0.0, 82.0, -23.0, 7.5])

    # High-frequency-biased skew operator.
    opC = np.array([-10.0, 30.0, -78.0, 0.0, 78.0, -30.0, 10.0])

    # Perturbed FD-like operator.
    perturb_pos = np.array([8.0, -2.5, 3.8])
    opD = fd3.copy()
    center = 3
    for j in range(1, 4):
        opD[center + j] += perturb_pos[j - 1]
        opD[center - j] -= perturb_pos[j - 1]

    # Random-looking but controlled skew operator.
    opE = np.array([-4.0, 15.0, -88.0, 0.0, 88.0, -15.0, 4.0])

    return {
        "A_manuscript_hidden": opA,
        "B_long_range": opB,
        "C_high_frequency": opC,
        "D_perturbed_FD": opD,
        "E_compact_biased": opE,
    }


# ================================================================
# Plot helpers
# ================================================================

def save_table(df, name):
    path = os.path.join(RESULT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  saved table: {path}")


def savefig(name):
    path = os.path.join(RESULT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved figure: {path}")


def plot_semilogy_from_df(df, xcol, ycols, labels, xlabel, ylabel, title, filename):
    plt.figure(figsize=(7.0, 4.5))
    for ycol, label in zip(ycols, labels):
        plt.semilogy(df[xcol], df[ycol], marker="o", linewidth=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.legend()
    savefig(filename)


# ================================================================
# Experiment driver helpers
# ================================================================

def make_training_system(w_true, R_learn, ntrain, noise_level, seed_offset=0, mmax=DEFAULT_MMAX):
    rng_fields = np.random.default_rng(SEED + seed_offset)
    rng_noise = np.random.default_rng(SEED + 1000 + seed_offset)
    E_all, H_all = random_fourier_fields(ntrain, N, L, mmax=mmax, rng=rng_fields)
    A = build_design_matrix(E_all, H_all, R_learn)
    b = build_targets(E_all, H_all, w_true, noise_level=noise_level, rng=rng_noise)
    return A, b


def train_default(w_true, R_learn=3, noise_level=0.2, ntrain=DEFAULT_NTRAIN, lam=DEFAULT_LAMBDA,
                  use_skew=True, use_moments=False, use_box=True, seed_offset=0):
    A, b = make_training_system(w_true, R_learn, ntrain, noise_level, seed_offset=seed_offset)
    w = learn_stencil_admm(
        A, b, R_learn, dx,
        lam=lam,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        use_skew=use_skew,
        use_moments=use_moments,
        use_box=use_box,
        verbose=False,
    )
    return w


# ================================================================
# Experiments
# ================================================================

def experiment_1_fd_recovery():
    print("\nExperiment 1: FD recovery check")
    rows = []
    for R in [1, 2, 3, 4]:
        w_fd = fd_stencil(R, dx)
        A, b = make_training_system(w_fd, R, ntrain=80, noise_level=0.0, seed_offset=GLOBAL_SEED_OFFSET, mmax=12)
        w_learned = learn_stencil_admm(
            A, b, R, dx,
            lam=1.0e-10,
            rho=DEFAULT_RHO,
            box=DEFAULT_BOX,
            max_iter=DEFAULT_ADMM_ITERS,
            use_skew=True,
            use_moments=True,
            use_box=True,
        )
        rows.append({
            "R": R,
            "expected_order": 2 * R,
            "fd_stencil": np.array2string(w_fd, precision=8),
            "learned_stencil": np.array2string(w_learned, precision=8),
            "relative_coeff_error": coefficient_error(w_learned, w_fd),
            "skew_violation": skew_violation(w_learned),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_1_fd_recovery.csv")
    return df


def experiment_2_hidden_clean(w_true):
    print("\nExperiment 2: Hidden operator learning from clean data")
    rows = []
    for R in [1, 2, 3]:
        w_learned = train_default(w_true, R_learn=R, noise_level=0.0, ntrain=DEFAULT_NTRAIN,
                                  lam=1.0e-10, use_skew=True, use_moments=False,
                                  use_box=True, seed_offset=GLOBAL_SEED_OFFSET)
        w_fd = fd_stencil(R, dx)
        eval_learned = evaluate_model(w_learned, w_true, "two_wave_exact")
        eval_fd = evaluate_model(w_fd, w_true, "two_wave_exact")
        rows.append({
            "R_learn": R,
            "FD_E_error": eval_fd["rel_E_L2"],
            "Learned_E_error": eval_learned["rel_E_L2"],
            "improvement_factor": eval_fd["rel_E_L2"] / max(eval_learned["rel_E_L2"], 1.0e-15),
            "Learned_H_error": eval_learned["rel_H_L2"],
            "coeff_error": coefficient_error(w_learned, w_true),
            "energy_drift": eval_learned["energy_drift"],
            "skew_violation": skew_violation(w_learned),
            "learned_stencil": np.array2string(w_learned, precision=8),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_2_hidden_clean.csv")

    plt.figure(figsize=(7.0, 4.5))
    plt.semilogy(df["R_learn"], df["FD_E_error"], marker="o", linewidth=2, label="FD")
    plt.semilogy(df["R_learn"], df["Learned_E_error"], marker="s", linewidth=2, label="Learned")
    plt.xlabel("Learned stencil radius R")
    plt.ylabel("Relative final-time E error")
    plt.title("Clean hidden-operator learning")
    plt.grid(False)
    plt.legend()
    savefig("experiment_2_hidden_clean_error.png")
    return df


def experiment_3_noise_sweep(w_true):
    print("\nExperiment 3: Many-noise-level robustness study")
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
    rows = []
    R = 3
    w_fd = fd_stencil(R, dx)
    ic = "two_wave_exact"
    eval_fd = evaluate_model(w_fd, w_true, ic)

    for idx, nl in enumerate(noise_levels):
        w_learned = train_default(w_true, R_learn=R, noise_level=nl, ntrain=DEFAULT_NTRAIN,
                                  lam=DEFAULT_LAMBDA, use_skew=True, use_moments=False,
                                  use_box=True, seed_offset=GLOBAL_SEED_OFFSET)
        ev = evaluate_model(w_learned, w_true, ic)
        rows.append({
            "noise_level": nl,
            "R": R,
            "FD_E_error": eval_fd["rel_E_L2"],
            "Learned_E_error": ev["rel_E_L2"],
            "FD_H_error": eval_fd["rel_H_L2"],
            "Learned_H_error": ev["rel_H_L2"],
            "improvement_factor": eval_fd["rel_E_L2"] / max(ev["rel_E_L2"], 1.0e-15),
            "coeff_error": coefficient_error(w_learned, w_true),
            "energy_drift": ev["energy_drift"],
            "skew_violation": skew_violation(w_learned),
            "coeff_norm": np.linalg.norm(w_learned),
            "learned_stencil": np.array2string(w_learned, precision=8),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_3_noise_sweep.csv")

    plot_semilogy_from_df(
        df,
        "noise_level",
        ["FD_E_error", "Learned_E_error"],
        ["FD", "Learned"],
        "Noise level",
        "Relative final-time E error",
        "Noise sweep: learned stencil vs fixed FD",
        "experiment_3_noise_sweep_error.png",
    )

    plot_semilogy_from_df(
        df,
        "noise_level",
        ["coeff_error", "energy_drift"],
        ["Coefficient error", "Energy drift"],
        "Noise level",
        "Error",
        "Noise sweep diagnostics",
        "experiment_3_noise_sweep_diagnostics.png",
    )
    return df


def experiment_4_many_initial_conditions(w_true):
    print("\nExperiment 4: Generalization across many initial conditions")
    R = 3
    noise_level = 0.2
    w_fd = fd_stencil(R, dx)
    w_learned = train_default(w_true, R_learn=R, noise_level=noise_level, ntrain=DEFAULT_NTRAIN,
                              lam=DEFAULT_LAMBDA, use_skew=True, use_moments=False,
                              use_box=True, seed_offset=GLOBAL_SEED_OFFSET)

    rows = []
    for ic in TEST_ICS:
        ev_fd = evaluate_model(w_fd, w_true, ic)
        ev_l = evaluate_model(w_learned, w_true, ic)
        rows.append({
            "initial_condition": ic,
            "noise_level": noise_level,
            "R": R,
            "FD_E_error": ev_fd["rel_E_L2"],
            "Learned_E_error": ev_l["rel_E_L2"],
            "FD_H_error": ev_fd["rel_H_L2"],
            "Learned_H_error": ev_l["rel_H_L2"],
            "improvement_factor_E": ev_fd["rel_E_L2"] / max(ev_l["rel_E_L2"], 1.0e-15),
            "energy_drift": ev_l["energy_drift"],
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_4_many_initial_conditions.csv")

    x = np.arange(len(df))
    width = 0.38
    plt.figure(figsize=(11.0, 5.0))
    plt.bar(x - width / 2, df["FD_E_error"], width, label="FD")
    plt.bar(x + width / 2, df["Learned_E_error"], width, label="Learned")
    plt.yscale("log")
    plt.xticks(x, df["initial_condition"], rotation=35, ha="right")
    plt.ylabel("Relative final-time E error")
    plt.title("Many initial conditions: FD vs learned stencil")
    plt.grid(False)
    plt.legend()
    savefig("experiment_4_many_initial_conditions_bar.png")
    return df


def experiment_5_many_hidden_operators(hidden_ops):
    print("\nExperiment 5: Many hidden skew-adjoint operators")
    R = 3
    noise_level = 0.2
    rows = []
    ic_subset = ["broadband_fourier", "gaussian_pulse", "two_wave_exact", "high_frequency_packet"]

    for k, (name, w_true) in enumerate(hidden_ops.items()):
        w_fd = fd_stencil(R, dx)
        w_learned = train_default(w_true, R_learn=R, noise_level=noise_level, ntrain=DEFAULT_NTRAIN,
                                  lam=DEFAULT_LAMBDA, use_skew=True, use_moments=False,
                                  use_box=True, seed_offset=GLOBAL_SEED_OFFSET)
        fd_errors = []
        learned_errors = []
        learned_energy = []
        for ic in ic_subset:
            fd_errors.append(evaluate_model(w_fd, w_true, ic)["rel_E_L2"])
            evl = evaluate_model(w_learned, w_true, ic)
            learned_errors.append(evl["rel_E_L2"])
            learned_energy.append(evl["energy_drift"])
        rows.append({
            "hidden_operator": name,
            "noise_level": noise_level,
            "R": R,
            "mean_FD_E_error": float(np.mean(fd_errors)),
            "mean_Learned_E_error": float(np.mean(learned_errors)),
            "improvement_factor": float(np.mean(fd_errors) / max(np.mean(learned_errors), 1.0e-15)),
            "coeff_error": coefficient_error(w_learned, w_true),
            "mean_energy_drift": float(np.mean(learned_energy)),
            "learned_stencil": np.array2string(w_learned, precision=8),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_5_many_hidden_operators.csv")

    x = np.arange(len(df))
    width = 0.38
    plt.figure(figsize=(9.5, 4.8))
    plt.bar(x - width / 2, df["mean_FD_E_error"], width, label="FD")
    plt.bar(x + width / 2, df["mean_Learned_E_error"], width, label="Learned")
    plt.yscale("log")
    plt.xticks(x, df["hidden_operator"], rotation=25, ha="right")
    plt.ylabel("Mean relative E error")
    plt.title("Different hidden skew-adjoint operators")
    plt.grid(False)
    plt.legend()
    savefig("experiment_5_many_hidden_operators.png")
    return df


def experiment_6_radius_study(w_true):
    print("\nExperiment 6: Learned stencil radius study")
    noise_level = 0.2
    rows = []
    ic_subset = ["gaussian_pulse", "two_wave_exact", "high_frequency_packet"]

    for R in [1, 2, 3, 4, 5]:
        w_fd = fd_stencil(R, dx)
        w_learned = train_default(w_true, R_learn=R, noise_level=noise_level, ntrain=DEFAULT_NTRAIN,
                                  lam=DEFAULT_LAMBDA, use_skew=True, use_moments=False,
                                  use_box=True, seed_offset=GLOBAL_SEED_OFFSET)
        fd_errors = []
        learned_errors = []
        for ic in ic_subset:
            fd_errors.append(evaluate_model(w_fd, w_true, ic)["rel_E_L2"])
            learned_errors.append(evaluate_model(w_learned, w_true, ic)["rel_E_L2"])
        rows.append({
            "R_learn": R,
            "noise_level": noise_level,
            "mean_FD_E_error": float(np.mean(fd_errors)),
            "mean_Learned_E_error": float(np.mean(learned_errors)),
            "improvement_factor": float(np.mean(fd_errors) / max(np.mean(learned_errors), 1.0e-15)),
            "coeff_error": coefficient_error(w_learned, w_true),
            "skew_violation": skew_violation(w_learned),
            "coeff_norm": np.linalg.norm(w_learned),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_6_radius_study.csv")

    plot_semilogy_from_df(
        df,
        "R_learn",
        ["mean_FD_E_error", "mean_Learned_E_error"],
        ["FD", "Learned"],
        "Learned radius R",
        "Mean relative E error",
        "Effect of stencil radius",
        "experiment_6_radius_study.png",
    )
    return df




def experiment_8_regularization_study(w_true):
    print("\nExperiment 8: Regularization study")
    R = 3
    noise_level = 0.5
    lambdas = [0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4]
    rows = []
    ic = "two_wave_exact"
    w_fd = fd_stencil(R, dx)
    fd_error = evaluate_model(w_fd, w_true, ic)["rel_E_L2"]

    for idx, lam in enumerate(lambdas):
        effective_lam = lam
        # A zero lambda is allowed here because the box and constraints regularize the solution numerically.
        w_learned = train_default(w_true, R_learn=R, noise_level=noise_level, ntrain=DEFAULT_NTRAIN,
                                  lam=effective_lam, use_skew=True, use_moments=False,
                                  use_box=True, seed_offset=GLOBAL_SEED_OFFSET)
        ev = evaluate_model(w_learned, w_true, ic)
        rows.append({
            "lambda": lam,
            "noise_level": noise_level,
            "FD_E_error": fd_error,
            "Learned_E_error": ev["rel_E_L2"],
            "improvement_factor": fd_error / max(ev["rel_E_L2"], 1.0e-15),
            "coeff_error": coefficient_error(w_learned, w_true),
            "coeff_norm": np.linalg.norm(w_learned),
            "energy_drift": ev["energy_drift"],
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_8_regularization_study.csv")

    # For plotting lambda=0 on a log axis, replace by a tiny display value.
    xplot = df["lambda"].replace(0.0, 1.0e-14)
    plt.figure(figsize=(7.0, 4.5))
    plt.loglog(xplot, df["Learned_E_error"], marker="o", linewidth=2, label="Learned E error")
    plt.loglog(xplot, df["coeff_error"], marker="s", linewidth=2, label="Coefficient error")
    plt.xlabel("Tikhonov regularization lambda")
    plt.ylabel("Error")
    plt.title("Regularization under noisy derivative data")
    plt.grid(False)
    plt.legend()
    savefig("experiment_8_regularization_study.png")
    return df


def experiment_9_constraint_ablation(w_true):
    print("\nExperiment 9: Constraint ablation study")
    R = 3
    noise_level = 0.5
    ic = "two_wave_exact"
    rows = []

    methods = [
        {
            "name": "unconstrained_LS",
            "use_skew": False,
            "use_moments": False,
            "use_box": False,
            "lam": 1.0e-8,
        },
        {
            "name": "skew_only",
            "use_skew": True,
            "use_moments": False,
            "use_box": False,
            "lam": 1.0e-8,
        },
        {
            "name": "skew_plus_box",
            "use_skew": True,
            "use_moments": False,
            "use_box": True,
            "lam": 0.0,
        },
        {
            "name": "skew_plus_box_plus_reg",
            "use_skew": True,
            "use_moments": False,
            "use_box": True,
            "lam": 1.0e-6,
        },
        {
            "name": "skew_plus_moments_plus_box_plus_reg",
            "use_skew": True,
            "use_moments": True,
            "use_box": True,
            "lam": 1.0e-6,
        },
    ]

    A, b = make_training_system(w_true, R, ntrain=DEFAULT_NTRAIN, noise_level=noise_level, seed_offset=GLOBAL_SEED_OFFSET)
    for method in methods:
        w = learn_stencil_admm(
            A, b, R, dx,
            lam=method["lam"],
            rho=DEFAULT_RHO,
            box=DEFAULT_BOX,
            max_iter=DEFAULT_ADMM_ITERS,
            tol=DEFAULT_TOL,
            use_skew=method["use_skew"],
            use_moments=method["use_moments"],
            use_box=method["use_box"],
        )
        ev = evaluate_model(w, w_true, ic)
        rows.append({
            "method": method["name"],
            "noise_level": noise_level,
            "rel_E_error": ev["rel_E_L2"],
            "rel_H_error": ev["rel_H_L2"],
            "energy_drift": ev["energy_drift"],
            "coeff_error": coefficient_error(w, w_true),
            "coeff_norm": np.linalg.norm(w),
            "skew_violation": skew_violation(w),
            "learned_stencil": np.array2string(w, precision=8),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_9_constraint_ablation.csv")

    x = np.arange(len(df))
    plt.figure(figsize=(10.0, 4.8))
    plt.bar(x, df["energy_drift"])
    plt.yscale("log")
    plt.xticks(x, df["method"], rotation=25, ha="right")
    plt.ylabel("Relative energy drift")
    plt.title("Constraint ablation: energy behavior")
    plt.grid(False)
    savefig("experiment_9_ablation_energy.png")

    plt.figure(figsize=(10.0, 4.8))
    plt.bar(x, df["rel_E_error"])
    plt.yscale("log")
    plt.xticks(x, df["method"], rotation=25, ha="right")
    plt.ylabel("Relative E error")
    plt.title("Constraint ablation: field error")
    plt.grid(False)
    savefig("experiment_9_ablation_error.png")
    return df


def experiment_10_long_time(w_true):
    print("\nExperiment 10: Long-time propagation and energy conservation")
    R = 3
    noise_level = 0.5
    times = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50,0.75, 1.0]
    ic = "two_wave_exact"
    w_fd = fd_stencil(R, dx)
    w_learned = train_default(w_true, R_learn=R, noise_level=noise_level, ntrain=DEFAULT_NTRAIN,
                              lam=DEFAULT_LAMBDA, use_skew=True, use_moments=False,
                              use_box=True, seed_offset=GLOBAL_SEED_OFFSET)
    rows = []
    for T in times:
        ev_fd = evaluate_model(w_fd, w_true, ic, T=T)
        ev_l = evaluate_model(w_learned, w_true, ic, T=T)
        rows.append({
            "T": T,
            "FD_E_error": ev_fd["rel_E_L2"],
            "Learned_E_error": ev_l["rel_E_L2"],
            "FD_energy_drift": ev_fd["energy_drift"],
            "Learned_energy_drift": ev_l["energy_drift"],
            "improvement_factor": ev_fd["rel_E_L2"] / max(ev_l["rel_E_L2"], 1.0e-15),
        })
    df = pd.DataFrame(rows)
    save_table(df, "experiment_10_long_time.csv")

    plot_semilogy_from_df(
        df,
        "T",
        ["FD_E_error", "Learned_E_error"],
        ["FD", "Learned"],
        "Final time T",
        "Relative E error",
        "Long-time field error",
        "experiment_10_long_time_error.png",
    )
    floor = 1e-18

    plt.figure(figsize=(8, 5))
    plt.semilogy(df["T"], safe_log_values(df["FD_energy_drift"], floor),
                 "o-", label="FD")
    plt.semilogy(df["T"], safe_log_values(df["Learned_energy_drift"], floor),
                 "o-", label="Learned")
    
    plt.xlabel("Final time $T$")
    plt.ylabel("Relative energy drift")
    plt.title("Energy conservation over increasing final times")
    plt.ylim(1e-18, 1e-13)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "experiment_10_long_time_energy.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    return df




def experiment_11_admm_fd_spadmm_comparison(w_true):
    """
    Dedicated comparison of:
        1. fixed finite difference (FD),
        2. original constrained ADMM,
        3. skew-parameterized structure-preserving ADMM (SP-ADMM).

    This experiment is designed for the paper section comparing solvers.
    """
    print("\nExperiment 11: FD vs ADMM vs Structure-Preserving ADMM")

    R = 3
    noise_level = 0.2

    # ------------------------------------------------------------
    # Part A: compare across many initial conditions
    # ------------------------------------------------------------
    w_fd = fd_stencil(R, dx)
    w_admm, info_admm, w_spadmm, info_sp = train_admm_and_spadmm_same_data(
        w_true,
        R_learn=R,
        noise_level=noise_level,
        ntrain=DEFAULT_NTRAIN,
        lam=DEFAULT_LAMBDA,
        seed_offset=GLOBAL_SEED_OFFSET,
    )

    rows = []
    for ic in TEST_ICS:
        ev_fd = evaluate_model(w_fd, w_true, ic)
        ev_admm = evaluate_model(w_admm, w_true, ic)
        ev_sp = evaluate_model(w_spadmm, w_true, ic)

        rows.append({
            "initial_condition": ic,
            "noise_level": noise_level,
            "R": R,
            "FD_E_error": ev_fd["rel_E_L2"],
            "ADMM_E_error": ev_admm["rel_E_L2"],
            "SP_ADMM_E_error": ev_sp["rel_E_L2"],
            "FD_energy_drift": ev_fd["energy_drift"],
            "ADMM_energy_drift": ev_admm["energy_drift"],
            "SP_ADMM_energy_drift": ev_sp["energy_drift"],
            "ADMM_skew_violation": skew_violation(w_admm),
            "SP_ADMM_skew_violation": skew_violation(w_spadmm),
            "ADMM_runtime_seconds": info_admm["runtime_seconds"],
            "SP_ADMM_runtime_seconds": info_sp["runtime_seconds"],
            "SP_ADMM_iterations": info_sp["iterations"],
        })

    df_ic = pd.DataFrame(rows)
    save_table(df_ic, "experiment_11_fd_admm_spadmm_initial_conditions.csv")

    print("\nSP-ADMM comparison across initial conditions:")
    print(df_ic[[
        "initial_condition",
        "FD_E_error",
        "ADMM_E_error",
        "SP_ADMM_E_error",
        "ADMM_energy_drift",
        "SP_ADMM_energy_drift",
        "ADMM_runtime_seconds",
        "SP_ADMM_runtime_seconds",
        "SP_ADMM_iterations",
    ]].to_string(index=False))

    x = np.arange(len(df_ic))
    width = 0.25
    plt.figure(figsize=(12.0, 5.2))
    plt.bar(x - width, df_ic["FD_E_error"], width, label="FD")
    plt.bar(x, df_ic["ADMM_E_error"], width, label="ADMM")
    plt.bar(x + width, df_ic["SP_ADMM_E_error"], width, label="SP-ADMM")
    plt.yscale("log")
    plt.xticks(x, df_ic["initial_condition"], rotation=35, ha="right")
    plt.ylabel("Relative final-time E error")
    plt.title("FD vs ADMM vs structure-preserving ADMM")
    plt.grid(False)
    plt.legend()
    savefig("experiment_11_fd_admm_spadmm_initial_conditions.png")

    # ------------------------------------------------------------
    # Part B: compare across noise levels
    # ------------------------------------------------------------
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
    ic = "two_wave_exact"
    ev_fd = evaluate_model(w_fd, w_true, ic)

    rows = []
    for idx, nl in enumerate(noise_levels):
        w_admm, info_admm, w_spadmm, info_sp = train_admm_and_spadmm_same_data(
            w_true,
            R_learn=R,
            noise_level=nl,
            ntrain=DEFAULT_NTRAIN,
            lam=DEFAULT_LAMBDA,
            seed_offset=120 + idx,
        )
        ev_admm = evaluate_model(w_admm, w_true, ic)
        ev_sp = evaluate_model(w_spadmm, w_true, ic)

        rows.append({
            "noise_level": nl,
            "R": R,
            "FD_E_error": ev_fd["rel_E_L2"],
            "ADMM_E_error": ev_admm["rel_E_L2"],
            "SP_ADMM_E_error": ev_sp["rel_E_L2"],
            "FD_energy_drift": ev_fd["energy_drift"],
            "ADMM_energy_drift": ev_admm["energy_drift"],
            "SP_ADMM_energy_drift": ev_sp["energy_drift"],
            "ADMM_skew_violation": skew_violation(w_admm),
            "SP_ADMM_skew_violation": skew_violation(w_spadmm),
            "ADMM_coeff_error": coefficient_error(w_admm, w_true),
            "SP_ADMM_coeff_error": coefficient_error(w_spadmm, w_true),
            "ADMM_runtime_seconds": info_admm["runtime_seconds"],
            "SP_ADMM_runtime_seconds": info_sp["runtime_seconds"],
            "SP_ADMM_iterations": info_sp["iterations"],
        })

    df_noise = pd.DataFrame(rows)
    save_table(df_noise, "experiment_11_fd_admm_spadmm_noise_sweep.csv")

    print("\nSP-ADMM comparison across noise levels:")
    print(df_noise[[
        "noise_level",
        "FD_E_error",
        "ADMM_E_error",
        "SP_ADMM_E_error",
        "ADMM_coeff_error",
        "SP_ADMM_coeff_error",
        "ADMM_skew_violation",
        "SP_ADMM_skew_violation",
    ]].to_string(index=False))

    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(df_noise["noise_level"], df_noise["FD_E_error"], "o-", linewidth=2, label="FD")
    plt.semilogy(df_noise["noise_level"], df_noise["ADMM_E_error"], "s-", linewidth=2, label="ADMM")
    plt.semilogy(df_noise["noise_level"], df_noise["SP_ADMM_E_error"], "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Noise level")
    plt.ylabel("Relative final-time E error")
    plt.title("Noise robustness: FD vs ADMM vs SP-ADMM")
    plt.grid(False)
    plt.legend()
    savefig("experiment_11_fd_admm_spadmm_noise_sweep.png")

    # ------------------------------------------------------------
    # Part C: energy drift diagnostics
    # ------------------------------------------------------------
    floor = 1.0e-18
    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(df_noise["noise_level"], safe_log_values(df_noise["FD_energy_drift"], floor),
                 "o-", linewidth=2, label="FD")
    plt.semilogy(df_noise["noise_level"], safe_log_values(df_noise["ADMM_energy_drift"], floor),
                 "s-", linewidth=2, label="ADMM")
    plt.semilogy(df_noise["noise_level"], safe_log_values(df_noise["SP_ADMM_energy_drift"], floor),
                 "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Noise level")
    plt.ylabel("Relative energy drift")
    plt.title("Energy preservation: FD vs ADMM vs SP-ADMM")
    plt.ylim(floor, 1.0e-12)
    plt.grid(False)
    plt.legend()
    savefig("experiment_11_fd_admm_spadmm_energy.png")

    # ------------------------------------------------------------
    # Part D: coefficient and skew diagnostics
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(df_noise["noise_level"], df_noise["ADMM_coeff_error"], "s-", linewidth=2, label="ADMM coeff. error")
    plt.semilogy(df_noise["noise_level"], df_noise["SP_ADMM_coeff_error"], "^-", linewidth=2, label="SP-ADMM coeff. error")
    plt.xlabel("Noise level")
    plt.ylabel("Relative coefficient error")
    plt.title("Coefficient recovery: ADMM vs SP-ADMM")
    plt.grid(False)
    plt.legend()
    savefig("experiment_11_admm_spadmm_coeff_error.png")

    return df_ic, df_noise





# ================================================================
# OVERRIDE EXPERIMENTS 1--10:
# Every experiment below compares FD, original ADMM, and SP-ADMM.
# These definitions intentionally replace the earlier experiment
# definitions with the same names.
# ================================================================

def train_admm_spadmm_same_data_custom(
    w_true,
    R_learn=3,
    noise_level=0.2,
    ntrain=DEFAULT_NTRAIN,
    lam=DEFAULT_LAMBDA,
    seed_offset=0,
    use_moments_admm=False,
    use_box=True,
):
    """
    Train original ADMM and SP-ADMM on the same design matrix and target vector.

    ADMM solves the full-stencil constrained problem.
    SP-ADMM solves the skew-parameterized reduced problem.

    Moment constraints are available for ADMM only.  SP-ADMM is kept as the
    skew-parameterized structure-preserving solver used in the hidden-operator
    experiments.
    """
    A, b = make_training_system(w_true, R_learn, ntrain, noise_level, seed_offset=seed_offset)

    t0 = time.time()
    w_admm = learn_stencil_admm(
        A, b, R_learn, dx,
        lam=lam,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        use_skew=True,
        use_moments=use_moments_admm,
        use_box=use_box,
        verbose=False,
    )
    info_admm = {
        "runtime_seconds": time.time() - t0,
        "skew_violation": skew_violation(w_admm),
    }

    w_spadmm, info_sp = learn_stencil_structure_preserving_admm(
        A, b, R_learn,
        lam=lam,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        alpha=1.5,
        adaptive_rho=True,
        use_box=use_box,
        verbose=False,
        return_info=True,
    )
    info_sp["skew_violation"] = skew_violation(w_spadmm)

    return w_admm, info_admm, w_spadmm, info_sp


def add_three_method_row(rows, label_key, label_value, w_fd, w_admm, info_admm,
                         w_spadmm, info_sp, w_true, ic, T=T_FINAL,
                         extra=None):
    """Append one row containing FD, ADMM, and SP-ADMM diagnostics."""
    ev_fd = evaluate_model(w_fd, w_true, ic, T=T)
    ev_admm = evaluate_model(w_admm, w_true, ic, T=T)
    ev_sp = evaluate_model(w_spadmm, w_true, ic, T=T)

    row = {
        label_key: label_value,
        "ic": ic,
        "T": T,

        "FD_E_error": ev_fd["rel_E_L2"],
        "ADMM_E_error": ev_admm["rel_E_L2"],
        "SP_ADMM_E_error": ev_sp["rel_E_L2"],

        "FD_H_error": ev_fd["rel_H_L2"],
        "ADMM_H_error": ev_admm["rel_H_L2"],
        "SP_ADMM_H_error": ev_sp["rel_H_L2"],

        "FD_energy_drift": ev_fd["energy_drift"],
        "ADMM_energy_drift": ev_admm["energy_drift"],
        "SP_ADMM_energy_drift": ev_sp["energy_drift"],

        "ADMM_coeff_error": coefficient_error(w_admm, w_true),
        "SP_ADMM_coeff_error": coefficient_error(w_spadmm, w_true),

        "ADMM_skew_violation": skew_violation(w_admm),
        "SP_ADMM_skew_violation": skew_violation(w_spadmm),

        "ADMM_runtime_seconds": info_admm.get("runtime_seconds", np.nan),
        "SP_ADMM_runtime_seconds": info_sp.get("runtime_seconds", np.nan),
        "SP_ADMM_iterations": info_sp.get("iterations", np.nan),
    }
    if extra is not None:
        row.update(extra)
    rows.append(row)


def plot_three_method_curves(df, xcol, y_fd, y_admm, y_sp, xlabel, ylabel,
                             title, filename, logx=False, logy=True):
    plt.figure(figsize=(7.4, 4.8))
    plot_fun = plt.loglog if (logx and logy) else plt.semilogy if logy else plt.plot
    plot_fun(df[xcol], df[y_fd], "o-", linewidth=2, label="FD")
    plot_fun(df[xcol], df[y_admm], "s-", linewidth=2, label="ADMM")
    plot_fun(df[xcol], df[y_sp], "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.legend()
    savefig(filename)


def plot_three_method_bars(df, label_col, y_fd, y_admm, y_sp, ylabel, title,
                           filename, rotation=30):
    x = np.arange(len(df))
    width = 0.25
    plt.figure(figsize=(12.0, 5.2))
    plt.bar(x - width, df[y_fd], width, label="FD")
    plt.bar(x, df[y_admm], width, label="ADMM")
    plt.bar(x + width, df[y_sp], width, label="SP-ADMM")
    plt.yscale("log")
    plt.xticks(x, df[label_col], rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.legend()
    savefig(filename)


def experiment_1_fd_recovery():
    print("\nExperiment 1: FD recovery check with FD, ADMM, and SP-ADMM")
    rows = []
    for R in [1, 2, 3, 4]:
        w_fd = fd_stencil(R, dx)

        # ADMM is given moment constraints in this verification test.
        # SP-ADMM is skew-parameterized and trained from the same clean FD data.
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_fd,
            R_learn=R,
            noise_level=0.0,
            ntrain=100,
            lam=1.0e-10,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=True,
            use_box=True,
        )

        rows.append({
            "R": R,
            "expected_order": 2 * R,
            "FD_coeff_error": 0.0,
            "ADMM_coeff_error_vs_FD": coefficient_error(w_admm, w_fd),
            "SP_ADMM_coeff_error_vs_FD": coefficient_error(w_spadmm, w_fd),
            "ADMM_skew_violation": skew_violation(w_admm),
            "SP_ADMM_skew_violation": skew_violation(w_spadmm),
            "ADMM_runtime_seconds": info_admm["runtime_seconds"],
            "SP_ADMM_runtime_seconds": info_sp["runtime_seconds"],
            "SP_ADMM_iterations": info_sp["iterations"],
            "fd_stencil": np.array2string(w_fd, precision=8),
            "admm_stencil": np.array2string(w_admm, precision=8),
            "sp_admm_stencil": np.array2string(w_spadmm, precision=8),
        })

    df = pd.DataFrame(rows)
    save_table(df, "experiment_1_fd_admm_spadmm_recovery.csv")

    floor = 1.0e-18
    plt.figure(figsize=(7.2, 4.8))
    plt.semilogy(df["R"], safe_log_values(df["FD_coeff_error"], floor), "o-", linewidth=2, label="FD exact")
    plt.semilogy(df["R"], safe_log_values(df["ADMM_coeff_error_vs_FD"], floor), "s-", linewidth=2, label="ADMM")
    plt.semilogy(df["R"], safe_log_values(df["SP_ADMM_coeff_error_vs_FD"], floor), "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Stencil radius R")
    plt.ylabel("Relative coefficient error vs FD")
    plt.title("FD recovery: FD vs ADMM vs SP-ADMM")
    plt.ylim(floor, max(1.0e-1, float(df[["ADMM_coeff_error_vs_FD","SP_ADMM_coeff_error_vs_FD"]].max().max())*10))
    plt.grid(False)
    plt.legend()
    savefig("experiment_1_fd_admm_spadmm_recovery.png")

    return df


def experiment_2_hidden_clean(w_true):
    print("\nExperiment 2: Hidden operator learning from clean data with FD, ADMM, and SP-ADMM")
    rows = []
    ic = "two_wave_exact"

    for R in [1, 2, 3]:
        w_fd = fd_stencil(R, dx)
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_true,
            R_learn=R,
            noise_level=0.0,
            ntrain=DEFAULT_NTRAIN,
            lam=1.0e-10,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=False,
            use_box=True,
        )
        add_three_method_row(
            rows, "R_learn", R, w_fd, w_admm, info_admm, w_spadmm, info_sp,
            w_true, ic,
            extra={"noise_level": 0.0}
        )

    df = pd.DataFrame(rows)
    save_table(df, "experiment_2_hidden_clean_fd_admm_spadmm.csv")
    plot_three_method_curves(
        df, "R_learn",
        "FD_E_error", "ADMM_E_error", "SP_ADMM_E_error",
        "Stencil radius R", "Relative final-time E error",
        "Clean hidden-operator learning: FD vs ADMM vs SP-ADMM",
        "experiment_2_hidden_clean_fd_admm_spadmm.png"
    )
    return df


def experiment_3_noise_sweep(w_true):
    print("\nExperiment 3: Noise sweep with FD, ADMM, and SP-ADMM")
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
    R = 3
    ic = "two_wave_exact"
    w_fd = fd_stencil(R, dx)
    rows = []

    for idx, nl in enumerate(noise_levels):
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_true,
            R_learn=R,
            noise_level=nl,
            ntrain=DEFAULT_NTRAIN,
            lam=DEFAULT_LAMBDA,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=False,
            use_box=True,
        )
        add_three_method_row(
            rows, "noise_level", nl, w_fd, w_admm, info_admm, w_spadmm, info_sp,
            w_true, ic,
            extra={"R": R}
        )

    df = pd.DataFrame(rows)
    save_table(df, "experiment_3_noise_sweep_fd_admm_spadmm.csv")
    plot_three_method_curves(
        df, "noise_level",
        "FD_E_error", "ADMM_E_error", "SP_ADMM_E_error",
        "Noise level", "Relative final-time E error",
        "Noise robustness: FD vs ADMM vs SP-ADMM",
        "experiment_3_noise_sweep_fd_admm_spadmm_error.png"
    )

    plt.figure(figsize=(7.5, 4.8))
    floor = 1.0e-18
    plt.semilogy(df["noise_level"], safe_log_values(df["FD_energy_drift"], floor), "o-", linewidth=2, label="FD")
    plt.semilogy(df["noise_level"], safe_log_values(df["ADMM_energy_drift"], floor), "s-", linewidth=2, label="ADMM")
    plt.semilogy(df["noise_level"], safe_log_values(df["SP_ADMM_energy_drift"], floor), "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Noise level")
    plt.ylabel("Relative energy drift")
    plt.title("Energy drift under noisy derivative data")
    plt.ylim(floor, 1.0e-12)
    plt.grid(False)
    plt.legend()
    savefig("experiment_3_noise_sweep_fd_admm_spadmm_energy.png")

    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(df["noise_level"], df["ADMM_coeff_error"], "s-", linewidth=2, label="ADMM coeff. error")
    plt.semilogy(df["noise_level"], df["SP_ADMM_coeff_error"], "^-", linewidth=2, label="SP-ADMM coeff. error")
    plt.xlabel("Noise level")
    plt.ylabel("Relative coefficient error")
    plt.title("Coefficient recovery under noisy derivative data")
    plt.grid(False)
    plt.legend()
    savefig("experiment_3_noise_sweep_admm_spadmm_coeff_error.png")

    return df


def experiment_4_many_initial_conditions(w_true):
    print("\nExperiment 4: Many initial conditions with FD, ADMM, and SP-ADMM")
    R = 3
    noise_level = 0.2
    w_fd = fd_stencil(R, dx)

    w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
        w_true,
        R_learn=R,
        noise_level=noise_level,
        ntrain=DEFAULT_NTRAIN,
        lam=DEFAULT_LAMBDA,
        seed_offset=GLOBAL_SEED_OFFSET,
        use_moments_admm=False,
        use_box=True,
    )

    rows = []
    for ic in TEST_ICS:
        add_three_method_row(
            rows, "initial_condition", ic, w_fd, w_admm, info_admm, w_spadmm, info_sp,
            w_true, ic,
            extra={"R": R, "noise_level": noise_level}
        )

    df = pd.DataFrame(rows)
    save_table(df, "experiment_4_initial_conditions_fd_admm_spadmm.csv")
    plot_three_method_bars(
        df, "initial_condition",
        "FD_E_error", "ADMM_E_error", "SP_ADMM_E_error",
        "Relative final-time E error",
        "Generalization across initial conditions: FD vs ADMM vs SP-ADMM",
        "experiment_4_initial_conditions_fd_admm_spadmm.png",
        rotation=35
    )
    return df


def experiment_5_many_hidden_operators(hidden_ops):
    print("\nExperiment 5: Many hidden operators with FD, ADMM, and SP-ADMM")
    R = 3
    noise_level = 0.2
    ic_subset = ["broadband_fourier", "gaussian_pulse", "two_wave_exact", "high_frequency_packet"]
    rows = []

    for k, (name, w_true_local) in enumerate(hidden_ops.items()):
        w_fd = fd_stencil(R, dx)
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_true_local,
            R_learn=R,
            noise_level=noise_level,
            ntrain=DEFAULT_NTRAIN,
            lam=DEFAULT_LAMBDA,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=False,
            use_box=True,
        )

        fd_errors, admm_errors, sp_errors = [], [], []
        fd_energy, admm_energy, sp_energy = [], [], []

        for ic in ic_subset:
            ev_fd = evaluate_model(w_fd, w_true_local, ic)
            ev_admm = evaluate_model(w_admm, w_true_local, ic)
            ev_sp = evaluate_model(w_spadmm, w_true_local, ic)
            fd_errors.append(ev_fd["rel_E_L2"])
            admm_errors.append(ev_admm["rel_E_L2"])
            sp_errors.append(ev_sp["rel_E_L2"])
            fd_energy.append(ev_fd["energy_drift"])
            admm_energy.append(ev_admm["energy_drift"])
            sp_energy.append(ev_sp["energy_drift"])

        rows.append({
            "hidden_operator": name,
            "R": R,
            "noise_level": noise_level,
            "mean_FD_E_error": float(np.mean(fd_errors)),
            "mean_ADMM_E_error": float(np.mean(admm_errors)),
            "mean_SP_ADMM_E_error": float(np.mean(sp_errors)),
            "mean_FD_energy_drift": float(np.mean(fd_energy)),
            "mean_ADMM_energy_drift": float(np.mean(admm_energy)),
            "mean_SP_ADMM_energy_drift": float(np.mean(sp_energy)),
            "ADMM_coeff_error": coefficient_error(w_admm, w_true_local),
            "SP_ADMM_coeff_error": coefficient_error(w_spadmm, w_true_local),
            "ADMM_skew_violation": skew_violation(w_admm),
            "SP_ADMM_skew_violation": skew_violation(w_spadmm),
            "ADMM_runtime_seconds": info_admm["runtime_seconds"],
            "SP_ADMM_runtime_seconds": info_sp["runtime_seconds"],
            "SP_ADMM_iterations": info_sp["iterations"],
        })

    df = pd.DataFrame(rows)
    save_table(df, "experiment_5_hidden_operators_fd_admm_spadmm.csv")
    plot_three_method_bars(
        df, "hidden_operator",
        "mean_FD_E_error", "mean_ADMM_E_error", "mean_SP_ADMM_E_error",
        "Mean relative E error",
        "Performance across hidden operators: FD vs ADMM vs SP-ADMM",
        "experiment_5_hidden_operators_fd_admm_spadmm.png",
        rotation=25
    )
    return df


def experiment_6_radius_study(w_true):
    print("\nExperiment 6: Radius study with FD, ADMM, and SP-ADMM")
    noise_level = 0.2
    ic_subset = ["gaussian_pulse", "two_wave_exact", "high_frequency_packet"]
    rows = []

    for R in [1, 2, 3, 4, 5]:
        w_fd = fd_stencil(R, dx)
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_true,
            R_learn=R,
            noise_level=noise_level,
            ntrain=DEFAULT_NTRAIN,
            lam=DEFAULT_LAMBDA,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=False,
            use_box=True,
        )

        fd_errors, admm_errors, sp_errors = [], [], []
        for ic in ic_subset:
            fd_errors.append(evaluate_model(w_fd, w_true, ic)["rel_E_L2"])
            admm_errors.append(evaluate_model(w_admm, w_true, ic)["rel_E_L2"])
            sp_errors.append(evaluate_model(w_spadmm, w_true, ic)["rel_E_L2"])

        rows.append({
            "R_learn": R,
            "noise_level": noise_level,
            "mean_FD_E_error": float(np.mean(fd_errors)),
            "mean_ADMM_E_error": float(np.mean(admm_errors)),
            "mean_SP_ADMM_E_error": float(np.mean(sp_errors)),
            "ADMM_coeff_error": coefficient_error(w_admm, w_true),
            "SP_ADMM_coeff_error": coefficient_error(w_spadmm, w_true),
            "ADMM_skew_violation": skew_violation(w_admm),
            "SP_ADMM_skew_violation": skew_violation(w_spadmm),
            "ADMM_runtime_seconds": info_admm["runtime_seconds"],
            "SP_ADMM_runtime_seconds": info_sp["runtime_seconds"],
            "SP_ADMM_iterations": info_sp["iterations"],
        })

    df = pd.DataFrame(rows)
    save_table(df, "experiment_6_radius_fd_admm_spadmm.csv")
    plot_three_method_curves(
        df, "R_learn",
        "mean_FD_E_error", "mean_ADMM_E_error", "mean_SP_ADMM_E_error",
        "Learned stencil radius R", "Mean relative E error",
        "Effect of stencil radius: FD vs ADMM vs SP-ADMM",
        "experiment_6_radius_fd_admm_spadmm.png"
    )
    return df


def experiment_7_training_size_study(w_true):
    print("\nExperiment 7: Training size study with FD, ADMM, and SP-ADMM")
    R = 3
    noise_level = 0.2
    sizes = [10**k for k in range(2, 5)]
    ic = "two_wave_exact"
    w_fd = fd_stencil(R, dx)
    rows = []

    for idx, ntrain in enumerate(sizes):
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_true,
            R_learn=R,
            noise_level=noise_level,
            ntrain=ntrain,
            lam=DEFAULT_LAMBDA,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=False,
            use_box=True,
        )
        add_three_method_row(
            rows, "n_train", ntrain, w_fd, w_admm, info_admm, w_spadmm, info_sp,
            w_true, ic,
            extra={"R": R, "noise_level": noise_level}
        )

    df = pd.DataFrame(rows)
    save_table(df, "experiment_7_training_size_fd_admm_spadmm.csv")
    plot_three_method_curves(
        df, "n_train",
        "FD_E_error", "ADMM_E_error", "SP_ADMM_E_error",
        "Number of training samples", "Relative final-time E error",
        "Effect of training size: FD vs ADMM vs SP-ADMM",
        "experiment_7_training_size_fd_admm_spadmm.png",
        logx=True,
        logy=True
    )
    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------
    eps = 1.0e-15
    df["SP_ADMM_speedup_vs_ADMM"] = (
        df["ADMM_runtime_seconds"] / np.maximum(df["SP_ADMM_runtime_seconds"], eps)
    )
    df["ADMM_time_per_sample"] = df["ADMM_runtime_seconds"] / df["n_train"]
    df["SP_ADMM_time_per_sample"] = df["SP_ADMM_runtime_seconds"] / df["n_train"]
    
    save_table(df, "experiment_7_training_size_fd_admm_spadmm.csv")
    
    print("\nTraining-size accuracy and runtime results:")
    print(df[[
        "n_train",
        "FD_E_error",
        "ADMM_E_error",
        "SP_ADMM_E_error",
        "ADMM_runtime_seconds",
        "SP_ADMM_runtime_seconds",
        "SP_ADMM_speedup_vs_ADMM",
        "SP_ADMM_iterations",
    ]].to_string(index=False))
    
    # ------------------------------------------------------------
    # Accuracy plot
    # ------------------------------------------------------------
    plot_three_method_curves(
        df, "n_train",
        "FD_E_error", "ADMM_E_error", "SP_ADMM_E_error",
        "Number of training samples", "Relative final-time E error",
        "Effect of training size: FD vs ADMM vs SP-ADMM",
        "experiment_7_training_size_fd_admm_spadmm.png",
        logx=True,
        logy=True
    )
    
    # ------------------------------------------------------------
    # Training-time plot: ADMM vs SP-ADMM
    # ------------------------------------------------------------
    plt.figure(figsize=(7.4, 4.8))
    plt.loglog(df["n_train"], df["ADMM_runtime_seconds"],
               "s-", linewidth=2, label="ADMM")
    plt.loglog(df["n_train"], df["SP_ADMM_runtime_seconds"],
               "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Number of training samples")
    plt.ylabel("Training time (seconds)")
    plt.title("Training time: ADMM vs SP-ADMM")
    plt.grid(False)
    plt.legend()
    savefig("experiment_7_training_time_admm_vs_spadmm.png")
    
    # ------------------------------------------------------------
    # Speedup plot
    # speedup > 1 means SP-ADMM is faster than ADMM
    # ------------------------------------------------------------
    plt.figure(figsize=(7.4, 4.8))
    plt.semilogx(df["n_train"], df["SP_ADMM_speedup_vs_ADMM"],
                 "o-", linewidth=2)
    plt.axhline(1.0, linestyle="--", linewidth=1.5)
    plt.xlabel("Number of training samples")
    plt.ylabel("ADMM time / SP-ADMM time")
    plt.title("SP-ADMM speedup relative to ADMM")
    plt.grid(False)
    savefig("experiment_7_speedup_spadmm_vs_admm.png")
    
    # ------------------------------------------------------------
    # Optional: time per training sample
    # ------------------------------------------------------------
    plt.figure(figsize=(7.4, 4.8))
    plt.loglog(df["n_train"], df["ADMM_time_per_sample"],
               "s-", linewidth=2, label="ADMM")
    plt.loglog(df["n_train"], df["SP_ADMM_time_per_sample"],
               "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Number of training samples")
    plt.ylabel("Training time per sample (seconds)")
    plt.title("Training cost per sample")
    plt.grid(False)
    plt.legend()
    savefig("experiment_7_training_time_per_sample.png")
    
    return df


def experiment_8_regularization_study(w_true):
    print("\nExperiment 8: Regularization study with FD, ADMM, and SP-ADMM")
    R = 3
    noise_level = 0.5
    lambdas = [0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4]
    ic = "two_wave_exact"
    w_fd = fd_stencil(R, dx)
    rows = []

    for idx, lam in enumerate(lambdas):
        w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
            w_true,
            R_learn=R,
            noise_level=noise_level,
            ntrain=DEFAULT_NTRAIN,
            lam=lam,
            seed_offset=GLOBAL_SEED_OFFSET,
            use_moments_admm=False,
            use_box=True,
        )
        add_three_method_row(
            rows, "lambda", lam, w_fd, w_admm, info_admm, w_spadmm, info_sp,
            w_true, ic,
            extra={"R": R, "noise_level": noise_level}
        )

    df = pd.DataFrame(rows)
    save_table(df, "experiment_8_regularization_fd_admm_spadmm.csv")
    xplot = df["lambda"].replace(0.0, 1.0e-14)

    plt.figure(figsize=(7.5, 4.8))
    plt.loglog(xplot, df["FD_E_error"], "o-", linewidth=2, label="FD")
    plt.loglog(xplot, df["ADMM_E_error"], "s-", linewidth=2, label="ADMM")
    plt.loglog(xplot, df["SP_ADMM_E_error"], "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Tikhonov regularization lambda")
    plt.ylabel("Relative final-time E error")
    plt.title("Regularization: FD vs ADMM vs SP-ADMM")
    plt.grid(False)
    plt.legend()
    savefig("experiment_8_regularization_fd_admm_spadmm_error.png")

    plt.figure(figsize=(7.5, 4.8))
    plt.loglog(xplot, df["ADMM_coeff_error"], "s-", linewidth=2, label="ADMM coeff. error")
    plt.loglog(xplot, df["SP_ADMM_coeff_error"], "^-", linewidth=2, label="SP-ADMM coeff. error")
    plt.xlabel("Tikhonov regularization lambda")
    plt.ylabel("Relative coefficient error")
    plt.title("Regularization coefficient diagnostics")
    plt.grid(False)
    plt.legend()
    savefig("experiment_8_regularization_admm_spadmm_coeff_error.png")

    return df


def experiment_9_constraint_ablation(w_true):
    print("\nExperiment 9: Constraint ablation with FD, ADMM variants, and SP-ADMM")
    R = 3
    noise_level = 0.5
    ic = "two_wave_exact"
    rows = []

    w_fd = fd_stencil(R, dx)
    ev_fd = evaluate_model(w_fd, w_true, ic)
    rows.append({
        "method": "FD",
        "rel_E_error": ev_fd["rel_E_L2"],
        "rel_H_error": ev_fd["rel_H_L2"],
        "energy_drift": ev_fd["energy_drift"],
        "coeff_error": coefficient_error(w_fd, w_true),
        "coeff_norm": np.linalg.norm(w_fd),
        "skew_violation": skew_violation(w_fd),
        "runtime_seconds": 0.0,
        "iterations": 0,
    })

    A, b = make_training_system(w_true, R, ntrain=DEFAULT_NTRAIN, noise_level=noise_level, seed_offset=GLOBAL_SEED_OFFSET)

    methods = [
        ("ADMM_unconstrained_LS", False, False, False, 1.0e-8),
        ("ADMM_skew_only", True, False, False, 1.0e-8),
        ("ADMM_skew_plus_box", True, False, True, 0.0),
        ("ADMM_skew_plus_box_plus_reg", True, False, True, 1.0e-6),
        ("ADMM_skew_plus_moments_plus_box_plus_reg", True, True, True, 1.0e-6),
    ]

    for name, use_skew, use_moments, use_box, lam in methods:
        t0 = time.time()
        w = learn_stencil_admm(
            A, b, R, dx,
            lam=lam,
            rho=DEFAULT_RHO,
            box=DEFAULT_BOX,
            max_iter=DEFAULT_ADMM_ITERS,
            tol=DEFAULT_TOL,
            use_skew=use_skew,
            use_moments=use_moments,
            use_box=use_box,
            verbose=False,
        )
        runtime = time.time() - t0
        ev = evaluate_model(w, w_true, ic)
        rows.append({
            "method": name,
            "rel_E_error": ev["rel_E_L2"],
            "rel_H_error": ev["rel_H_L2"],
            "energy_drift": ev["energy_drift"],
            "coeff_error": coefficient_error(w, w_true),
            "coeff_norm": np.linalg.norm(w),
            "skew_violation": skew_violation(w),
            "runtime_seconds": runtime,
            "iterations": np.nan,
        })

    w_sp, info_sp = learn_stencil_structure_preserving_admm(
        A, b, R,
        lam=DEFAULT_LAMBDA,
        rho=DEFAULT_RHO,
        box=DEFAULT_BOX,
        max_iter=DEFAULT_ADMM_ITERS,
        tol=DEFAULT_TOL,
        alpha=1.5,
        adaptive_rho=True,
        use_box=True,
        verbose=False,
        return_info=True,
    )
    ev_sp = evaluate_model(w_sp, w_true, ic)
    rows.append({
        "method": "SP_ADMM_skew_parameterized",
        "rel_E_error": ev_sp["rel_E_L2"],
        "rel_H_error": ev_sp["rel_H_L2"],
        "energy_drift": ev_sp["energy_drift"],
        "coeff_error": coefficient_error(w_sp, w_true),
        "coeff_norm": np.linalg.norm(w_sp),
        "skew_violation": skew_violation(w_sp),
        "runtime_seconds": info_sp["runtime_seconds"],
        "iterations": info_sp["iterations"],
    })

    df = pd.DataFrame(rows)
    df["noise_level"] = noise_level
    df["R"] = R
    save_table(df, "experiment_9_ablation_fd_admm_spadmm.csv")

    x = np.arange(len(df))
    plt.figure(figsize=(12.0, 5.0))
    plt.bar(x, df["rel_E_error"])
    plt.yscale("log")
    plt.xticks(x, df["method"], rotation=30, ha="right")
    plt.ylabel("Relative E error")
    plt.title("Constraint ablation: FD, ADMM variants, and SP-ADMM")
    plt.grid(False)
    savefig("experiment_9_ablation_fd_admm_spadmm_error.png")

   
    
    floor = 1.0e-17
    energy_plot = safe_log_values(df["energy_drift"], floor)
    
    plt.figure(figsize=(12.0, 5.0))
    plt.bar(x, energy_plot)
    plt.yscale("log")
    plt.xticks(x, df["method"], rotation=30, ha="right")
    plt.ylabel("Relative energy drift")
    plt.title("Energy behavior: FD, ADMM variants, and SP-ADMM")
    
    # Put the lower y-limit below the plotting floor so tiny bars are visible
    plt.ylim(1.0e-18, 1.0e-10)
    
    plt.grid(False)
    savefig("experiment_9_ablation_fd_admm_spadmm_energy.png")
    return df


def experiment_10_long_time(w_true):
    print("\nExperiment 10: Increasing final time with FD, ADMM, and SP-ADMM")
    R = 3
    noise_level = 0.2
    times = [0.01, 0.25, 0.50, 0.75, 1.0,1.5, 2.0]
    ic = "two_wave_exact"
    w_fd = fd_stencil(R, dx)

    w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
        w_true,
        R_learn=R,
        noise_level=noise_level,
        ntrain=DEFAULT_NTRAIN,
        lam=DEFAULT_LAMBDA,
        seed_offset=GLOBAL_SEED_OFFSET,
        use_moments_admm=False,
        use_box=True,
    )

    rows = []
    for T in times:
        add_three_method_row(
            rows, "T", T, w_fd, w_admm, info_admm, w_spadmm, info_sp,
            w_true, ic, T=T,
            extra={"R": R, "noise_level": noise_level}
        )

    df = pd.DataFrame(rows)
    save_table(df, "experiment_10_long_time_fd_admm_spadmm.csv")

    plot_three_method_curves(
        df, "T",
        "FD_E_error", "ADMM_E_error", "SP_ADMM_E_error",
        "Final time T", "Relative E error",
        "Error growth over increasing final times",
        "experiment_10_long_time_fd_admm_spadmm_error.png"
    )

    floor = 1.0e-18
    plt.figure(figsize=(8.0, 5.0))
    plt.semilogy(df["T"], safe_log_values(df["FD_energy_drift"], floor), "o-", linewidth=2, label="FD")
    plt.semilogy(df["T"], safe_log_values(df["ADMM_energy_drift"], floor), "s-", linewidth=2, label="ADMM")
    plt.semilogy(df["T"], safe_log_values(df["SP_ADMM_energy_drift"], floor), "^-", linewidth=2, label="SP-ADMM")
    plt.xlabel("Final time $T$")
    plt.ylabel("Relative energy drift")
    plt.title("Energy conservation over increasing final times")
    plt.ylim(floor, 1.0e-10)
    plt.grid(False)
    plt.legend()
    savefig("experiment_10_long_time_fd_admm_spadmm_energy.png")

    return df



# ================================================================
# Main driver
# ================================================================
# ================================================================
# Experiment 12: Layered-medium Maxwell wave propagation
# ================================================================

def layered_permittivity(Nloc, Lloc, eps1=1.0, eps2=4.0, x_interface=0.55):
    """
    Piecewise-constant permittivity for a 1D layered medium.
    """
    x = np.linspace(0.0, Lloc, Nloc, endpoint=False)
    eps = eps1 * np.ones(Nloc)
    eps[x >= x_interface] = eps2
    return x, eps


def layered_initial_condition(Nloc, Lloc, eps1=1.0, mu=1.0,
                              x0=0.22, sigma=0.045):
    """
    Right-going Gaussian pulse in the left material.

    For the sign convention used in this code,
        E_t = H_x,  H_t = E_x,
    a right-going pulse has H = -sqrt(eps/mu) E.
    """
    x = np.linspace(0.0, Lloc, Nloc, endpoint=False)
    E0 = np.exp(-((x - x0) / sigma) ** 2)
    H0 = -np.sqrt(eps1 / mu) * E0
    return H0, E0


def layered_energy(H, E, eps, dxloc, mu=1.0):
    """
    Weighted electromagnetic energy for the layered Maxwell system.
    """
    return 0.5 * dxloc * (mu * np.sum(H**2) + np.sum(eps * E**2))


def maxwell_layered_block_operator(w, eps, mu=1.0):
    """
    Block operator for the variable-coefficient layered Maxwell system.

    U = [H; E]

        H_t = (1/mu) D E,
        E_t = eps^{-1} D H.

    If D is skew-adjoint, this preserves the weighted electromagnetic
    energy up to time-integration and roundoff errors.
    """
    Nloc = len(eps)
    D = periodic_convolution_matrix(w, Nloc)
    Z = csr_matrix((Nloc, Nloc))
    Einv = diags(1.0 / eps, 0, format="csr")

    return bmat(
        [
            [Z, (1.0 / mu) * D],
            [Einv @ D, Z],
        ],
        format="csr",
    )


def evolve_layered_maxwell(w, H0, E0, eps, T, mu=1.0):
    """
    Evolve the layered-medium Maxwell system using a matrix exponential.
    """
    Aop = maxwell_layered_block_operator(w, eps, mu=mu)
    U0 = np.concatenate([H0, E0])
    UT = expm_multiply(T * Aop, U0)
    Nloc = len(H0)
    return UT[:Nloc], UT[Nloc:]
def noisy_fd_stencil(w_fd, noise_level=0.05, rng=None, mode="relative"):
    """
    Create a skew-preserving noisy finite-difference stencil.

    This perturbs only the positive-side FD coefficients and then reflects
    them antisymmetrically so that the resulting stencil still satisfies
    w_{-j} = -w_j and w_0 = 0.

    Parameters
    ----------
    w_fd : array
        Standard finite-difference stencil.
    noise_level : float
        Relative noise level. For example, 0.05 means about 5 percent
        coefficient noise.
    rng : numpy random generator
        Random number generator.
    mode : str
        "relative" perturbs each coefficient multiplicatively.
        "additive" adds noise scaled by the coefficient norm.

    Returns
    -------
    w_noisy : array
        Skew-symmetric noisy FD stencil.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    R = (len(w_fd) - 1) // 2
    a_fd = w_fd[R + 1:].copy()  # positive-side coefficients [w_1,...,w_R]

    if mode == "relative":
        a_noisy = a_fd * (1.0 + noise_level * rng.normal(size=R))
    elif mode == "additive":
        scale = np.linalg.norm(a_fd) / np.sqrt(R)
        a_noisy = a_fd + noise_level * scale * rng.normal(size=R)
    else:
        raise ValueError("mode must be 'relative' or 'additive'.")

    w_noisy = np.zeros_like(w_fd)
    for j in range(1, R + 1):
        w_noisy[R + j] = a_noisy[j - 1]
        w_noisy[R - j] = -a_noisy[j - 1]

    return w_noisy

def experiment_12_layered_medium_maxwell():
    """
    Realistic Maxwell example: Gaussian pulse propagation through a
    two-layer dielectric medium.

    This experiment compares FD, ADMM, and SP-ADMM against a high-order
    reference derivative operator.
    """
    print("\nExperiment 12: Layered-medium Maxwell wave propagation")

    # Use the same global grid so the learned stencils are compatible.
    Nloc = N
    Lloc = L
    dxloc = dx

    # Layered material parameters.
    eps1 = 1.0
    eps2 = 4.0
    mu = 1.0
    x_interface = 0.55
    T_layer = 0.55

    x, eps = layered_permittivity(
        Nloc, Lloc, eps1=eps1, eps2=eps2, x_interface=x_interface
    )

    H0, E0 = layered_initial_condition(
        Nloc, Lloc, eps1=eps1, mu=mu, x0=0.22, sigma=0.045
    )

    # Reference derivative: higher-order FD stencil.
    # The comparison methods use compact radius R=3 stencils.
    R = 3
    R_ref = 6
    w_ref = fd_stencil(R_ref, dxloc)
    w_fd = fd_stencil(R, dxloc)
    
    # Add a noisy finite-difference baseline.
    # This keeps skew-symmetry, so it remains energy-preserving in structure.
    fd_noise_level = 0.05
    rng_fd_noise = np.random.default_rng(SEED + GLOBAL_SEED_OFFSET + 1200)
    w_fd_noisy = noisy_fd_stencil(
        w_fd,
        noise_level=fd_noise_level,
        rng=rng_fd_noise,
        mode="relative",
    )
    # Train ADMM and SP-ADMM to learn a compact radius-3 approximation
    # of the high-order reference derivative.
    w_admm, info_admm, w_spadmm, info_sp = train_admm_spadmm_same_data_custom(
        w_ref,
        R_learn=R,
        noise_level=0.05,
        ntrain=DEFAULT_NTRAIN,
        lam=DEFAULT_LAMBDA,
        seed_offset=GLOBAL_SEED_OFFSET + 12,
        use_moments_admm=False,
        use_box=True,
    )

    # Reference solution.
    H_ref, E_ref = evolve_layered_maxwell(
        w_ref, H0, E0, eps, T_layer, mu=mu
    )

    # Numerical solutions.
    H_fd, E_fd = evolve_layered_maxwell(
        w_fd, H0, E0, eps, T_layer, mu=mu
    )
    H_fd_noisy, E_fd_noisy = evolve_layered_maxwell(
    w_fd_noisy, H0, E0, eps, T_layer, mu=mu
     )
    H_admm, E_admm = evolve_layered_maxwell(
        w_admm, H0, E0, eps, T_layer, mu=mu
    )
    H_sp, E_sp = evolve_layered_maxwell(
        w_spadmm, H0, E0, eps, T_layer, mu=mu
    )

    E_initial_energy = layered_energy(H0, E0, eps, dxloc, mu=mu)

    rows = []
    for method, H_num, E_num, w_num in [
        ("FD", H_fd, E_fd, w_fd),
        ("ADMM", H_admm, E_admm, w_admm),
        ("SP-ADMM", H_sp, E_sp, w_spadmm),
    ]:
        final_energy = layered_energy(H_num, E_num, eps, dxloc, mu=mu)
        rows.append({
            "method": method,
            "R": R,
            "T": T_layer,
            "eps1": eps1,
            "eps2": eps2,
            "x_interface": x_interface,
            "E_L2_error": rel_l2(E_num, E_ref, dxloc),
            "H_L2_error": rel_l2(H_num, H_ref, dxloc),
            "E_Linf_error": rel_linf(E_num, E_ref),
            "H_Linf_error": rel_linf(H_num, H_ref),
            "energy_drift": abs(final_energy - E_initial_energy)
                            / max(abs(E_initial_energy), 1.0e-15),
            "skew_violation": skew_violation(w_num),
        })

    df = pd.DataFrame(rows)
    save_table(df, "experiment_12_layered_medium_maxwell.csv")

    print("\nLayered-medium Maxwell results:")
    print(df.to_string(index=False))

    # ------------------------------------------------------------
    # Plot 1: electric-field profiles
    # ------------------------------------------------------------
    plt.figure(figsize=(9.0, 5.0))
    plt.plot(x, E0, "k--", linewidth=1.5, label="Initial E")
    plt.plot(x, E_ref, linewidth=2.5, label="Reference")
    plt.plot(x, E_fd, linewidth=2.0, label="FD")
    plt.plot(x, E_admm, linewidth=2.0, label="ADMM")
    plt.plot(x, E_sp, linewidth=2.0, label="SP-ADMM")
    plt.axvline(x_interface, linestyle=":", linewidth=2.0, label="Interface")
    plt.xlabel("$x$")
    plt.ylabel("$E(x,T)$")
    plt.title("Layered-medium Maxwell propagation")
    plt.grid(False)
    plt.legend()
    savefig("experiment_12_layered_medium_profiles.png")

    # ------------------------------------------------------------
    # Plot 2: final-time field error
    # ------------------------------------------------------------
    xbar = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(xbar - width / 2, df["E_L2_error"], width, label="$E$ error")
    plt.bar(xbar + width / 2, df["H_L2_error"], width, label="$H$ error")
    plt.yscale("log")
    plt.xticks(xbar, df["method"])
    plt.ylabel("Relative $L^2$ error")
    plt.title("Layered-medium field error")
    plt.grid(False)
    plt.legend()
    savefig("experiment_12_layered_medium_error.png")

    # ------------------------------------------------------------
    # Plot 3: energy drift
    # ------------------------------------------------------------
    floor = 1.0e-18
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(xbar, safe_log_values(df["energy_drift"], floor))
    plt.yscale("log")
    plt.xticks(xbar, df["method"])
    plt.ylabel("Relative weighted energy drift")
    plt.title("Layered-medium energy behavior")
    plt.ylim(floor, 1.0e-10)
    plt.grid(False)
    savefig("experiment_12_layered_medium_energy.png")

    return df


# ================================================================
# Experiment 13: ADMM vs SP-ADMM iteration convergence
# ================================================================

def experiment_13_admm_spadmm_iteration_convergence(w_true):
    """
    Compare convergence of standard constrained ADMM and reduced SP-ADMM
    on exactly the same clean radius-3 hidden-operator training problem.

    This experiment is designed to clarify that, with consistent
    regularization, both formulations target the same skew-adjoint constrained
    solution. Differences at a fixed iteration budget therefore reflect
    optimization/convergence behavior rather than different admissible classes.
    """
    print("\nExperiment 13: ADMM vs SP-ADMM iteration convergence")

    R = 3
    noise_level = 0.0
    ntrain = DEFAULT_NTRAIN
    lam = 1.0e-10
    ic = "two_wave_exact"
    iteration_budgets = [10, 30, 100, 300, 1000, 3000, 10000]

    # Build the training system once so every run uses identical A and b.
    A, b = make_training_system(
        w_true,
        R_learn=R,
        ntrain=ntrain,
        noise_level=noise_level,
        seed_offset=GLOBAL_SEED_OFFSET,
    )

    # Exact equality-constrained ridge solution.  The radius-3 hidden stencil
    # lies well inside the box [-DEFAULT_BOX, DEFAULT_BOX], so the box is
    # inactive for this clean test and this gives the common converged target.
    G, h = combine_constraints(R, dx, use_skew=True, use_moments=False)
    w_reference = solve_equality_constrained_ridge(
        A, b, G=G, h=h, lam=lam
    )
    ev_reference = evaluate_model(w_reference, w_true, ic, T=T_FINAL)
    reference_coeff_error = coefficient_error(w_reference, w_true)
    reference_E_error = ev_reference["rel_E_L2"]

    rows = []

    for max_iter in iteration_budgets:
        # --------------------------------------------------------
        # Standard constrained ADMM
        # --------------------------------------------------------
        w_admm, info_admm = learn_stencil_admm(
            A,
            b,
            R,
            dx,
            lam=lam,
            rho=DEFAULT_RHO,
            box=DEFAULT_BOX,
            max_iter=max_iter,
            tol=DEFAULT_TOL,
            use_skew=True,
            use_moments=False,
            use_box=True,
            verbose=False,
            return_info=True,
        )

        # --------------------------------------------------------
        # Reduced structure-preserving ADMM
        # --------------------------------------------------------
        w_sp, info_sp = learn_stencil_structure_preserving_admm(
            A,
            b,
            R,
            lam=lam,
            rho=DEFAULT_RHO,
            box=DEFAULT_BOX,
            max_iter=max_iter,
            tol=DEFAULT_TOL,
            alpha=1.5,
            adaptive_rho=True,
            use_box=True,
            verbose=False,
            return_info=True,
        )

        ev_admm = evaluate_model(w_admm, w_true, ic, T=T_FINAL)
        ev_sp = evaluate_model(w_sp, w_true, ic, T=T_FINAL)

        rows.append({
            "iteration_budget": max_iter,
            "ADMM_iterations": info_admm["iterations"],
            "SP_ADMM_iterations": info_sp["iterations"],
            "ADMM_coeff_error": coefficient_error(w_admm, w_true),
            "SP_ADMM_coeff_error": coefficient_error(w_sp, w_true),
            "reference_coeff_error": reference_coeff_error,
            "ADMM_E_error": ev_admm["rel_E_L2"],
            "SP_ADMM_E_error": ev_sp["rel_E_L2"],
            "reference_E_error": reference_E_error,
            "ADMM_H_error": ev_admm["rel_H_L2"],
            "SP_ADMM_H_error": ev_sp["rel_H_L2"],
            "ADMM_energy_drift": ev_admm["energy_drift"],
            "SP_ADMM_energy_drift": ev_sp["energy_drift"],
            "ADMM_skew_violation": skew_violation(w_admm),
            "SP_ADMM_skew_violation": skew_violation(w_sp),
            "ADMM_primal_residual": info_admm["primal_residual"],
            "ADMM_dual_residual": info_admm["dual_residual"],
            "ADMM_equality_residual": info_admm["equality_residual"],
            "SP_ADMM_primal_residual": info_sp["primal_residual"],
            "SP_ADMM_dual_residual": info_sp["dual_residual"],
            "ADMM_runtime_seconds": info_admm["runtime_seconds"],
            "SP_ADMM_runtime_seconds": info_sp["runtime_seconds"],
            "distance_ADMM_to_SP": coefficient_error(w_admm, w_sp),
            "distance_ADMM_to_reference": coefficient_error(w_admm, w_reference),
            "distance_SP_to_reference": coefficient_error(w_sp, w_reference),
        })

    df = pd.DataFrame(rows)
    save_table(df, "experiment_13_admm_spadmm_iteration_convergence.csv")

    print("\nADMM/SP-ADMM convergence results:")
    print(
        df[[
            "iteration_budget",
            "ADMM_iterations",
            "SP_ADMM_iterations",
            "ADMM_coeff_error",
            "SP_ADMM_coeff_error",
            "ADMM_E_error",
            "SP_ADMM_E_error",
            "distance_ADMM_to_SP",
            "distance_ADMM_to_reference",
            "distance_SP_to_reference",
        ]].to_string(index=False)
    )

    # ------------------------------------------------------------
    # Figure 1: coefficient convergence
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.8))
    plt.loglog(
        df["iteration_budget"], df["ADMM_coeff_error"],
        "s-", linewidth=2, label="ADMM"
    )
    plt.loglog(
        df["iteration_budget"], df["SP_ADMM_coeff_error"],
        "^-", linewidth=2, label="SP-ADMM"
    )
    plt.axhline(
        reference_coeff_error, linestyle="--", linewidth=1.5,
        label="Constrained optimum"
    )
    plt.xlabel("Maximum iteration budget")
    plt.ylabel("Relative coefficient error")
    plt.title("ADMM and SP-ADMM convergence to the hidden stencil")
    plt.grid(False)
    plt.legend()
    savefig("experiment_13_admm_spadmm_iteration_coeff_error.png")

    # ------------------------------------------------------------
    # Figure 2: final-time field accuracy
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.8))
    plt.loglog(
        df["iteration_budget"], df["ADMM_E_error"],
        "s-", linewidth=2, label="ADMM"
    )
    plt.loglog(
        df["iteration_budget"], df["SP_ADMM_E_error"],
        "^-", linewidth=2, label="SP-ADMM"
    )
    plt.axhline(
        reference_E_error, linestyle="--", linewidth=1.5,
        label="Constrained optimum"
    )
    plt.xlabel("Maximum iteration budget")
    plt.ylabel("Relative final-time E error")
    plt.title("Field accuracy versus optimization iteration budget")
    plt.grid(False)
    plt.legend()
    savefig("experiment_13_admm_spadmm_iteration_field_error.png")

    # ------------------------------------------------------------
    # Figure 3: distance to the common constrained optimum
    # ------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.8))
    plt.loglog(
        df["iteration_budget"], df["distance_ADMM_to_reference"],
        "s-", linewidth=2, label="ADMM to optimum"
    )
    plt.loglog(
        df["iteration_budget"], df["distance_SP_to_reference"],
        "^-", linewidth=2, label="SP-ADMM to optimum"
    )
    plt.xlabel("Maximum iteration budget")
    plt.ylabel("Relative stencil distance")
    plt.title("Convergence to the common constrained optimum")
    plt.grid(False)
    plt.legend()
    savefig("experiment_13_admm_spadmm_iteration_distance.png")

    return df


def main():
    start = time.time()
    print("============================================================")
    print("Energy-Conserving Maxwell Stencil Learning Experiments")
    print("============================================================")
    print(f"N={N}, dx={dx:.6e}, L={L}, T_FINAL={T_FINAL}")
    print(f"Saving results to: {RESULT_DIR}")

    hidden_ops = hidden_operator_dictionary(dx)
    w_true = hidden_ops["A_manuscript_hidden"]

    print("\nTrue hidden stencil used for main experiments:")
    print(w_true)
    print(f"Skew violation of true stencil: {skew_violation(w_true):.3e}")

    experiment_1_fd_recovery()
    experiment_2_hidden_clean(w_true)
    experiment_3_noise_sweep(w_true)
    experiment_4_many_initial_conditions(w_true)
    experiment_5_many_hidden_operators(hidden_ops)
    experiment_6_radius_study(w_true)
    experiment_7_training_size_study(w_true)
    experiment_8_regularization_study(w_true)
    experiment_9_constraint_ablation(w_true)
    experiment_10_long_time(w_true)
    experiment_11_admm_fd_spadmm_comparison(w_true)
    experiment_12_layered_medium_maxwell()
    experiment_13_admm_spadmm_iteration_convergence(w_true)
    
    

    elapsed = time.time() - start
    print("\nAll experiments finished.")
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"Check the folder: {RESULT_DIR}")


if __name__ == "__main__":
    main()
