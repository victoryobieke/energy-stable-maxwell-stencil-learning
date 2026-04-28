#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import expm_multiply

# ============================================================
# Noisy-data experiment for R = 1, 2, 3
#
# Hidden data-generating operator:
#   nonstandard skew-adjoint radius-3 stencil
#
# Learned operators:
#   ADMM learned stencils with R = 1, 2, 3
#
# Comparisons:
#   True nonstandard vs Standard FD_R vs ADMM_R
#
# Time evolution:
#   Matrix exponential exact-in-time evolution for each
#   semi-discrete Maxwell operator.
#
# Initial conditions:
#   1. single_mode
#   2. multi_mode
#   3. gaussian_pulse
#   4. two_wave_exact
#
# Accuracy is measured against the true nonstandard operator.
# ============================================================

np.random.seed(7)

# -----------------------------
# Global parameters
# -----------------------------
L = 1.0
N = 128
dx = L / N
x = np.arange(N) * dx

R_TRUE = 3
R_LIST = [1, 2, 3]

n_sims = 80
m_max = 6

# Try 0.20 or 0.90 for the paper tables
noise_level = 0.90

lam_admm = 1.0e-8
rho = 1.0
max_admm_iter = 500
tol = 1.0e-12

M_box = 10.0 / dx

# T is the final time. Nt only controls how many snapshots are saved.
dt_snapshot = 0.4 * dx
Nt = 400
T = Nt * dt_snapshot
num_times = Nt + 1

SHOW_FIGURES = True

output_dir = "Picture_noisy_FD_vs_ADMM_R123"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 1. Random smooth fields for training data
# ============================================================

def random_fourier_state(x, L, m_max):
    """
    Generate random smooth periodic Maxwell fields E and H.
    """
    E = np.zeros_like(x)
    H = np.zeros_like(x)

    for m in range(1, m_max + 1):
        a_m = np.random.randn()
        b_m = np.random.randn()
        phi_m = 2.0 * np.pi * np.random.rand()
        psi_m = 2.0 * np.pi * np.random.rand()

        E += a_m * np.sin(2.0 * np.pi * m * x / L + phi_m)
        H += b_m * np.sin(2.0 * np.pi * m * x / L + psi_m)

    return E, H


def generate_training_samples(n_sims, x, L, m_max):
    """
    Generate fixed training samples so all R values use the same data.
    """
    samples = []

    for _ in range(n_sims):
        E, H = random_fourier_state(x, L, m_max)
        samples.append((E, H))

    return samples


# ============================================================
# 2. Stencils and derivative matrices
# ============================================================

def standard_fd_stencil(R, dx):
    """
    Standard centered finite-difference stencil.
    """
    if R == 1:
        coeff = np.array([-1, 0, 1], dtype=float) / 2.0

    elif R == 2:
        coeff = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0

    elif R == 3:
        coeff = np.array([-1, 9, -45, 0, 45, -9, 1], dtype=float) / 60.0

    else:
        raise ValueError("This script supports R = 1, 2, or 3.")

    return coeff / dx


def make_nonstandard_skew_stencil(R, dx):
    """
    Construct a nonstandard skew-symmetric derivative stencil.

    This is deliberately different from the classical FD stencil,
    but normalized so that

        sum_{ell=-R}^R ell*w_ell = 1/dx.

    Therefore it is still a consistent first-derivative stencil.
    """
    if R != 3:
        raise ValueError("This nonstandard example is written for R_TRUE = 3.")

    # Positive-side raw coefficients for offsets +1,+2,+3.
    # These are deliberately different from FD6.
    p = np.array([0.95, -0.22, 0.07], dtype=float)

    offsets_pos = np.arange(1, R + 1, dtype=float)

    # For a skew stencil:
    # sum_{ell=-R}^R ell*w_ell = 2*sum_{ell>0} ell*w_ell.
    scale = 1.0 / (2.0 * np.sum(offsets_pos * p))
    p = scale * p / dx

    w = np.zeros(2 * R + 1)

    for ell in range(1, R + 1):
        w[R + ell] = p[ell - 1]
        w[R - ell] = -p[ell - 1]

    w[R] = 0.0

    return w


def derivative_matrix_from_stencil(w, R, N):
    """
    Build periodic convolution matrix D_w.
    """
    D = np.zeros((N, N))

    for i in range(N):
        for p, ell in enumerate(range(-R, R + 1)):
            j = (i + ell) % N
            D[i, j] += w[p]

    return D


def pad_stencil_to_radius(w, R, R_target):
    """
    Pad a smaller stencil of radius R into radius R_target.
    This allows coefficient comparison against the true radius-3 stencil.
    """
    padded = np.zeros(2 * R_target + 1)

    for ell in range(-R, R + 1):
        padded[R_target + ell] = w[R + ell]

    return padded


# ============================================================
# 3. Build noisy training system A w ≈ b
# ============================================================

def local_patch(u, i, R):
    """
    Periodic local stencil patch:
    (u_{i-R}, ..., u_i, ..., u_{i+R}).
    """
    Nloc = len(u)

    return np.array([u[(i + ell) % Nloc] for ell in range(-R, R + 1)])


def build_clean_targets_from_true_operator(samples, D_true):
    """
    Build clean derivative targets from the hidden true operator.

    Clean targets:
        E_t = D_true H,
        H_t = D_true E.

    Ordering:
        for each sample, for each grid point:
            target for D H,
            target for D E.
    """
    targets_clean = []

    for E, H in samples:
        dE_dt = D_true @ H
        dH_dt = D_true @ E

        for i in range(len(E)):
            targets_clean.append(dE_dt[i])
            targets_clean.append(dH_dt[i])

    return np.array(targets_clean)


def build_design_matrix_for_radius(samples, R):
    """
    Build design matrix A for a candidate learned stencil radius R.

    The row ordering matches build_clean_targets_from_true_operator.
    """
    rows = []

    for E, H in samples:
        for i in range(len(E)):
            rows.append(local_patch(H, i, R))
            rows.append(local_patch(E, i, R))

    return np.vstack(rows)


# ============================================================
# 4. Skew-adjoint constraint Cw = d
# ============================================================

def build_skew_constraint(R):
    """
    Stencil ordering:
        [w_{-R}, ..., w_0, ..., w_R]

    Enforce:
        w_0 = 0,
        w_{-ell} + w_{ell} = 0.
    """
    n = 2 * R + 1
    C = np.zeros((R + 1, n))

    # w_0 = 0
    C[0, R] = 1.0

    # w_{-ell} + w_{ell} = 0
    for ell in range(1, R + 1):
        C[ell, R - ell] = 1.0
        C[ell, R + ell] = 1.0

    d = np.zeros(R + 1)

    return C, d


# ============================================================
# 5. ADMM solver
# ============================================================

def solve_admm_energy_constrained(
    A,
    b,
    C,
    d,
    lam=1e-8,
    M_box=100.0,
    rho=1.0,
    max_iter=500,
    tol=1e-12
):
    """
    Solve

        min_w 0.5||Aw-b||^2 + lam/2||w||^2
        s.t.  Cw=d,
              -M_box <= w <= M_box

    using ADMM.

    This enforces skew-adjointness and hence energy conservation.
    """
    n = A.shape[1]
    m = C.shape[0]

    AtA = A.T @ A
    Atb = A.T @ b

    z = np.zeros(n)
    u = np.zeros(n)
    w = np.zeros(n)

    KKT = np.block([
        [AtA + (lam + rho) * np.eye(n), C.T],
        [C, np.zeros((m, m))]
    ])

    history = {
        "objective": [],
        "eq_residual": [],
        "primal_residual": []
    }

    for k in range(max_iter):
        rhs = np.concatenate([Atb + rho * (z - u), d])
        sol = np.linalg.solve(KKT, rhs)

        w = sol[:n]

        z_old = z.copy()

        # Box projection
        z = np.clip(w + u, -M_box, M_box)

        # Dual update
        u = u + w - z

        obj = (
            0.5 * np.linalg.norm(A @ w - b) ** 2
            + 0.5 * lam * np.linalg.norm(w) ** 2
        )
        eq_res = np.linalg.norm(C @ w - d)
        primal_res = np.linalg.norm(w - z)

        history["objective"].append(obj)
        history["eq_residual"].append(eq_res)
        history["primal_residual"].append(primal_res)

        if eq_res < tol and primal_res < tol and np.linalg.norm(z - z_old) < tol:
            break

    return w, history


# ============================================================
# 6. Matrix exponential Maxwell solver
# ============================================================

def evolve_maxwell_matrix_exponential(D, E0, H0, T, num_times, dx):
    """
    Exact-in-time evolution of the semi-discrete Maxwell system

        E_t = D H,
        H_t = D E,

    using the matrix exponential.

    We write U = [H; E], so

        U_t = A U,
        A = [[0, D],
             [D, 0]].

    Returns:
        times, E_hist, H_hist, energy
    """
    Nloc = len(E0)

    D_sparse = csr_matrix(D)
    Z = csr_matrix((Nloc, Nloc))

    Aop = bmat(
        [
            [Z, D_sparse],
            [D_sparse, Z]
        ],
        format="csr"
    )

    U0 = np.concatenate([H0.copy(), E0.copy()])

    times = np.linspace(0.0, T, num_times)

    U_hist = expm_multiply(
        Aop,
        U0,
        start=0.0,
        stop=T,
        num=num_times,
        endpoint=True
    )

    H_hist = U_hist[:, :Nloc]
    E_hist = U_hist[:, Nloc:]

    energy = 0.5 * dx * np.sum(E_hist**2 + H_hist**2, axis=1)

    return times, E_hist, H_hist, energy


# ============================================================
# 7. Initial conditions, including two-wave exact profile
# ============================================================

def periodic_gaussian(x, x0, sigma, L):
    """
    Periodic Gaussian using shortest distance on periodic domain.
    """
    dist = np.minimum(np.abs(x - x0), L - np.abs(x - x0))

    return np.exp(-(dist ** 2) / (2.0 * sigma ** 2))


def F_profile(s, L):
    """
    Right-moving smooth periodic wave packet for the exact Maxwell solution.
    """
    x1 = 0.30 * L
    kappa1 = 20.0

    envelope1 = np.exp(
        kappa1 * (np.cos(2.0 * np.pi * (s - x1) / L) - 1.0)
    )

    return (
        envelope1 * np.cos(12.0 * np.pi * s / L)
        + 0.25 * np.sin(4.0 * np.pi * s / L + 0.3)
    )


def G_profile(s, L):
    """
    Left-moving smooth periodic wave packet for the exact Maxwell solution.
    """
    x2 = 0.70 * L
    kappa2 = 16.0

    envelope2 = np.exp(
        kappa2 * (np.cos(2.0 * np.pi * (s - x2) / L) - 1.0)
    )

    return 0.35 * envelope2 * np.sin(8.0 * np.pi * s / L + 0.5)


def exact_two_wave_solution(x, t, L):
    """
    Exact solution of the continuous normalized 1D Maxwell system:

        E_t = H_x,
        H_t = E_x.

    General form:
        E(x,t) = F(x-t) + G(x+t),
        H(x,t) = -F(x-t) + G(x+t).

    In this noisy-data experiment, we use this at t=0 only as
    a smooth broadband initial condition.
    """
    F = F_profile(x - t, L)
    G = G_profile(x + t, L)

    E = F + G
    H = -F + G

    return E, H


def make_initial_conditions(x, L):
    initial_conditions = {}

    # Single Fourier mode
    E0 = np.sin(2.0 * np.pi * x / L)
    H0 = np.cos(2.0 * np.pi * x / L)
    initial_conditions["single_mode"] = (E0, H0)

    # Smooth multi-mode profile
    E0 = (
        np.sin(2.0 * np.pi * x / L)
        + 0.35 * np.sin(4.0 * np.pi * x / L + 0.4)
        + 0.15 * np.cos(6.0 * np.pi * x / L)
    )

    H0 = (
        np.cos(2.0 * np.pi * x / L)
        + 0.25 * np.cos(4.0 * np.pi * x / L + 0.2)
        - 0.10 * np.sin(6.0 * np.pi * x / L)
    )

    initial_conditions["multi_mode"] = (E0, H0)

    # Localized periodic Gaussian pulse
    pulse = periodic_gaussian(x, x0=0.35 * L, sigma=0.06 * L, L=L)
    carrier = np.cos(12.0 * np.pi * x / L)

    E0 = pulse * carrier
    H0 = -E0
    initial_conditions["gaussian_pulse"] = (E0, H0)

    # Two-wave exact Maxwell initial condition
    # Exact for the continuous Maxwell system, but used here as initial data.
    E0, H0 = exact_two_wave_solution(x, 0.0, L)
    initial_conditions["two_wave_exact"] = (E0, H0)

    return initial_conditions


# ============================================================
# 8. Diagnostics and plotting
# ============================================================

def rel_error(u, u_ref):
    return np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref)


def save_stencil_comparison_plot(learned_by_R, w_true):
    """
    Plot true nonstandard, FD, and ADMM stencils for R = 1,2,3.

    For R=1 and R=2, the true radius-3 stencil is restricted to
    the same offsets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, R in zip(axes, R_LIST):
        offsets = np.arange(-R, R + 1)

        w_fd = learned_by_R[R]["w_fd"]
        w_admm = learned_by_R[R]["w_admm"]

        true_start = R_TRUE - R
        true_end = R_TRUE + R + 1
        w_true_restricted = w_true[true_start:true_end]

        ax.plot(
            offsets,
            w_true_restricted,
            marker="^",
            linewidth=2,
            label="True nonstandard"
        )

        ax.plot(
            offsets,
            w_fd,
            marker="o",
            linewidth=2,
            label="Standard FD"
        )

        ax.plot(
            offsets,
            w_admm,
            marker="s",
            linestyle="--",
            linewidth=2,
            label="ADMM learned"
        )

        ax.set_title(f"R = {R}")
        ax.set_xlabel("stencil offset")
        ax.set_ylabel("coefficient")
        ax.legend()
        ax.grid(False)

    fname = os.path.join(output_dir, "stencil_true_FD_ADMM_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_final_profile_grid(case_name, x, results_R, E_true):
    """
    For one initial condition, plot final-time E for R=1,2,3.

    Each panel compares:
        True nonstandard, Standard FD, ADMM learned.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, R in zip(axes, R_LIST):
        data = results_R[R]

        ax.plot(x, E_true[-1], linewidth=2.5, label="True nonstandard")
        ax.plot(x, data["E_fd"][-1], linewidth=2, label="Standard FD")
        ax.plot(x, data["E_admm"][-1], "--", linewidth=2, label="ADMM learned")

        ax.set_title(f"R = {R}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$E(x,T)$")
        ax.legend()
        ax.grid(False)

    fig.suptitle(f"Final-time electric field: {case_name.replace('_', ' ')}")

    fname = os.path.join(output_dir, f"final_profiles_{case_name}_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_energy_error_grid(case_name, times, results_R, energy_true):
    """
    For one initial condition, plot energy error for R=1,2,3.

    Each panel compares:
        True nonstandard, Standard FD, ADMM learned.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, R in zip(axes, R_LIST):
        data = results_R[R]

        energy_fd = data["energy_fd"]
        energy_admm = data["energy_admm"]

        ax.plot(
            times,
            energy_true - energy_true[0],
            linewidth=2.5,
            label="True nonstandard"
        )

        ax.plot(
            times,
            energy_fd - energy_fd[0],
            linewidth=2,
            label="Standard FD"
        )

        ax.plot(
            times,
            energy_admm - energy_admm[0],
            "--",
            linewidth=2,
            label="ADMM learned"
        )

        ax.set_title(f"R = {R}")
        ax.set_xlabel("time")
        ax.set_ylabel(r"$\mathcal{E}^n-\mathcal{E}^0$")
        ax.legend()
        ax.grid(False)

    fig.suptitle(f"Energy error: {case_name.replace('_', ' ')}")

    fname = os.path.join(output_dir, f"energy_errors_{case_name}_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_2d_spacetime_grid(case_name, times, results_R, E_true):
    """
    For one initial condition, create a 3x3 heatmap grid.

    Rows: R = 1,2,3
    Columns: True nonstandard, Standard FD, ADMM learned
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)

    for row, R in enumerate(R_LIST):
        data = results_R[R]

        panels = [
            ("True nonstandard", E_true),
            ("Standard FD", data["E_fd"]),
            ("ADMM learned", data["E_admm"])
        ]

        vmin = min(np.min(panel[1]) for panel in panels)
        vmax = max(np.max(panel[1]) for panel in panels)

        for col, (title, E_hist) in enumerate(panels):
            ax = axes[row, col]

            im = ax.imshow(
                E_hist,
                aspect="auto",
                origin="lower",
                extent=[0.0, L, times[0], times[-1]],
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(f"{title}, R={R}")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Space-time electric field: {case_name.replace('_', ' ')}")

    fname = os.path.join(output_dir, f"spacetime2D_{case_name}_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 9. Main experiment
# ============================================================

print("\nConstructing hidden true nonstandard operator...")

w_true = make_nonstandard_skew_stencil(R_TRUE, dx)
D_true = derivative_matrix_from_stencil(w_true, R_TRUE, N)

print("True radius-3 stencil:")
print(w_true)

# Fixed training samples for all R
print("\nGenerating training samples...")
training_samples = generate_training_samples(n_sims, x, L, m_max)

print("Building clean targets from true operator...")
b_clean = build_clean_targets_from_true_operator(training_samples, D_true)

sigma = noise_level * np.std(b_clean)
noise = sigma * np.random.randn(*b_clean.shape)
b_noisy = b_clean + noise

print(f"Noise level: {noise_level:.2f}")
print(f"Noise sigma: {sigma:.6e}")

# Learn stencils for R = 1,2,3
learned_by_R = {}

for R in R_LIST:
    print("\n" + "=" * 80)
    print(f"Learning ADMM stencil for R = {R}")
    print("=" * 80)

    A = build_design_matrix_for_radius(training_samples, R)
    C, d = build_skew_constraint(R)

    w_admm, hist = solve_admm_energy_constrained(
        A,
        b_noisy,
        C,
        d,
        lam=lam_admm,
        M_box=M_box,
        rho=rho,
        max_iter=max_admm_iter,
        tol=tol
    )

    w_fd = standard_fd_stencil(R, dx)

    D_admm = derivative_matrix_from_stencil(w_admm, R, N)
    D_fd = derivative_matrix_from_stencil(w_fd, R, N)

    # Pad to radius 3 for coefficient comparison
    w_admm_pad = pad_stencil_to_radius(w_admm, R, R_TRUE)
    w_fd_pad = pad_stencil_to_radius(w_fd, R, R_TRUE)

    admm_coeff_err = np.linalg.norm(w_admm_pad - w_true) / np.linalg.norm(w_true)
    fd_coeff_err = np.linalg.norm(w_fd_pad - w_true) / np.linalg.norm(w_true)

    skew_res_admm = np.linalg.norm(C @ w_admm - d)
    skew_res_fd = np.linalg.norm(C @ w_fd - d)

    print("Standard FD stencil:")
    print(w_fd)
    print("ADMM learned stencil:")
    print(w_admm)
    print(f"FD skew residual:       {skew_res_fd:.6e}")
    print(f"ADMM skew residual:     {skew_res_admm:.6e}")
    print(f"FD coeff error vs true: {fd_coeff_err:.6e}")
    print(f"ADMM coeff error true:  {admm_coeff_err:.6e}")
    print(f"ADMM iterations:        {len(hist['objective'])}")

    learned_by_R[R] = {
        "w_fd": w_fd,
        "w_admm": w_admm,
        "D_fd": D_fd,
        "D_admm": D_admm,
        "fd_coeff_err": fd_coeff_err,
        "admm_coeff_err": admm_coeff_err,
        "fd_skew_res": skew_res_fd,
        "admm_skew_res": skew_res_admm,
        "admm_iterations": len(hist["objective"])
    }

save_stencil_comparison_plot(learned_by_R, w_true)

# Simulations for all initial conditions
initial_conditions = make_initial_conditions(x, L)

all_rows = []

for case_name, (E0, H0) in initial_conditions.items():
    print("\n" + "=" * 80)
    print(f"Running initial condition: {case_name}")
    print("=" * 80)

    times, E_true, H_true, energy_true = evolve_maxwell_matrix_exponential(
        D_true,
        E0,
        H0,
        T,
        num_times,
        dx
    )

    results_R = {}

    for R in R_LIST:
        D_fd = learned_by_R[R]["D_fd"]
        D_admm = learned_by_R[R]["D_admm"]

        _, E_fd, H_fd, energy_fd = evolve_maxwell_matrix_exponential(
            D_fd,
            E0,
            H0,
            T,
            num_times,
            dx
        )

        _, E_admm, H_admm, energy_admm = evolve_maxwell_matrix_exponential(
            D_admm,
            E0,
            H0,
            T,
            num_times,
            dx
        )

        fd_rel_E = rel_error(E_fd[-1], E_true[-1])
        admm_rel_E = rel_error(E_admm[-1], E_true[-1])

        fd_rel_H = rel_error(H_fd[-1], H_true[-1])
        admm_rel_H = rel_error(H_admm[-1], H_true[-1])

        fd_energy_drift = np.max(np.abs(energy_fd - energy_fd[0]))
        admm_energy_drift = np.max(np.abs(energy_admm - energy_admm[0]))
        true_energy_drift = np.max(np.abs(energy_true - energy_true[0]))

        print(
            f"R={R} | "
            f"FD rel_E={fd_rel_E:.6e}, ADMM rel_E={admm_rel_E:.6e} | "
            f"FD energy={fd_energy_drift:.3e}, ADMM energy={admm_energy_drift:.3e}"
        )

        results_R[R] = {
            "E_fd": E_fd,
            "H_fd": H_fd,
            "energy_fd": energy_fd,
            "E_admm": E_admm,
            "H_admm": H_admm,
            "energy_admm": energy_admm,
            "fd_rel_E": fd_rel_E,
            "admm_rel_E": admm_rel_E,
            "fd_rel_H": fd_rel_H,
            "admm_rel_H": admm_rel_H,
            "fd_energy_drift": fd_energy_drift,
            "admm_energy_drift": admm_energy_drift,
            "true_energy_drift": true_energy_drift
        }

        all_rows.append({
            "case": case_name,
            "R": R,
            "method": "True nonstandard",
            "rel_E_error_vs_true": 0.0,
            "rel_H_error_vs_true": 0.0,
            "max_energy_drift": true_energy_drift,
            "skew_residual": 0.0,
            "coeff_error_vs_true": 0.0
        })

        all_rows.append({
            "case": case_name,
            "R": R,
            "method": "Standard FD",
            "rel_E_error_vs_true": fd_rel_E,
            "rel_H_error_vs_true": fd_rel_H,
            "max_energy_drift": fd_energy_drift,
            "skew_residual": learned_by_R[R]["fd_skew_res"],
            "coeff_error_vs_true": learned_by_R[R]["fd_coeff_err"]
        })

        all_rows.append({
            "case": case_name,
            "R": R,
            "method": "ADMM learned",
            "rel_E_error_vs_true": admm_rel_E,
            "rel_H_error_vs_true": admm_rel_H,
            "max_energy_drift": admm_energy_drift,
            "skew_residual": learned_by_R[R]["admm_skew_res"],
            "coeff_error_vs_true": learned_by_R[R]["admm_coeff_err"]
        })

    save_final_profile_grid(case_name, x, results_R, E_true)
    save_energy_error_grid(case_name, times, results_R, energy_true)
    save_2d_spacetime_grid(case_name, times, results_R, E_true)


# Save CSV diagnostics
csv_path = os.path.join(output_dir, "FD_vs_ADMM_noisy_R123_diagnostics.csv")

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "case",
        "R",
        "method",
        "rel_E_error_vs_true",
        "rel_H_error_vs_true",
        "max_energy_drift",
        "skew_residual",
        "coeff_error_vs_true"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in all_rows:
        writer.writerow(row)


# Save stencil summary
summary_path = os.path.join(output_dir, "stencil_summary_R123.txt")

with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("Hidden true nonstandard radius-3 stencil\n")
    f.write("=" * 80 + "\n")
    f.write(str(w_true) + "\n\n")
    f.write(f"noise_level = {noise_level}\n")
    f.write(f"noise_sigma = {sigma:.16e}\n\n")

    for R in R_LIST:
        f.write("=" * 80 + "\n")
        f.write(f"R = {R}\n")
        f.write("=" * 80 + "\n")

        f.write("Standard FD stencil:\n")
        f.write(str(learned_by_R[R]["w_fd"]) + "\n")
        f.write(f"FD skew residual: {learned_by_R[R]['fd_skew_res']:.16e}\n")
        f.write(f"FD coeff error vs true: {learned_by_R[R]['fd_coeff_err']:.16e}\n\n")

        f.write("ADMM learned stencil:\n")
        f.write(str(learned_by_R[R]["w_admm"]) + "\n")
        f.write(f"ADMM skew residual: {learned_by_R[R]['admm_skew_res']:.16e}\n")
        f.write(f"ADMM coeff error vs true: {learned_by_R[R]['admm_coeff_err']:.16e}\n")
        f.write(f"ADMM iterations: {learned_by_R[R]['admm_iterations']}\n\n")


print("\n" + "=" * 80)
print("Noisy-data FD vs ADMM experiment for R=1,2,3 complete.")
print("Time evolution used: matrix exponential exact-in-time evolution.")
print("Figures saved in:", output_dir)
print("CSV diagnostics saved to:", csv_path)
print("Stencil summary saved to:", summary_path)
print("=" * 80)

print("\nTwo-wave exact initial-condition outputs include:")
print(os.path.join(output_dir, "final_profiles_two_wave_exact_R123.png"))
print(os.path.join(output_dir, "energy_errors_two_wave_exact_R123.png"))
print(os.path.join(output_dir, "spacetime2D_two_wave_exact_R123.png"))