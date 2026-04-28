#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import expm_multiply

# ============================================================
# ADMM learned-stencil Maxwell experiment
# Three initial conditions, R = 1, 2, 3
#
# Time evolution:
#   Matrix exponential exact-in-time evolution
#
# Energy + final profile plots
# ============================================================

np.random.seed(7)

# -----------------------------
# Global parameters
# -----------------------------
L = 1.0
N = 64
dx = L / N
x = np.arange(N) * dx

R_LIST = [1, 2, 3]

n_sims = 200
m_max = 5

lam = 1.0e-6
rho = 1.0
max_admm_iter = 300
tol = 1.0e-12

# T is the final time.
# num_times controls how many snapshots are saved for plotting.
dt_snapshot = 0.5 * dx
Nt = 300
T = Nt * dt_snapshot
num_times = Nt + 1

SHOW_FIGURES = True

output_dir = "Picture_initial_conditions_R123"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 1. Spectral derivative
# ============================================================

def spectral_derivative(u, L):
    """
    Compute du/dx spectrally on a periodic grid.
    """
    Nloc = len(u)
    dxloc = L / Nloc
    k = 2.0 * np.pi * np.fft.fftfreq(Nloc, d=dxloc)
    uhat = np.fft.fft(u)
    dudx = np.fft.ifft(1j * k * uhat).real
    return dudx


# ============================================================
# 2. Random Fourier training states
# ============================================================

def random_fourier_state(x, L, m_max):
    """
    Generate one random smooth periodic Maxwell state (E,H).
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


# ============================================================
# 3. Build least-squares system Aw ≈ b
# ============================================================

def local_patch(u, i, R):
    """
    Periodic local stencil patch:
    (u_{i-R}, ..., u_i, ..., u_{i+R}).
    """
    Nloc = len(u)
    return np.array([u[(i + ell) % Nloc] for ell in range(-R, R + 1)])


def build_training_system(n_sims, x, L, m_max, R):
    """
    Build least-squares system Aw ≈ b.

    We train one convolution stencil w so that

        D_w H ≈ partial_t E = partial_x H,
        D_w E ≈ partial_t H = partial_x E.
    """
    rows = []
    targets = []

    for _ in range(n_sims):
        E, H = random_fourier_state(x, L, m_max)

        dE_dx = spectral_derivative(E, L)
        dH_dx = spectral_derivative(H, L)

        dE_dt = dH_dx
        dH_dt = dE_dx

        for i in range(len(x)):
            rows.append(local_patch(H, i, R))
            targets.append(dE_dt[i])

            rows.append(local_patch(E, i, R))
            targets.append(dH_dt[i])

    A = np.vstack(rows)
    b = np.array(targets)

    return A, b


# ============================================================
# 4. Constraints: skew-adjointness + moment conditions
# ============================================================

def build_skew_constraint(R):
    """
    Stencil ordering:
        [w_{-R}, ..., w_0, ..., w_R]

    Skew constraints:
        w_0 = 0,
        w_{-ell} + w_{ell} = 0.
    """
    n = 2 * R + 1
    C = np.zeros((R + 1, n))

    # Center coefficient w_0
    C[0, R] = 1.0

    # Pairwise skew-symmetry
    for ell in range(1, R + 1):
        C[ell, R - ell] = 1.0
        C[ell, R + ell] = 1.0

    d = np.zeros(R + 1)

    return C, d


def build_moment_constraint(R, dx):
    """
    Moment constraints for formal 2R-th order centered derivative.

    Stencil ordering:
        [w_{-R}, ..., w_0, ..., w_R]

    Conditions:
        sum ell*w_ell = 1/dx,
        sum ell^q*w_ell = 0 for q = 3,5,...,2R-1.
    """
    offsets = np.arange(-R, R + 1, dtype=float)

    rows = []
    rhs = []

    # First derivative consistency
    rows.append(offsets)
    rhs.append(1.0 / dx)

    # Higher odd moments vanish
    for q in range(3, 2 * R, 2):
        rows.append(offsets ** q)
        rhs.append(0.0)

    B = np.vstack(rows)
    r = np.array(rhs)

    return B, r


def build_combined_constraint(R, dx):
    """
    Combine Cw=d and Bw=r into Gw=h.
    """
    C, d = build_skew_constraint(R)
    B, r = build_moment_constraint(R, dx)

    G = np.vstack([C, B])
    h = np.concatenate([d, r])

    return C, d, B, r, G, h


# ============================================================
# 5. ADMM solver with combined equality constraint Gw = h
# ============================================================

def solve_admm(
    A,
    b,
    G,
    h,
    lam=1e-6,
    M_box=100.0,
    rho=1.0,
    max_iter=300,
    tol=1e-12
):
    """
    Solve

        min_w  0.5 ||Aw - b||^2 + lam/2 ||w||^2
        s.t.   Gw = h,
               -M_box <= w <= M_box,

    using ADMM with auxiliary variable z.
    """
    n = A.shape[1]
    m = G.shape[0]

    AtA = A.T @ A
    Atb = A.T @ b

    z = np.zeros(n)
    u = np.zeros(n)
    w = np.zeros(n)

    KKT = np.block([
        [AtA + (lam + rho) * np.eye(n), G.T],
        [G, np.zeros((m, m))]
    ])

    history = {
        "objective": [],
        "eq_residual": [],
        "primal_residual": []
    }

    for k in range(max_iter):
        rhs = np.concatenate([Atb + rho * (z - u), h])

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
        eq_res = np.linalg.norm(G @ w - h)
        primal_res = np.linalg.norm(w - z)

        history["objective"].append(obj)
        history["eq_residual"].append(eq_res)
        history["primal_residual"].append(primal_res)

        if eq_res < tol and primal_res < tol and np.linalg.norm(z - z_old) < tol:
            break

    return w, history


# ============================================================
# 6. Standard centered finite-difference stencils
# ============================================================

def standard_fd_stencil(R, dx):
    """
    Return standard centered FD stencil of radius R.
    """
    if R == 1:
        coeff = np.array([-1, 0, 1], dtype=float) / 2.0

    elif R == 2:
        coeff = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0

    elif R == 3:
        coeff = np.array([-1, 9, -45, 0, 45, -9, 1], dtype=float) / 60.0

    else:
        raise ValueError("This script supports R = 1, 2, 3.")

    return coeff / dx


# ============================================================
# 7. Build derivative matrix from stencil
# ============================================================

def derivative_matrix_from_stencil(w, R, N):
    """
    Build periodic convolution matrix D_w from stencil w.
    """
    D = np.zeros((N, N))

    for i in range(N):
        for p, ell in enumerate(range(-R, R + 1)):
            j = (i + ell) % N
            D[i, j] += w[p]

    return D


# ============================================================
# 8. Matrix exponential Maxwell solver
# ============================================================

def evolve_maxwell_matrix_exponential(D, E0, H0, T, num_times, dx):
    """
    Exact-in-time evolution of the semi-discrete Maxwell system

        E_t = D H,
        H_t = D E,

    using the matrix exponential.

    We use U = [H; E], so

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
# 9. Initial conditions
# ============================================================

def periodic_gaussian(x, x0, sigma, L):
    """
    Periodic Gaussian using shortest distance on periodic domain.
    """
    dist = np.minimum(np.abs(x - x0), L - np.abs(x - x0))
    return np.exp(-(dist ** 2) / (2.0 * sigma ** 2))


def make_initial_conditions(x, L):
    """
    Return dictionary of named initial conditions.
    """
    initial_conditions = {}

    # IC 1: Single Fourier mode
    E0 = np.sin(2.0 * np.pi * x / L)
    H0 = np.cos(2.0 * np.pi * x / L)
    initial_conditions["single_mode"] = (E0, H0)

    # IC 2: Smooth multi-mode wave
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

    # IC 3: Localized periodic Gaussian pulse
    pulse = periodic_gaussian(x, x0=0.35 * L, sigma=0.06 * L, L=L)
    carrier = np.cos(12.0 * np.pi * x / L)

    E0 = pulse * carrier
    H0 = -E0

    initial_conditions["gaussian_pulse"] = (E0, H0)

    return initial_conditions


# ============================================================
# 10. Learn stencils for R = 1,2,3
# ============================================================

def learn_stencils_for_all_R():
    """
    Learn one ADMM stencil for each R in R_LIST.
    """
    learned = {}

    for R in R_LIST:
        print("\n" + "=" * 70)
        print(f"Learning stencil for R = {R}")
        print("=" * 70)

        A, b = build_training_system(n_sims, x, L, m_max, R)

        C, d, B, r, G, h = build_combined_constraint(R, dx)

        # Box bound should scale like derivative coefficients ~ 1/dx.
        M_box = 10.0 / dx

        w_admm, hist = solve_admm(
            A,
            b,
            G,
            h,
            lam=lam,
            M_box=M_box,
            rho=rho,
            max_iter=max_admm_iter,
            tol=tol
        )

        w_fd = standard_fd_stencil(R, dx)

        D_admm = derivative_matrix_from_stencil(w_admm, R, N)
        D_fd = derivative_matrix_from_stencil(w_fd, R, N)

        skew_res = np.linalg.norm(C @ w_admm - d)
        mom_res = np.linalg.norm(B @ w_admm - r)
        coeff_err = np.linalg.norm(w_admm - w_fd) / np.linalg.norm(w_fd)

        print("ADMM learned stencil:", w_admm)
        print("Standard FD stencil: ", w_fd)
        print(f"Skew residual ||Cw-d||: {skew_res:.6e}")
        print(f"Moment residual ||Bw-r||: {mom_res:.6e}")
        print(f"Relative coefficient error: {coeff_err:.6e}")
        print(f"ADMM iterations: {len(hist['objective'])}")

        learned[R] = {
            "w_admm": w_admm,
            "w_fd": w_fd,
            "D_admm": D_admm,
            "D_fd": D_fd,
            "skew_res": skew_res,
            "mom_res": mom_res,
            "coeff_err": coeff_err,
            "iterations": len(hist["objective"])
        }

    return learned


# ============================================================
# 11. Run one case for one R
# ============================================================

def run_case_for_R(case_name, E0, H0, R, data_R):
    """
    Run FD and ADMM simulations for one initial condition and one R
    using matrix exponential exact-in-time evolution.
    """
    times, E_fd, H_fd, energy_fd = evolve_maxwell_matrix_exponential(
        data_R["D_fd"],
        E0,
        H0,
        T,
        num_times,
        dx
    )

    _, E_admm, H_admm, energy_admm = evolve_maxwell_matrix_exponential(
        data_R["D_admm"],
        E0,
        H0,
        T,
        num_times,
        dx
    )

    rel_E_error = np.linalg.norm(E_admm[-1] - E_fd[-1]) / np.linalg.norm(E_fd[-1])
    rel_H_error = np.linalg.norm(H_admm[-1] - H_fd[-1]) / np.linalg.norm(H_fd[-1])

    max_energy_drift_fd = np.max(np.abs(energy_fd - energy_fd[0]))
    max_energy_drift_admm = np.max(np.abs(energy_admm - energy_admm[0]))

    return {
        "case": case_name,
        "R": R,
        "times": times,
        "E_fd": E_fd,
        "H_fd": H_fd,
        "energy_fd": energy_fd,
        "E_admm": E_admm,
        "H_admm": H_admm,
        "energy_admm": energy_admm,
        "rel_E_error": rel_E_error,
        "rel_H_error": rel_H_error,
        "max_energy_drift_fd": max_energy_drift_fd,
        "max_energy_drift_admm": max_energy_drift_admm
    }


# ============================================================
# 12. Plot functions
# ============================================================

def save_initial_condition_plot(initial_conditions):
    """
    Plot all three initial conditions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)

    for ax, (case_name, (E0, H0)) in zip(axes, initial_conditions.items()):
        ax.plot(x, E0, linewidth=2, label=r"$E(x,0)$")
        ax.plot(x, H0, "--", linewidth=2, label=r"$H(x,0)$")
        ax.set_title(case_name.replace("_", " "))
        ax.set_xlabel("x")
        ax.set_ylabel("field")
        ax.legend()
        ax.grid(False)

    fname = os.path.join(output_dir, "initial_conditions_all.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_final_profile_grid(results_by_case):
    """
    For each initial condition, create a 1x3 plot for R=1,2,3
    comparing final-time FD and ADMM electric fields.
    """
    for case_name, results_R in results_by_case.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        for ax, R in zip(axes, R_LIST):
            result = results_R[R]

            ax.plot(x, result["E_fd"][-1], linewidth=2, label="Standard FD")
            ax.plot(x, result["E_admm"][-1], "--", linewidth=2, label="ADMM learned")

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


def save_energy_error_grid(results_by_case):
    """
    For each initial condition, create a 1x3 plot for R=1,2,3
    comparing energy error for FD and ADMM.
    """
    for case_name, results_R in results_by_case.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        for ax, R in zip(axes, R_LIST):
            result = results_R[R]

            times = result["times"]
            energy_fd = result["energy_fd"]
            energy_admm = result["energy_admm"]

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


def save_2d_spacetime_grid(results_by_case):
    """
    For each initial condition, create a 1x3 2D space-time heatmap
    for R=1,2,3 showing the ADMM-learned electric field E(x,t).
    """
    for case_name, results_R in results_by_case.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        for ax, R in zip(axes, R_LIST):
            result = results_R[R]

            times = result["times"]
            E_admm = result["E_admm"]

            im = ax.imshow(
                E_admm,
                aspect="auto",
                origin="lower",
                extent=[0.0, L, times[0], times[-1]]
            )

            ax.set_title(f"R = {R}")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, shrink=0.85)

        fig.suptitle(f"Space-time electric field: {case_name.replace('_', ' ')}")

        fname = os.path.join(output_dir, f"spacetime2D_{case_name}_R123.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")

        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


# ============================================================
# 13. Main experiment
# ============================================================

print("\nLearning ADMM stencils for R = 1, 2, 3...")
learned = learn_stencils_for_all_R()

initial_conditions = make_initial_conditions(x, L)
save_initial_condition_plot(initial_conditions)

results_by_case = {}
diagnostics_rows = []

for case_name, (E0, H0) in initial_conditions.items():
    print("\n" + "=" * 70)
    print(f"Running initial condition: {case_name}")
    print("=" * 70)

    results_by_case[case_name] = {}

    for R in R_LIST:
        result = run_case_for_R(case_name, E0, H0, R, learned[R])
        results_by_case[case_name][R] = result

        row = {
            "case": case_name,
            "R": R,
            "rel_E_error": result["rel_E_error"],
            "rel_H_error": result["rel_H_error"],
            "max_energy_drift_fd": result["max_energy_drift_fd"],
            "max_energy_drift_admm": result["max_energy_drift_admm"],
            "skew_residual": learned[R]["skew_res"],
            "moment_residual": learned[R]["mom_res"],
            "coefficient_error": learned[R]["coeff_err"]
        }

        diagnostics_rows.append(row)

        print(
            f"R={R}: "
            f"rel_E={result['rel_E_error']:.6e}, "
            f"rel_H={result['rel_H_error']:.6e}, "
            f"energy ADMM drift={result['max_energy_drift_admm']:.6e}"
        )

# Save plots
save_final_profile_grid(results_by_case)
save_energy_error_grid(results_by_case)
save_2d_spacetime_grid(results_by_case)

# Save diagnostics CSV
csv_path = os.path.join(output_dir, "diagnostics_initial_conditions_R123.csv")

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "case",
        "R",
        "rel_E_error",
        "rel_H_error",
        "max_energy_drift_fd",
        "max_energy_drift_admm",
        "skew_residual",
        "moment_residual",
        "coefficient_error"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in diagnostics_rows:
        writer.writerow(row)

# Save stencil summary
stencil_path = os.path.join(output_dir, "stencil_summary_R123.txt")

with open(stencil_path, "w") as f:
    for R in R_LIST:
        f.write("=" * 70 + "\n")
        f.write(f"R = {R}\n")
        f.write("=" * 70 + "\n")
        f.write("ADMM learned stencil:\n")
        f.write(str(learned[R]["w_admm"]) + "\n\n")
        f.write("Standard FD stencil:\n")
        f.write(str(learned[R]["w_fd"]) + "\n\n")
        f.write(f"Skew residual: {learned[R]['skew_res']:.16e}\n")
        f.write(f"Moment residual: {learned[R]['mom_res']:.16e}\n")
        f.write(f"Coefficient relative error: {learned[R]['coeff_err']:.16e}\n")
        f.write(f"ADMM iterations: {learned[R]['iterations']}\n\n")

print("\n" + "=" * 70)
print("All experiments completed.")
print("Time evolution used: matrix exponential exact-in-time evolution.")
print("Figures saved in:", output_dir)
print("Diagnostics CSV:", csv_path)
print("Stencil summary:", stencil_path)
print("=" * 70)