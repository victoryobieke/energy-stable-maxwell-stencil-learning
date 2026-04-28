#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import expm_multiply

# ============================================================
# Convergence study for ADMM-learned Maxwell stencils
#
# More complicated exact Maxwell solution:
#
#   E(x,t) = F(x-t) + G(x+t)
#   H(x,t) = -F(x-t) + G(x+t)
#
# Produces:
#   1. Exact initial/final 1D plot
#   2. Horizontal final-time profile plot for R=1,2,3
#   3. Separate exact 2D space-time heatmap
#   4. Horizontal ADMM R=1,2,3 2D space-time heatmaps
#   5. Horizontal energy-error plot for R=1,2,3
#   6. Convergence plots
#   7. Convergence CSV
# ============================================================

np.random.seed(7)

# -----------------------------
# Global parameters
# -----------------------------
L = 1.0
T = 0.25

R_LIST = [1, 2, 3]
N_LIST = [64, 128, 256, 512]

n_sims = 300
m_max_train = 12

lam = 1.0e-6
rho = 1.0
max_admm_iter = 300
tol = 1.0e-12

ENFORCE_MOMENT_CONSTRAINTS = True
SHOW_FIGURES = True

output_dir = "Picture_convergence_complicated_exact"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 1. Spectral derivative for training data
# ============================================================

def spectral_derivative(u, L):
    """
    Compute du/dx spectrally on a periodic grid.
    """
    N = len(u)
    dx = L / N
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return np.fft.ifft(1j * k * np.fft.fft(u)).real


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


def local_patch(u, i, R):
    """
    Periodic local patch:
        (u_{i-R}, ..., u_i, ..., u_{i+R}).
    """
    N = len(u)
    return np.array([u[(i + ell) % N] for ell in range(-R, R + 1)])


def build_training_system(n_sims, x, L, m_max, R):
    """
    Build least-squares system A w ≈ b.

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
# 3. Constraints
# ============================================================

def build_constraints(R, dx, enforce_moments=False):
    """
    Stencil ordering:
        [w_{-R}, ..., w_0, ..., w_R]

    Always enforce skew-symmetry:
        w_0 = 0,
        w_{-ell} + w_{ell} = 0.

    Optionally enforce moment conditions for formal 2R order:
        sum_l l w_l = 1/dx,
        sum_l l^q w_l = 0, q=3,5,...,2R-1.
    """
    n = 2 * R + 1
    offsets = np.arange(-R, R + 1)

    rows = []
    rhs = []

    # Center coefficient w_0 = 0
    row = np.zeros(n)
    row[R] = 1.0
    rows.append(row)
    rhs.append(0.0)

    # Pairwise skew symmetry
    for ell in range(1, R + 1):
        row = np.zeros(n)
        row[R - ell] = 1.0
        row[R + ell] = 1.0
        rows.append(row)
        rhs.append(0.0)

    if enforce_moments:
        # First derivative consistency
        row = offsets.astype(float)
        rows.append(row)
        rhs.append(1.0 / dx)

        # Higher odd moments vanish
        for q in range(3, 2 * R, 2):
            row = offsets.astype(float) ** q
            rows.append(row)
            rhs.append(0.0)

    C = np.vstack(rows)
    d = np.array(rhs)

    return C, d


# ============================================================
# 4. ADMM solver
# ============================================================

def solve_admm(A, b, C, d, lam, M, rho, max_iter, tol):
    """
    Solve

        min_w 0.5 ||Aw-b||^2 + lam/2 ||w||^2
        s.t.  Cw = d,
              -M <= w <= M,

    using ADMM.
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

        z = np.clip(w + u, -M, M)
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
# 5. Standard centered FD stencils
# ============================================================

def standard_fd_stencil(R, dx):
    """
    Standard centered FD derivative stencils.

    R=1 gives 2nd order.
    R=2 gives 4th order.
    R=3 gives 6th order.
    """
    if R == 1:
        coeff = np.array([-1, 0, 1], dtype=float) / 2.0

    elif R == 2:
        coeff = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0

    elif R == 3:
        coeff = np.array([-1, 9, -45, 0, 45, -9, 1], dtype=float) / 60.0

    else:
        raise ValueError("Only R=1,2,3 are implemented.")

    return coeff / dx


# ============================================================
# 6. Sparse derivative matrix
# ============================================================

def derivative_matrix_sparse(w, R, N):
    """
    Build sparse periodic convolution derivative matrix D_w.
    """
    rows = []
    cols = []
    vals = []

    for i in range(N):
        for p, ell in enumerate(range(-R, R + 1)):
            j = (i + ell) % N
            rows.append(i)
            cols.append(j)
            vals.append(w[p])

    return csr_matrix((vals, (rows, cols)), shape=(N, N))


# ============================================================
# 7. More complicated exact Maxwell solution
# ============================================================

def F_profile(s, L):
    """
    Right-moving smooth periodic wave packet.
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
    Left-moving smooth periodic wave packet.
    """
    x2 = 0.70 * L
    kappa2 = 16.0

    envelope2 = np.exp(
        kappa2 * (np.cos(2.0 * np.pi * (s - x2) / L) - 1.0)
    )

    return 0.35 * envelope2 * np.sin(8.0 * np.pi * s / L + 0.5)


def exact_solution(x, t, L):
    """
    Exact solution of the normalized 1D Maxwell system

        E_t = H_x,
        H_t = E_x.

    General form:
        E(x,t) = F(x-t) + G(x+t),
        H(x,t) = -F(x-t) + G(x+t).
    """
    F = F_profile(x - t, L)
    G = G_profile(x + t, L)

    E = F + G
    H = -F + G

    return E, H


# ============================================================
# 8. Semi-discrete exact-in-time evolution
# ============================================================

def evolve_semidiscrete_exact(D, E0, H0, T):
    """
    Exact time integration of U_t = A U using expm_multiply.

    U = [H; E],
    A = [[0, D],
         [D, 0]].
    """
    N = len(E0)
    Z = csr_matrix((N, N))

    Aop = bmat([[Z, D], [D, Z]], format="csr")
    U0 = np.concatenate([H0, E0])

    UT = expm_multiply(T * Aop, U0)

    H_T = UT[:N]
    E_T = UT[N:]

    return E_T, H_T


def evolve_semidiscrete_exact_history(D, E0, H0, T, num_times=201):
    """
    Exact-in-time semi-discrete evolution at many time levels.

    Returns:
        times, E_hist, H_hist
    """
    N = len(E0)
    Z = csr_matrix((N, N))

    Aop = bmat([[Z, D], [D, Z]], format="csr")
    U0 = np.concatenate([H0, E0])

    times = np.linspace(0.0, T, num_times)

    U_hist = expm_multiply(
        Aop,
        U0,
        start=0.0,
        stop=T,
        num=num_times,
        endpoint=True
    )

    H_hist = U_hist[:, :N]
    E_hist = U_hist[:, N:]

    return times, E_hist, H_hist


# ============================================================
# 9. Error, rates, and energy
# ============================================================

def rel_l2_error(u, u_exact, dx):
    """
    Relative discrete L2 error.
    """
    num = np.sqrt(dx * np.sum((u - u_exact) ** 2))
    den = np.sqrt(dx * np.sum(u_exact ** 2))
    return num / den


def compute_rates(errors):
    """
    Compute grid-refinement convergence rates.
    """
    rates = [np.nan]

    for j in range(1, len(errors)):
        rates.append(np.log(errors[j - 1] / errors[j]) / np.log(2.0))

    return rates


def compute_energy_history(E_hist, H_hist, dx):
    """
    Compute discrete electromagnetic energy over time:
        E^n = 0.5 dx sum_i (E_i^2 + H_i^2).
    """
    return 0.5 * dx * np.sum(E_hist**2 + H_hist**2, axis=1)


# ============================================================
# 10. Plotting helpers
# ============================================================

def save_exact_profile_initial_final(output_dir):
    """
    Plot exact E and H at t=0 and t=T on a fine grid.
    """
    N_plot = 1000
    x_plot = np.linspace(0.0, L, N_plot, endpoint=False)

    E0, H0 = exact_solution(x_plot, 0.0, L)
    ET, HT = exact_solution(x_plot, T, L)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(x_plot, E0, linewidth=2, label=r"$E(x,0)$")
    axes[0].plot(x_plot, H0, "--", linewidth=2, label=r"$H(x,0)$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("field")
    axes[0].set_title("Exact solution at t=0")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].plot(x_plot, ET, linewidth=2, label=r"$E(x,T)$")
    axes[1].plot(x_plot, HT, "--", linewidth=2, label=r"$H(x,T)$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("field")
    axes[1].set_title("Exact solution at final time")
    axes[1].legend()
    axes[1].grid(False)

    fname = os.path.join(output_dir, "exact_solution_initial_final.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_final_profiles_exact_solution_R123(x, E_exact_T, E_admm_by_R, E_fd_by_R, output_dir):
    """
    Horizontal final-time profile plot for R=1,2,3.

    Uses the style:
        Standard FD solid
        ADMM learned dashed
        Exact black dotted
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, R in zip(axes, R_LIST):
        ax.plot(x, E_fd_by_R[R], linewidth=2, label="Standard FD")
        ax.plot(x, E_admm_by_R[R], "--", linewidth=2, label="ADMM learned")
        ax.plot(x, E_exact_T, ":", linewidth=2.5, label="Exact")

        ax.set_title(f"R = {R}")
        ax.set_xlabel("x")
        ax.set_ylabel(r"$E(x,T)$")
        ax.legend()
        ax.grid(False)

    fig.suptitle("Final-time electric field: exact Maxwell solution")

    fname = os.path.join(output_dir, "final_profiles_exact_solution_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_spacetime2D_exact_solution_R123(x, times, E_admm_hist_by_R, output_dir):
    """
    Horizontal 2D ADMM space-time plot for R=1,2,3.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    vmin = min(np.min(E_admm_hist_by_R[R]) for R in R_LIST)
    vmax = max(np.max(E_admm_hist_by_R[R]) for R in R_LIST)

    for ax, R in zip(axes, R_LIST):
        E_hist = E_admm_hist_by_R[R]

        im = ax.imshow(
            E_hist,
            aspect="auto",
            origin="lower",
            extent=[0.0, L, times[0], times[-1]],
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"R = {R}")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("Space-time electric field: exact Maxwell solution")

    fname = os.path.join(output_dir, "spacetime2D_exact_solution_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_spacetime2D_exact_E(x, times, E_exact_hist, output_dir):
    """
    Separate exact 2D space-time plot.
    """
    fig = plt.figure(figsize=(6.5, 4.5))

    im = plt.imshow(
        E_exact_hist,
        aspect="auto",
        origin="lower",
        extent=[0.0, L, times[0], times[-1]]
    )

    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Exact space-time electric field")
    plt.colorbar(im, shrink=0.85)
    plt.grid(False)

    fname = os.path.join(output_dir, "spacetime2D_exact_E.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_energy_errors_exact_solution_R123(times, energy_by_R, output_dir):
    """
    Horizontal energy error plot for R=1,2,3.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    for ax, R in zip(axes, R_LIST):
        energy_fd = energy_by_R[R]["FD"]
        energy_admm = energy_by_R[R]["ADMM"]

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
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\mathcal{E}^n-\mathcal{E}^0$")
        ax.legend()
        ax.grid(False)

    fig.suptitle("Energy error: exact Maxwell solution")

    fname = os.path.join(output_dir, "energy_errors_exact_solution_R123.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_convergence_plot(R, dx_values, err_E_learned, err_E_fd, output_dir):
    """
    Plot convergence curve for one R.
    """
    plt.figure(figsize=(9, 6))

    plt.loglog(
        dx_values,
        err_E_learned,
        "o-",
        linewidth=3,
        markersize=8,
        label="ADMM learned"
    )

    plt.loglog(
        dx_values,
        err_E_fd,
        "s--",
        linewidth=3,
        markersize=8,
        label=f"Standard FD, R={R}"
    )

    ref_order = 2 * R
    ref = err_E_fd[0] * (np.array(dx_values) / dx_values[0]) ** ref_order

    plt.loglog(
        dx_values,
        ref,
        ":",
        linewidth=3,
        label=f"Reference slope {ref_order}"
    )

    plt.gca().invert_xaxis()
    plt.xlabel(r"$\Delta x$", fontsize=16)
    plt.ylabel(r"Relative $L^2$ error in $E(\cdot,T)$", fontsize=16)
    plt.title(f"Spatial convergence, R={R}", fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    fname = os.path.join(output_dir, f"convergence_R{R}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()


# ============================================================
# 11. Main convergence loop
# ============================================================

save_exact_profile_initial_final(output_dir)

# Storage for finest-grid plots
E_admm_T_by_R = {}
E_fd_T_by_R = {}
E_exact_T_saved = None
x_final_saved = None

E_admm_hist_by_R = {}
E_exact_hist_saved = None
times_plot_saved = None
x_plot_saved = None

energy_by_R = {}

all_rows = []

for R in R_LIST:
    print("\n" + "=" * 70)
    print(f"Convergence test for R = {R}")
    print("=" * 70)

    dx_values = []
    err_E_learned = []
    err_H_learned = []
    err_E_fd = []
    err_H_fd = []

    for N in N_LIST:
        dx = L / N
        x = np.arange(N) * dx

        print(f"\nTraining and testing N = {N}, dx = {dx:.6e}")

        A, b = build_training_system(n_sims, x, L, m_max_train, R)

        M_box = 10.0 / dx

        C, d = build_constraints(
            R,
            dx,
            enforce_moments=ENFORCE_MOMENT_CONSTRAINTS
        )

        w_learned, hist = solve_admm(
            A,
            b,
            C,
            d,
            lam=lam,
            M=M_box,
            rho=rho,
            max_iter=max_admm_iter,
            tol=tol
        )

        w_fd = standard_fd_stencil(R, dx)

        D_learned = derivative_matrix_sparse(w_learned, R, N)
        D_fd = derivative_matrix_sparse(w_fd, R, N)

        E0, H0 = exact_solution(x, 0.0, L)
        E_exact_T, H_exact_T = exact_solution(x, T, L)

        E_learned_T, H_learned_T = evolve_semidiscrete_exact(
            D_learned,
            E0,
            H0,
            T
        )

        E_fd_T, H_fd_T = evolve_semidiscrete_exact(
            D_fd,
            E0,
            H0,
            T
        )

        # Store only finest grid data for final profile, 2D, and energy plots
        if N == max(N_LIST):
            E_admm_T_by_R[R] = E_learned_T
            E_fd_T_by_R[R] = E_fd_T

            if E_exact_T_saved is None:
                E_exact_T_saved = E_exact_T
                x_final_saved = x.copy()

            num_times_plot = 201

            times_plot, E_admm_hist, H_admm_hist = evolve_semidiscrete_exact_history(
                D_learned,
                E0,
                H0,
                T,
                num_times=num_times_plot
            )

            _, E_fd_hist, H_fd_hist = evolve_semidiscrete_exact_history(
                D_fd,
                E0,
                H0,
                T,
                num_times=num_times_plot
            )

            E_admm_hist_by_R[R] = E_admm_hist

            energy_by_R[R] = {
                "ADMM": compute_energy_history(E_admm_hist, H_admm_hist, dx),
                "FD": compute_energy_history(E_fd_hist, H_fd_hist, dx)
            }

            if E_exact_hist_saved is None:
                E_exact_hist = np.zeros_like(E_admm_hist)

                for k, tk in enumerate(times_plot):
                    E_exact_hist[k, :], _ = exact_solution(x, tk, L)

                E_exact_hist_saved = E_exact_hist
                times_plot_saved = times_plot
                x_plot_saved = x.copy()

        eE_l = rel_l2_error(E_learned_T, E_exact_T, dx)
        eH_l = rel_l2_error(H_learned_T, H_exact_T, dx)

        eE_f = rel_l2_error(E_fd_T, E_exact_T, dx)
        eH_f = rel_l2_error(H_fd_T, H_exact_T, dx)

        eq_res = np.linalg.norm(C @ w_learned - d)
        coeff_err = np.linalg.norm(w_learned - w_fd) / np.linalg.norm(w_fd)

        dx_values.append(dx)
        err_E_learned.append(eE_l)
        err_H_learned.append(eH_l)
        err_E_fd.append(eE_f)
        err_H_fd.append(eH_f)

        print("ADMM learned stencil:", w_learned)
        print("FD stencil:          ", w_fd)
        print(f"constraint residual: {eq_res:.3e}")
        print(f"coefficient rel err: {coeff_err:.3e}")
        print(f"E error learned:     {eE_l:.3e}")
        print(f"E error FD:          {eE_f:.3e}")

    rates_E_learned = compute_rates(err_E_learned)
    rates_H_learned = compute_rates(err_H_learned)
    rates_E_fd = compute_rates(err_E_fd)
    rates_H_fd = compute_rates(err_H_fd)

    for j, N in enumerate(N_LIST):
        all_rows.append({
            "R": R,
            "N": N,
            "dx": dx_values[j],
            "err_E_learned": err_E_learned[j],
            "rate_E_learned": rates_E_learned[j],
            "err_H_learned": err_H_learned[j],
            "rate_H_learned": rates_H_learned[j],
            "err_E_fd": err_E_fd[j],
            "rate_E_fd": rates_E_fd[j],
            "err_H_fd": err_H_fd[j],
            "rate_H_fd": rates_H_fd[j],
        })

    save_convergence_plot(
        R,
        dx_values,
        err_E_learned,
        err_E_fd,
        output_dir
    )


# ============================================================
# 12. Save requested plots
# ============================================================

save_final_profiles_exact_solution_R123(
    x_final_saved,
    E_exact_T_saved,
    E_admm_T_by_R,
    E_fd_T_by_R,
    output_dir
)

save_spacetime2D_exact_E(
    x_plot_saved,
    times_plot_saved,
    E_exact_hist_saved,
    output_dir
)

save_spacetime2D_exact_solution_R123(
    x_plot_saved,
    times_plot_saved,
    E_admm_hist_by_R,
    output_dir
)

save_energy_errors_exact_solution_R123(
    times_plot_saved,
    energy_by_R,
    output_dir
)


# ============================================================
# 13. Save CSV
# ============================================================

csv_path = os.path.join(output_dir, "convergence_results.csv")

with open(csv_path, "w", newline="") as csvfile:
    fieldnames = [
        "R",
        "N",
        "dx",
        "err_E_learned",
        "rate_E_learned",
        "err_H_learned",
        "rate_H_learned",
        "err_E_fd",
        "rate_E_fd",
        "err_H_fd",
        "rate_H_fd"
    ]

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in all_rows:
        writer.writerow(row)


# ============================================================
# 14. Print summary table
# ============================================================

print("\n" + "=" * 90)
print("Convergence summary")
print("=" * 90)

for R in R_LIST:
    print(f"\nR = {R}, expected classical FD order = {2*R}")
    print("N        dx            err_E_learned     rate      err_E_fd          rate")
    print("-" * 80)

    rows_R = [row for row in all_rows if row["R"] == R]

    for row in rows_R:
        rate_l = row["rate_E_learned"]
        rate_f = row["rate_E_fd"]

        rate_l_str = "--" if np.isnan(rate_l) else f"{rate_l:.2f}"
        rate_f_str = "--" if np.isnan(rate_f) else f"{rate_f:.2f}"

        print(
            f"{row['N']:<8d} "
            f"{row['dx']:<13.6e} "
            f"{row['err_E_learned']:<17.6e} "
            f"{rate_l_str:<8s} "
            f"{row['err_E_fd']:<17.6e} "
            f"{rate_f_str:<8s}"
        )

print("\nSaved results to:", csv_path)
print("Saved figures in:", output_dir)

print("\nMain figures:")
print(os.path.join(output_dir, "final_profiles_exact_solution_R123.png"))
print(os.path.join(output_dir, "spacetime2D_exact_E.png"))
print(os.path.join(output_dir, "spacetime2D_exact_solution_R123.png"))
print(os.path.join(output_dir, "energy_errors_exact_solution_R123.png"))
print(os.path.join(output_dir, "convergence_R1.png"))
print(os.path.join(output_dir, "convergence_R2.png"))
print(os.path.join(output_dir, "convergence_R3.png"))