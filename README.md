# Energy-Stable Maxwell Stencil Learning

This repository contains the reproducibility code and numerical results for the manuscript

**“An Energy-Stable Approach for Learning Derivative Operators from Noisy Data for Maxwell’s Equations.”**

The code learns compact spatial derivative stencils for a one-dimensional Maxwell system while enforcing the skew-adjoint structure required for discrete energy conservation.

The repository compares three approaches:

- **FD:** classical centered finite-difference stencils,
- **ADMM:** constrained stencil learning in the full coefficient space,
- **SP-ADMM:** a skew-parameterized structure-preserving ADMM formulation.

In SP-ADMM, skew-adjointness is enforced by construction through a reduced parameterization of the stencil coefficients.

---

## Repository structure

```text
energy-stable-maxwell-stencil-learning/
│
├── README.md
├── requirements.txt
│
├── scripts/
│   └── main.py
│
└── results_stencil_learning/
    ├── *.csv
    └── *.png

The main experiment script is
scripts/main.py

and the generated numerical tables and figures are stored in
results_stencil_learning/

Requirements
The experiments require Python 3 and the following packages:
numpy
scipy
pandas
matplotlib

Install the dependencies with
pip install -r requirements.txt

Running the experiments
From the root of the repository, run
python scripts/main.py

The script creates the output directory automatically if it does not already exist:
results_stencil_learning/

All CSV tables and figures used for the numerical study are written to this directory.
Experiments
The main script reproduces the complete numerical study:
1. Finite-difference recovery check
2. Clean hidden-operator learning
3. Noise robustness
4. Generalization across initial conditions
5. Performance across multiple hidden skew-adjoint operators
6. Stencil-radius study
7. Training-set-size and runtime study
8. Regularization study
9. Constraint ablation
10. Long-time propagation and energy behavior
11. Direct FD/ADMM/SP-ADMM comparison
12. Layered-medium Maxwell propagation
13. ADMM versus SP-ADMM convergence study
14. Multi-seed robustness study
15. Radius-four SP-ADMM convergence diagnostic
Reproducibility settings
Unless otherwise stated, the main experiments use
N = 128
L = 1
T_final = 0.02
training samples = 200
maximum Fourier mode = 18
lambda = 1e-8
rho = 10
box bound = 250
default ADMM iteration budget = 300
stopping tolerance = 1e-10

The principal hidden radius-three stencil is
[-6.2222, 19.5556, -84.4444, 0.0, 84.4444, -19.5556, 6.2222]

with entries ordered as
(w_-3, w_-2, w_-1, w_0, w_1, w_2, w_3).

The hidden stencil is used only to generate synthetic derivative targets and is not supplied to the learning algorithms.
Multi-seed study
To evaluate sensitivity to random training-field and noise realizations, Experiment 14 repeats the stochastic studies using ten independent seed offsets:
11, 22, 33, 44, 55, 66, 77, 88, 99, 111

The repository includes both the raw trial data and summary statistics containing the mean, sample standard deviation, and 95% confidence interval.
ADMM and SP-ADMM comparison
ADMM and SP-ADMM optimize the same skew-adjoint stencil class when the regularization terms are normalized consistently.
The distinction is computational:
- ADMM works with the full 2R+1 coefficient vector and imposes skew-adjointness through equality constraints and a KKT system.
- SP-ADMM works directly with the R independent positive-side coefficients.
- The SP-ADMM implementation also uses over-relaxation and adaptive penalty updates.
Experiment 13 verifies that the two methods approach the same constrained solution when sufficiently converged and that differences observed at a fixed iteration budget are convergence effects rather than different limiting solutions.
Radius-four diagnostic
Experiment 15 investigates the larger SP-ADMM coefficient error observed for R = 4 under the default 300-iteration budget.
The diagnostic shows that the error decreases steadily as the iteration budget is increased and that the SP-ADMM solution approaches the direct reduced optimum. Thus, the observed R = 4 discrepancy is a finite-iteration convergence effect rather than failure of the skew-parameterized formulation.
Energy conservation
For periodic convolution operators, an antisymmetric stencil satisfies
D^T = -D,

which yields discrete energy conservation for the semi-discrete Maxwell system.
The numerical experiments use matrix-exponential time evolution to isolate spatial-operator error. Energy variations near 1e-15–1e-16 should therefore be interpreted as floating-point roundoff rather than meaningful differences in conservation quality.
Output files
Each experiment produces one or more:
- .csv files containing numerical results,
- .png figures corresponding to the manuscript plots.
Examples include
experiment_13_admm_spadmm_iteration_convergence.csv
experiment_14_multiseed_noise_summary.csv
experiment_14_multiseed_hidden_operators_summary.csv
experiment_15_radius4_spadmm_convergence.csv

and their corresponding figures.
Reproducibility
The repository is intended to provide the complete code required to reproduce the numerical experiments reported in the manuscript.
If the script is run with the supplied parameter values and random seeds, it will regenerate the numerical results and figures in results_stencil_learning/.
Authors
Victory C. Obieke
Oregon State University
Ameh Emmanuel Sunday
Cornell University
