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
