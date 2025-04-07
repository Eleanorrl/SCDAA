# SCDAA

SCDAA is a research and experimental project focused on implementing and comparing various algorithms in the actor-critic family for solving control problems. The project includes implementations of:

- **Actor-Critic**: Combines policy-based (actor) and value-based (critic) methods.
- **Actor-Only**: Uses a policy-gradient approach where the actor is solely responsible for decision making.
- **Critic-Only**: Focuses on value estimation to guide learning.
- **Soft LQR**: Implements a soft variant of the Linear Quadratic Regulator (LQR) and its solver.
- **lqr-solver**: A module for solving the LQR problem using a soft constraint approach.
- **SCDAA-Zheyan Lu.ipynb**: A Jupyter Notebook that demonstrates experiments, visualizations, and comparisons of the different approaches.

> **Note for Teachers:** Please refer to the Jupyter Notebook file `SCDAA-Zheyan Lu.ipynb` to run the code, reproduce the results, and generate the graphs.

---

## Overview

SCDAA provides a flexible codebase for exploring different reinforcement learning approaches within control systems. This repository is ideal for researchers and educators alike, offering:
 
- Multiple algorithm implementations for performance comparison.
- Tools for solving classical control problems like LQR with soft constraints.
- An interactive environment via Jupyter Notebook for visualizing results and tweaking experimental parameters.

---

### Reference
- D.Siˇska. Continuous-time relaxed entropy regularized control, policy gradient and LQR. https://www.maths.ed.ac.uk/∼dsiska/relaxed control and pol grad.pdf.2024.
- G. dos Reis and D.Siˇska. Stochastic Control and Dynamic Asset Allocation. https://www.maths.ed.ac.uk/∼dsiska/LecNotesSCDAA.pdf. 2021.
