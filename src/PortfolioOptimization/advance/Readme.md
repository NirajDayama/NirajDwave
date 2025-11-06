# Portfolio Optimization with Side Constraints (10 Assets)

This repository documents the solution to a constrained Markowitz-style portfolio optimization problem solved using the **D-Wave Leap Hybrid Non-Linear (NL) Sampler**. The goal is to find the minimum-risk (minimum variance) portfolio subject to several real-world financial and regulatory constraints.

---

## 1. Problem Formulation and Objective

The problem is formulated as a Quadratic Unconstrained Binary Optimization (QUBO) analogue suitable for hybrid quantum solvers. Since the D-Wave NL Sampler natively handles non-linear constraints and integer variables, the problem is defined directly in terms of minimizing the continuous portfolio variance.

### Decision Variables

The continuous portfolio weight ($w_i$) for Asset $i$ is **discretized** into an integer variable, $x_i$, which represents the percentage allocation scaled by 100.

$$
x_i \in \mathbb{Z} \quad \text{where } 0 \le x_i \le 100
$$

The actual continuous weight is $w_i = x_i / 100$.

### Objective Function (Minimize Risk)

The solver minimizes the **Scaled Portfolio Variance** ($\sigma^2_p$):

$$
\text{Minimize} \quad \text{Objective} = \sum_{i=1}^{10} \sum_{j=1}^{10} x_i x_j \Sigma_{i,j}
$$

where $\Sigma_{i,j}$ is the covariance between assets $i$ and $j$.

---

## 2. Implemented Constraints

In addition to the standard budget and return constraints, four crucial side constraints were enforced to reflect real-world trading limits.

| Constraint | Type | Description | Formulation Detail |
| :--- | :--- | :--- | :--- |
| **C1** | Budget | Total allocation must be 100%. | $$\sum_{i=1}^{10} x_i = 100$$ |
| **C2** | Min. Return | Portfolio return must meet a target. | $$\sum_{i=1}^{10} x_i R_i \geq 5.0000 \quad (R_{\text{target}}=5\%)$$ |
| **C3** | Position Size | Limits on the size of any single investment. | **Enforced by Variable Bounds:** $2 \le x_i \le 30$ |
| **C4a/C4b** | Sector Limits | Max exposure to specific market segments. | **Sector 1 (Assets 1-5):** $\sum_{i=1}^{5} x_i \le 60$ |
| **C4b/C4b** | Sector Limits | | **Sector 2 (Assets 6-10):** $\sum_{i=6}^{10} x_i \le 60$ |

---

## 3. Execution Summary

The model was built using `dwave.optimization.Model` and submitted to the **LeapHybridNLSampler**. The detailed log for the model setup is shown below: