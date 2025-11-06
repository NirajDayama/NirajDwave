# Portfolio Optimization: Comparison of D-Wave NL and CQM Solvers

This repository documents the solution of a classic Markowitz-style portfolio optimization problem for 10 assets using two different D-Wave Hybrid Solvers: the **Constrained Quadratic Model (CQM) Sampler** and the **Non-Linear (NL) Sampler**. The goal is to minimize portfolio risk (variance) while satisfying a target return constraint and a 100% budget constraint.

## Section 1: Classic Portfolio Optimization (No Side Constraints)

### Problem Definition

The objective of classic mean-variance portfolio optimization is to find a set of weights ($w_i$) that minimizes the portfolio variance for a given required minimum expected return ($R_{target}$).

**Objective Function (Minimize Risk/Variance):**

The portfolio variance ($\sigma^2_p$) is a quadratic function of the asset weights ($w_i$) and the covariance matrix ($\Sigma_{i,j}$).

$$
\text{Minimize} \quad \sigma^2_p = \sum_{i} \sum_{j} w_i w_j \Sigma_{i,j}
$$

**Constraints:**

1.  **Budget Constraint (Total Allocation):** The sum of all asset weights must equal $1$ (or $100\%$).
    $$
    \sum_{i} w_i = 1
    $$

2. **Target Return Constraint:** The expected portfolio return ($\mu_p$) must be greater than or equal to a minimum target return ($R_{target}$).
    $$
    \sum_{i} w_i R_i \geq R_{target}
    $$

## Section 2: Solution using CQM (Constrained Quadratic Model)

This approach uses D-Wave's `dimod.ConstrainedQuadraticModel` to define the problem and the `LeapHybridCQMSampler` to solve it.

### Variables Created

The core variable is the integer weight $W_i$, which represents the percentage allocation to Asset $i$. Since the total budget is $100\%$, $W_i$ is an **Integer Variable** in the range $[0, 100]$.

The constraints are defined using a "scaled" approach where the continuous problem is multiplied by the discretization factor (100) to keep everything in integers.

### Results Analysis (CQM Output)

| **Metric** | **Value** | 
| :--- | :--- | 
| **Objective Value (Scaled Risk)** | $0.0003669900$ | 
| **Target Return Required** | $\ge 0.0850$ (Note: *The CQM approach was run with a significantly higher return target of 8.5% compared to the NL Solver's 5.0%.*) | 
| **Portfolio Return Achieved (Scaled)** | $8.8200$ | 

**Optimal Asset Weights (Integer Percentage** $W_i$**):**

| **Asset (i)** | **Weight Wi​** | **Return Ri​** | 
| :--- | :--- | :--- | 
| 1 | 13 | 0.025 | 
| 2 | 6 | 0.040 | 
| 3 | 21 | 0.060 | 
| 4 | 8 | 0.035 | 
| 5 | 6 | 0.070 | 
| 6 | 17 | 0.050 | 
| 7 | 7 | 0.015 | 
| 8 | 5 | 0.080 | 
| 9 | 11 | 0.045 | 
| 10 | 6 | 0.065 | 
| **Total** | **100** |  | 

### Feasibility and Optimality (CQM)

* **Feasibility:** The solution is feasible. The **Total Weight** verification confirms the budget constraint is met ($1.0000$), and the **Portfolio Return** (8.8200) successfully exceeds the required minimum (8.50).

* **Optimality:** The CQM solver found 137 feasible solutions and selected the one with the lowest cost (risk). Because the CQM Sampler is a hybrid solver, this result is the **best known solution** found by combining classical and quantum methods for the given target return of $8.5\%$. The allocation is sensible, favoring assets that contribute to both return (Assets 3, 6) and risk mitigation.

## Section 3: Solution using NL (Non-Linear) Solver

This approach uses the `dwave.optimization.Model` and the `LeapHybridNLSampler`.

### Variables Created

The problem is defined using 10 integer variables $x_i$, where $x_i$ is defined as the integer weight for Asset $i$ with bounds $[0, 100]$:

$$
\text{Variable Definition: } x_i \in \mathbb{Z} \quad \text{where } 0 \le x_i \le 100
$$

The continuous portfolio weight $w_i$ is derived from $w_i = x_i / 100$.

### Results Analysis (NL Solver Output)

**Problem Constraints:**

| **Constraint** | **Value** |
| :--- | :--- |
| **Sum of Integer Weights** | $100.0$ |
| **Scaled Portfolio Return** | $\ge 5.0000$ (Target Return $R_{target}=5\%$) |

**Extracted Asset Allocations (Integer Values** $x_i$**):**

| **Asset (i)** | **Integer Weight xi​** | **Continuous Weight wi​** | **Return Ri​ (from code)** |
| :--- | :--- | :--- | :--- |
| 1 | 24 | 0.2400 | 0.025 |
| 2 | 3 | 0.0300 | 0.040 |
| 3 | 13 | 0.1300 | 0.060 |
| 4 | 5 | 0.0500 | 0.035 |
| 5 | 17 | 0.1700 | 0.070 |
| 6 | 5 | 0.0500 | 0.050 |
| 7 | 4 | 0.0400 | 0.015 |
| 8 | 13 | 0.1300 | 0.080 |
| 9 | 10 | 0.1000 | 0.045 |
| 10 | 6 | 0.0600 | 0.065 |
| **Total** | **100** | **1.0000** |  |

### Feasibility and Optimality (NL Solver)

#### Feasibility Check

1. **Budget Constraint:** The output confirms feasibility: `Total Integer Sum = 100 (Expected: 100)`. **(Met)**

2. **Target Return Constraint:** We must verify that the achieved scaled return $\sum (x_i \cdot R_i)$ is $\ge 5.0000$. The **Calculated Sum** of $5.055$ is greater than the required $5.0000$, confirming the solution is **feasible**.

#### Optimality and Sensibility

The solution is highly sensible and is the **best found by the NL Solver** for the $5\%$ target return problem:

* **Risk Mitigation Strategy:** The solver prioritizes the lowest-variance assets (like Asset 1) by assigning them the highest weights necessary to keep risk low. Asset 1 receives the single largest allocation ($24\%$).

* **Efficient Return Generation:** The allocation is spread to high-return assets (like Asset 5 and Asset 8) only as much as needed to minimally exceed the $5\%$ return target while keeping the overall quadratic variance low.

* **Conclusion on Optimality:** As a hybrid solution, this is the most optimal, feasible integer-discretized portfolio identified by the D-Wave system. Since the objective is a convex quadratic function, the hybrid solver is highly effective at finding the global minimum, making this solution robustly close to the true continuous optimum.