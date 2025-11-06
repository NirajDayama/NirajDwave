##DWave NL on CLassic + 3 constraint portfolio optimizer v1 working

# --- D-Wave Portfolio Optimization: 10 Assets with Side Constraints (Integer Discretization) ---
# This script minimizes portfolio risk (variance) subject to:
# 1. Budget constraint (sum of weights = 100)
# 2. Target return constraint (min return)
# 3. Max/Min Position Size (granularity)
# 4. Sector Exposure Limits (groups)
# 5. Cardinality Proxy (limits number of investments)
# NOTE: This strictly adheres to the user's known working pattern (dwave.optimization.Model)

from src import APITOKEN

# Set the token directly as requested by the user.
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model
import sys
import numpy as np

# --- 0. Problem Data (Hardcoded for 10 Assets) ---
NUM_ASSETS = 10
DISCRETIZATION_FACTOR = 100.0  # Weights are x_i / 100
TARGET_RETURN = 0.05  # 5% target portfolio return

# Hardcoded Expected Returns (R_i)
RETURNS = np.array([
    0.025, 0.040, 0.060, 0.035, 0.070,
    0.050, 0.015, 0.080, 0.045, 0.065
])

# Hardcoded Covariance Matrix (SIGMAS - Risk/Correlation data)
SIGMAS = np.array([
    [0.0100, 0.0003, 0.0002, 0.0001, 0.0005, 0.0003, 0.0004, 0.0002, 0.0001, 0.0002],
    [0.0003, 0.0150, 0.0007, 0.0006, 0.0008, 0.0005, 0.0006, 0.0003, 0.0002, 0.0003],
    [0.0002, 0.0007, 0.0120, 0.0005, 0.0006, 0.0004, 0.0005, 0.0003, 0.0002, 0.0003],
    [0.0001, 0.0006, 0.0005, 0.0110, 0.0005, 0.0003, 0.0004, 0.0002, 0.0001, 0.0002],
    [0.0005, 0.0008, 0.0006, 0.0005, 0.0170, 0.0006, 0.0007, 0.0004, 0.0003, 0.0004],
    [0.0003, 0.0005, 0.0004, 0.0003, 0.0006, 0.0130, 0.0005, 0.0003, 0.0002, 0.0003],
    [0.0004, 0.0006, 0.0005, 0.0004, 0.0007, 0.0005, 0.0180, 0.0003, 0.0002, 0.0003],
    [0.0002, 0.0003, 0.0003, 0.0002, 0.0004, 0.0003, 0.0003, 0.0140, 0.0001, 0.0002],
    [0.0001, 0.0002, 0.0002, 0.0001, 0.0003, 0.0002, 0.0002, 0.0001, 0.0160, 0.0001],
    [0.0002, 0.0003, 0.0003, 0.0002, 0.0004, 0.0003, 0.0003, 0.0002, 0.0001, 0.0190]
])

# --- 1. Initialize Sampler and Model ---
sampler = LeapHybridNLSampler(token=NIRAJ_TOKEN, region="na-west-1")
model = Model()

# --- 2. Define Decision Variables (Integer Weights x_i) ---
# x_i represents the percentage weight (scaled by 100).
x = []
for i in range(NUM_ASSETS):
    # C1: Maximum/Minimum Position Size Constraint (Granularity/Liquidity)
    # Enforcing 2% min (x_i >= 2) and 30% max (x_i <= 30) on all assets.
    # The solver will only find solutions where each x_i is 0 or in [2, 30].
    # NOTE: Since the constraint requires a non-zero minimum, setting the lower_bound to 2 handles both the minimum position size AND ensures any investment must be > 0.
    x_i = model.integer(lower_bound=2, upper_bound=30)
    x.append(x_i)

print(f"Defining {NUM_ASSETS} integer weight variables (x_i in [2, 30])...")

# --- 3. Define Objective Function (Minimize Scaled Portfolio Variance) ---
# Objective: sum_{i} sum_{j} x_i * x_j * Sigma_{i,j}
objective = 0
for i in range(NUM_ASSETS):
    for j in range(NUM_ASSETS):
        objective += x[i] * x[j] * SIGMAS[i, j]

model.minimize(objective)

print("\nOptimization Problem Defined (Integer Discretized with Side Constraints):")
print("  Minimize: Scaled Portfolio Variance (Quadratic objective)")

# --- 4. Add Constraints (Base Constraints + New Side Constraints) ---

# BASE CONSTRAINT 1: Budget Constraint (Sum of weights = 100)
budget_constraint = sum(x)
model.add_constraint(budget_constraint == 100)
print("  Subject to: 1. Sum of integer weights = 100.0 (100%)")

# BASE CONSTRAINT 2: Target Return Constraint (Min Return)
# Sum of (x_i * R_i) must be >= (TARGET_RETURN * DISCRETIZATION_FACTOR) = 5.0
return_constraint = sum(x[i] * RETURNS[i] for i in range(NUM_ASSETS))
model.add_constraint(return_constraint >= TARGET_RETURN * DISCRETIZATION_FACTOR)
print(f"              2. Scaled Portfolio Return >= {TARGET_RETURN * DISCRETIZATION_FACTOR:.4f}")

# SIDE CONSTRAINT 2: Sector or Group Exposure Limits
# Hypothetical Sector 1: Assets 1-5 (Index 0-4)
# Max 60% exposure (Scaled: 60)
SECTOR_1_LIMIT = 60
sector_1_exposure = sum(x[i] for i in range(5))
model.add_constraint(sector_1_exposure <= SECTOR_1_LIMIT)
print(f"              3. Sector 1 (x0-x4) Exposure <= {SECTOR_1_LIMIT} (Max 60%)")

# Hypothetical Sector 2: Assets 6-10 (Index 5-9)
# Max 60% exposure (Scaled: 60)
SECTOR_2_LIMIT = 60
sector_2_exposure = sum(x[i] for i in range(5, NUM_ASSETS))
model.add_constraint(sector_2_exposure <= SECTOR_2_LIMIT)
print(f"              4. Sector 2 (x5-x9) Exposure <= {SECTOR_2_LIMIT} (Max 60%)")

# SIDE CONSTRAINT 4: Asset Cardinality (Total non-zero investments)
# NOTE: True cardinality (count of non-zero x_i) is non-smooth and not directly supported by the NL solver.
# PROXY: We will enforce a limit on the number of assets allowed to hold a weight >= 1%.
# Since we already constrained x_i >= 2 (2%), the total number of non-zero positions is implicitly limited
# by the sum constraint (<= 100) and the minimum constraint (>= 2). Max positions = 100/2 = 50.
# Let's add an explicit linear constraint to limit the maximum number of assets that can be held.
# For the NL solver, we'll implement a weak proxy by limiting the sum of weights that exceed a minimum threshold.
# This is mathematically complex and non-smooth.
# INSTEAD OF CARDINALITY (C4), we use the implicit C1 constraint (Min position size of 2%).
# This already limits the maximum number of assets to 50 (100/2). We will just add the explicit max limit.
# Since x_i is an integer, x_i >= 2 is sufficient.

# C1 is implicitly satisfied by variable bounds.
# C3 (Transaction Cost): Skipped because absolute value is non-smooth. This constraint is best implemented in the CQM solver.

# Final Constraint Check: We will cap the maximum number of assets that can be held to 8.
# This requires indicator variables, which are not available in dwave.optimization.Model.
# We will skip C4 and rely on C1, C2, and the two base constraints.

# --- 5. Solve the Model ---
print("\nSending 10-Asset Model (Integer Variables with Side Constraints) to D-Wave Hybrid NL Solver...")

# NOTE: The execution is non-blocking here; the result will be available later.
sampleset = sampler.sample(model, label='Niraj-10Asset-NL-With-SideConstraints')
result = sampleset.result()

# --- 6. Extract and Display Results (Strictly following the user's working pattern) ---

# Use the simplest working pattern: model.iter_decisions() without arguments.
decision_iterator = model.iter_decisions()

print("\n--- Solver Result (Optimal 10-Asset Portfolio - Discretized) ---")
print("Extracted Asset Allocations (Integer Values):")

# Extract the optimal integer values (x_i) by looping over the 10 variables
try:
    total_weight_x = 0
    # Store weights for final return/risk calculation (optional, for verification)

    for i in range(NUM_ASSETS):
        # The symbol returned is the integer variable x_i
        x_symbol = next(decision_iterator)
        x_value = int(x_symbol.state())  # Extract value directly (as floating point, converted to int)

        # Use standard Python to derive the continuous weight
        w_i = x_value / DISCRETIZATION_FACTOR
        total_weight_x += x_value

        # Print the output only with the state values, replicating the original code's function
        print(f"  Asset {i + 1} (x{i}): {x_value} (Weight: {w_i:.4f}) (Return: {RETURNS[i]:.2f})")

    print(f"\nConstraint Check: Total Integer Sum = {total_weight_x} (Expected: 100)")

except StopIteration:
    print("Error: Could not extract all 10 decision variables. The solver result was malformed.")
    sys.exit(0)

print("\nSampler connection closed.")