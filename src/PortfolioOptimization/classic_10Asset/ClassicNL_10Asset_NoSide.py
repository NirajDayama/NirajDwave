# --- D-Wave Portfolio Optimization: 10 Assets (Integer Discretization) ---
# This script minimizes portfolio risk (variance) subject to budget and target return
# constraints by DISCRETIZING the portfolio weights into integers (0 to 100).
# NOTE: This version strictly adheres to the variable extraction pattern of the user's
# known working code and avoids all objective value extraction API calls.

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
    [0.0100, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001],  # 1
    [0.0002, 0.0150, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001],  # 2
    [0.0001, 0.0002, 0.0200, 0.0002, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0002],  # 3
    [0.0002, 0.0001, 0.0002, 0.0120, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0001],  # 4
    [0.0001, 0.0002, 0.0001, 0.0001, 0.0250, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001],  # 5
    [0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0180, 0.0002, 0.0001, 0.0001, 0.0002],  # 6
    [0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0080, 0.0002, 0.0001, 0.0001],  # 7
    [0.0002, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0002, 0.0300, 0.0002, 0.0001],  # 8
    [0.0001, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0140, 0.0002],  # 9
    [0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001, 0.0002, 0.0220],  # 10
])

# --- 1. Sampler Setup ---
try:
    sampler = LeapHybridNLSampler(token=NIRAJ_TOKEN, region="na-west-1")
except Exception as e:
    if "Authentication failed" in str(e) or "access token" in str(e):
        print("Error: D-Wave Authentication failed. Please ensure APITOKEN.NIRAJ_TOKEN is correctly configured.")
    else:
        print(f"Error initializing LeapHybridNLSampler: {e}")
    sys.exit(1)

# --- 2. Model Creation ---
model = Model()

# --- 3. Define Integer Variables (Discretized Weights) ---
# x_i represents the fraction of the budget * 100. (e.g., x_i=25 means 25% allocation).
print(f"Defining {NUM_ASSETS} integer weight variables (x_i in [0, 100])...")

x = {}
for i in range(NUM_ASSETS):
    # Use model.integer() as requested
    x[i] = model.integer(lower_bound=0, upper_bound=int(DISCRETIZATION_FACTOR))

# --- 4. Objective Function: Minimize Risk (Portfolio Variance) ---
# Objective: Minimize Z = (1/D^2) * sum_{i,j} (x_i * x_j * Sigma_ij)
# We minimize the sum portion, as the factor (1/D^2) is constant.
portfolio_variance_scaled = sum(
    x[i] * x[j] * SIGMAS[i, j]
    for i in range(NUM_ASSETS)
    for j in range(NUM_ASSETS)
)
model.minimize(portfolio_variance_scaled)

# --- 5. Constraints ---

# 5.1. Budget Constraint: Total discrete weights must sum to the discretization factor (100)
# Sum(x_i) = 100
model.add_constraint(sum(x[i] for i in range(NUM_ASSETS)) == DISCRETIZATION_FACTOR)

# 5.2. Target Return Constraint: Portfolio expected return must meet the target
# Sum(x_i * R_i / 100) >= TARGET_RETURN  ->  Sum(x_i * R_i) >= 100 * TARGET_RETURN
scaled_target_return = DISCRETIZATION_FACTOR * TARGET_RETURN
portfolio_return_scaled = sum(x[i] * RETURNS[i] for i in range(NUM_ASSETS))
model.add_constraint(portfolio_return_scaled >= scaled_target_return)

print("\nOptimization Problem Defined (Integer Discretized):")
print(f"  Minimize: Scaled Portfolio Variance (Quadratic objective)")
print(f"  Subject to: 1. Sum of integer weights = {DISCRETIZATION_FACTOR} (100%)")
print(f"              2. Scaled Portfolio Return >= {scaled_target_return:.4f}")

# --- 6. Solve and Sample ---
try:
    # Re-adding model.into_file for strict adherence to the original code's presence
    model.into_file("Niraj_10Asset_Model_Saved")

    print("\nSending 10-Asset Model (Integer Variables) to D-Wave Hybrid NL Solver...")
    sampleset = sampler.sample(model, label='Niraj-10Asset-Integer-Portfolio')
    # Keeping result = sampleset.result() for strict adherence to the original code's presence
    result = sampleset.result()

    if not result:
        print("Solver returned an empty result set.")
        sys.exit(0)

    # --- 7. Extract and Print Results (STRICT ADHERENCE TO WORKING PATTERN) ---

    # Use the simplest working pattern: model.iter_decisions() without arguments.
    decision_iterator = model.iter_decisions()

    print("\n--- Solver Result (Optimal 10-Asset Portfolio - Discretized) ---")
    print("Extracted Asset Allocations (Integer Values):")

    # Extract the optimal integer values (x_i) by looping over the 10 variables
    try:
        total_weight_x = 0
        for i in range(NUM_ASSETS):
            # The symbol returned is the integer variable x_i
            x_symbol = next(decision_iterator)
            x_value = int((x_symbol.state()))  # Should be an integer already

            # Use standard Python to derive the continuous weight
            w_i = x_value / DISCRETIZATION_FACTOR
            total_weight_x += x_value

            # Print the output only with the state values, replicating the original code's function
            print(f"  Asset {i + 1} (x{i}): {x_value} (Weight: {w_i:.4f}) (Return: {RETURNS[i]:.2f})")

        print(f"\nConstraint Check: Total Integer Sum = {total_weight_x} (Expected: 100)")

    except StopIteration:
        print("Error: Could not extract all 10 decision variables. The solver result was malformed.")
        sys.exit(0)

    # All non-essential D-Wave API calls have been removed.

except Exception as e:
    print(f"\nAn error occurred during sampling: {e}")
    sys.exit(1)

# Clean up sampler connection
sampler.close()
print("\nSampler connection closed.")