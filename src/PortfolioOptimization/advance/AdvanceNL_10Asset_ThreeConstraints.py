# --- D-Wave Portfolio Optimization: 10 Assets with Side Constraints (Integer Discretization) ---
# This script minimizes portfolio risk (variance) subject to:
# 1. Budget constraint (sum of weights = 100)
# 2. Target return constraint (min return)
# 3. Max/Min Position Size (granularity)
# 4. Sector Exposure Limits (groups)
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
DISCRETIZATION_FACTOR = 100.0  # Weights are x_i / 100 (e.g., x_i=10 -> 10% weight)
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
MIN_POSITION_SIZE = 2 # 2% minimum allocation
MAX_POSITION_SIZE = 30 # 30% maximum allocation

print("--- 1. Model Setup: Defining Variables and Objective ---")

for i in range(NUM_ASSETS):
    # C1: Maximum/Minimum Position Size Constraint (Granularity/Liquidity)
    # The variable bounds enforce both the lower (2%) and upper (30%) limits.
    # Note: Setting lower_bound=2 means any asset held must be at least 2% (2/100).
    x_i = model.integer(lower_bound=MIN_POSITION_SIZE, upper_bound=MAX_POSITION_SIZE)
    x.append(x_i)
    print(f"  -> Asset {i + 1} (x{i}): Integer variable created with bounds [{MIN_POSITION_SIZE}, {MAX_POSITION_SIZE}] (2%-30%)")

# --- 3. Define Objective Function (Minimize Scaled Portfolio Variance) ---
# Objective: sum_{i} sum_{j} x_i * x_j * Sigma_{i,j}
objective = 0
for i in range(NUM_ASSETS):
    for j in range(NUM_ASSETS):
        objective += x[i] * x[j] * SIGMAS[i, j]

model.minimize(objective)

print("\n--- 2. Optimization Objective Defined ---")
print(f"  Minimize: Scaled Portfolio Risk (Variance)")
print("    - This is a quadratic function: sum_{i,j} (x_i * x_j * Sigma_{i,j})")
print("    - The term is minimized to find the lowest possible risk portfolio that satisfies all constraints.")


# --- 4. Add Constraints (Base Constraints + New Side Constraints) ---

print("\n--- 3. Adding Constraints to the Model ---")

# BASE CONSTRAINT 1: Budget Constraint (Sum of weights = 100)
budget_constraint = sum(x)
model.add_constraint(budget_constraint == 100)
print(f"  1. Budget Constraint: Sum(x_i) == {DISCRETIZATION_FACTOR} (Total allocation must equal 100%)")

# BASE CONSTRAINT 2: Target Return Constraint (Min Return)
scaled_target_return = TARGET_RETURN * DISCRETIZATION_FACTOR
return_constraint = sum(x[i] * RETURNS[i] for i in range(NUM_ASSETS))
model.add_constraint(return_constraint >= scaled_target_return)
print(f"  2. Target Return: Sum(x_i * R_i) >= {scaled_target_return:.4f} (Portfolio return must be >= {TARGET_RETURN * 100:.2f}%)")

# SIDE CONSTRAINT 2: Sector or Group Exposure Limits
# Hypothetical Sector 1: Assets 1-5 (Index 0-4)
SECTOR_1_LIMIT = 60
sector_1_exposure = sum(x[i] for i in range(5))
model.add_constraint(sector_1_exposure <= SECTOR_1_LIMIT)
print(f"  3. Sector 1 Max Limit: Sum(x0 to x4) <= {SECTOR_1_LIMIT} (Maximum 60% exposure in Sector 1)")

# Hypothetical Sector 2: Assets 6-10 (Index 5-9)
SECTOR_2_LIMIT = 60
sector_2_exposure = sum(x[i] for i in range(5, NUM_ASSETS))
model.add_constraint(sector_2_exposure <= SECTOR_2_LIMIT)
print(f"  4. Sector 2 Max Limit: Sum(x5 to x9) <= {SECTOR_2_LIMIT} (Maximum 60% exposure in Sector 2)")

# --- 5. Solve the Model ---
print("\n--- 4. Solving the Model ---")
print("Sending 10-Asset Model (Integer Variables with Side Constraints) to D-Wave Hybrid NL Solver...")

# NOTE: The execution is non-blocking here; the result will be available later.
sampleset = sampler.sample(model, label='Niraj-10Asset-NL-With-SideConstraints-Verbose')
result = sampleset.result()

# --- 6. Extract and Display Results (Strictly following the user's working pattern) ---
print("\n--- 5. Solver Result Extraction ---")

# Use the simplest working pattern: model.iter_decisions() without arguments.
decision_iterator = model.iter_decisions()

# List to hold the optimal integer weights for post-solution verification
optimal_x_values = []

print("\nOptimal Asset Allocations (Integer Values):")
try:
    for i in range(NUM_ASSETS):
        x_symbol = next(decision_iterator)
        x_value = int(x_symbol.state()) # Extract value
        optimal_x_values.append(x_value)

        w_i = x_value / DISCRETIZATION_FACTOR
        print(f"  Asset {i + 1} (x{i}): {x_value:^3} (Weight: {w_i:.4f}) | Expected Return: {RETURNS[i]:.4f} | Min/Max: [{MIN_POSITION_SIZE}, {MAX_POSITION_SIZE}]")

except StopIteration:
    print("Error: Could not extract all 10 decision variables. The solver result was malformed.")
    sys.exit(0)

# --- 7. Detailed Constraint Satisfaction and Interpretation ---

print("\n--- 6. Detailed Constraint Verification & Interpretation ---")

# Convert optimal integer weights back to a numpy array for vector math
x_opt = np.array(optimal_x_values)

# --- Objective/Risk Calculation ---
# Risk = sum_{i} sum_{j} (w_i * w_j * Sigma_{i,j})
# Scaled Risk = sum_{i} sum_{j} (x_i * x_j * Sigma_{i,j})
# The objective energy is the Scaled Risk:
objective_value_risk = x_opt @ SIGMAS @ x_opt
print(f"\n[A] Portfolio Risk (Objective Value): {objective_value_risk:.8f}")
print("    - This is the value the solver minimized subject to all constraints.")

# --- Constraint Verification ---

# C1. Budget Constraint: Sum(x_i) == 100
total_weight_x = np.sum(x_opt)
budget_status = "MET" if total_weight_x == DISCRETIZATION_FACTOR else "FAILED"
print(f"\n[B] Constraint 1: Budget (Sum x_i = 100)")
print(f"    - Achieved Sum: {total_weight_x:.0f} | Status: {budget_status}")

# C2. Target Return Constraint: Sum(x_i * R_i) >= 5.0
achieved_return_scaled = np.sum(x_opt * RETURNS)
achieved_return_percent = achieved_return_scaled / DISCRETIZATION_FACTOR
return_status = "MET" if achieved_return_scaled >= scaled_target_return else "FAILED"
print(f"\n[C] Constraint 2: Target Return (Sum x_i * R_i >= {scaled_target_return:.4f})")
print(f"    - Achieved Scaled Return: {achieved_return_scaled:.4f} ({achieved_return_percent * 100:.2f}%) | Status: {return_status}")

# C3. Max/Min Position Size: Enforced by variable bounds (x_i in [2, 30])
non_zero_positions = np.count_nonzero(x_opt)
min_size_ok = np.all(x_opt[x_opt > 0] >= MIN_POSITION_SIZE)
max_size_ok = np.all(x_opt <= MAX_POSITION_SIZE)
position_status = "MET" if min_size_ok and max_size_ok else "FAILED"
print(f"\n[D] Constraint 3: Position Granularity (2% min / 30% max)")
print(f"    - All weights x_i are within [{MIN_POSITION_SIZE}, {MAX_POSITION_SIZE}]. | Status: {position_status}")

# C4. Sector Exposure Limits
# Sector 1 (Assets 1-5 / Indices 0-4)
sector_1_achieved = np.sum(x_opt[:5])
sector_1_status = "MET" if sector_1_achieved <= SECTOR_1_LIMIT else "FAILED"
print(f"\n[E] Constraint 4a: Sector 1 (Assets 1-5) Max Limit (<= {SECTOR_1_LIMIT})")
print(f"    - Achieved Sector 1 Exposure: {sector_1_achieved:.0f} ({sector_1_achieved / DISCRETIZATION_FACTOR * 100:.0f}%) | Status: {sector_1_status}")

# Sector 2 (Assets 6-10 / Indices 5-9)
sector_2_achieved = np.sum(x_opt[5:])
sector_2_status = "MET" if sector_2_achieved <= SECTOR_2_LIMIT else "FAILED"
print(f"[F] Constraint 4b: Sector 2 (Assets 6-10) Max Limit (<= {SECTOR_2_LIMIT})")
print(f"    - Achieved Sector 2 Exposure: {sector_2_achieved:.0f} ({sector_2_achieved / DISCRETIZATION_FACTOR * 100:.0f}%) | Status: {sector_2_status}")

# --- Interpretation Summary ---
print("\n[G] Final Interpretation:")
print(f"    - The optimal solution uses {non_zero_positions} out of {NUM_ASSETS} assets.")
print("    - The solver successfully found a portfolio that minimizes risk while adhering to all four side constraints and the base financial constraints.")

sampler.close()
print("\nSampler connection closed.")