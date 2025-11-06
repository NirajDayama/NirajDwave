import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler
import sys
import json
from src import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

# --- 1. Problem Data (Integer Weights W_i where 0 <= W_i <= 100) ---
N = 10 # Number of assets
# Expected Returns (mu_i * 100, used in linear constraint)
RETURNS_PERC = np.array([8.0, 12.0, 7.0, 9.5, 11.0, 6.5, 10.5, 13.0, 9.0, 11.5])
# Variances (sigma_ii / 10000, used in quadratic objective)
VARIANCES_SCALED = np.array([0.0015, 0.0030, 0.0010, 0.0022, 0.0028, 0.0012, 0.0025, 0.0035, 0.0018, 0.0029]) / 10000
# Constant Covariance (sigma_ij / 10000)
COVARIANCE_OFF_DIAG_SCALED = 0.0002 / 10000
R_TARGET_PERC = 8.5 # Target return for the portfolio in percentage units (9.0%)

# --- 2. Build Scaled Covariance Matrix (Sigma_Scaled) ---
# Create an N x N matrix initialized with off-diagonal scaled covariance
SIGMA_SCALED = np.full((N, N), COVARIANCE_OFF_DIAG_SCALED)
# Set the diagonal elements to the scaled variances
np.fill_diagonal(SIGMA_SCALED, VARIANCES_SCALED)

# --- 3. Model Creation ---
cqm = dimod.ConstrainedQuadraticModel()

# --- 4. Define Decision Variables (Integer Percentage Weights) ---
# W_i: Integer percentage weight of asset i (0 <= W_i <= 100)
W = [dimod.Integer(f'W_{i+1}', lower_bound=0, upper_bound=100) for i in range(N)]

# --- 5. Define Objective Function: Minimize Scaled Portfolio Risk ---
# Objective Z = sum(W_i * W_j * (sigma_ij / 10000))
objective = 0
for i in range(N):
    for j in range(N):
        # NOTE: This quadratic interaction is now valid because W[i] is a dimod.Integer
        objective += W[i] * W[j] * SIGMA_SCALED[i, j]

cqm.set_objective(objective)

# --- 6. Add Constraints ---

# Constraint 1: Budget Constraint (Sum of percentage weights must equal 100)
# sum(W_i) == 100
budget_sum = sum(W)
cqm.add_constraint(budget_sum == 100, label="Budget_Constraint_Integer")

# Constraint 2: Target Return Constraint (Portfolio return >= R_target in percentage units)
# sum(W_i * mu_i) >= R_TARGET_PERC
return_sum = sum(W[i] * RETURNS_PERC[i] for i in range(N))
cqm.add_constraint(return_sum >= R_TARGET_PERC, label="Target_Return_Constraint_Integer")

print("--- REVISED CQM Model Built Successfully (with Integer Weights) ---")
print(f"Objective: Minimize a Quadratic function of 10 Integer variables.")
print(f"Number of constraints: {len(cqm.constraints)}")
print("------------------------------------------------------------------")

# --- 7. Solve the CQM (Using User's Sampler Configuration) ---
sampler = LeapHybridCQMSampler(
    token=NIRAJ_TOKEN,
    region="na-west-1"
)

print(f"Submitting problem to LeapHybridCQMSampler...")

# Submit the CQM to the hybrid sampler
sampleset = sampler.sample_cqm(cqm, time_limit=25, label="Niraj-Portfolio-CQM-Integer-Solver")

print("...Solving complete.")

# --- 8. Extract and Display Results (Using User's Extraction Logic) ---

# Filter the sampleset to only include feasible solutions
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

if not feasible_sampleset:
    print("\n--- ERROR: No feasible solutions found! ---")
    sys.exit(1)

print(f"\n--- Found {len(feasible_sampleset)} Feasible Solutions ---")

# Find and display the single best solution found
print("\n--- Best Solution Found (Lowest Cost/Risk) ---")
best_sample = feasible_sampleset.first
best_sample_dict = {k: v for k, v in best_sample.sample.items()}

# Calculate and print actual portfolio return and total weight for verification
# Note: W_i values are integers (e.g., 30)
best_return = sum(best_sample_dict[f'W_{i+1}'] * RETURNS_PERC[i] for i in range(N)) / 100.0
total_weight = sum(best_sample_dict.values()) / 100.0

print(f"Portfolio Risk (Objective Value, Scaled): {best_sample.energy:.10f}")
print(f"Portfolio Return (Verification): {best_return:.4f} (Required: >= {R_TARGET_PERC/100:.4f})")
print(f"Total Weight (Verification): {total_weight:.4f} (Required: 1.0000)")
print("\nOptimal Asset Weights (W_i - Integer Percentage):")
print(json.dumps(best_sample_dict, indent=2))

print("------------------------------------------------------------------")