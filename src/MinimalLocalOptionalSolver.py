# --- D-Wave Network Flow Optimization (Simplified Model) ---
# This script solves a simplified 5-node minimum cost problem.
#
# *** PROBLEM SIMPLIFICATION ***
# We assume all arcs have unlimited capacity and we don't care about the
# specific path the energy takes. This abstracts the network into a
# single "pool" of energy.
#
# The problem becomes:
# Total Supply (Generation + Storage Depletion) must equal Total Demand (Sinks).
#
# This reduces the problem from 9 complex variables to 3 simple variables,
# making it solvable by the local ExactCQMSolver.
#
# Total Supply = g1 + g2 + STORAGE_INIT
# Total Demand = DEMAND_N4 + DEMAND_N5 + s (final storage)
#
# Constraint: g1 + g2 + STORAGE_INIT = DEMAND_N4 + DEMAND_N5 + s
# Rearranged: g1 + g2 - s = (DEMAND_N4 + DEMAND_N5) - STORAGE_INIT
#
# ***********************************

# D-Wave token (not used if USE_LOCAL_SOLVER is True)
import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN
# === STUDENT PRACTICE TOGGLE ===
# Set this to True to use the local 'ExactCQMSolver' (no API key needed).
# The simplified model is small enough for this to work.
USE_LOCAL_SOLVER = True
# ===============================


# Import required D-Wave libraries
from dwave.system import LeapHybridCQMSampler
import dimod
# Import the local ExactCQMSolver
from dimod import ExactCQMSolver
import sys  # Used for exiting gracefully
import json  # For pretty-printing the final result

# --- 1. Problem Definition ---
# Define the constants for the problem.

# Source Node Costs
COST_G1 = 2.0  # Cost per unit for Source N1
COST_G2 = 3.5  # Cost per unit for Source N2

# Source Node Max Generation (Capacity)
MAX_G1 = 15  # Max generation for Source N1
MAX_G2 = 10  # Max generation for Source N2

# Storage Node (N3) Properties
STORAGE_INIT = 10  # Initial storage at N3
STORAGE_MAX = 20  # Max storage capacity at N3

# Sink Node Demands
DEMAND_N4 = 10  # Demand required at Sink N4
DEMAND_N5 = 15  # Demand required at Sink N5

# (Arc Capacities are removed as they are no longer needed)

# --- 2. Model Creation ---

# Initialize an empty Constrained Quadratic Model
cqm = dimod.ConstrainedQuadraticModel()

# --- 3. Define Decision Variables ---
# We only need 3 variables for this simplified model.

# Generation Variables (Sources)
# g1: Generation at N1. Bounds: 0 <= g1 <= 15
g1 = dimod.Integer('g1', lower_bound=0, upper_bound=MAX_G1)
# g2: Generation at N2. Bounds: 0 <= g2 <= 10
g2 = dimod.Integer('g2', lower_bound=0, upper_bound=MAX_G2)

# Storage Variable
# s: Final storage at N3. Bounds: 0 <= s <= 20
s = dimod.Integer('s', lower_bound=0, upper_bound=STORAGE_MAX)

# (Flow variables x_ij are removed)

# --- 4. Define Objective Function ---
# The objective is unchanged: minimize the total generation cost.
# Z = 2.0*g1 + 3.5*g2
cqm.set_objective(COST_G1 * g1 + COST_G2 * g2)

# --- 5. Add Constraints ---
# We now have only ONE constraint: the global energy balance.
# g1 + g2 - s = (DEMAND_N4 + DEMAND_N5) - STORAGE_INIT
# g1 + g2 - s = (10 + 15) - 10
# g1 + g2 - s = 15

# Calculate the Right-Hand Side (RHS) of the equation
total_demand = DEMAND_N4 + DEMAND_N5
rhs = total_demand - STORAGE_INIT

cqm.add_constraint(g1 + g2 - s == rhs, label="Global_Energy_Balance")

# Note: The generation/storage capacity constraints (g1 <= 15, g2 <= 10, s <= 20)
# are already defined by the `upper_bound` of the dimod.Integer variables.

print("--- Simplified CQM Model Built Successfully ---")
print(f"Objective: Minimize {cqm.objective.to_polystring()}")
print(f"Number of constraints: {len(cqm.constraints)}")
print(f"Constraint: g1 + g2 - s == {rhs}")
print("--------------------------------------")

# --- 6. Solve the CQM ---

if USE_LOCAL_SOLVER:
    # --- LOCAL SOLVER (No API Key) ---
    # Use dimod's built-in exact solver. This runs locally.
    # The search space is now 16*11*21 = 3,696 cases (vs 10^14 before)
    print(f"Submitting problem to local dimod.ExactCQMSolver...")
    sampler = ExactCQMSolver()
    sampleset = sampler.sample_cqm(cqm)

else:
    # --- D-WAVE CLOUD SOLVER (API Key Required) ---
    # Set up the sampler configuration from your NIRAJ__CQM.py file
    print(f"Submitting problem to LeapHybridCQMSampler...")
    sampler = LeapHybridCQMSampler(
        token=NIRAJ_TOKEN,
        region="na-west-1"  # Added from your NL solver file for robustness
    )
    # Submit the CQM to the hybrid sampler
    sampleset = sampler.sample_cqm(cqm, time_limit=5, label="Niraj-Network-CQM-Solver-Simple")

print("...Solving complete.")

# --- 7. Extract and Display Results ---

# Filter the sampleset to only include feasible solutions
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

if not feasible_sampleset:
    print("\n--- ERROR: No feasible solutions found! ---")
    print("This could be due to a conflicting model, a short time limit, or a solver issue.")
    print("\nFull Sampleset (for debugging):")
    print(sampleset)
    sys.exit(1)  # Exit if no solution is found

print(f"\n--- Found {len(feasible_sampleset)} Feasible Solutions ---")

# Iterate over all feasible solutions found
for i, sample_data in enumerate(feasible_sampleset.data()):
    print(f"\n--- Feasible Solution {i + 1} ---")

    # Get the dictionary of variable values for this sample
    sample = sample_data.sample

    # Get the objective function value (total cost) for this sample
    energy = sample_data.energy

    print(f"  Objective Cost (Z): {energy}")

    print("\n  Generation Variables:")
    print(f"    g1 (Source N1): {sample['g1']} (Max: {MAX_G1})")
    print(f"    g2 (Source N2): {sample['g2']} (Max: {MAX_G2})")

    print("\n  Storage Variable (N3):")
    print(f"    s (Final Storage): {sample['s']} (Initial: {STORAGE_INIT}, Max: {STORAGE_MAX})")

    # (Flow variable printouts are removed)
    print("---------------------------------")

# Finally, print the single best solution found
print("\n--- Best Solution Found (Lowest Cost) ---")
best_sample = feasible_sampleset.first
print(f"Objective: {best_sample.energy}")
print("Sample:")
print(best_sample.sample.items())

