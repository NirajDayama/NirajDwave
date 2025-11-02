# --- D-Wave Network Flow Optimization using CQM ---
# This script solves the 5-node minimum cost network flow problem
# by building a Constrained Quadratic Model (CQM) directly using the `dimod` library
# and solving it with the LeapHybridCQMSampler.

from src import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

# Import required D-Wave libraries
from dwave.system import LeapHybridCQMSampler
import dimod
import sys  # Used for exiting gracefully
import json # <-- ADD THIS LINE

# --- 1. Problem Definition ---
# Define the constants for the 5-node network problem.

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

# Arc Capacities (Assuming 100 for all as in the app.py example)
ARC_CAP = 100

# --- 2. Model Creation ---

# Initialize an empty Constrained Quadratic Model
cqm = dimod.ConstrainedQuadraticModel()

# --- 3. Define Decision Variables ---
# We use dimod.Integer() to create integer variables with specified bounds.

# Generation Variables (Sources)
# g1: Generation at N1. Bounds: 0 <= g1 <= 15
g1 = dimod.Integer('g1', lower_bound=0, upper_bound=MAX_G1)
# g2: Generation at N2. Bounds: 0 <= g2 <= 10
g2 = dimod.Integer('g2', lower_bound=0, upper_bound=MAX_G2)

# Storage Variable
# s: Final storage at N3. Bounds: 0 <= s <= 20
s = dimod.Integer('s', lower_bound=0, upper_bound=STORAGE_MAX)

# Flow Variables (Arcs)
# x_ij represents the flow from node i to node j. Bounds: 0 <= x_ij <= 100
x13 = dimod.Integer('x_N1_N3', lower_bound=0, upper_bound=ARC_CAP)
x23 = dimod.Integer('x_N2_N3', lower_bound=0, upper_bound=ARC_CAP)
x34 = dimod.Integer('x_N3_N4', lower_bound=0, upper_bound=ARC_CAP)
x35 = dimod.Integer('x_N3_N5', lower_bound=0, upper_bound=ARC_CAP)
x14 = dimod.Integer('x_N1_N4', lower_bound=0, upper_bound=ARC_CAP)
x25 = dimod.Integer('x_N2_N5', lower_bound=0, upper_bound=ARC_CAP)

# --- 4. Define Objective Function ---
# The objective is to minimize the total generation cost: Z = 2.0*g1 + 3.5*g2
# We set this as the objective for the CQM.
cqm.set_objective(COST_G1 * g1 + COST_G2 * g2)

# --- 5. Add Constraints ---
# Constraints ensure flow conservation (mass balance) at each node.
# Format: cqm.add_constraint(expression, sense, rhs, label)
# '==' means "equal to"
# '<=' means "less than or equal to"
# '>=' means "greater than or equal to"

# Node 1 (Source): Outflow == Generation
# (x13 + x14) - 0 = g1  =>  x13 + x14 - g1 == 0
cqm.add_constraint(x13 + x14 - g1 == 0, label="N1_Source_Balance")

# Node 2 (Source): Outflow == Generation
# (x23 + x25) - 0 = g2  =>  x23 + x25 - g2 == 0
cqm.add_constraint(x23 + x25 - g2 == 0, label="N2_Source_Balance")

# Node 3 (Storage): Inflow - Outflow == Final Storage - Initial Storage
# (x13 + x23) - (x34 + x35) = s - 10
# => x13 + x23 - x34 - x35 - s == -10
cqm.add_constraint(x13 + x23 - x34 - x35 - s == -STORAGE_INIT, label="N3_Storage_Balance")

# Node 4 (Sink): Inflow == Demand
# (x34 + x14) - 0 = 10  =>  x34 + x14 == 10
cqm.add_constraint(x34 + x14 == DEMAND_N4, label="N4_Sink_Balance")

# Node 5 (Sink): Inflow == Demand
# (x35 + x25) - 0 = 15  =>  x35 + x25 == 15
cqm.add_constraint(x35 + x25 == DEMAND_N5, label="N5_Sink_Balance")

# Note: The generation/storage capacity constraints (g1 <= 15, g2 <= 10, s <= 20)
# are already defined by the `upper_bound` of the dimod.Integer variables.

print("--- CQM Model Built Successfully ---")
print(f"Objective: Minimize {cqm.objective.to_polystring()}")
print(f"Number of constraints: {len(cqm.constraints)}")
print("--------------------------------------")

# --- 6. Solve the CQM ---

# Set up the sampler configuration from your NIRAJ__CQM.py file
sampler = LeapHybridCQMSampler(
    token=NIRAJ_TOKEN,
    # endpoint="https://cloud.dwavesys.com/sapi", # Endpoint is often not needed if region is set
    # solver="hybrid_constrained_quadratic_model_version1p" # Not needed, token finds default
    region="na-west-1"  # Added from your NL solver file for robustness
)

print(f"Submitting problem to LeapHybridCQMSampler...")

# Submit the CQM to the hybrid sampler
# We add a time limit (in seconds) as good practice
sampleset = sampler.sample_cqm(cqm, time_limit=5, label="Niraj-Network-CQM-Solver")

print("...Solving complete.")

# --- 7. Extract and Display Results ---

# Filter the sampleset to only include feasible solutions
# This is a critical step when using CQM solvers.
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

if not feasible_sampleset:
    print("\n--- ERROR: No feasible solutions found! ---")
    print("This could be due to a conflicting model, a short time limit, or a solver issue.")
    print("\nFull Sampleset (for debugging):")
    print(sampleset)
    sys.exit(1)  # Exit if no solution is found

print(f"\n--- Found {len(feasible_sampleset)} Feasible Solutions ---")

# Iterate over all feasible solutions found (often just 1, but could be more)
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

    print("\n  Flow Variables (Arcs):")
    print(f"    x_N1_N3 (Flow N1->N3): {sample['x_N1_N3']}")
    print(f"    x_N2_N3 (Flow N2->N3): {sample['x_N2_N3']}")
    print(f"    x_N3_N4 (Flow N3->N4): {sample['x_N3_N4']}")
    print(f"    x_N3_N5 (Flow N3->N5): {sample['x_N3_N5']}")
    print(f"    x_N1_N4 (Flow N1->N4): {sample['x_N1_N4']}")
    print(f"    x_N2_N5 (Flow N2->N5): {sample['x_N2_N5']}")
    print("---------------------------------")

# Finally, print the single best solution found
print("\n--- Best Solution Found (Lowest Cost) ---")
best_sample = feasible_sampleset.first
print(f"Objective: {best_sample.energy}")
print("Sample:")
# Format the dictionary for cleaner printing
best_sample_dict = {k: v for k, v in best_sample.sample.items()}
print(json.dumps(best_sample_dict, indent=2))
