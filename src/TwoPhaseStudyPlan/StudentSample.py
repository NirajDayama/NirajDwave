# --- Student's D-Wave Solver (Local Machine, No API Key) ---
#
# This script builds a small optimization problem, solves it locally,
# and includes functions to export the model for a teacher to solve
# on the D-Wave cloud.
#
# Problem: "Minimal Factory Upgrade"
# You must decide:
#   1. (Binary)  Should we upgrade the assembly line? (Cost: 500)
#   2. (Integer) How many new robot arms to install? (Cost: 150 per arm, max 5)
#   3. (Real)    What should the new line speed be? (Cost: 10.5 per m/s, max 10.0 m/s)
#
# Constraints:
#   - You can only install robot arms IF you upgrade (e.g., arms <= 5 * upgrade)
#   - You can only set the line speed IF you upgrade (e.g., speed <= 10 * upgrade)
#   - The line speed must be at least 1.2 m/s for every arm installed
#
# Objective: Minimize the total cost.

import dimod
from dimod import ExactCQMSolver
import sys
import json
import os
import pickle  # <-- 1. IMPORT THE STANDARD PICKLE LIBRARY
import shutil  # <-- 1. IMPORT SHUTIL
import tempfile  # <-- 1. IMPORT TEMPFILE

# --- 1. Problem Definition ---

# Define constants for the problem
COST_UPGRADE = 500.0
COST_PER_ARM = 150.0
COST_PER_SPEED = 10.5
MAX_ARMS = 5
MAX_SPEED = 10.0  # m/s
MIN_SPEED_PER_ARM = 1.2

# Define file names for export/import
MODEL_EXPORT_FILE = "my_model_for_teacher.dat"  # <-- 2. Use .dat for binary model
RESULTS_IMPORT_FILE = "results_from_teacher.json"


def build_cqm():
    """
    Builds the Constrained Quadratic Model (CQM) for the factory problem.
    """
    print("--- Building CQM ---")
    cqm = dimod.ConstrainedQuadraticModel()

    # --- 2. Define Decision Variables ---

    # Binary Variable (Boolean)
    # b_upgrade: 1 if we upgrade, 0 if not
    b_upgrade = dimod.Binary('b_upgrade')

    # Integer Variable (Discrete)
    # i_arms: Number of robot arms to install (0 to 5)
    i_arms = dimod.Integer('i_arms', lower_bound=0, upper_bound=MAX_ARMS)

    # Real Variable (Continuous) - SIMULATION
    # CQM samplers (like ExactCQMSolver) do not support true Real variables.
    # We simulate them by using a scaled Integer.
    # To get 2 decimal places of precision for 10.00 m/s:
    # We create an integer from 0 to 1000 and divide by 100.0
    r_speed_scaled = dimod.Integer('r_speed_scaled', lower_bound=0, upper_bound=int(MAX_SPEED * 100))

    # We define our "real" variable r_speed in terms of the scaled integer
    r_speed = r_speed_scaled / 100.0

    print(f"Variables: {', '.join(cqm.variables)}")

    # --- 3. Define Objective Function ---
    # Minimize: 500*b_upgrade + 150*i_arms + 10.5*r_speed
    objective = (COST_UPGRADE * b_upgrade) + \
                (COST_PER_ARM * i_arms) + \
                (COST_PER_SPEED * r_speed)

    cqm.set_objective(objective)
    print(f"Objective: Minimize {objective.to_polystring()}")

    # --- 4. Add Constraints ---

    # C1: Can only install arms if we upgrade
    # i_arms <= MAX_ARMS * b_upgrade  (e.g., i_arms <= 5 * 0 is 0)
    cqm.add_constraint(i_arms - (MAX_ARMS * b_upgrade) <= 0, label="C1_arm_upgrade_gate")

    # C2: Can only set speed if we upgrade
    # r_speed <= MAX_SPEED * b_upgrade
    # r_speed_scaled / 100.0 <= 10.0 * b_upgrade
    # r_speed_scaled <= 1000 * b_upgrade
    cqm.add_constraint(r_speed_scaled - (MAX_SPEED * 100 * b_upgrade) <= 0, label="C2_speed_upgrade_gate")

    # C3: Speed must be >= 1.2 m/s per arm
    # r_speed >= MIN_SPEED_PER_ARM * i_arms
    # r_speed_scaled / 100.0 >= 1.2 * i_arms
    # r_speed_scaled >= 120 * i_arms
    # 0 >= 120 * i_arms - r_speed_scaled
    cqm.add_constraint((MIN_SPEED_PER_ARM * 100 * i_arms) - r_speed_scaled <= 0, label="C3_speed_per_arm")

    print(f"Number of constraints: {len(cqm.constraints)}")

    return cqm


def solve_locally(cqm):
    """
    Solves the CQM using the local ExactCQMSolver.
    This requires NO API key.
    """
    print("\n--- 1. Solving Locally (ExactCQMSolver) ---")

    # Check if the problem is too large (more than ~1,000,000 cases)
    # Search space = (2 options) * (6 options) * (1001 options) = 12,012
    # This is very small and fast!
    search_space = 1
    for var in cqm.variables:
        if cqm.vartype(var) == dimod.BINARY:
            search_space *= 2
        elif cqm.vartype(var) == dimod.INTEGER:
            # Use int() to ensure standard Python integer for calculation
            search_space *= (int(cqm.upper_bound(var)) - int(cqm.lower_bound(var)) + 1)

    print(f"Total search space to check: {search_space} combinations.")

    if search_space > 2_000_000:  # Safety break
        print("Error: Problem is too large for local solver!")
        print("Please export the model and send it to your teacher.")
        return None

    sampler = ExactCQMSolver()
    sampleset = sampler.sample_cqm(cqm)

    print("...Local solving complete.")
    return sampleset


def export_model_to_file(cqm, filename):
    """
    Exports the CQM model to a binary file.
    This uses the .to_file() method as a context manager, which
    yields a temporary file object.
    """
    print(f"\n--- 2. Exporting Model to {filename} ---")

    # --- 3. THIS IS THE FIX ---
    # Your hint was correct. .to_file() is a context manager.
    # We copy the yielded temporary file object into our new file.
    try:
        with cqm.to_file() as temp_file:
            # temp_file is now an open, binary file-like object
            # We open our destination file in 'wb' (write binary)
            with open(filename, 'wb') as f:
                # shutil.copyfileobj copies the contents efficiently
                shutil.copyfileobj(temp_file, f)

        print(f"Model successfully saved. Please send this file to your teacher.")
    except AttributeError:
        # This error means the dimod library is too old.
        print("\n-------------------------------------------------------------")
        print("FATAL ERROR: Your 'dimod' library is too old.")
        print("The '.to_file()' method does not exist.")
        print("Please upgrade your D-Wave libraries to use this feature:")
        print("  pip install --upgrade dimod dwave-system")
        print("-------------------------------------------------------------")
    except Exception as e:
        print(f"Error: Could not export model. {e}")


def import_results_from_file(filename):
    """
    Imports the SampleSet results from a JSON file.
    """
    print(f"\n--- 3. Importing Results from {filename} ---")
    if not os.path.exists(filename):
        print(f"Note: Results file not found: {filename}")
        print("You can get this from your teacher after exporting your model.")
        return None

    try:
        # SampleSet serialization still uses json and .from_serializable()
        # This part has always been correct.
        with open(filename, 'r') as f:
            sampleset = dimod.SampleSet.from_serializable(json.load(f))
        print("...Results loaded successfully.")
        return sampleset
    except Exception as e:
        print(f"Error reading results file. It may be corrupted. {e}")
        return None


def print_solution(sampleset):
    """
    Prints the best feasible solution from a sampleset.
    """
    if sampleset is None:
        return

    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

    if not feasible_sampleset:
        print("\n--- ERROR: No feasible solutions found! ---")
        return

    best_sample = feasible_sampleset.first
    sample_dict = best_sample.sample
    energy = best_sample.energy

    print(f"\n--- Best Feasible Solution (Cost: {energy}) ---")

    # --- Re-interpret the variables ---

    # 1. Binary
    upgrade_val = sample_dict['b_upgrade']
    print(f"\n  Upgrade Facility: {'YES' if upgrade_val == 1 else 'NO'}")

    # 2. Integer
    arms_val = sample_dict['i_arms']
    print(f"  Robot Arms: {int(arms_val)} (Max: {MAX_ARMS})")

    # 3. Simulated Real
    # We must un-scale the 'r_speed_scaled' variable
    speed_scaled_val = sample_dict['r_speed_scaled']
    speed_val = speed_scaled_val / 100.0
    print(f"  Line Speed: {speed_val:.2f} m/s (Max: {MAX_SPEED:.2f})")

    print("\n  Full Sample Dictionary:")
    # Cast NumPy types (like int32) to standard Python int()
    print(json.dumps({k: int(v) for k, v in sample_dict.items()}, indent=2))
    print("---------------------------------")


# --- Main Workflow ---
if __name__ == "__main__":

    # Build the CQM
    my_cqm = build_cqm()

    # --- OPTION A: Solve locally ---
    local_sampleset = solve_locally(my_cqm)
    if local_sampleset:
        print_solution(local_sampleset)

    # --- OPTION B: Export/Import (if problem is too big) ---

    # 1. Student exports the model
    export_model_to_file(my_cqm, MODEL_EXPORT_FILE)

    # 2. (Teacher runs their script and sends back the results file)
    #    For this demo, we assume the teacher has done this.
    #    If you don't have the results file yet, this step will fail.

    # 3. Student imports the teacher's results
    cloud_sampleset = import_results_from_file(RESULTS_IMPORT_FILE)
    if cloud_sampleset:
        print("\n--- SOLUTION FROM TEACHER (D-Wave Cloud) ---")
        print_solution(cloud_sampleset)


