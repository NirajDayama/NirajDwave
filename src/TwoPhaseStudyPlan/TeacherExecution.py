# --- Teacher's D-Wave Solver (Cloud API Key) ---
#
# This script is for the teacher ONLY.
# It loads a CQM model file from a student, solves it using the
# LeapHybridCQMSampler, and exports the results for the student.

import dimod
from dwave.system import LeapHybridCQMSampler
import sys
import json
import os

from src import APITOKEN
NIRAJ_TOKEN = APITOKEN.NIRAJ_TOKEN

# Note: pickle, shutil, and tempfile are not needed here
# We only need the libraries to load the model and run the solver

# --- CONFIGURATION ---
# !! REPLACE WITH YOUR D-WAVE API TOKEN !!
TEACHER_TOKEN = NIRAJ_TOKEN  # Use your own token
TIME_LIMIT_SEC = 10  # Time limit to give the hybrid solver

# Define file names for import/export
MODEL_IMPORT_FILE = "my_model_for_teacher.dat"  # File from student (binary)
RESULTS_EXPORT_FILE = "results_from_teacher.json"  # File to send back (json)

# --- 1. Load Student's Model ---
print(f"--- Loading model from {MODEL_IMPORT_FILE} ---")

if not os.path.exists(MODEL_IMPORT_FILE):
    print(f"Error: Model file not found: {MODEL_IMPORT_FILE}")
    print("Please get the model file from your student.")
    sys.exit(1)

try:
    # --- THIS IS THE CORRECT METHOD ---
    # Use the matching .from_file() method
    # It must be opened in 'rb' (read binary) mode
    with open(MODEL_IMPORT_FILE, 'rb') as f:
        cqm = dimod.ConstrainedQuadraticModel.from_file(f)

    print("...Model loaded successfully.")
    print(f"Objective: {cqm.objective.to_polystring()}")
    print(f"Variables: {list(cqm.variables)}")
    print(f"Constraints: {len(cqm.constraints)}")

except AttributeError:
    # This error means the dimod library is too old.
    print("\n-------------------------------------------------------------")
    print("FATAL ERROR: Your 'dimod' library is too old.")
    print("The '.from_file()' method does not exist.")
    print("Please upgrade your D-Wave libraries to use this feature:")
    print("  pip install --upgrade dimod dwave-system")
    print("-------------------------------------------------------------")
    sys.exit(1)
except Exception as e:
    print(f"Error: Could not read model file. {e}")
    sys.exit(1)

# --- 2. Solve on D-Wave Cloud ---
print(f"\n--- Submitting to LeapHybridCQMSampler (Time Limit: {TIME_LIMIT_SEC}s) ---")

if not TEACHER_TOKEN or "YOUR_TOKEN_HERE" in TEACHER_TOKEN:
    print("Error: TEACHER_TOKEN is not set. Please add your API key.")
    sys.exit(1)

sampler = LeapHybridCQMSampler(token=TEACHER_TOKEN)

try:
    sampleset = sampler.sample_cqm(
        cqm,
        time_limit=TIME_LIMIT_SEC,
        label="Teacher-Solve-for-Student"
    )
    print("...Solving complete.")

    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    if not feasible_sampleset:
        print("\nWARNING: No feasible solutions found by the hybrid solver.")
        print("The model may be infeasible, or the time limit too short.")
    else:
        best_energy = feasible_sampleset.first.energy
        print(f"Best feasible solution found with cost: {best_energy}")

except Exception as e:
    print(f"Error during D-Wave API call: {e}")
    sys.exit(1)

# --- 3. Export Results for Student ---
print(f"\n--- Exporting results to {RESULTS_EXPORT_FILE} ---")

try:
    # We use .to_serializable() to prepare the SampleSet for JSON
    # This part was (and still is) correct!
    with open(RESULTS_EXPORT_FILE, 'w') as f:
        json.dump(sampleset.to_serializable(), f)
    print("...Results saved successfully.")
    print(f"Please send {RESULTS_EXPORT_FILE} back to your student.")

except Exception as e:
    print(f"Error saving results: {e}")
    sys.exit(1)

