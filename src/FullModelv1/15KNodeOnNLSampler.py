# --- Large-Scale D-Wave NL Solver (Cloud API Key) ---
#
# This script builds a large-scale, hard-coded MINIMUM COST FLOW problem
# and solves it using the LeapHybridNLSampler.
#
# This is a translation of the CQM version to the Non-Linear (NL)
# dwave.optimization.Model API.

from dwave.optimization import Model
# We will use Python's built-in `sum()` function, so `quicksum` is not needed.

from dwave.system import LeapHybridNLSampler
import sys
import json
import os
import traceback  # <-- Import traceback for better error logging
from src import APITOKEN  # <-- Changed per user request

# --- 1. CONFIGURATION ---

# !! REPLACE WITH YOUR D-WAVE API TOKEN !!
TEACHER_TOKEN = APITOKEN.NIRAJ_TOKEN  # <-- Changed per user request
TIME_LIMIT_SEC = 25  # <-- Changed per user request

# --- 2. Hard-Coded Problem Data ---
# (Data is identical to the CQM version)

# 8 Source Nodes: (type, cost per unit, max generation)
SOURCE_DATA = [
    # Solar/Wind (Cheapest)
    {"type": "Solar", "cost": 2.0, "max_gen": 3000},
    {"type": "Solar", "cost": 2.1, "max_gen": 3000},
    {"type": "Wind", "cost": 2.2, "max_gen": 4000},
    {"type": "Wind", "cost": 2.3, "max_gen": 4000},
    # Nuclear/Thermal (Intermediate)
    {"type": "Nuclear", "cost": 3.0, "max_gen": 10000},
    {"type": "Thermal", "cost": 3.5, "max_gen": 8000},
    {"type": "Thermal", "cost": 3.6, "max_gen": 8000},
    # Hydro (Costliest - Peaker)
    {"type": "Hydro", "cost": 5.0, "max_gen": 12000}
]
NUM_SOURCES = len(SOURCE_DATA)
TOTAL_MAX_GEN = sum(s["max_gen"] for s in SOURCE_DATA)  # 52,000

# 2 Battery Nodes: (max storage capacity)
BATTERY_DATA = [
    {"max_cap": 10000, "initial_cap": 10000, "min_cap": 1000},  # Starts full (100%), min 10%
    {"max_cap": 8000, "initial_cap": 8000, "min_cap": 800}  # Starts full (100%), min 10%
]
NUM_BATTERIES = len(BATTERY_DATA)
TOTAL_INITIAL_STORAGE = sum(b["initial_cap"] for b in BATTERY_DATA)  # 18,000

# 15 Sink (Downstream) Trans-shipment Nodes
DEMAND_PER_SINK_NODE = 1000 * 4.5  # 4,500 units per node
NUM_SINK_NODES = 15
TOTAL_DEMAND = NUM_SINK_NODES * DEMAND_PER_SINK_NODE  # 67,500

# 25 Trans-shipment (TS) nodes in total
NODE_NAMES = [f"TS_S{i}" for i in range(NUM_SOURCES)] + \
             [f"TS_B{j}" for j in range(NUM_BATTERIES)] + \
             [f"TS_D{k}" for k in range(NUM_SINK_NODES)]
NUM_TS_NODES = len(NODE_NAMES)  # 25

# Max flow for any single arc in the mesh
MAX_ARC_FLOW = 10000  # 10,000 units


def build_large_nl_model():
    """
    Builds the 25-node, 610-variable Minimum Cost Flow NL Model.
    """
    print("--- Building Large-Scale Complex NL Model ---")
    # Use dwave.optimization.Model
    model = Model()

    # --- 2a. Define Decision Variables ---

    # We must use setattr() to dynamically assign labeled variables to the model

    # 8 Integer variables for generation
    g = []
    for i in range(NUM_SOURCES):
        var_label = f"g{i}"
        # Create variable with keywords only (no positional args)
        var = model.integer(lower_bound=0, upper_bound=SOURCE_DATA[i]["max_gen"])
        # Attach it to the model so the library can find its label
        setattr(model, var_label, var)
        g.append(var)

    # 2 Integer variables for final battery storage
    s = []
    for j in range(NUM_BATTERIES):
        var_label = f"s{j}"
        var = model.integer(lower_bound=BATTERY_DATA[j]["min_cap"], upper_bound=BATTERY_DATA[j]["max_cap"])
        setattr(model, var_label, var)
        s.append(var)

    # 600 (25 * 24) Integer variables for arc flow
    x = {}
    for k_idx, k_name in enumerate(NODE_NAMES):
        for l_idx, l_name in enumerate(NODE_NAMES):
            if k_idx == l_idx:
                continue  # No self-loops
            var_label = f"x_{k_name}_{l_name}"
            var = model.integer(lower_bound=0, upper_bound=MAX_ARC_FLOW)
            setattr(model, var_label, var)
            x[(k_name, l_name)] = var

    print(f"Total variables: {len(g)} (gen) + {len(s)} (storage) + {len(x)} (arcs) = {len(g) + len(s) + len(x)}")

    # --- 2b. Define Objective Function ---
    # Use Python's built-in `sum()` function
    objective = sum(SOURCE_DATA[i]["cost"] * g[i] for i in range(NUM_SOURCES))
    model.minimize(objective)

    # --- 2c. Add Constraints (Flow Conservation at each TS Node) ---

    # Loop over all 25 TS nodes
    for k_idx, k_name in enumerate(NODE_NAMES):

        # Use Python's built-in `sum()` function
        flow_in = sum(x[(l_name, k_name)] for l_idx, l_name in enumerate(NODE_NAMES) if l_idx != k_idx)
        flow_out = sum(x[(k_name, l_name)] for l_idx, l_name in enumerate(NODE_NAMES) if l_idx != k_idx)

        # Use model.add_constraint()

        if 0 <= k_idx < 8:
            # --- Type 1: Source TS Node (Nodes 0-7) ---
            model.add_constraint(flow_in - flow_out + g[k_idx] == 0)

        elif 8 <= k_idx < 10:
            # --- Type 2: Battery TS Node (Nodes 8-9) ---
            j = k_idx - 8  # Battery index (0 or 1)
            initial_storage = BATTERY_DATA[j]["initial_cap"]
            final_storage_var = s[j]
            model.add_constraint(flow_in - flow_out + (initial_storage - final_storage_var) == 0)

        else:
            # --- Type 3: Sink TS Node (Nodes 10-24) ---
            model.add_constraint(flow_in - flow_out == DEMAND_PER_SINK_NODE)

    # The `dwave.optimization.Model` object uses the `.num_constraints` property
    print(f"Total constraints: {model.num_constraints}")

    # Return all the variable lists/dicts so we can read their .state() later
    return model, g, s, x


def print_solution(model, g_vars, s_vars, x_vars):
    """
    Prints a formatted summary of the best solution.
    Reads results directly from the .state() of the model's variables.
    """

    # Check if a solution was found (the state will be populated)
    # We can check the state of the first variable.
    try:
        # Try to read the state. If it's None, no solution was found.
        first_state = g_vars[0].state()
        if first_state is None:
            print("\n--- ERROR: No feasible solutions found! ---")
            print("This could mean the problem is infeasible or the time limit was too short.")
            return
    except Exception as e:
        print(f"\n--- ERROR: Failed to read solution state. Details: {e} ---")
        return

    # --- THIS IS THE FIX ---
    # Manually recalculate the energy from the decision variable states,
    # as model.objective.state() is not available.
    energy = 0
    total_gen = 0
    print("\nSource Generation:")
    for i in range(NUM_SOURCES):
        # Read the .state() from the variable object
        gen = int(g_vars[i].state())
        cost = SOURCE_DATA[i]['cost']
        energy += gen * cost  # Recalculate cost

        total_gen += gen
        max_gen = SOURCE_DATA[i]['max_gen']
        print(f"  - S{i} ({SOURCE_DATA[i]['type']:<7} Cost ${cost:.2f}): {gen: >7,} / {max_gen: >7,} units")

    print("\n--- Optimal Solution Found ---")
    print(f"Minimal Generation Cost: ${energy:,.2f}")
    print(f"  Total Generation: {total_gen:,.0f} units")
    # --- END FIX ---

    print("\nBattery Storage:")
    total_discharge = 0
    for j in range(NUM_BATTERIES):
        # Read the .state() from the variable object
        final = int(s_vars[j].state())
        initial = BATTERY_DATA[j]['initial_cap']
        discharged = initial - final
        total_discharge += discharged

        final_pct = 100 * (final / initial)
        print(f"  - Battery {j}: {final: >7,} / {initial: >7,} units ({final_pct:.1f}% final)")
        print(f"    Discharged: {discharged: >7,}")
    print(f"  Total Discharged: {total_discharge:,.0f} units")

    print("\n--- Energy Balance Check ---")
    total_supply = total_gen + total_discharge
    print(f"  Total Supply (Gen + Discharge): {total_supply:,.0f}")
    print(f"  Total Demand (Sinks):         {TOTAL_DEMAND:,.0f}")
    print(f"  Surplus/Deficit:              {total_supply - TOTAL_DEMAND:,.0f}")

    print("\n--- Non-Zero Arc Flows (Top 50) ---")
    count = 0
    total_flow = 0
    for (k_name, l_name), x_var in x_vars.items():

        # Read the .state() from the variable object
        flow = int(x_var.state())
        total_flow += flow

        if flow > 0:
            count += 1
            if count <= 50:  # Only print the first 50 to avoid spam
                print(f"  {k_name: <7} -> {l_name: <7} : {flow:,.0f} units")
    if count > 50:
        print(f"  ...and {count - 50} more non-zero flows.")
    print(f"  Total flow across all {len(x_vars)} arcs: {total_flow:,.0f} units")


# --- Main Execution ---
if __name__ == "__main__":

    # 1. Build the large-scale model
    # Get the model AND all the variable lists/dicts back
    model, g_vars, s_vars, x_vars = build_large_nl_model()

    # 2. Check for API Token
    if not TEACHER_TOKEN:
        print("Error: TEACHER_TOKEN is not set. Please add your API key.")
        sys.exit(1)

    # 3. Set up the Hybrid Sampler
    print(f"\n--- Submitting to LeapHybridNLSampler (Time Limit: {TIME_LIMIT_SEC}s) ---")
    # Use LeapHybridNLSampler
    sampler = LeapHybridNLSampler(token=TEACHER_TOKEN)

    try:
        # 4. Solve the NL Model (API Call)
        future = sampler.sample(
            model,
            time_limit=TIME_LIMIT_SEC,
            label="Large-Complex-Network-Solve-NL"
        )

        print("...Waiting for D-Wave server to return results...")

        # Call .result() to block. This populates the `model` object.
        # We can discard the return value, as we don't need it.
        result_object = future.result()

        print("...Solving complete.")

    except Exception as e:
        print(f"\n--- ERROR during D-Wave API call ---")
        print("This error happened while calling sampler.sample() or future.result().")
        print(f"Error details: {e}")
        sys.exit(1)

    try:
        # 5. Print the formatted solution (Local Code)

        # Pass the stateful 'model' and the variable lists
        # to the new print_solution function.
        print_solution(model, g_vars, s_vars, x_vars)

    except Exception as e:
        print(f"\n--- ERROR during local solution processing ---")
        print("This error happened in the print_solution() function.")
        # This will print the full traceback with line numbers, as requested
        traceback.print_exc()
        sys.exit(1)

