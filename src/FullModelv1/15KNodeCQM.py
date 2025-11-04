# --- Large-Scale D-Wave Solver (Cloud API Key) ---
#
# This script builds a large-scale, hard-coded MINIMUM COST FLOW problem
# and solves it using the LeapHybridCQMSampler.
#
# Problem: 15,010 Node Complex Network
#   - 8 Source Nodes (Generators)
#   - 2 Battery Nodes (Storage)
#   - 15 Downstream Trans-shipment Nodes (representing 15,000 sinks)
#   - 10 Upstream Trans-shipment Nodes (connected to sources/batteries)
#
# Network Structure:
#   - 8 Sources -> 8 "Source TS" Nodes
#   - 2 Batteries <-> 2 "Battery TS" Nodes
#   - All 25 TS nodes (8 Source + 2 Battery + 15 Sink-facing) are in a
#     fully-connected mesh.
#   - 15 "Sink TS" Nodes must satisfy a large, fixed demand.
#
# Objective: Minimize total generation cost to meet all demand.

import dimod
from dwave.system import LeapHybridCQMSampler
import sys
import json
import os

# --- 1. CONFIGURATION ---

# !! REPLACE WITH YOUR D-WAVE API TOKEN !!
from src import APITOKEN
TEACHER_TOKEN = APITOKEN.NIRAJ_TOKEN  # Use your own token
TIME_LIMIT_SEC = 25

# --- 2. Hard-Coded Problem Data ---

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
# Each represents 1000 sinks with an average demand of 4.5 units
DEMAND_PER_SINK_NODE = 1000 * 4.5  # 4,500 units per node
NUM_SINK_NODES = 15
# Total demand is 15 * 4,500 = 67,500
TOTAL_DEMAND = NUM_SINK_NODES * DEMAND_PER_SINK_NODE

# CRITICAL CONSTRAINT: Total Demand (67,500) > Total Max Gen (52,000)
# This forces the batteries to discharge to meet the 15,500 unit deficit.
# Max possible supply = 52,000 (Gen) + 16,200 (Battery discharge) = 68,200
# This confirms the problem is feasible but tight.

# We have 25 Trans-shipment (TS) nodes in total
# TS Node 0-7:   Connected to Sources
# TS Node 8-9:   Connected to Batteries
# TS Node 10-24: Connected to Sinks
NODE_NAMES = [f"TS_S{i}" for i in range(NUM_SOURCES)] + \
             [f"TS_B{j}" for j in range(NUM_BATTERIES)] + \
             [f"TS_D{k}" for k in range(NUM_SINK_NODES)]
NUM_TS_NODES = len(NODE_NAMES)  # 25

# Max flow for any single arc in the mesh
MAX_ARC_FLOW = 10000  # 10,000 units


def build_large_cqm():
    """
    Builds the 25-node, 610-variable Minimum Cost Flow CQM.
    """
    print("--- Building Large-Scale Complex CQM ---")
    cqm = dimod.ConstrainedQuadraticModel()

    # --- 2a. Define Decision Variables ---

    # 8 Integer variables for generation
    # *** FIX IS HERE: Use keyword arguments 'lower_bound' and 'upper_bound' ***
    g = [dimod.Integer(f"g{i}", lower_bound=0, upper_bound=SOURCE_DATA[i]["max_gen"]) for i in range(NUM_SOURCES)]

    # 2 Integer variables for final battery storage
    # *** FIX IS HERE: Use keyword arguments 'lower_bound' and 'upper_bound' ***
    s = [dimod.Integer(f"s{j}", lower_bound=BATTERY_DATA[j]["min_cap"], upper_bound=BATTERY_DATA[j]["max_cap"]) for j in
         range(NUM_BATTERIES)]

    # 600 (25 * 24) Integer variables for arc flow in the trans-shipment mesh
    # x[k][l] = flow from TS Node k to TS Node l
    # *** FIX IS HERE: Use keyword arguments 'lower_bound' and 'upper_bound' ***
    x = {}
    for k_idx, k_name in enumerate(NODE_NAMES):
        for l_idx, l_name in enumerate(NODE_NAMES):
            if k_idx == l_idx:
                continue  # No self-loops
            x[(k_name, l_name)] = dimod.Integer(f"x_{k_name}_{l_name}", lower_bound=0, upper_bound=MAX_ARC_FLOW)

    print(f"Total variables: {len(g)} (gen) + {len(s)} (storage) + {len(x)} (arcs) = {len(g) + len(s) + len(x)}")

    # --- 2b. Define Objective Function ---
    # Minimize total generation cost
    objective = dimod.quicksum(SOURCE_DATA[i]["cost"] * g[i] for i in range(NUM_SOURCES))
    cqm.set_objective(objective)

    # --- 2c. Add Constraints (Flow Conservation at each TS Node) ---
    #
    # General Formula:
    # Sum(Flow In) - Sum(Flow Out) + Local_Supply = Local_Demand

    # Loop over all 25 TS nodes
    for k_idx, k_name in enumerate(NODE_NAMES):

        # Sum(Flow In) = Sum(flow from all other nodes 'l' *to* 'k')
        flow_in = dimod.quicksum(x[(l_name, k_name)] for l_idx, l_name in enumerate(NODE_NAMES) if l_idx != k_idx)

        # Sum(Flow Out) = Sum(flow from 'k' *to* all other nodes 'l')
        flow_out = dimod.quicksum(x[(k_name, l_name)] for l_idx, l_name in enumerate(NODE_NAMES) if l_idx != k_idx)

        # The rest depends on the node type

        if 0 <= k_idx < 8:
            # --- Type 1: Source TS Node (Nodes 0-7) ---
            # Local_Supply = Generation g[k_idx]
            # Local_Demand = 0
            # Constraint: flow_in - flow_out + g[k_idx] == 0
            cqm.add_constraint(flow_in - flow_out + g[k_idx] == 0, label=f"balance_{k_name}")

        elif 8 <= k_idx < 10:
            # --- Type 2: Battery TS Node (Nodes 8-9) ---
            # Local_Supply/Demand = Net discharge
            # Net Discharge = Initial_Storage - Final_Storage
            j = k_idx - 8  # Battery index (0 or 1)
            initial_storage = BATTERY_DATA[j]["initial_cap"]
            final_storage_var = s[j]
            # Constraint: flow_in - flow_out + (initial - final) == 0
            cqm.add_constraint(flow_in - flow_out + (initial_storage - final_storage_var) == 0,
                               label=f"balance_{k_name}")

        else:
            # --- Type 3: Sink TS Node (Nodes 10-24) ---
            # Local_Supply = 0
            # Local_Demand = DEMAND_PER_SINK_NODE
            # Constraint: flow_in - flow_out == DEMAND_PER_SINK_NODE
            cqm.add_constraint(flow_in - flow_out == DEMAND_PER_SINK_NODE, label=f"balance_{k_name}")

    print(f"Total constraints: {len(cqm.constraints)}")

    return cqm, x


def print_solution(sampleset, x_vars):
    """
    Prints a formatted summary of the best solution.
    """
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

    if not feasible_sampleset:
        print("\n--- ERROR: No feasible solutions found! ---")
        print("This could mean the problem is infeasible or the time limit was too short.")
        return

    best = feasible_sampleset.first
    sample = best.sample
    energy = best.energy

    print("\n--- Optimal Solution Found ---")
    print(f"Minimal Generation Cost: ${energy:,.2f}")

    print("\nSource Generation:")
    total_gen = 0
    for i in range(NUM_SOURCES):
        gen = int(sample[f"g{i}"])
        total_gen += gen
        max_gen = SOURCE_DATA[i]['max_gen']
        cost = SOURCE_DATA[i]['cost']
        print(f"  - S{i} ({SOURCE_DATA[i]['type']:<7} Cost ${cost:.2f}): {gen: >7,} / {max_gen: >7,} units")
    print(f"  Total Generation: {total_gen:,.0f} units")

    print("\nBattery Storage:")
    total_discharge = 0
    for j in range(NUM_BATTERIES):
        final = int(sample[f"s{j}"])
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
        flow = int(sample[list(x_var.variables)[0]])
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
    cqm, x_vars = build_large_cqm()

    # 2. Check for API Token
    if not TEACHER_TOKEN or "YOUR_TOKEN_HERE" in TEACHER_TOKEN:
        print("Error: TEACHER_TOKEN is not set. Please add your API key.")
        sys.exit(1)

    # 3. Set up the Hybrid Sampler
    print(f"\n--- Submitting to LeapHybridCQMSampler (Time Limit: {TIME_LIMIT_SEC}s) ---")
    sampler = LeapHybridCQMSampler(token=TEACHER_TOKEN)

    try:
        # 4. Solve the CQM
        sampleset = sampler.sample_cqm(
            cqm,
            time_limit=TIME_LIMIT_SEC,
            label="Large-Complex-Network-Solve"
        )
        print("...Solving complete.")

        # 5. Print the formatted solution
        print_solution(sampleset, x_vars)

    except Exception as e:
        print(f"\n--- ERROR during D-Wave API call ---")
        print(e)
        sys.exit(1)