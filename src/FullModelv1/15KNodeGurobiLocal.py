# --- Large-Scale Gurobi MIP Solver (Local) ---
#
# This script builds and solves the large-scale MINIMUM COST FLOW problem
# using the local Gurobi MIP solver to find the exact optimal solution.
#
# The problem data and formulation are identical to the D-Wave
# solver scripts for a direct comparison.

import gurobipy as gp
from gurobipy import GRB
import sys
import os
import time

# --- 1. Hard-Coded Problem Data ---
# (Data is identical to the D-Wave CQM/NL versions)

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

# 2 Battery Nodes: (max storage capacity)
BATTERY_DATA = [
    {"max_cap": 10000, "initial_cap": 10000, "min_cap": 1000},  # Starts full (100%), min 10%
    {"max_cap": 8000, "initial_cap": 8000, "min_cap": 800}  # Starts full (100%), min 10%
]
NUM_BATTERIES = len(BATTERY_DATA)

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


def build_and_solve_gurobi():
    """
    Builds and solves the 25-node, 610-variable Minimum Cost Flow MIP.
    """
    try:
        print("--- Building Large-Scale Gurobi MIP Model ---")
        # --- 2. Build Gurobi Model ---
        # Create a new model
        model = gp.Model("Large_Network_Flow_Gurobi")

        # --- 3. Define Decision Variables ---

        # 8 Integer variables for generation
        g = {}
        for i in range(NUM_SOURCES):
            g[i] = model.addVar(vtype=GRB.INTEGER,
                                lb=0,
                                ub=SOURCE_DATA[i]["max_gen"],
                                name=f"g{i}")

        # 2 Integer variables for final battery storage
        s = {}
        for j in range(NUM_BATTERIES):
            s[j] = model.addVar(vtype=GRB.INTEGER,
                                lb=BATTERY_DATA[j]["min_cap"],
                                ub=BATTERY_DATA[j]["max_cap"],
                                name=f"s{j}")

        # 600 (25 * 24) Integer variables for arc flow
        x = {}
        for k_idx, k_name in enumerate(NODE_NAMES):
            for l_idx, l_name in enumerate(NODE_NAMES):
                if k_idx == l_idx:
                    continue  # No self-loops
                x[(k_name, l_name)] = model.addVar(vtype=GRB.INTEGER,
                                                   lb=0,
                                                   ub=MAX_ARC_FLOW,
                                                   name=f"x_{k_name}_{l_name}")

        print(f"Total variables: {model.NumVars}")

        # --- 4. Define Objective Function ---
        # Minimize total generation cost
        objective = gp.quicksum(SOURCE_DATA[i]["cost"] * g[i] for i in range(NUM_SOURCES))
        model.setObjective(objective, GRB.MINIMIZE)

        # --- 5. Add Constraints (Flow Conservation at each TS Node) ---

        # Loop over all 25 TS nodes
        for k_idx, k_name in enumerate(NODE_NAMES):

            # Sum(Flow In) = Sum(flow from all other nodes 'l' *to* 'k')
            flow_in = gp.quicksum(x[(l_name, k_name)] for l_idx, l_name in enumerate(NODE_NAMES) if l_idx != k_idx)

            # Sum(Flow Out) = Sum(flow from 'k' *to* all other nodes 'l')
            flow_out = gp.quicksum(x[(k_name, l_name)] for l_idx, l_name in enumerate(NODE_NAMES) if l_idx != k_idx)

            if 0 <= k_idx < 8:
                # --- Type 1: Source TS Node (Nodes 0-7) ---
                model.addConstr(flow_in - flow_out + g[k_idx] == 0, name=f"balance_{k_name}")

            elif 8 <= k_idx < 10:
                # --- Type 2: Battery TS Node (Nodes 8-9) ---
                j = k_idx - 8  # Battery index (0 or 1)
                initial_storage = BATTERY_DATA[j]["initial_cap"]
                final_storage_var = s[j]
                model.addConstr(flow_in - flow_out + (initial_storage - final_storage_var) == 0,
                                name=f"balance_{k_name}")

            else:
                # --- Type 3: Sink TS Node (Nodes 10-24) ---
                model.addConstr(flow_in - flow_out == DEMAND_PER_SINK_NODE, name=f"balance_{k_name}")

        print(f"Total constraints: {model.NumConstrs}")

        # --- 6. Solve Model ---
        print("\n--- Solving with Gurobi MIP Solver ---")
        start_time = time.time()
        model.optimize()
        solve_time = time.time() - start_time
        print(f"...Solving complete in {solve_time:.2f} seconds.")

        return model, g, s, x

    except gp.GurobiError as e:
        print(f"\n--- GUROBI ERROR ---")
        print(f"Error code {e.errno}: {e}")
        print("Please ensure your Gurobi license is set up correctly.")
        sys.exit(1)
    except ImportError:
        print("\n--- IMPORT ERROR ---")
        print("Gurobi (gurobipy) not found or not installed.")
        print("Please run: pip install gurobipy")
        sys.exit(1)


def print_solution_gurobi(model, g_vars, s_vars, x_vars):
    """
    Prints a formatted summary of the best solution from Gurobi.
    """

    if model.Status != GRB.OPTIMAL:
        print("\n--- ERROR: No optimal solution found! ---")
        print(f"Gurobi solver status code: {model.Status}")
        return

    # --- Manually recalculate energy (same as NL script) ---
    energy = 0
    total_gen = 0
    print("\nSource Generation:")
    for i in range(NUM_SOURCES):
        # *** KEY CHANGE: Use the .X attribute to get the value ***
        gen = int(g_vars[i].X)
        cost = SOURCE_DATA[i]['cost']
        energy += gen * cost  # Recalculate cost

        total_gen += gen
        max_gen = SOURCE_DATA[i]['max_gen']
        print(f"  - S{i} ({SOURCE_DATA[i]['type']:<7} Cost ${cost:.2f}): {gen: >7,} / {max_gen: >7,} units")

    print("\n--- Optimal Solution Found ---")
    print(f"Minimal Generation Cost: ${energy:,.2f} (Exact MIP Solution)")
    print(f"  Total Generation: {total_gen:,.0f} units")

    print("\nBattery Storage:")
    total_discharge = 0
    for j in range(NUM_BATTERIES):
        # *** KEY CHANGE: Use the .X attribute to get the value ***
        final = int(s_vars[j].X)
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

        # *** KEY CHANGE: Use the .X attribute to get the value ***
        flow = int(x_var.X)
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

    model, g, s, x = build_and_solve_gurobi()

    # If model building and solving was successful, print the solution
    if model and model.Status == GRB.OPTIMAL:
        print_solution_gurobi(model, g, s, x)
