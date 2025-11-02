import json
import time
import sys
import os
import copy
import base64
import io
from typing import List, Dict, Any, Tuple

# Third-party libraries
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback
    import dash_cytoscape as cyto
    import dimod
    from dimod import ExactCQMSolver
    from dwave.optimization import Model
    from dwave.system import LeapHybridNLSampler, LeapHybridCQMSampler
    import pandas as pd
except ImportError as e:
    print(f"ERROR: Missing required Python package: {e}")
    print("Please install all necessary libraries:")
    print("pip install dash dash-cytoscape dimod dwave-optimization dwave-system pandas openpyxl")
    # We will let the app run, but Dash will fail if a component is missing
    if 'dash_cytoscape' not in sys.modules:
        cyto = None  # Graceful failure for cyto
    if 'pandas' not in sys.modules:
        pd = None  # Graceful failure for pandas

# --- CONFIGURATION ---

import APITOKEN
DWAVE_TOKEN = APITOKEN.NIRAJ_TOKEN

## DWAVE_TOKEN = os.environ.get("DWAVE_API_TOKEN", "YOUR_API_TOKEN_HERE")
if DWAVE_TOKEN == "YOUR_API_TOKEN_HERE":
    print("WARNING: DWAVE_API_TOKEN not set. Using placeholder.")

# Default Problem Setup (Your 5-Node Example)
DEFAULT_PROBLEM_STATE = {
    "nodes": {
        "N1": {"type": "Source", "cost": 2.0, "max_g": 15, "pos": [50, 50]},
        "N2": {"type": "Source", "cost": 3.5, "max_g": 10, "pos": [50, 200]},
        "N3": {"type": "Storage", "s_init": 10, "s_max": 20, "pos": [250, 125]},
        "N4": {"type": "Sink", "demand": 10, "pos": [450, 50]},
        "N5": {"type": "Sink", "demand": 15, "pos": [450, 200]}
    },
    "arcs": [
        {"id": "A13", "source": "N1", "target": "N3", "capacity": 100},
        {"id": "A23", "source": "N2", "target": "N3", "capacity": 100},
        {"id": "A34", "source": "N3", "target": "N4", "capacity": 100},
        {"id": "A35", "source": "N3", "target": "N5", "capacity": 100},
        {"id": "A14", "source": "N1", "target": "N4", "capacity": 100},
        {"id": "A25", "source": "N2", "target": "N5", "capacity": 100}
    ]
}


# --- MODELING FUNCTION (Updated Solver Logic) ---

def build_and_solve_network_flow(
        problem_data: Dict[str, Any], sampler_type: str, time_limit: int, api_token: str
) -> Dict[str, Any]:
    """Builds the optimization model, solves it using the selected sampler, and extracts results."""

    nodes = problem_data['nodes']
    arcs = problem_data['arcs']

    model = Model()
    g_vars = {}
    s_var = None
    x_vars = {}

    # Pre-process network structure
    sources = [k for k, v in nodes.items() if v.get('type') == 'Source']
    sinks = [k for k, v in nodes.items() if v.get('type') == 'Sink']
    storage_nodes = [k for k, v in nodes.items() if v.get('type') == 'Storage']
    all_nodes = list(nodes.keys())

    objective = 0

    # 2. Define Variables and Objective
    for name, data in nodes.items():
        if data.get('type') == 'Source':
            g_vars[name] = model.integer(lower_bound=0, upper_bound=data.get('max_g', 0))
            objective += data.get('cost', 0.0) * g_vars[name]

        elif data.get('type') == 'Storage' and name in storage_nodes:
            s_var = model.integer(lower_bound=0, upper_bound=data.get('s_max', 0))

    # Flow variables x_i,j
    for arc in arcs:
        source, target, capacity = arc['source'], arc['target'], arc['capacity']
        var_name = f"x_{source}_{target}"
        x_vars[(source, target)] = model.integer(lower_bound=0, upper_bound=capacity)

    model.minimize(objective)

    # 3. Add Constraints (Flow Conservation)
    for src in sources:
        outflow = sum(x_vars.get((src, dst_node), 0) for dst_node in all_nodes if (src, dst_node) in x_vars)
        inflow = sum(x_vars.get((src_node, src), 0) for src_node in all_nodes if (src_node, src) in x_vars)
        model.add_constraint(outflow - inflow == g_vars[src])

    for snk in sinks:
        demand = nodes[snk].get('demand', 0)
        outflow = sum(x_vars.get((snk, dst_node), 0) for dst_node in all_nodes if (snk, dst_node) in x_vars)
        inflow = sum(x_vars.get((src_node, snk), 0) for src_node in all_nodes if (src_node, snk) in x_vars)
        model.add_constraint(inflow - outflow == demand)

    for storage in storage_nodes:
        if s_var is None: continue
        s_init = nodes[storage].get('s_init', 0)
        outflow = sum(x_vars.get((storage, dst_node), 0) for dst_node in all_nodes if (storage, dst_node) in x_vars)
        inflow = sum(x_vars.get((src_node, storage), 0) for src_node in all_nodes if (src_node, storage) in x_vars)
        model.add_constraint(inflow - outflow == s_var - s_init)

    # --- Solving Logic ---

    results = {}
    solver_name = ""
    solver_time = 0.0

    # ------------------------------------------------------------------
    # --- Branch 1: Classical Solver (dimod.ExactCQMSolver)
    # ------------------------------------------------------------------
    if sampler_type == 'CLASSICAL':
        solver_name = "dimod.ExactCQMSolver"
        cqm = model.to_cqm()
        sampler = ExactCQMSolver()

        start_time = time.perf_counter()
        sampleset = sampler.sample_cqm(cqm)
        solver_time = time.perf_counter() - start_time

        if not sampleset:
            return {"status": "FAILED", "message": "ExactCQMSolver returned no samples."}

        # Filter for feasible solutions only
        feasible_samples = sampleset.filter(lambda d: d.is_feasible)
        if not feasible_samples:
            return {"status": "FAILED", "message": "ExactCQMSolver found no feasible solutions."}

        best_sample_dict = feasible_samples.first.sample
        final_objective = feasible_samples.first.energy

        results = {
            "objective": float(final_objective),
            "status": "COMPLETED (Classical)",
            "sampler": solver_name,
            "solver_time": solver_time,
        }

        # Extract variables from the sample dictionary by label
        for src in sources:
            results[f"g_{src}"] = int(best_sample_dict.get(f"g_{src}", 0))
        if storage_nodes:
            results["s"] = int(best_sample_dict.get("s", 0))
        for arc in arcs:
            results[f"x_{arc['source']}_{arc['target']}"] = int(
                best_sample_dict.get(f"x_{arc['source']}_{arc['target']}", 0))

        return results

    # ------------------------------------------------------------------
    # --- Branch 2: D-Wave NL Hybrid Solver
    # ------------------------------------------------------------------
    elif sampler_type == 'NL':
        sampler = LeapHybridNLSampler(token=api_token)
        solver_name = "LeapHybridNLSampler"

        start_time = time.perf_counter()
        sampleset = sampler.sample(model, time_limit=time_limit, label="Network-Flow-NL")
        solver_time = time.perf_counter() - start_time

        if not sampleset or sampleset.first.energy == dimod.UNFEASIBLE_ENERGY:
            return {"status": "FAILED", "message": "Solver returned an unfeasible result or no samples."}

    # ------------------------------------------------------------------
    # --- Branch 3: D-Wave CQM Hybrid Solver
    # ------------------------------------------------------------------
    elif sampler_type == 'CQM':
        cqm = model.to_cqm()
        sampler = LeapHybridCQMSampler(token=api_token)
        solver_name = "LeapHybridCQMSampler"

        start_time = time.perf_counter()
        sampleset = sampler.sample_cqm(cqm, time_limit=time_limit, label="Network-Flow-CQM")
        solver_time = time.perf_counter() - start_time

        if not sampleset or not sampleset.first.is_feasible:
            return {"status": "FAILED", "message": "CQM solver returned an unfeasible result or no samples."}

    # 4. Extract Results (Hybrid Solvers)
    best_sample = sampleset.first
    model.set_state(best_sample.state)  # Use the model's state-setting feature
    final_objective = model.objective.state

    results = {
        "objective": float(final_objective),
        "status": "COMPLETED (Hybrid)",
        "sampler": solver_name,
        "solver_time": solver_time,
    }

    decision_iterator = model.iter_decisions()

    for src in sources:
        var_symbol = next(decision_iterator)
        results[f"g_{src}"] = int(var_symbol.state())

    if storage_nodes:
        var_symbol = next(decision_iterator)
        results["s"] = int(var_symbol.state())

    for arc in arcs:
        var_symbol = next(decision_iterator)
        results[f"x_{arc['source']}_{arc['target']}"] = int(var_symbol.state())

    return results


# --- CYTOSCAPE UTILITIES ---

# Styles for the graph visualization
CYTOSCAPE_STYLESHEET = [
    {'selector': 'node',
     'style': {'content': 'data(label)', 'text-valign': 'center', 'text-halign': 'center', 'font-size': '10px',
               'color': '#FFFFFF', 'background-color': '#6C757D'}},
    {'selector': '.Source', 'style': {'background-color': '#28A745', 'shape': 'square', 'width': 30, 'height': 30}},
    {'selector': '.Sink', 'style': {'background-color': '#DC3545', 'shape': 'triangle', 'width': 30, 'height': 30}},
    {'selector': '.Storage', 'style': {'background-color': '#007BFF', 'shape': 'circle', 'width': 30, 'height': 30}},
    {'selector': 'edge', 'style': {'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#495057',
                                   'target-arrow-color': '#495057', 'width': 2, 'content': 'data(label)',
                                   'font-size': '10px', 'color': '#C9D1D9', 'text-background-color': '#000',
                                   'text-background-opacity': 0.7}},
    {'selector': ':selected',
     'style': {'border-width': '3px', 'border-color': '#FFC107', 'border-opacity': 1.0, 'line-color': '#FFC107',
               'target-arrow-color': '#FFC107'}},
    {'selector': '.Flow',
     'style': {'line-color': '#FFC107', 'target-arrow-color': '#FFC107', 'width': 4, 'label': 'data(flowLabel)'}},
]


def generate_cytoscape_elements(state: Dict[str, Any], results: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Converts the problem state dictionary into Cytoscape elements."""
    elements = []

    # 1. Nodes
    for node_id, data in state['nodes'].items():
        node_type = data.get('type', 'Unknown')

        label = node_id
        if node_type == 'Source':
            label += f" ($g\le{data.get('max_g')})"
        elif node_type == 'Sink':
            label += f" ($D={data.get('demand')})"
        elif node_type == 'Storage':
            label += f" ($S_i={data.get('s_init')})"

        elements.append({
            'data': {
                'id': node_id, 'label': label, 'type': node_type,
                'cost': data.get('cost'), 'max_g': data.get('max_g'),
                'demand': data.get('demand'), 's_init': data.get('s_init'),
                's_max': data.get('s_max'),
            },
            'position': {'x': data.get('pos', [0, 0])[0], 'y': data.get('pos', [0, 0])[1]},
            'classes': node_type
        })

    # 2. Edges (Arcs)
    for arc in state['arcs']:
        flow = results.get(f"x_{arc['source']}_{arc['target']}", 0) if results else 0
        classes = 'Flow' if flow > 0 and results else ''

        elements.append({
            'data': {
                'id': arc['id'], 'source': arc['source'], 'target': arc['target'],
                'capacity': arc['capacity'], 'label': f"Cap: {arc['capacity']}",
                'flowLabel': f"Flow: {flow}", 'flow': flow,
            },
            'classes': classes
        })

    return elements


# --- FILE PARSING UTILITIES ---

def parse_xlsx_to_state(contents: str) -> Dict[str, Any]:
    """Parses a base64-encoded XLSX file into the problem state dictionary."""
    if not pd:
        raise ImportError("Pandas library is not installed. Cannot parse XLSX.")

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Read all sheets
    xls_data = pd.read_excel(io.BytesIO(decoded), sheet_name=None)

    if "Nodes" not in xls_data or "Arcs" not in xls_data:
        raise ValueError("XLSX file must contain 'Nodes' and 'Arcs' sheets.")

    new_state = {"nodes": {}, "arcs": []}

    # Process Nodes
    nodes_df = xls_data["Nodes"].where(pd.notnull(xls_data["Nodes"]), None)
    # Use 'id' as the index for the dictionary
    nodes_df = nodes_df.set_index('id', drop=False)
    nodes_dict = nodes_df.to_dict(orient='index')

    for node_id, data in nodes_dict.items():
        new_state['nodes'][node_id] = {
            "type": data.get('type'),
            "cost": data.get('cost'),
            "max_g": data.get('max_g'),
            "demand": data.get('demand'),
            "s_init": data.get('s_init'),
            "s_max": data.get('s_max'),
            "pos": [data.get('pos_x', 0), data.get('pos_y', 0)]
        }

    # Process Arcs
    arcs_df = xls_data["Arcs"].where(pd.notnull(xls_data["Arcs"]), None)
    new_state['arcs'] = arcs_df.to_dict(orient='records')

    return new_state


def parse_json_to_state(contents: str) -> Dict[str, Any]:
    """Parses a base64-encoded JSON file into the problem state dictionary."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return json.loads(decoded.decode('utf-8'))


# --- DASH APP SETUP ---
app = dash.Dash(__name__)
app.title = "D-Wave Network Flow Editor & Solver"

INPUT_STYLE = {"padding": "8px", "borderRadius": "4px", "border": "1px solid #ccc", "marginBottom": "10px",
               "width": "100%", "backgroundColor": "#f8f8f8"}
BUTTON_STYLE = {"backgroundColor": "#1F77B4", "color": "white", "padding": "10px", "borderRadius": "5px",
                "border": "none", "width": "100%", "fontWeight": "bold", "cursor": "pointer"}
EDIT_INPUT_STYLE = {"padding": "5px", "borderRadius": "3px", "border": "1px solid #ccc", "width": "calc(100% - 70px)",
                    "marginRight": "10px"}
GRAPH_BUTTON_STYLE = {"backgroundColor": "#6C757D", "color": "white", "padding": "5px 10px", "borderRadius": "5px",
                      "border": "none", "cursor": "pointer", "margin": "0 5px"}

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "maxWidth": "1600px", "margin": "0 auto", "padding": "20px"},
    children=[
        dcc.Store(id='problem-state', data=DEFAULT_PROBLEM_STATE),
        dcc.Store(id='results-state', data={}),
        dcc.Store(id='last-clicked-element', data={}),
        dcc.Store(id='selected-nodes-store', data=[]),  # Store for arc creation

        html.H1("D-Wave Hybrid Network Flow Editor & Solver", style={"textAlign": "center", "color": "#1F77B4"}),
        html.Hr(),

        # --- MAIN EDITING AND SOLVER PANEL ---
        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                # 1. GRAPH VISUALIZATION (Main Content)
                html.Div(
                    style={"flex": 3, "minWidth": "600px", "height": "600px", "border": "1px solid #ddd",
                           "borderRadius": "8px", "padding": "5px"},
                    children=[
                        html.H3("Network Diagram", style={"textAlign": "center"}),
                        (cyto.Cytoscape if cyto else html.Div)(  # Conditional rendering
                            id='network-graph',
                            layout={'name': 'preset'},
                            style={'width': '100%', 'height': '450px'},
                            elements=generate_cytoscape_elements(DEFAULT_PROBLEM_STATE),
                            stylesheet=CYTOSCAPE_STYLESHEET,
                            autounselectify=False,
                            boxSelectionEnabled=True,
                        ),
                        html.P(id="graph-info", style={"fontSize": "small", "marginTop": "5px", "textAlign": "center"}),

                        # Graph Editing Buttons
                        html.Div(
                            style={"display": "flex", "justifyContent": "center", "padding": "10px",
                                   "borderTop": "1px solid #eee"},
                            children=[
                                html.Button("Add Node", id="add-node-button", n_clicks=0, style=GRAPH_BUTTON_STYLE),
                                html.Button("Add Arc (Select 2 Nodes)", id="add-arc-button", n_clicks=0,
                                            style={**GRAPH_BUTTON_STYLE, "backgroundColor": "#28A745"}),
                                html.Button("Delete Selected", id="delete-button", n_clicks=0,
                                            style={**GRAPH_BUTTON_STYLE, "backgroundColor": "#DC3545"}),
                            ]
                        ),

                        # Upload Component
                        dcc.Upload(
                            id='upload-topography',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select a JSON or XLSX File')
                            ]),
                            style={
                                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                'textAlign': 'center', 'margin': '10px 0'
                            },
                            multiple=False
                        ),
                    ]
                ),

                # 2. SIDE PANEL (Editing and Solver Config)
                html.Div(
                    style={"flex": 1, "minWidth": "350px", "maxHeight": "600px", "overflowY": "auto"},
                    children=[
                        # Element Editor Panel
                        html.Div(
                            id="element-editor-panel",
                            style={"padding": "15px", "border": "1px solid #ddd", "borderRadius": "8px",
                                   "marginBottom": "20px"},
                            children=[
                                html.H3("Edit Node/Arc Attributes", style={"color": "#DC3545"}),
                                html.Div("Click a Node or Arc in the diagram to edit its attributes.",
                                         id="editor-content")
                            ]
                        ),

                        # Solver Configuration Panel
                        html.Div(
                            style={"padding": "15px", "border": "1px solid #ddd", "borderRadius": "8px"},
                            children=[
                                html.H3("Solver Configuration", style={"color": "#FF7F0E"}),
                                html.Label("D-Wave API Token:",
                                           style={"fontWeight": "bold", "display": "block", "marginTop": "10px"}),
                                dcc.Input(id="api-token-input", type="password", value=DWAVE_TOKEN, style=INPUT_STYLE),

                                html.Label("Select Sampler:",
                                           style={"fontWeight": "bold", "display": "block", "marginTop": "10px"}),
                                dcc.Dropdown(
                                    id="sampler-select",
                                    options=[
                                        {"label": "Leap Hybrid NL Sampler", "value": "NL"},
                                        {"label": "Leap Hybrid CQM Sampler", "value": "CQM"},
                                        {"label": "Classical Exact Solver", "value": "CLASSICAL"},
                                    ],
                                    value="NL", clearable=False, style={"marginBottom": "10px"}
                                ),

                                html.Label("Time Limit (seconds):",
                                           style={"fontWeight": "bold", "display": "block", "marginTop": "10px"}),
                                dcc.Input(id="time-limit-input", type="number", value=5, min=1, max=120,
                                          style=INPUT_STYLE),

                                html.Button("RUN OPTIMIZATION", id="run-button", n_clicks=0,
                                            style={**BUTTON_STYLE, "marginTop": "20px"}),
                                html.Button("Export Model (.nl/.cqm)", id="export-button", n_clicks=0,
                                            style={**BUTTON_STYLE, "backgroundColor": "#9467BD", "marginTop": "10px"}),
                            ]
                        ),
                    ]
                )
            ]
        ),

        html.Hr(style={"marginTop": "20px"}),

        # --- RESULTS PANEL ---
        html.H2("Optimization Results", style={"color": "#D62728", "marginTop": "20px"}),
        html.Div(id="solving-status", style={"padding": "10px", "fontWeight": "bold", "backgroundColor": "#F0F0F0"}),
        html.Div(id="results-output", style={"marginTop": "10px"})
    ]
)


# --- CALLBACKS ---

# 1. Update Editor Panel on Click
@callback(
    [
        Output('editor-content', 'children'),
        Output('last-clicked-element', 'data'),
    ],
    [
        Input('network-graph', 'tapNodeData'),
        Input('network-graph', 'tapEdgeData'),
    ],
    State('problem-state', 'data')
)
def display_element_editor(node_data, edge_data, current_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div("Click a Node or Arc to edit."), {}

    # Determine which input triggered the callback
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[1]

    data = node_data if trigger_id == 'tapNodeData' else edge_data

    if not data:
        return html.Div("Click a Node or Arc to edit."), {}

    is_node = 'type' in data
    element_id = data['id']

    if is_node:
        node_data = current_state['nodes'].get(element_id, {})
        editor_title = f"Editing Node: {element_id}"

        inputs = [
            html.H4(editor_title, style={"marginBottom": "10px"}),
            html.Label("Type:", style={"fontWeight": "bold", "display": "block"}),
            dcc.Dropdown(
                id={'type': 'attr-input', 'id': 'type'},
                options=[
                    {'label': 'Source', 'value': 'Source'},
                    {'label': 'Sink', 'value': 'Sink'},
                    {'label': 'Storage', 'value': 'Storage'},
                ],
                value=node_data.get('type'), clearable=False, style={"marginBottom": "10px"}
            )
        ]

        node_type = node_data.get('type')
        if node_type == 'Source':
            inputs.extend([
                html.Label("Cost ($/unit):", style={"fontWeight": "bold", "display": "block"}),
                dcc.Input(id={'type': 'attr-input', 'id': 'cost'}, type='number', value=node_data.get('cost'),
                          style=EDIT_INPUT_STYLE),
                html.Label("Max Generation:", style={"fontWeight": "bold", "display": "block", "marginTop": "10px"}),
                dcc.Input(id={'type': 'attr-input', 'id': 'max_g'}, type='number', value=node_data.get('max_g'),
                          style=EDIT_INPUT_STYLE),
            ])
        elif node_type == 'Sink':
            inputs.extend([
                html.Label("Demand Value:", style={"fontWeight": "bold", "display": "block"}),
                dcc.Input(id={'type': 'attr-input', 'id': 'demand'}, type='number', value=node_data.get('demand'),
                          style=EDIT_INPUT_STYLE),
            ])
        elif node_type == 'Storage':
            inputs.extend([
                html.Label("Initial Charge (S_init):", style={"fontWeight": "bold", "display": "block"}),
                dcc.Input(id={'type': 'attr-input', 'id': 's_init'}, type='number', value=node_data.get('s_init'),
                          style=EDIT_INPUT_STYLE),
                html.Label("Max Charge (S_max):",
                           style={"fontWeight": "bold", "display": "block", "marginTop": "10px"}),
                dcc.Input(id={'type': 'attr-input', 'id': 's_max'}, type='number', value=node_data.get('s_max'),
                          style=EDIT_INPUT_STYLE),
            ])

        inputs.append(html.P(f"Position: ({int(node_data['pos'][0])}, {int(node_data['pos'][1])})",
                             style={"fontSize": "small", "marginTop": "15px"}))
        return html.Div(inputs), {'id': element_id, 'is_node': True}

    else:  # Arc/Edge
        arc_index = next((i for i, a in enumerate(current_state['arcs']) if a['id'] == element_id), None)
        if arc_index is None: return "Error: Arc not found.", {}

        arc_data = current_state['arcs'][arc_index]
        editor_title = f"Editing Arc: {element_id} ({arc_data['source']} -> {arc_data['target']})"

        inputs = [
            html.H4(editor_title, style={"marginBottom": "10px"}),
            html.Label("Capacity:", style={"fontWeight": "bold", "display": "block"}),
            dcc.Input(id={'type': 'attr-input', 'id': 'capacity'}, type='number', value=arc_data.get('capacity'),
                      style=EDIT_INPUT_STYLE),
        ]
        return html.Div(inputs), {'id': element_id, 'is_node': False, 'index': arc_index}


# 2. Update Problem State on Input Change
@callback(
    Output('problem-state', 'data', allow_duplicate=True),
    Input({'type': 'attr-input', 'id': dash.ALL}, 'value'),
    State('last-clicked-element', 'data'),
    State('problem-state', 'data'),
    prevent_initial_call=True
)
def update_problem_state_from_editor(values, clicked_element, current_state):
    """Updates the internal problem state when an attribute input field changes."""
    if not clicked_element or not clicked_element.get('id'):
        raise dash.exceptions.PreventUpdate

    state = copy.deepcopy(current_state)
    element_id = clicked_element['id']

    ctx_inputs = dash.ctx.inputs_list[0]
    if not ctx_inputs:
        raise dash.exceptions.PreventUpdate

    for item in ctx_inputs:
        attr_key = item['id']['id']
        new_value = item['value']

        try:
            if attr_key == 'cost':
                val = float(new_value)
            elif attr_key == 'type':
                val = new_value
            else:
                val = int(new_value)
        except (ValueError, TypeError):
            continue

        if clicked_element['is_node']:
            if element_id in state['nodes']:
                state['nodes'][element_id][attr_key] = val
        else:
            arc_index = clicked_element.get('index')
            if arc_index is not None and arc_index < len(state['arcs']):
                state['arcs'][arc_index][attr_key] = val

    return state


# 3. Graph Redraw (Triggered by problem-state or result-state change)
@callback(
    [
        Output('network-graph', 'elements'),
        Output('network-graph', 'layout'),
        Output('graph-info', 'children'),
    ],
    [
        Input('problem-state', 'data'),
        Input('results-state', 'data'),
    ]
)
def update_graph_and_info(state, results):
    if not state: state = DEFAULT_PROBLEM_STATE  # Handle empty state
    elements = generate_cytoscape_elements(state, results)
    layout = {'name': 'preset'}

    num_nodes = len(state.get('nodes', {}))
    num_arcs = len(state.get('arcs', []))
    info_message = f"{num_nodes} Nodes, {num_arcs} Arcs. Drag to layout. Click to edit."

    return elements, layout, info_message


# 4. Run Optimization and Update Results State
@callback(
    [
        Output("solving-status", "children"),
        Output("results-output", "children"),
        Output("results-state", "data"),
    ],
    Input("run-button", "n_clicks"),
    State("problem-state", "data"),
    State("sampler-select", "value"),
    State("time-limit-input", "value"),
    State("api-token-input", "value"),
    prevent_initial_call=True
)
def run_solver(n_clicks, problem_data, sampler_type, time_limit, api_token):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    status_message = html.Div([
        f"Solving with {sampler_type} Sampler...",
        dcc.Loading(type="default", children=html.Div(id="loading-spinner"), fullscreen=False, style={"height": "20px"})
    ])

    # 1. Validation
    for node_id, data in problem_data['nodes'].items():
        if data.get('type') == 'Source' and (data.get('cost') is None or data.get('max_g') is None):
            return "ERROR: Source node attributes are incomplete.", None, {}
        # Add more validation as needed...

    # 2. Call the core optimization logic
    try:
        results = build_and_solve_network_flow(
            problem_data, sampler_type, int(time_limit), api_token
        )
    except Exception as e:
        return f"CRITICAL RUNTIME ERROR: {e}", None, {}

    # 3. Check for failure states
    if results.get("status") == "FAILED":
        return f"SOLVER FAILED: {results.get('message', 'Unspecified error.')}", None, {}

    # --- SUCCESSFUL RESULT DISPLAY (Copied from previous, still valid) ---
    header_style = {"backgroundColor": "#E0E0E0", "padding": "10px", "borderRadius": "5px", "marginTop": "10px"}
    status_content = html.Div(
        [
            html.P(f"STATUS: {results['status']}", style={"fontWeight": "bold", "color": "green"}),
            html.P(f"Sampler: {results['sampler']}"),
            html.P(f"Total Solve Time: {results.get('solver_time', 'N/A'):.2f} seconds"),
        ], style=header_style
    )
    objective_content = html.Div(
        [html.H3(f"Minimum Total Cost (Objective Z): ${results['objective']:.2f}",
                 style={"color": "#1F77B4", "fontSize": "20px"})],
        style={"marginTop": "15px", "marginBottom": "15px"}
    )

    variable_rows = []
    flow_rows = []

    for key, value in results.items():
        if key.startswith('g_N') or key == 's':
            desc = f"Generation {key.split('_')[1]}" if key.startswith('g_N') else "Final Storage (s)"
            variable_rows.append(html.Tr([html.Td(desc), html.Td(value)]))
        elif key.startswith('x_'):
            parts = key.split('_')
            desc = f"Flow {parts[1]} -> {parts[2]}"
            flow_rows.append(html.Tr([html.Td(desc), html.Td(value)]))

    variables_table = html.Div(
        [
            html.H4("Source & Storage Variables:", style={"color": "#2CA02C"}),
            html.Table(
                [html.Thead(html.Tr([html.Th("Variable"), html.Th("Optimal Value")])), html.Tbody(variable_rows)],
                style={"borderCollapse": "collapse", "width": "100%", "marginTop": "10px"}, className="results-table"
            ),
            html.H4("Optimal Flow Variables ($x_{i,j}$):", style={"color": "#2CA02C", "marginTop": "20px"}),
            html.Table(
                [html.Thead(html.Tr([html.Th("Arc Flow"), html.Th("Optimal Value")])), html.Tbody(flow_rows)],
                style={"borderCollapse": "collapse", "width": "100%", "marginTop": "10px"}, className="results-table"
            ),
        ], style={"padding": "10px", "border": "1px solid #ddd", "borderRadius": "5px"}
    )

    return status_content, html.Div([objective_content, variables_table]), results


# 5. Handle Node Dragging (Update position in problem state)
@callback(
    Output('problem-state', 'data', allow_duplicate=True),
    Input('network-graph', 'mouseupNode'),
    State('problem-state', 'data'),
    prevent_initial_call=True
)
def update_node_position_on_drag(node_data, current_state):
    """Updates the position of a node in the internal state after it is dragged."""
    if node_data:
        state = copy.deepcopy(current_state)
        node_id = node_data['data']['id']
        x, y = node_data['position']['x'], node_data['position']['y']

        if node_id in state['nodes']:
            state['nodes'][node_id]['pos'] = [x, y]
            return state
    raise dash.exceptions.PreventUpdate


# 6. Store selected nodes (for arc creation)
@callback(
    Output('selected-nodes-store', 'data'),
    Input('network-graph', 'selectedNodeData')
)
def store_selected_nodes(selected_nodes):
    if selected_nodes:
        return [node['id'] for node in selected_nodes]
    return []


# 7. Add/Delete Node/Arc Buttons
@callback(
    Output('problem-state', 'data'),
    [
        Input('add-node-button', 'n_clicks'),
        Input('add-arc-button', 'n_clicks'),
        Input('delete-button', 'n_clicks')
    ],
    [
        State('problem-state', 'data'),
        State('selected-nodes-store', 'data'),
        State('network-graph', 'selectedEdgeData')
    ],
    prevent_initial_call=True
)
def modify_graph(add_node_clicks, add_arc_clicks, delete_clicks, current_state, selected_nodes, selected_edges):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    state = copy.deepcopy(current_state)

    if button_id == 'add-node-button':
        # Add a new node
        new_id = f"N_new_{add_node_clicks}"
        state['nodes'][new_id] = {"type": "Sink", "demand": 0, "pos": [50, 100 + (add_node_clicks * 10)]}

    elif button_id == 'add-arc-button':
        # Add an arc between two selected nodes
        if len(selected_nodes) == 2:
            src, dst = selected_nodes
            new_id = f"A_{src}_{dst}"
            # Check if arc already exists
            if not any(a['id'] == new_id for a in state['arcs']):
                state['arcs'].append({"id": new_id, "source": src, "target": dst, "capacity": 100})
        else:
            print("Warning: Must select exactly two nodes to add an arc.")

    elif button_id == 'delete-button':
        # Delete selected nodes
        if selected_nodes:
            for node_id in selected_nodes:
                state['nodes'].pop(node_id, None)
            # Filter out arcs connected to the deleted nodes
            state['arcs'] = [
                arc for arc in state['arcs']
                if arc['source'] not in selected_nodes and arc['target'] not in selected_nodes
            ]

        # Delete selected edges
        if selected_edges:
            edge_ids_to_delete = {edge['id'] for edge in selected_edges}
            state['arcs'] = [arc for arc in state['arcs'] if arc['id'] not in edge_ids_to_delete]

    return state


# 8. Handle File Upload
@callback(
    [
        Output('problem-state', 'data', allow_duplicate=True),
        Output('solving-status', 'children', allow_duplicate=True)
    ],
    Input('upload-topography', 'contents'),
    State('upload-topography', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    try:
        if filename.endswith('.json'):
            new_state = parse_json_to_state(contents)
            return new_state, html.Div(f"Successfully loaded {filename}", style={"color": "green"})
        elif filename.endswith('.xlsx'):
            if not pd:
                raise ImportError("Pandas/Openpyxl not installed. Cannot read XLSX.")
            new_state = parse_xlsx_to_state(contents)
            return new_state, html.Div(f"Successfully loaded {filename}", style={"color": "green"})
        else:
            return dash.no_update, html.Div("Error: File must be .json or .xlsx", style={"color": "red"})
    except Exception as e:
        return dash.no_update, html.Div(f"Error parsing {filename}: {e}", style={"color": "red"})


# 9. Simulated Export (same as before)
@callback(
    Output("solving-status", "children", allow_duplicate=True),
    Input("export-button", "n_clicks"),
    State("sampler-select", "value"),
    prevent_initial_call=True
)
def export_model(n_clicks, sampler_type):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    ext = 'nl' if sampler_type == 'NL' else 'cqm'
    return html.Div(
        f"Simulated Export successful! The {sampler_type} model would typically be exported as a .{ext} file.",
        style={"backgroundColor": "#9467BD", "color": "white", "padding": "10px", "borderRadius": "5px"}
    )


if __name__ == '__main__':
    if not all([dash, cyto, dimod, pd]):
        print("\n--- FATAL ERROR ---")
        print("One or more required libraries are missing.")
        print("Please review the errors above and install all dependencies.")
        print("pip install dash dash-cytoscape dimod dwave-optimization dwave-system pandas openpyxl")
        sys.exit(1)

    app.run_server(debug=True)

