import traceback
import dash
import pandas as pd
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import networkx as nx
from dash.dependencies import Input, Output, State
import dash_interactive_graphviz
from graph_processor import GraphProcessor
from graph_visualization import GraphVisualizator
import io
import _pickle as pickle
import base64

graph_files = ['Example1.csv', 'Example2.csv']
graph_processors = [GraphProcessor(filename) for filename in graph_files]
graph_visualizators = [GraphVisualizator(gp) for gp in graph_processors]

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                assets_folder="./assets",
                title="Applet UMA")
IMATH_LOGO = src = dash.get_asset_url('logo1.svg')
MODAL_EXAMPLE = dash.get_asset_url('example.svg')

navbar = dbc.Navbar(
    dbc.Container(
        [
            dcc.Store(id='graph-store', data={'selected_graph_index': 0}),
            dcc.Store(id='custom-gp'),
            dcc.Store(id='custom-gv'),
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=IMATH_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("UMA Applet", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://imath.pixel-online.org",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                [
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Example 1", id="navitem-1", href="#", active=True)),
                            dbc.NavItem(dbc.NavLink("Example 2", id="navitem-2", href="#")),
                        ],
                        className="me-auto",  # Align NavItems to the left
                    ),
                    dbc.NavItem(
                        dbc.InputGroup(
                            [
                                dcc.Upload(
                                    id='upload-data',
                                    children=[
                                        dbc.Button("Upload File", id="upload-file"),
                                    ],
                                    multiple=False,
                                ),
                                dbc.Button(
                                    id="open-graph-format",
                                    color="transparent",
                                    children=[
                                        html.I(className="bi bi-question-circle", style={"color": "white"})
                                    ],
                                )
                            ],
                            className="ms-auto",
                        )
                    ),
                ],
                className="ms-auto",  # Align Upload to the right
                id="navbar-collapse",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)

modalFormat = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("File Format")),
                dbc.ModalBody("You can upload a .CSV file for checking your own concept maps/graphs. The data should "
                              "be formatted in the following way:"),
                html.Code("node_name;child_reference1;child_reference2;child_reference3", className="text-center"),
                dbc.ModalBody("For example, the following:"),
                html.Code([
                    "A;2;;",
                    html.Br(),
                    "B;1;;",
                    html.Br(),
                    "C;1;2;"],
                    className="text-center"
                ),
                dbc.ModalBody("Would generate:"),
                dbc.Container(children=[
                    html.Img(src=MODAL_EXAMPLE)
                ],
                    className="text-center",
                    fluid=True
                ),
            ],
            id="modal-graph-format",
            size="lg",
            is_open=False,
        ),
    ]
)

modalError = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Error")),
                dbc.ModalBody("", id="modal-body-text"),  # Added an id to the ModalBody
            ],
            id="modal-graph-error",
            size="lg",
            is_open=False,
        ),
    ]
)

modalGraph = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Information about the displayed Graph Data")),
                dbc.ModalBody("Understanding the structure of a concept map is essential for effective learning, "
                              "Thus we some provide metrics that can help with checking the complexity of the graph,"
                              "allowing you to make informed decisions for evaluating or improving a graph.:"),
                html.Ul([
                    html.Li("Nodes: Nodes represent individual concepts within the map. A lower number of nodes often "
                            "implies a more focused concept map."),
                    html.Li("Edges: Edges signify relationships between nodes. Fewer edges generally indicate a "
                            "simpler structure, allowing a easier navigation"),
                    html.Li("Descendants: The number of descendants measures the layers of connected concepts "
                            "stemming from a selected node. A smaller count suggests a more straightforward "
                            "structure."),
                ]),
                dbc.ModalBody("By clicking on a node, you can access the number of descendants for that certain node.")
            ],
            id="modal-graph-info",
            size="lg",
            is_open=False,
        ),
    ]
)

modalView = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Information about Layout Metrics")),
                dbc.ModalBody(children=[
                    "Organizing information in a visually appealing approach is very important for clarity "
                              "in concept maps. When you select a specific graph layout algorithm (provided by "
                              "Graphviz, you can check the features of each layout in ",
                              html.A("the Graphviz documentation", href="https://graphviz.org/docs/layouts/",
                                     target="_blank"),
                              "), the following metrics are calculated by checking the generated SVG: "]),
                html.Ul([
                    html.Li("Crossings Between Edges: The number of crossings between edges in the layout. A lower "
                            "count indicates a more organized representation and clearer for the student."),
                    html.Li("Dummy Nodes: Dummy nodes are introduced during the visualization process to optimize "
                            "layout. Monitoring the number of dummy nodes generated provides insights into the "
                            "algorithm's efficiency in handling complex relationships and the clear representation "
                            "that we can obtain, so the lower the better for a hierarchical structure."),
                    html.Li("Median distance: The number of descendants measures the layers of connected concepts "
                            "stemming from a selected node. A smaller count suggests a more straightforward "
                            "structure and a easier way to navigate."),
                ]),
            ],
            id="modal-view-info",
            size="lg",
            is_open=False,
        ),
    ]
)



engineDropdown = html.Div([
    html.Label(['Layout engine:'], style={'font-weight': 'bold'}),
    dcc.Dropdown(
        id='engine-dropdown',
        options=['dot', 'neato', 'twopi', 'circo', 'fdp', 'osage'],
        value='dot',
        clearable=False,
        className="w-50"
    ),
])

app.layout = html.Div([
    navbar,
    modalFormat,
    modalError,
    modalGraph,
    modalView,
    html.Div([
        dbc.Col(
            dash_interactive_graphviz.DashInteractiveGraphviz(id="gv", engine="dot",
                                                              style={"height": "50vh", "width": "100%"}),
            style={"height": "50vh", "width": "100vw"}
        ),
    ], className="overflow-hidden"
    ),
    dbc.Container([
        dbc.Row(
            [
                html.Div(children=[
                    html.Div([
                        html.H4(style={"display": "flex", "align-items": "center"},
                                children=[
                                    dbc.Button(
                                        id="open-graph-info",
                                        color="transparent",
                                        children=[
                                            html.I(className="bi bi-question-circle", style={"color": "black"})
                                        ],
                                    ), "Graph data:"]),
                        html.Hr(style={"margin": "0px 0"}),
                        # Graph
                        html.Div(id='graph-nodes-info'),
                        html.Div(id='graph-edges-info'),
                        html.Div(id='node-info'),
                        html.Div(id='descendants-info'),
                        html.Br()
                    ],
                        className="col-sm border rounded bg-light"
                    ),
                    html.Br(),
                    html.Div(children=[
                        html.H4(style={"display": "flex", "align-items": "center"},
                                children=[
                                    dbc.Button(
                                        id="open-view-info",
                                        color="transparent",
                                        children=[
                                            html.I(className="bi bi-question-circle",
                                                   style={"color": "black", "display": "inline"})
                                        ],
                                    ),
                                    "Visualization data:"]),
                        html.Hr(style={"margin": "0px 0"}),
                        # Change layout
                        engineDropdown,
                        # Graph visualization metrics
                        dcc.Loading(
                            id="loading",
                            type="circle",
                            children=[
                                html.Div(id='dummy-nodes'),
                                html.Div(id='crossings-edges'),
                                html.Div(id='median-distance'),
                                html.Br(),
                            ]
                        )
                    ],
                        className="col-sm border rounded bg-light "
                    ),
                ],
                    className="col-sm-3 "),
                html.Div([
                    html.Div([
                        html.Br(style={"display": "block", "margin": "5px 0"}),
                        html.P([
                            "Our interactive app is crafted to serve as a visual tool for assessing critical elements "
                            "within a concept map. Inspired by pertinent research, notably [",
                            html.A("1", href="#reference-1"),
                            "],  which underscores the significance of map structure in the learning process, "
                            "our app focuses on salience metrics to evaluate the quality of concept maps."]),
                        html.Br(style={"display": "block", "margin": "5px 0"}),
                        html.P(
                            "Users have the ability to explore how metrics evolve based on the chosen layout for the "
                            "same dataset. Additionally, they can upload their own .CSV data and experiment with "
                            "various visualizations. Some noteworthy indicators related to the graph itself, "
                            "such as the number of nodes or edges, can provide valuable insights. A higher number of "
                            "nodes or edges might pose challenges for students in grasping concepts. It could be "
                            "beneficial to consider consolidating multiple concepts into a single node to present a "
                            "more comprehensive overview."),
                        html.Br(style={"display": "block", "margin": "5px 0"}),
                        html.P(
                            "Beyond basic graph metrics, other data points like the number of crossings between "
                            "edges, the distance between related concepts (nodes), or the generation of dummy nodes "
                            "for proper visualization by levels, reveal the visual complexity of the graph. This "
                            "complexity may impede students' navigation through the graph and hinder effective "
                            "learning. Understanding these aspects becomes crucial in enhancing the educational value "
                            "of the app."),
                        html.Hr(style={"margin": "0.1em auto"}),
                        html.P([
                            "1. Krieglstein, F., Schneider, S., Beege, M., & Rey, G. D. (2022).",
                            html.A(
                                "How the design and complexity of concept maps influence cognitive learning processes. ",
                                href="https://doi.org/10.1007/s11423-022-10083-2"),
                            "Educational technology research and development, 70(1), 99-118. ",
                            html.Br(),
                            "doi:",
                            html.A("10.1007/s11423-022-10083-2", href="https://doi.org/10.1007/s11423-022-10083-2")],
                            id="reference-1"),

                    ],
                        className="col-sm border rounded bg-light h-100")
                ],
                    className="col-sm")
            ]
        )
    ],
        fluid=True),
])


def get_graph_object(stored_data, custom_data, graph_list):
    if custom_data is None:
        selected_index = stored_data['selected_graph_index']
        return graph_list[selected_index]
    else:
        return pickle.loads(base64.b64decode(custom_data))


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    if r'.csv' not in filename:
        raise ValueError("File is not CSV")
    decoded = base64.b64decode(content_string)
    return decoded.decode()


def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    [Output('custom-gp', 'data'),
     Output('custom-gv', 'data'),
     Output("modal-graph-error", "is_open"),
     Output("modal-body-text", "children"),
     ],
    [Input('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def read_custom_file(contents, filename):
    if contents is not None:
        try:
            parsed = parse_contents(contents, filename)
            gp = GraphProcessor(io.StringIO(parsed))
            gv = GraphVisualizator(gp)

            custom_gp = pickle.dumps(gp)
            decoded_gp = base64.b64encode(custom_gp).decode('utf8')
            custom_gv = pickle.dumps(gv)
            decoded_gv = base64.b64encode(custom_gv).decode('utf8')

            return decoded_gp, decoded_gv, False, None
        except Exception as e:
            error_message = f"Error: {str(e)}"
            return None, None, True, error_message

    return None, None, False, None


@app.callback(
    Output('gv', 'dot_source'),
    [Input('graph-store', 'data'),
     Input('custom-gp', 'data')],
)
def load_graph(stored_data, custom_gp):
    selected_gp = get_graph_object(stored_data, custom_gp, graph_processors)
    figure = selected_gp.dot_data
    return figure


@app.callback(
    [Output('graph-store', 'data'),
     Output('custom-gp', 'clear_data'),
     Output('custom-gv', 'clear_data')],
    [Input(f"navitem-{i}", "n_clicks") for i in range(1, len(graph_files) + 1)],
    prevent_initial_call=True
)
def update_selected_graph(*inputs):
    ctx = dash.callback_context
    if ctx.triggered_id is None:
        return {'selected_graph_index': 0}
    triggered_button = int(ctx.triggered_id.split("-")[-1])

    return {'selected_graph_index': triggered_button - 1}, True, True


@app.callback(
    Output('gv', 'engine'),
    [Input('engine-dropdown', 'value')],
    [State('graph-store', 'data'),
     State('custom-gv', 'data')],
    prevent_initial_call=True
)
def change_view(value, stored_data, custom_gv):
    gv = get_graph_object(stored_data, custom_gv, graph_visualizators)
    gv.change_algorithm(value)
    return value


@app.callback(
    [Output('graph-nodes-info', 'children'),
     Output('graph-edges-info', 'children')],
    [Input('gv', 'dot_source'),
     Input('gv', 'engine')],
    [State('graph-store', 'data'),
     State('custom-gp', 'data')]
)
def graph_info(source, engine, stored_data, custom_gp):
    gp = get_graph_object(stored_data, custom_gp, graph_processors)
    nodes = len(gp.graph.nodes())
    string_nodes = f"Number of Nodes: {nodes}"
    edges = len(gp.graph.edges())
    string_edges = f"Number of Edges: {edges}"
    return string_nodes, string_edges


@app.callback(
    [
        Output('dummy-nodes', 'children'),
        Output('crossings-edges', 'children'),
        Output('median-distance', 'children'),
    ],
    [Input('gv', 'dot_source'),
     Input('gv', 'engine')],
    [State('graph-store', 'data'),
     State('custom-gv', 'data')]
)
def visualization_info(source, engine, stored_data, custom_gv):
    selected_gv = get_graph_object(stored_data, custom_gv, graph_visualizators)
    if selected_gv.prog is not engine:
        selected_gv.change_algorithm(engine)
    dummy_nodes = selected_gv.dummy_nodes
    string_nodes = f"Number of Dummy Nodes: {dummy_nodes}"
    crossing_edges = selected_gv.crossings
    string_crossings = f"Number of Crossings: {crossing_edges}"
    median_distance_matrix = selected_gv.distance_matrix
    nodes = list(median_distance_matrix.keys())
    df = pd.DataFrame(median_distance_matrix, index=nodes)
    string_distance = f"Median of Distance: {df.stack().median()}"
    return string_nodes, string_crossings, string_distance


@app.callback(
    [Output('node-info', 'children'),
     Output('descendants-info', 'children')],
    [Input('gv', 'selected_node')],
    [State('graph-store', 'data'),
     State('custom-gp', 'data')]
)
def selected_node_info(selected, stored_data, custom_gp):
    selected_gp = get_graph_object(stored_data, custom_gp, graph_processors)
    G = selected_gp.graph
    if selected is not None:
        num_descendants = len(nx.descendants(G, selected))
        descendants = f"Number of Descendants: {num_descendants}"
        return selected, descendants
    else:
        return "No node selected", ""


@app.callback(
    [Output(f"navitem-{i}", "active") for i in range(1, 3)],
    [Input(f"navitem-{i}", "n_clicks") for i in range(1, 3)] + [Input("upload-data", "contents")],
    prevent_initial_call=True,
)
def update_active(*inputs):
    button_clicks = inputs[:-1]
    upload_clicks = inputs[-1]

    ctx = dash.callback_context

    triggered_button = None

    for i in range(1, 3):
        if f"navitem-{i}" in ctx.triggered_id:
            triggered_button = i
            break

    # If a NavItem is clicked, deactivate the other ones
    if triggered_button is not None:
        return [button == triggered_button for button in range(1, 3)]

    # If the Upload button is clicked, deactivate all NavItems
    if upload_clicks:
        return [False] * len(button_clicks)

    return [True, False]  # Default state


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


app.callback(
    Output("modal-graph-info", "is_open"),
    Input("open-graph-info", "n_clicks"),
    State("modal-graph-info", "is_open"),
)(toggle_modal)

app.callback(
    Output("modal-view-info", "is_open"),
    Input("open-view-info", "n_clicks"),
    State("modal-view-info", "is_open"),
)(toggle_modal)

app.callback(
    Output("modal-graph-format", "is_open"),
    Input("open-graph-format", "n_clicks"),
    State("modal-graph-format", "is_open"),
)(toggle_modal)

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",debug=False)
