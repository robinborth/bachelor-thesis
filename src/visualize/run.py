from difflib import SequenceMatcher
from itertools import chain

import dash
import dash_cytoscape as cyto
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from dash import dcc, html
from dash.dependencies import Input, Output, State
from DEPRECATED_data.util import load_data
from scipy.stats import hmean

####################################################################################################
# GLOBAL VARIABLES TO CHANGE
####################################################################################################

GRAPH_SPACE = 2000
SEED = 0
CAUSAL_GRAPH = "flight_logs"  # 'flight_logs"
BOOTSTRAP = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"

####################################################################################################
# BASE SETTINGS
####################################################################################################

app = dash.Dash(title="Causal Graph Visualisation", external_stylesheets=[BOOTSTRAP])
nlp = spacy.load("en_core_web_lg")
causal_graph = load_data(CAUSAL_GRAPH)
graph = nx.from_pandas_edgelist(
    causal_graph,
    "cause",
    "effect",
    edge_attr="weight",
    create_using=nx.DiGraph(seed=SEED),
)
nlp_nodes = {node: nlp(node) for node in graph.nodes}

####################################################################################################
# UTILS
####################################################################################################


def similarity_score(d1, d2):
    has_vector = lambda t1, t2: t1.vector_norm and t2.vector_norm
    sequence_similarity = lambda t1, t2: SequenceMatcher(None, t1.text, t2.text).ratio()
    similarity_matrix = [
        [
            t1.similarity(t2) if has_vector(t1, t2) else sequence_similarity(t1, t2)
            for t2 in d2
        ]
        for t1 in d1
    ]
    similarity_df = pd.DataFrame(similarity_matrix)
    score = np.mean(
        [similarity_df.max(axis=1).mean(), similarity_df.max(axis=0).mean()]
    )
    return max(0, score)


def search(nlp_nodes, search_node):
    search_node = nlp(search_node.lower())
    score_nodes = [
        {"node": node, "score": similarity_score(nlp_nodes[node], search_node)}
        for node in nlp_nodes
    ]
    return sorted(score_nodes, key=lambda item: item["score"], reverse=True)


def search_pair(graph, nlp_nodes, search_cause, search_effect):
    search_cause, search_effect = nlp(search_cause.lower()), nlp(search_effect.lower())
    score_edges = []
    for cause, effect in graph.edges:
        score_cause = similarity_score(search_cause, nlp_nodes[cause])
        score_effect = similarity_score(search_effect, nlp_nodes[effect])
        score = hmean([score_cause, score_effect])
        score_edges.append(dict(cause=cause, effect=effect, score=score))
    return sorted(score_edges, key=lambda item: item["score"], reverse=True)


def weight_node(graph, node):
    successors = ((node, effect) for effect in graph.successors(node))
    predecessors = ((cause, node) for cause in graph.predecessors(node))
    return sum(
        [
            int(graph.get_edge_data(*edge)["weight"])
            for edge in chain(successors, predecessors)
        ]
    )


####################################################################################################
# BASE NODES + EDGES SETTINGS
####################################################################################################

pos = nx.spring_layout(graph, seed=SEED)

nodes = [
    {
        "data": {"id": node, "label": node, "mentioned": weight_node(graph, node)},
        "position": {"x": pos[node][0] * GRAPH_SPACE, "y": pos[node][1] * GRAPH_SPACE},
    }
    for node in graph.nodes
]
edges = [
    {"data": {"source": source, "target": target, "weight": weight}}
    for source, target, weight in graph.edges(data="weight")
]

elements = nodes + edges

####################################################################################################
# STYLESHEET SETTINGS
####################################################################################################

node_mentioned_range = [1, 3, 10, 30, 100]

node_color = [
    "rgb(95,0,164)" "rgb(160,27,155)",
    "rgb(200,67,123)",
    "rgb(249,151,63)",
    "rgb(241,245,33)",
]

stylesheet_node_base = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "width": 5,
            "height": 5,
            "font-size": 2,
            "background-color": "rgb(23,7,140)",
            "opacity": 0.5,
        },
    },
]

stylesheet_node_heatmap = [
    {
        "selector": f"[mentioned > {mentioned}]",
        "style": {"background-color": color, "opacity": 0.6 + (0.1 * index)},
    }
    for index, (mentioned, color) in enumerate(zip(node_mentioned_range, node_color))
]

edge_weight_range = [1, 3]

edge_color = [
    "rgb(200,67,123)",
    "rgb(241,245,33)",
]

stylesheet_edge_base = [
    {
        "selector": "edge",
        "style": {
            # 'label': 'data(weight)',
            "font-size": 5,
            "line-color": "rgb(95,0,164)",
            "opacity": 0.1,
            "width": 0.6,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#000000",
            "arrow-scale": 0.4,
        },
    },
]


stylesheet_edge_weights = [
    {
        "selector": f"[weight > {weight}]",
        "style": {
            "background-color": color,
            "opacity": 0.1 + (index + 1) * 0.05,
            "width": 0.6 + (index + 1) * 0.1,
        },
    }
    for index, (weight, color) in enumerate(zip(edge_weight_range, edge_color))
]

stylesheet = (
    stylesheet_node_base
    + stylesheet_node_heatmap
    + stylesheet_edge_base
    + stylesheet_edge_weights
)

####################################################################################################
# LAYOUT
####################################################################################################

search_node_layout = [
    html.Div(children="Search Node:", className="h4 pt-3"),
    html.Div(
        [
            html.Div(
                [
                    dcc.Input(
                        id="input-node-search",
                        type="text",
                        value="",
                        className="form-control flex",
                    ),
                ],
                className="col",
            ),
            html.Div(
                [
                    html.Button(
                        id="submit-node-search",
                        n_clicks=0,
                        children="Search",
                        className="btn btn-dark",
                    ),
                ],
                className="col",
            ),
        ],
        className="row",
    ),
]

search_edge_layout = [
    html.Div(children="Search Cause-Effect Pair:", className="h4 pt-3"),
    html.Div(
        [
            html.Div(
                [
                    dcc.Input(
                        id="input-cause-search",
                        type="text",
                        value="",
                        className="form-control flex",
                    ),
                ],
                className="col",
            ),
            html.Div(
                [
                    dcc.Input(
                        id="input-effect-search",
                        type="text",
                        value="",
                        className="form-control flex",
                    ),
                ],
                className="col",
            ),
            html.Div(
                [
                    html.Button(
                        id="submit-edge-search",
                        n_clicks=0,
                        children="Search",
                        className="btn btn-dark",
                    ),
                ],
                className="col",
            ),
        ],
        className="row",
    ),
]

search_result_layout = [
    dcc.Slider(
        min=0,
        max=50,
        step=2,
        value=8,
        id="search-slider",
        tooltip={"placement": "bottom", "always_visible": True},
        className="p-0 pt-3",
    ),
    html.Div(children="Results:", className="h4 pt-5"),
    html.Div(
        children=dcc.Loading(
            id="loading-search",
            type="default",
            children=html.Div(
                id="output-search", className="overflow-auto", style={"height": "500px"}
            ),
        ),
        className="pt-1",
    ),
]

graph_layout = [
    cyto.Cytoscape(
        id="causal-graph",
        style={"width": "100%", "height": "800px"},
        layout={"name": "preset"},
        elements=elements,
        stylesheet=stylesheet,
    )
]

app.layout = html.Div(
    [
        html.Div(children="Causal Graph Visualisation", className="h1"),
        html.Div(
            [
                html.Div(
                    [
                        *search_node_layout,
                        *search_edge_layout,
                        *search_result_layout,
                    ],
                    className="col-5",
                ),
                html.Div(graph_layout, className="col-7"),
            ],
            className="row",
        ),
    ],
    className="container py-5",
)


####################################################################################################
# DASH APP FUNCTIONALITY
####################################################################################################


def update_node_search(slider_value, node):
    if not node:
        return dcc.Markdown("*noting found*")
    to_markdown = (
        lambda item, mark="*": f"*{mark}{item['node']}*{mark} *({item['score']:.2f})*  \n"
    )
    search_result = search(nlp_nodes, node)
    return dcc.Markdown(
        "".join(
            chain(
                map(to_markdown, search_result[:3]),
                (to_markdown(item, mark="") for item in search_result[3:slider_value]),
            )
        )
    )


def update_edge_search(slider_value, cause, effect):
    if not (cause and effect):
        return dcc.Markdown("*noting found*")
    to_markdown = (
        lambda item, mark="*": f"*{mark}{item['cause']} => {item['effect']}*{mark} *({item['score']:.2f})*  \n"
    )
    search_result = search_pair(graph, nlp_nodes, cause, effect)
    return dcc.Markdown(
        "".join(
            chain(
                map(to_markdown, search_result[:3]),
                (to_markdown(item, mark="") for item in search_result[3:slider_value]),
            )
        )
    )


def next_node(node, slider_value, successor=True):
    """
    :param node: The target node
    :param slider_value: How many nodes the search return
    :param successor: If successor is true get all nodes that are the effect of the target node else get all
    the causes of the target node
    :return: The Markdown text for the node which is either the effect or target nodes + the corresponding weight
    """
    weight = lambda pre, suc: next(
        (w for u, v, w in graph.edges(data="weight") if u == pre and v == suc), -1
    )
    nodes_graph = graph.successors(node) if successor else graph.predecessors(node)
    node_dict = {
        n: weight(node, n) if successor else weight(n, node) for n in nodes_graph
    }
    to_markdown = lambda nd: "  ".join(f"*{n} ({w})*  \n" for n, w in nd if w > 0)
    markdown = to_markdown(
        sorted(node_dict.items(), key=lambda item: item[1], reverse=True)[:slider_value]
    )
    return markdown if markdown else "*nothing found*"


def update_edge_causal_graph(slider_value, edge_data):
    cause, effect, weight = (
        edge_data["source"],
        edge_data["target"],
        edge_data["weight"],
    )
    return dcc.Markdown(
        f"""
    **{cause} => {effect}** *({weight})*:

    *(node) =>* **{cause}**

    {next_node(cause, slider_value, successor=False)}

    **{effect}** *=> (node)*

    {next_node(effect, slider_value, successor=True)}
    """
    )


def update_node_causal_graph(slider_value, node_data):
    node, weight = node_data["label"], node_data["mentioned"]
    return dcc.Markdown(
        f"""
    **{node}** *({weight})*:

    *(node) =>* **{node}**

    {next_node(node, slider_value, successor=False)}

    **{node}** *=> (node)*

    {next_node(node, slider_value, successor=True)}
    """
    )


@app.callback(
    Output("output-search", "children"),
    Input("causal-graph", "tapNodeData"),
    Input("causal-graph", "tapEdgeData"),
    Input("submit-node-search", "n_clicks"),
    Input("submit-edge-search", "n_clicks"),
    State("search-slider", "value"),
    State("input-cause-search", "value"),
    State("input-effect-search", "value"),
    State("input-node-search", "value"),
)
def update_results(
    node_data, edge_data, btn_node, btn_edge, slider_value, cause, effect, node
):
    ctx, trigger = dash.callback_context.triggered[0]["prop_id"].split(".")
    if ctx == "causal-graph" and trigger == "tapNodeData":
        return update_node_causal_graph(slider_value, node_data)
    if ctx == "causal-graph":
        return update_edge_causal_graph(slider_value, edge_data)
    if ctx == "submit-node-search":
        return update_node_search(slider_value, node)
    return update_edge_search(slider_value, cause, effect)


if __name__ == "__main__":
    app.run_server(debug=True)
