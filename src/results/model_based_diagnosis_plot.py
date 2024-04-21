#%%
import math
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import plot

from src.results.graph_characteristics_report import paths_to_crash
from src.utils import load_data, load_graph, load_digraph
import json
import graph_characteristics_report as gcr
import networkx as nx
from src.results import ACA_plots

def path_log_check(path, log):
    node_counter = 0
    path_length = len(path)
    for node in path:
        if node in log.index and log[node] == 0:
            return False
        elif node in log.index and log[node] == 1:
            node_counter += 1
    if node_counter > 1:
        return True


def mbd_find_happened(logs: pd.DataFrame, paths: List[List[str]]) -> pd.Series:
    occured_path = pd.Series(index=logs.index).apply(lambda _: [])
    for index, log in logs.iterrows():
        for path in paths:
            if path_log_check(path, log):
                occured_path[index].append(path)
    return occured_path


def find_root_causes(explanations):
    explanation_count = pd.Series(index=explanations.index).apply(lambda x: [])
    for index, row in explanations.items():
        for i in row:
            for event in logs.columns:
                if (
                    event in i
                    and event not in explanation_count[index]
                    and event != "crash"
                ):
                    explanation_count[index].append(event)
    return explanation_count


def root_cause_logs(explanation_count):
    mbd = pd.DataFrame(index=explanation_count.index, columns=logs.columns)
    mbd = mbd.fillna(0)
    for k, v in explanation_count.items():
        for node in logs.columns:
            if node in v:
                mbd[node][k] = 1
    return mbd


def upset_mbd_plot(mbd):
    cols = list(logs.columns)
    cols.remove("crash")
    existing_root_causes = list(mbd.columns[(mbd.sum() > 0)])
    df_upset_plot = (
        mbd[[*existing_root_causes, "crash"]].groupby(existing_root_causes).count()
    )
    df_upset_plot = df_upset_plot.sort_values(by="crash", ascending=False)
    df_upset_plot = df_upset_plot[1:]
    fig = plt.figure(figsize=(14, 9))
    plot(df_upset_plot["crash"], sort_by="cardinality")
    plt.savefig("../../results/plots/diagnosis.pdf")
    plt.show()


def load_logs() -> pd.DataFrame:
    logs = load_data("pipeline/06_diagnosis/df_events.csv")
    logs.set_index("id", inplace=True)
    return logs[logs["crash"] == 1]

def build_node_for_causal_canvas(name,formula, is_exo):
    x,y = generate_x_y()
    true = True
    false = False
    if is_exo:
        node = {
            "x": x,
            "y": y,
            "title": name,
            "value": false,
            "id": name,
            "formula": formula,
            "isExogenousVariable": true
        }
    else:
        node = {
            "x": x,
            "y": y,
            "title": name,
            "value": false,
            "id": name,
            "formula": formula,
            "isExogenousVariable": false
        }
    return node

def build_acyclic_graph(g):
    paths = gcr.paths_to_crash(g)
    true = 'true'
    false = 'false'
    g_acyclic = nx.DiGraph()
    for path in paths:
        if len(path) > 1:
            for k,v in enumerate(path[:-1]):
                g_acyclic.add_edge(path[k], path[k+1])
        else:
            g_acyclic.add_edge(path[0], path[1])
    return g_acyclic

def build_monitored_graph(g):
    logs = load_data('pipeline/06_diagnosis/df_events.csv')
    monitored = []
    logs.columns = [i if ' ' not in i else i.replace(' ','_') for i in logs.columns]
    for i in logs.columns:
        for node in g.nodes:
            if i == node:
                monitored.append(i)

    # remove the exos that were not monitored
    g_observable = nx.DiGraph()
    for event in monitored:
        paths_one_exo = nx.all_shortest_paths(g, event, 'crash')
        for path in paths_one_exo:
            for k,v in enumerate(path[:-1]):
                g_observable.add_edge(path[k],path[k+1])
    return g_observable

def edit_node_name(g,char1,char2):
    mapping = {}
    edit_space = lambda x:x.replace(char1, char2)
    for i in g.nodes:
        mapping.update({i:edit_space(i)})
    g = nx.relabel_nodes(g, mapping)
    return g

def build_causalmodel_for_canvas(g):
    canvas_file = {"id": "0d8a462e-e3bf-45e8-a294-c90fce8a9f0e",
         "directed": True,
         "title": "Rock-Throwing",
         "nodes": [
             ]}

    graph = edit_node_name(g,' ','_')
    graph = edit_node_name(graph,'-','_')
    graph = edit_node_name(graph, '.', '_')

    for node in graph.nodes():
        formula = ""
        preds = list(graph.predecessors(node))
        if len(preds)==1:
            for pred in preds:
                formula = preds[0]
            canvas_file["nodes"].append(build_node_for_causal_canvas(node, formula, False))
        if len(preds) > 1:
            formula = preds[0]
            for pred in preds[1:]:
                formula += "|" + pred
            canvas_file["nodes"].append(build_node_for_causal_canvas(node, formula, False))
        if len(preds) == 0:
            canvas_file["nodes"].append(build_node_for_causal_canvas("Exo_"+node,formula,True))
            canvas_file["nodes"].append(build_node_for_causal_canvas(node,"Exo_"+node,False))


    with open('data.causalmodel', 'w') as f:
        json.dump(canvas_file, f, indent=1)
    return canvas_file

def update_canvas_file(incidents, canvas_file):
    # exos = [node['title'] for node in model_for_canvas['nodes'] if 'Exo' in node['title']]
    for node in canvas_file['nodes']:
        for incident in incidents:
            if incident == node['title']:
                node['value'] = True
                pass
    return canvas_file

def build_causalmodel_for_ACA(file_name, graph_name):
    original_file = json.load(open('{}.causalmodel'.format(file_name),'r'))
    endos = []
    exos = []
    for node in original_file['nodes']:
        node['formula'] = node['formula'].replace('!', '~')
        if node['isExogenousVariable'] == True:
            exos.append(node['title'])
        else:
            endos.append({'formula':node['formula'],'name':node['title']})

    ACA_file = {'endos': endos, 'exos': exos, 'name': file_name}
    with open('ACA_file_'+graph_name+'.causalmodel', 'w') as f:
        json.dump(ACA_file, f, indent=1)
    return ACA_file

def generate_x_y():
    import math
    import random
    R = 1000
    r = R
    theta = random.random() * 2 * math.pi
    x = 0 + r * round(math.cos(theta),2)
    y = 0 + r * round(math.sin(theta),2)
    return x,y

def plot_digraph(g,name):
    # sort nodes alphabetically to make the comparison of graphs easier in the paper
    H = nx.DiGraph()
    H.add_nodes_from(sorted(g.nodes(data=True)))
    H.add_edges_from(g.edges(data=True))

    plt.close()
    # plt.margins(x=0.1,y=0)
    nx.draw(H, pos=nx.circular_layout(H), with_labels=True, node_size=1000, arrowsize=30,
            node_color='darkgrey', font_size=24)
    plt.savefig('{}.pdf'.format(name),)
    plt.show()

def plot_adjacent_nodes(g,node):
    parents = []
    children = []
    for edge in g.edges:
        if edge[1] == node:
            parents.append(edge[0])
        if edge[0] == node:
            children.append(edge[1])
    g_sub = g.subgraph([node, *parents, *children])
    nx.draw(g_sub, pos=nx.circular_layout(g_sub), with_labels=True, node_size=1000, arrowsize=30,
            node_color='darkgrey', font_size=24)
    plt.show()

def graph_density(graph):
    den = len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1))
    return den

if __name__ == "__main__":
    for i in ['baseline', 'lemma', 'pos', 'root']:
        graph = load_digraph(i)
        graph_acyclic = build_acyclic_graph(graph)
        graph_monitored = build_monitored_graph(graph_acyclic)
        plot_digraph(graph_monitored,i)
        model_for_canvas = build_causalmodel_for_canvas(graph_monitored)
        model_for_ACA = build_causalmodel_for_ACA('data',i)

    incidents = ACA_plots.extract_variables_from_logs('../../src/mbd/LogAnalyzer/examples/11e03cadf00000039.log')
    mapped_incidents = ACA_plots.map_events(incidents)
    updated_canvas = update_canvas_file(mapped_incidents, model_for_canvas)

    graph = load_graph("baseline")
    paths = paths_to_crash(graph)
    logs = load_logs()

    explanations = mbd_find_happened(logs, paths)
    explanation_count = find_root_causes(explanations)
    mbd = root_cause_logs(explanation_count)
    upset_mbd_plot(mbd)