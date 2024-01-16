#%% Manual concept graphs
import numpy as np
import pandas as pd
import networkx as nx

#%%  Test graph -----------------------------------------------------------------------------------------
# @brief get simple test graph
def get_simplifiedKnowledgeGraph():
    '''
    Returns simple concept map.
        Parameters: 
            None
        Returns:
            snG: Networkx graph with concepts as nodes and directed links as their dependences
    '''
    nodes = {
        "K1": dict(color="Red"),
        "K2": dict(color="Red"),
        "K3": dict(color="Red"),
        "K4": dict(color="Red"),
        "K5": dict(color="Blue"),
        "K6": dict(color="Blue"),
    }
    node_attrs = {
        "K1":{"Qs": [1,2,3,4,5,6,7,8]}, 
        "K2":{"Qs": [9,10,11,12,13,14]}, 
        "K3":{"Qs": [15,16,17,18,19,20]}, 
        "K4":{"Qs": [21,22,23,24,25,26]}, 
        "K5":{"Qs": [27,28,29,30,31,32]}, 
        "K6":{"Qs": [33,34,35,36,37,38]}}
    edges = [
        ("K1", "K2", "Strong"),
        ("K1", "K3", "Weak"),
        ("K2", "K4", "Strong"),
        ("K2", "K5", "Strong"),
        ("K3", "K5", "Strong"),
        ("K5", "K6", "Strong"),
    ]

    snG = nx.DiGraph(name='Concept map')
    
    snG.add_nodes_from(n for n in nodes.items())
    snG.add_edges_from((u, v, {"type": label}) for u, v, label in edges)
    nx.set_node_attributes(snG, node_attrs)

    return snG

# @brief get manual concept maps
def get_ConceptMap(code=None):
    '''
    Returns concept map.
        Parameters: 
            code
        Returns:
            snG: Networkx graph with concepts as nodes and directed links as their dependences
    '''

    if code=='mat_comp':

        nodes = {
            'computation',
            'determinant', 
            'inverse',
            'rank',
            'advanced_MC'
        }
        node_attrs = {
            'computation': {"Qs": {10, 0,  27, 14, 16, 2, 5}},
            'determinant': {"Qs": {12, 15, 19, 18, 6, 7, 17}}, 
            'inverse': {"Qs": {1, 11, 3, 25, 38, 31}},
            'rank': {"Qs": {4, 20, 9, 22, 23, 37, 32, 29, 30}},
            'advanced_MC': {"Qs": {21, 8, 24, 13, 20, 26, 34, 36, 35, 28, 33, 37, 39}}
        }

        edges = [
            ('computation', 'determinant', 0.9),
            ('computation', 'inverse', 0.8),
            ('computation', 'rank', 0.7),
            ('computation', 'advanced_MC', 0.5),
            ('determinant', 'inverse', 0.6),
            ('determinant', 'rank', 0.6),
            ('determinant', 'advanced_MC', 0.5),
            ('inverse', 'advanced_MC', 0.4),
            ('rank', 'inverse', 0.7),
            ('rank', 'advanced_MC', 0.8)
        ]

        # Creat graph
        conM = nx.MultiDiGraph(name='Matrix computation CM')

        # Add nodes
        conM.add_nodes_from(nodes)
        nx.set_node_attributes(conM, node_attrs)

        # Add and weight edges
        conM.add_edges_from((u, v, {"w": w}) for u, v, w in edges)

        return conM

    if code=='mat_comp_merged':

        nodes = {
            'C_1',
            'C_2', 
            'C_3',
            'C_4',
            'C_5',
            'C_6'
        }
        node_attrs = {
            'C_1': {"Qs": {0, 2, 5, 10, 16, 27, 14}},
            'C_2': {"Qs": {1, 11, 25, 3, 31, 38}},
            'C_3': {"Qs": {4, 13, 17, 6, 7, 9, 22, 23}},
            'C_4': {"Qs": {20, 12, 15, 19, 18}},
            'C_5': {"Qs": {35, 36, 8, 26, 32, 33}},
            'C_6': {"Qs": {21, 28, 24, 29, 30, 34, 37, 39}}
        }
        edges = [
            ('C_1', 'C_2', 0.9),
            ('C_1', 'C_3', 0.5),
            ('C_2', 'C_3', 0.8),
            ('C_3', 'C_4', 0.7),
            ('C_3', 'C_5', 0.6),
            ('C_4', 'C_5', 0.4),
            ('C_5', 'C_6', 0.7),
        ]

        # Creat graph
        conM = nx.MultiDiGraph(name='Matrix computation CM')

        # Add nodes
        conM.add_nodes_from(nodes)
        nx.set_node_attributes(conM, node_attrs)

        # Add and weight edges
        conM.add_edges_from((u, v, {"w": w}) for u, v, w in edges)

        return conM


# %%
