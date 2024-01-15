# This are tools for concept map editor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import plotly.graph_objects as go
from io import BytesIO
import base64



# ---------------------------------------------------------------------------------
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
        

# ---------------------------------------------------------------------------------
def load_conM(concept_map_path, conM_fn):
    ''' Load concept map
        Parameters
        Return
    '''
    conM = pickle.load(open(concept_map_path + conM_fn, 'rb'))

    return conM
# Run
#concept_map_path = '01-ConceptMaps/'
#topic_nm = 'mat_comp'
#conM = load_conM(concept_map_path, topic_nm)


# ---------------------------------------------------------------------------------
def save_conM(conM, concept_map_path, conM_fn):
    ''' Save concept map
        Parameters
        Return
    '''
    pickle.dump(conM, open(concept_map_path + conM_fn, 'wb'))

    return 0
    
    
def plot_conceptMap(conM, label_code='all'):
    np.random.seed(0)
    qs_attrs = nx.get_node_attributes(conM, 'Qs')
    if label_code == 'all':
        labels = {l: (l, qs_attrs[l]) for l in qs_attrs}
    elif label_code == 'keys_only':    
        labels = {l: l for l in qs_attrs}
    
    CM_node_attrs = nx.get_node_attributes(conM, 'Qs')
    node_sizes = [1000 * len(CM_node_attrs[u]) for u in CM_node_attrs]

    fig, ax = plt.subplots()
    nx.draw(conM, labels=labels, node_size=node_sizes, pos=nx.spring_layout(conM), with_labels=True, verticalalignment='top', ax=ax)
    ax.set_title(conM.name)

    # Save the plot to a BytesIO object
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
   
    plt.close()

    # Convert the plot to base64 for displaying in Dash
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    
    return img_base64


# ---------------------------------------------------------------------------------
def print_conceptMap(conM):
    '''
    Print conM nodes and questions.
    '''
    conc_qs_dc = nx.get_node_attributes(conM, 'Qs')
    nodes_dc = list(conM.nodes)
    all_qs = []
    print ('Concepts')
    for c in nodes_dc:
        if c in conc_qs_dc.keys():
            all_qs.extend(list(conc_qs_dc[c]))
            print (c, conc_qs_dc[c])
        else:
            print (c, {})

    link_w_dc = nx.get_edge_attributes(conM, 'w')
    print('\nLinks')
    for e in link_w_dc:
        print (e[0]+ ' ---' + str(link_w_dc[e]) + '---> ' + e[1])

    return




# ---------------------------------------------------------------------------------
def get_concepts(conM):
    '''
    Get concepts of concept map
    '''
    return list(conM.nodes)
#l = get_concepts(conM)
#print(l)


# ---------------------------------------------------------------------------------
def get_concepts_as_dc(conM):
    '''
    Get concepts of concept map
    '''
    concepts_lst = list(conM.nodes)
    concepts_dc = [{'label':'concept '+str(i), 'value':c} for i, c in enumerate(concepts_lst)]
    
    return concepts_dc

#concepts_dc = get_concepts_as_dc(conM)

# ---------------------------------------------------------------------------------
def get_questions(conM):
    '''
    Get concept questions as a dictionary
    '''
    return nx.get_node_attributes(conM, 'Qs')
#qs = get_questions(conM)
#print (qs)

# ---------------------------------------------------------------------------------
def get_links(conM):
    '''
    Get dictionary of links among concepts where links are adjacency lists with weights
    '''
    link_ws = nx.get_edge_attributes(conM, 'w')
    c_dc = {}
    for c in conM.nodes:
        c_dc.update({c:[]})
    for e in conM.edges:
        e_w = link_ws[e]
        c_dc[e[0]].append([e[1], e_w])

    return c_dc
#links_dc = get_links(conM)
#print (links_dc)

# ---------------------------------------------------------------------------------
def add_concept(conM, c):
    '''
    Add concept
    '''
    conM1 = conM.copy()
    conM1.add_node(c, Qs={})

    return conM1
#c = 'decomp'
#conM1 = add_concept(conM, c)
#cet.print_conceptMap(conM1)

# ---------------------------------------------------------------------------------
def add_question_to_concept(conM, c, qID):
    '''
    Add question qID to a given concept c
    '''
    
    conM1 = conM.copy()
    node_qs = nx.get_node_attributes(conM1, 'Qs')
    if node_qs[c] == {}:
        conM1.nodes[c].update({'Qs': {qID}})    
    else:
        conM1.nodes[c].update({'Qs': node_qs[c].union({qID})})
    return conM1
#c = 'decomp'
#conM2 = add_question_to_concept(conM1, c, 48)
#conM3 = add_question_to_concept(conM2, c, 101)
#cet.print_conceptMap(conM3)

# ---------------------------------------------------------------------------------
def remove_question_from_concpet(conM, c, qID):
    '''
    Remove question from concept
    '''
    conM1 = conM.copy()
    node_qs = nx.get_node_attributes(conM1, 'Qs')
    if node_qs[c] != {}:
        conM1.nodes[c].update({'Qs': node_qs[c].difference({qID})})
    return conM1
#c = 'rank'
#qID = 30
#conM2 = remove_question_from_concpet(conM1, c, qID)
#cet.print_conceptMap(conM2)

# ---------------------------------------------------------------------------------
def add_link(conM, c1, c2, w):
    '''
    Add link from concepts c1 and c2 weighted by w
    '''
    conM1 = conM.copy()
    conM1.add_edge(c1, c2)
    conM1.edges[c1, c2, 0].update({'w':w})
    return conM1

#c1 = 'rank'
#c2 = 'decomp'
#w = 0.4444
#conM3 = add_link(conM1, c1, c2, w)
#cet.print_conceptMap(conM3)
#cet.plot_conceptMap(conM3)

# ---------------------------------------------------------------------------------
def remove_concept(conM, c):
    '''
    Remove concept from the map
    '''
    conM1 = conM.copy()
    conM1.remove_node(c)
    return conM1

#conM4 = remove_concept(conM3, c)
#cet.print_conceptMap(conM4)

# ---------------------------------------------------------------------------------
def remove_link(conM, c1, c2):
    '''
    Remove link
    '''
    #print(c1, c2)
    conM1 = conM.copy()
    conM1.remove_edge(c1, c2, 0)
    return conM1

#c1 = 'rank'
#c2 = 'inverse'
#conM4 = remove_link(conM3, c1, c2)
#cet.print_conceptMap(conM4)


# ---------------------------------------------------------------------------------
def set_link_weight(conM, c1, c2, w):
    '''
    (Re)set link weight (overwrite it)
    '''
    conM1 = conM.copy()
    conM1.edges[c1, c2, 0].update({'w': w})
    return conM1

# c1 = 'rank'
# c2 = 'inverse'
# w = 0.11
# conM5 = set_link_weight(conM, c1, c2, w)
# cet.print_conceptMap(conM5)


def networkx_to_plotly(graph):
    pos = nx.spring_layout(graph)

    edge_x, edge_y = zip(*pos)
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x, node_y = zip(*pos.values())
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Add labels to the nodes
    node_text = list(graph.nodes)
    node_trace.text = node_text

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0)
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    return fig



# @brief: Draw concept map
def draw_concept_map(value, conM):
    '''
        Draw concept map with labels
    '''

    np.random.seed(0)

    # Create a Plotly figure from the NetworkX graph
    pos = nx.spring_layout(conM)


    node_Qs_dc = nx.get_node_attributes(conM, 'Qs')
    node_sizes = [6*len(node_Qs_dc[n]) for n in node_Qs_dc]

    link_w_dc = nx.get_edge_attributes(conM, 'w')
    w_strs = ['w: '+str(link_w_dc[e]) for e in conM.edges]

    edge_x = []
    edge_y = []
    for edge in conM.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    arr_x, arr_y = [], []
    arr_ax, arr_ay = [], []
    for edge in conM.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        arr_ax.extend([x0])
        arr_ay.extend([y0])
        arr_x.extend([x1])
        arr_y.extend([y1])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        text = w_strs
    )

    node_x = []
    node_y = []
    for node in conM.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=list(conM.nodes),
        mode='markers',
        hoverinfo='text',
        marker=dict(size=node_sizes,
                    line=dict(width=4, 
                              color='blue'), 
                    color='blue')
    )

    # Create a Plotly figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=10, l=10, r=10, t=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    # Add annotations
    for ax, ay, x, y, w_str in zip(arr_ax, arr_ay, arr_x, arr_y, w_strs):
        fig.add_annotation(ax=ax, axref = 'x', ay=ay, ayref = 'y',
                        x=x, xref = 'x', y=y, yref = 'y',
                        arrowwidth = 3, arrowhead = 3,
                        showarrow=True)
        fig.add_annotation(x=(x+ax)/2, y=(y+ay)/2, 
                           text=w_str, 
                           font=dict(size=18, color='blue'),
                           showarrow=False)

    fig.update_layout()
    return fig

#%%








#%%
def plot_conceptMap_AK(conM, label_code='all'):
    '''
    Produces a plot of a concept map with sized concepts (nodes) and listed questions.
        Parameters: 
            conM: conceptual map as a directed networkx graph labeled by questions.
            label_code: how to label nodes
                'all': plot all question idnexes 
                'keys_only': plot only concept keys if available
        Returns:
            ax: axis handle of the plot
            the plot itself 
    '''
    np.random.seed(0)

    qs_attrs = nx.get_node_attributes(conM, 'Qs') 
    if label_code == 'all':
        labels = {l: (l, qs_attrs[l]) for l in qs_attrs}
    elif label_code == 'keys_only':    
        labels = {l: l for l in qs_attrs}
    
    
    CM_node_attrs = nx.get_node_attributes(conM, 'Qs')
    node_sizes = [1000*len(CM_node_attrs[u]) for u in CM_node_attrs]

    nx.draw(conM, labels=labels, node_size=node_sizes, pos=nx.spring_layout(conM), with_labels=True, verticalalignment='top')
    #nx.draw_networkx_edge_labels(conM, pos=nx.spring_layout(conM))
    ax = plt.gca()
    #ax.set_title(conM.name)
    plt.show()

    return ax
