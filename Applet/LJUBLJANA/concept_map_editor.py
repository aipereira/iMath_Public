
#%%
# Installs
# pip install dash-cytoscape
# pip install openpyxl
# networkx

from conM_editor_tools import *

import dash
from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import pandas as pd  
import matplotlib
from openpyxl import load_workbook
import pickle
from dash.dependencies import Input, Output, State
import plotly.express as px
import networkx as nx
import dash_cytoscape as cyto
from dash import html
#import base64
#import time



matplotlib.use('Agg')
os.listdir()

# https://docs.faculty.ai/user-guide/apps/examples/dash_file_upload_download.html

#%% Settings
app = dash.Dash(__name__)
# Get Concept Map
concept_map_path = 'ConceptMaps/'
conM_fn = 'mat_comp_manual_conM.pickle'
debugQ = False


conM = load_conM(concept_map_path, conM_fn)
conM2 = conM.copy()
# elements = nx.readwrite.json_graph.cytoscape_data(conM)['elements']

# Test plot
plot = plot_conceptMap(conM)

# Test concepts
concepts = get_concepts(conM)
#concepts_dc = get_concepts_as_dc(conM)

# Get questions
questions = get_questions(conM)

# links
links = get_links(conM)


# Layout of the app
app.layout = html.Div([

    
    # Title & instructions
    # https://docs.faculty.ai/user-guide/apps/examples/dash_file_upload_download.html
    html.Div([
        html.H2('Concept map editor'),
        html.Div([
            html.P('Uplod concept map file'),
        ]),
    ]),

    dcc.Upload(
        id='load-concept-map', 
        children=html.Div([
            'Select file',
        ]),
        style={
            'width': '10%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '2px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Do not allow multiple files to be uploaded
        multiple=False,
    ),
    html.Ul(id='file-list'),

    # Show current concept lists
    html.Div(children=[
        html.Div([
            html.H4('List of concepts (link start)'),
            dcc.Checklist(
                id='concepts_1',
                options=concepts,
                value=[],
                labelStyle={'display': 'block'},
            ),
        ]),

        html.Div([
            html.H4('List of concepts (link end)'),
            dcc.Checklist(
                id='concepts_2',
                options=concepts,
                value=[],
                labelStyle={'display': 'block'},
            ),
        ]),        

        html.Div([
            # Modify concept map - all interaction
            html.H4('Modify concept map'),

            # Add concept
            html.Div(children=[

                html.Label('Add concept / question', style={'width': '240px', 'height':'30px'}),
                dcc.Input(
                    id='1-add-concept-text',
                    type='text',
                    value='',
                    placeholder='Concept text',
                    style={'width': '180px'}
                ),
                dcc.Input(
                    id='1-add-concept-quest_id',
                    type='number',
                    value=1,
                    placeholder='Question ID',
                    style={'width': '180px'}
                ),
                html.Button('Add concept', id='1-submit-button', style={'width':'200px', 'height':'30px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '1px', 'width':'650px', 'height':'30px'}),

                        # Add link
            html.Div(children=[
                html.Label('Add question', style={'width': '240px', 'height':'30px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='6-add-question-concept',
                    options=concepts,
                    value=[],
                    placeholder='Concept 1',
                    style={'width':'290px', 'height':'30px', 'display': 'inline-block'},
                ),
                dcc.Input(
                    id='6-add-question-quest_id',
                    type='number',
                    value=0,
                    placeholder='qID',
                    style={'width':'60px', 'height':'30px', 'display': 'inline-block'},
                ),
                html.Button('Add question', id='6-submit-button', style={'width':'200px', 'height':'30px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '1px', 'width':'650px','height':'30px'}),


            # Add link
            html.Div(children=[
                html.Label('Add link', style={'width': '240px', 'height':'30px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='2-add-link-concept_1',
                    options=concepts,
                    value=[],
                    placeholder='Concept 1',
                    style={'width':'145px', 'height':'30px', 'display': 'inline-block'},
                ),
                dcc.Dropdown(
                    id='2-add-link-concept_2',
                    options=concepts,
                    value=[],
                    placeholder='Concept 2',
                    style={'width':'145px', 'height':'30px', 'display': 'inline-block'},
                ),
                dcc.Input(
                    id='2-add-link-weight',
                    type='number',
                    value=0,
                    placeholder='w',
                    style={'width':'60px', 'height':'30px', 'display': 'inline-block'},
                ),
                html.Button('Add link', id='2-submit-button', style={'width':'200px', 'height':'30px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '1px', 'width':'650px','height':'30px'}),


            # Modify the weight of a link
            html.Div(children=[
                html.Label('Modify link weight', style={'width': '240px', 'height':'30px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='3-modify-weight-concept_1',
                    options=concepts,
                    value=[],
                    placeholder='Concept 1',
                    style={'width':'145px', 'height':'30px', 'display': 'inline-block'},
                ),
                dcc.Dropdown(
                    id='3-modify-weight-concept_2',
                    options=concepts,
                    value=[],
                    placeholder='Concept 2',
                    style={'width':'145px', 'height':'30px', 'display': 'inline-block'},
                ),
                dcc.Input(
                    id='3-modify-weight-weight',
                    type='number',
                    value=0,
                    placeholder='w',
                    style={'width':'60px', 'height':'30px', 'display': 'inline-block'},
                ),
                html.Button('Enter weight', id='3-submit-button', style={'width':'200px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '1px', 'width':'650px','height':'30px'}),

            # Remove concept
            html.Div(children=[
                html.Label('Remove concept', style={'width': '240px', 'height':'30px'}),
                dcc.Dropdown(
                    id='4-remove-concept-concpet',
                    options=concepts,
                    value=[],
                    style={'width':'360px', 'height':'30px', 'display': 'inline-block'},
                    placeholder='Concept',
                ),
                html.Button('Remove concept', id='4-submit-button', style={'width':'200px', 'height':'30px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '1px', 'width':'650px', 'height':'30px'}),


            # Remove the link
            html.Div([
                html.Label('Remove link', style={'width': '240px', 'height':'30px'}),
                dcc.Dropdown(
                    id='5-remove-link-concept_1',
                    options=concepts,
                    value=[],
                    style={'width':'180px', 'height':'30px', 'display': 'inline-block'},
                    placeholder='Concept 1',
                ),
                dcc.Dropdown(
                    id='5-remove-link-concept_2',
                    options=concepts,
                    value=[],
                    style={'width':'180px', 'height':'30px', 'display': 'inline-block'},
                    placeholder='Concept 2',
                ),
                html.Button('Remove link', id='5-submit-button', style={'width':'200px', 'height':'30px'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '1px', 'width':'650px', 'height':'30px'}),

        ]),
    
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'margin-bottom': '20px', 'width':'90%', 'height': '200px'}),


    # Link UI
    html.Div(children=[
        #html.H5('System response:'),
        html.Div(id='0-load-concept-map'),
        html.Div(id='1-add-concept'), 
        html.Div(id='6-add-question'),
        html.Div(id='2-add-link'),
        html.Div(id='3-modify-weight'),
        html.Div(id='4-remove-concept'),
        html.Div(id='5-remove-link'),
    ], style={'height':'80px', 'display':'none'}),
   
    # Show and save
    dcc.Graph(id='plot-concept-map', style={'width':'400pt', 'height':'350pt'}), 
    html.Button('Update Concept map plot', id='update-concept-map-plot', style={'width':'200px'}),
    html.Div(id='store_concept_map'),
    html.Button('Save Concpept map', id='save-concept-map', style={'width':'200px'}), 

    #dcc.Upload(
    #    id='store-concept-map', 
    #    children=html.Div([
    #        'Save file',
    #    ]),
    #    multiple=False,
    #),
    #html.Ul(id='store-file-list')

])

# @brief add concept to a list of concepts
def add_concept_to_list(concept_added):

    global conM
    concepts_lst = list(get_concepts(conM))
    if (concept_added!=None) & (concept_added!=''):
        concepts_upd_lst = concepts_lst + [concept_added]
    else:
        concepts_upd_lst = concepts_lst
    
    if debugQ:
        print ('@add_concept_to_list - updates:', concepts_upd_lst)
    return concepts_upd_lst


# Refreshing menues ====================================================================
# https://community.plotly.com/t/updating-a-dropdown-menus-contents-dynamically/4920
# https://www.angela1c.com/projects/dash/basic_callbacks1/
# https://stackoverflow.com/questions/60512452/how-to-update-chained-dropdown-value-in-dash-dcc





# Update concept lists
@app.callback(
    dash.dependencies.Output('concepts_1', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def update_concept_list(concept_added, upload_inp):
    if debugQ:
        print ('@Dependences - concept list 1: ', concept_added)
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

@app.callback(
    dash.dependencies.Output('concepts_2', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_concept_list(concept_added, upload_inp):
    if debugQ:
        print ('@Dependences - concept list 2: ', concept_added)
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Add question
@app.callback(
    dash.dependencies.Output('6-add-question-concept', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def update_concept_list(concept_added, upload_inp):
    if debugQ:
        print ('@Dependences - add question ', concept_added)
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Add link 1
@app.callback(
    dash.dependencies.Output('2-add-link-concept_1', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    if debugQ:
        print ('@Dependences add link: concept ', concept_added, ' From upload ', upload_inp)
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst


# Add link 2
@app.callback(
    dash.dependencies.Output('2-add-link-concept_2', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Modify weight
@app.callback(
    dash.dependencies.Output('3-modify-weight-concept_1', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Modify weight
@app.callback(
    dash.dependencies.Output('3-modify-weight-concept_2', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Remove concept
@app.callback(
    dash.dependencies.Output('4-remove-concept-concpet', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Remove concept
@app.callback(
    dash.dependencies.Output('5-remove-link-concept_1', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst

# Remove link
@app.callback(
    dash.dependencies.Output('5-remove-link-concept_2', 'options'), # Use data to refresh,
    [dash.dependencies.Input('1-add-concept-text', 'value'),  # Provides data 
     dash.dependencies.Input('file-list', 'children')]
)
def dash_update_dropdowns(concept_added, upload_inp):
    concepts_upd_lst = add_concept_to_list(concept_added)
    return concepts_upd_lst






# Modification callbacks =========================================================================
# Callback to upload file
@app.callback(
    Output("file-list", "children"),
    [Input("load-concept-map", "filename"), 
     Input("load-concept-map", "contents")],
    prevent_initial_call=True,
)
def dash_upload_conM_file(uploaded_filename, uploaded_file_content):

    global conM
    global last_conM_fn
    if (uploaded_filename != None):
        if debugQ:
            print ('@Uplod file ', uploaded_filename)
        # Load file name
        last_conM_fn = uploaded_filename
        conM = load_conM(concept_map_path, uploaded_filename)
        #result = file_name

        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'
    
    return None 


# Callback to update the output based on the input values
@app.callback(
    Output(component_id='1-add-concept', component_property='children'),
    [Input(component_id='1-submit-button', component_property='n_clicks'),
     Input(component_id='1-add-concept-text', component_property='value'), 
     Input(component_id='1-add-concept-quest_id', component_property='value')],
    prevent_initial_call=True,
)
def dash_add_concept(n_clicks, concept_name, question_id):

    global conM
    # Check it
    if not isinstance(question_id, (int)):
        question_id = -1
    if not isinstance(n_clicks, (int)):
        n_clicks = -1
    if (n_clicks > 0) & (concept_name != '') & (question_id >= 0):
        if debugQ:
            print ('@Add concept', concept_name)
         # modify concept map
        conM = add_concept(conM, concept_name)
        conM = add_question_to_concept(conM, concept_name, question_id)
        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'

   
# Add question
@app.callback(
    Output(component_id='6-add-question', component_property='children'),
    [Input(component_id='6-submit-button', component_property='n_clicks'),
     Input(component_id='6-add-question-concept', component_property='value'), 
     Input(component_id='6-add-question-quest_id', component_property='value')],
    prevent_initial_call=True,
)
def dash_add_question(n_clicks, concept_name, question_id):

    global conM
    # Check it
    if not isinstance(question_id, (int)):
        question_id = -1
    if not isinstance(n_clicks, (int)):
        n_clicks = -1
    if (n_clicks > 0) & (concept_name != '') & (question_id >= 0):
        if debugQ:
            print ('@Add question', concept_name)
         # modify concept map
        conM = add_question_to_concept(conM, concept_name, question_id)
        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'



# Callback to update the output based on the input values
@app.callback(
    Output(component_id='2-add-link', component_property='children'),
    [Input(component_id='2-submit-button', component_property='n_clicks'),
     Input(component_id='2-add-link-concept_1', component_property='value'), 
     Input(component_id='2-add-link-concept_2', component_property='value'),
     Input(component_id='2-add-link-weight', component_property='value')],
    prevent_initial_call=True,
)
def dash_add_link(n_clicks, concept_1, concept_2, weight):
    
    global conM
    if not isinstance(weight, (int, float)):
        weight = -1
    if not isinstance(n_clicks, (int)):
        n_clicks = -1
    if (n_clicks > 0) & (len(concept_1) > 0) & (len(concept_2) > 0) & (weight >= 0):
        if debugQ:
            print ('@ Add link', weight)

        conM = add_link(conM, concept_1, concept_2, weight)
        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'
    
    return None # return_val


# Callback to update the output based on the input values
@app.callback(
    Output(component_id='3-modify-weight', component_property='children'),
    [Input(component_id='3-submit-button', component_property='n_clicks'),
     Input(component_id='3-modify-weight-concept_1', component_property='value'), 
     Input(component_id='3-modify-weight-concept_2', component_property='value'),
     Input(component_id='3-modify-weight-weight', component_property='value')],
    prevent_initial_call=True,
)
def dash_modify_weight(n_clicks, concept_1, concept_2, weight):
    
    global conM
    if not isinstance(n_clicks, (int)):
        n_clicks = -1
    if not isinstance(weight, (int, float)):
        weight = -1
    if (n_clicks > 0) & (len(concept_1) > 0) & (len(concept_2) > 0) & (weight >= 0):
        if debugQ:
            print ('@Modify link weight', weight)
        conM = set_link_weight(conM, concept_1, concept_2, weight)
        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'

    return None #return_val

    

# Remove concept
@app.callback(
    Output(component_id='4-remove-concept', component_property='children'),
    [Input(component_id='4-submit-button', component_property='n_clicks'),
     Input(component_id='4-remove-concept-concpet', component_property='value')],
    prevent_initial_call=True,
)
def dash_remove_concept(n_clicks, concept):
    global conM
    #print ('n_clicks', n_clicks)
    if not isinstance(n_clicks, (int)):
        n_clicks = -1
    if (n_clicks > 0) & (len(concept) > 0):
        if debugQ:
            print ('@Remove concept', concept)
        conM =  remove_concept(conM2, concept)
        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'
    
    return None # return_val


# Callbck remove link
@app.callback(
    Output(component_id='5-remove-link', component_property='children'),
    [Input(component_id='5-submit-button', component_property='n_clicks'),
     Input(component_id='5-remove-link-concept_1', component_property='value'), 
     Input(component_id='5-remove-link-concept_2', component_property='value')],
    prevent_initial_call=True,
)
def dash_remove_link(n_clicks, concept_1, concept_2):
    global conM
    if not isinstance(n_clicks, (int)):
        n_clicks = -1
    if (n_clicks > 0) & (len(concept_1) > 0) & (len(concept_2) > 0):
        concs =  concept_1 + ' :=: ' + concept_2
        if debugQ:
            print ('@ Remove link', concs)
        conM = remove_link(conM, concept_1, concept_2)
        return_val = 'OK' 
    else:
        return_val = 'Set proper values!'  

    return None # return_val


# Callback to update the graph based on checkbox and input values
@app.callback(
    Output(component_id='plot-concept-map', component_property='figure'),
    [Input(component_id='update-concept-map-plot', component_property='n_clicks'),
     dash.dependencies.Input('file-list', 'children')],
)
def dash_update_graph(n_clicks, inp_val):
    # Call the plot_conceptMap function to get the base64 image
    # img_base64 = plot_conceptMap(conM)
    #if not isinstance(n_clicks, (int)):
    #    n_clicks = -1

    if debugQ:
        print ('@ Graph update')

    global conM
    figure = draw_concept_map(None, conM)

    # For debuging only
    if debugQ:
        print_conceptMap(conM)

    return figure


@app.callback(
    Output(component_id='store_concept_map', component_property='children'),
    [Input(component_id='save-concept-map', component_property='n_clicks')],
    prevent_initial_call=True,
)
def dash_save_graph(n_clicks):

    if debugQ:
        print ('@ Graph store 1')
    
    global conM
    global last_conM_fn
    mod_conM_fn = 'mod_' + last_conM_fn
    save_conM(conM, concept_map_path, mod_conM_fn)
    if debugQ:
        print ('@ Graph store: ', concept_map_path, mod_conM_fn)
    
    return None


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8100)


# %%
