import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

#For 3d Animmations
from plotly.subplots import make_subplots

#Criação da tabela do usuário
import base64
import io
import pandas as pd
from dash import dash_table

#Dados exemplo
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

#Biblioteca para k-means
from sklearn.cluster import KMeans

#Plotar
import plotly.express as px

#Plotar Frames
import plotly.graph_objects as go

#Global
dataFrame = None    #Save the user's dataFrame
dataFrameNum = None
x_axis = None       #Value x axis
y_axis = None       #Value y axis
z_axis = None       #Value z axis
x_aux = None
y_aux = None
z_aux = None
fig = None          #Figure
configFig = []      #Figure for back and forward buttons  
indexFig = 0        #Indexes of the actual Figure
n_back_aux = 0      #Indexes auxiliar of the actual Figure
frames = []         #Frames button play

frames_list = []    #List of frams after press Play
steps_list = []     #List using for generate graphs after presse Play

exampleChoosed = 0  #Auxiliary control variable (Random data)
dataUserChoosed = 0 #Auxiliary control variable (User data)

"""
#Logo images input
with open('C:\CodigosInvetigacao\Codigo1\Giulia\IPB_Giulia\iMath-Project-main\k_means\logo.png', 'rb') as f:
    image_data1 = f.read()
    encoded_image1 = base64.b64encode(image_data1).decode('utf-8')

with open('C:\CodigosInvetigacao\Codigo1\Giulia\IPB_Giulia\iMath-Project-main\k_means\logo1.png', 'rb') as f:
    image_data2 = f.read()
    encoded_image2 = base64.b64encode(image_data2).decode('utf-8')
"""
#TAG1: Funções iniciais
# Gera um valor aleatório
cluster_std = random.uniform(1, 10)
centers = round(random.uniform(2, 10))
#Conjunto de dados exemplo
features, true_labels = make_blobs(
    n_samples=300,
    centers=centers,
    cluster_std=cluster_std,
    random_state=42
)
#TAG1 FIM

# Padronizar as features dos dados
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#Função para gerar o grafico base
def first_time():
    num_dimensions = scaled_features.shape[1]
    if num_dimensions == 2:
        fig = px.scatter(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            title='K-means',
            template='plotly_white',
        )
    elif num_dimensions == 3:
        fig = px.scatter_3d(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            z=scaled_features[:, 2],
            title='Data',
            template='plotly_white',
        )
    else:
        raise ValueError("Número de dimensões inválido. Apenas 2D e 3D são suportados.")
    
    fig.update_layout(
        title={
            'text': 'K-means',
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'black', 'size': 50},
            'x': 0.5,
            'y': 0.95
        }
    )
    global configFig 
    configFig.append(fig)
    return fig

#First time but no append in the vector
def first_time_noAppend():
    num_dimensions = scaled_features.shape[1]
    if num_dimensions == 2:
        fig = px.scatter(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            title='K-means',
            template='plotly_white',
        )
    elif num_dimensions == 3:
        fig = px.scatter_3d(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            z=scaled_features[:, 2],
            title='Data',
            template='plotly_white',
        )
    else:
        raise ValueError("Número de dimensões inválido. Apenas 2D e 3D são suportados.")
    
    fig.update_layout(
        title={
            'text': 'K-means',
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'black', 'size': 50},
            'x': 0.5,
            'y': 0.95
        }
    )
    return fig

#Função para gerara primeira interação do algoritmo k-means    
def first_time_cluster(clustersValue, iter_value):
    global centroids, predicted_labels 
    # Configurações do algoritmo de agrupamento k-means
    kmeans = KMeans(
        init="random",
        n_clusters=clustersValue,
        n_init=1,
        max_iter=iter_value,
        random_state=42
    )
    kmeans.fit(scaled_features)
    #Labels Iniciais
    predicted_labels = kmeans.labels_
    #Centroides iniciais
    centroids = kmeans.cluster_centers_
    
    # Criar figura usando a biblioteca Plotly
    num_dimensions = scaled_features.shape[1]
    if num_dimensions == 2:
        fig = px.scatter(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            title='K-means',
            color=predicted_labels,
            template='plotly_white',
        )
        fig.add_scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            marker=dict(
                symbol='x',
                color='red',
                size=8,
                line=dict(color='black', width=0.5),
                opacity=0.8,
            ),
            name='Centroids'
        )
        fig.update_layout(
            title={
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'color': 'black', 'size': 50},
                'x': 0.5,
                'y': 0.95
            },
            xaxis=dict(
                title=x_axis,
                title_font=dict(size=18)
            ),
            yaxis=dict(
                title=y_axis,
                title_font=dict(size=18)
            )
        )
    elif num_dimensions == 3:
        fig = px.scatter_3d(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            z=scaled_features[:, 2],
            title='K-means',
            color=predicted_labels,
            template='plotly_white',
        )
        fig.add_scatter3d(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            z=kmeans.cluster_centers_[:, 2],
            mode='markers',
            marker=dict(
                symbol='x',
                color='red',
                size=5,
                line=dict(color='black', width=0.5),
                opacity=0.8
            ),
            name='Centroids'
        )
        fig.update_layout(
            title={
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'color': 'black', 'size': 50},
                'x': 0.5,
                'y': 0.95
            },
            scene=dict(
                xaxis=dict(
                    title=x_axis,
                    title_font=dict(size=18)
                ),
                yaxis=dict(
                    title=y_axis,
                    title_font=dict(size=18)
                ),
                zaxis=dict(
                    title=z_axis,
                    title_font=dict(size=18)
                )
            )
        )
    else:
        raise ValueError("Número de dimensões inválido. Apenas 2D e 3D são suportados.")

    
    # Ajustar a posição da legenda
    fig.update_layout(
        legend=dict(
            x=0.77,   # Define a posição horizontal da legenda (0-1)
            y=0.1,   # Define a posição vertical da legenda (0-1)
            bgcolor='rgb(255,248,220)',  # Define a cor de fundo da legenda como preto
            bordercolor="black",  # Define a cor da borda da legenda como branco
            borderwidth=1  # Define a largura da borda da legenda
        ) 
    )
    global configFig 
    configFig.append(fig)
    return fig

#Função para gerar uma nova interação do algoritmo k-means
def update_centroids(cluster_number, iter_value):
    global centroids, predicted_labels 
    kmeans = KMeans(
        init=centroids,
        n_clusters=cluster_number,
        n_init=1,
        max_iter=iter_value,
        random_state=42
    )

    kmeans.fit(scaled_features)
    predicted_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    num_dimensions = scaled_features.shape[1]
    if num_dimensions == 2:
        fig = px.scatter(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            color=predicted_labels,
            labels={'color': 'Cluster'},
            title='K-means',
            template='plotly_white',
        )
        fig.add_scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            marker=dict(
                symbol='x',
                color='red',
                size=8,
                line=dict(color='black', width=0.5),
                opacity=0.8
            ),
            name='Centroids'
        )
        fig.update_layout(
            title={
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'color': 'black', 'size': 50},
                'x': 0.5,
                'y': 0.95
            },
            legend=dict(
                x=0.77,
                y=0.1,
                bgcolor='rgb(255,248,220)',
                bordercolor="black",
                borderwidth=1
            ),
            xaxis=dict(
                title=x_axis,
                title_font=dict(size=18)
            ),
            yaxis=dict(
                title=y_axis,
                title_font=dict(size=18)
            ) 
        )
    elif num_dimensions == 3:
        fig = px.scatter_3d(
            x=scaled_features[:, 0],
            y=scaled_features[:, 1],
            z=scaled_features[:, 2],
            color=predicted_labels,
            labels={'color': 'Cluster'},
            title='K-means',
            template='plotly_white',
        )
        fig.add_scatter3d(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            z=kmeans.cluster_centers_[:, 2],
            mode='markers',
            marker=dict(
                symbol='x',
                color='red',
                size=5,
                line=dict(color='black', width=0.5),
                opacity=0.8
            ),
            name='Centroids'
        )
        fig.update_layout(
            title={
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'color': 'black', 'size': 50},
                'x': 0.5,
                'y': 0.95
            },
            legend=dict(
                x=0.77,
                y=0.1,
                bgcolor='rgb(255,248,220)',
                bordercolor="black",
                borderwidth=1
            ),
            scene=dict(
                xaxis=dict(
                    title=x_axis,
                    title_font=dict(size=18)
                ),
                yaxis=dict(
                    title=y_axis,
                    title_font=dict(size=18)
                ),
                zaxis=dict(
                    title=z_axis,
                    title_font=dict(size=18)
                )
            )
        )
    global configFig 
    configFig.append(fig)
    return fig
        

##############--Inicio do Dash--################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

app.layout = html.Div([

        #Graph
        html.Div([
                    dcc.Graph(
                        id='k-means-first',
                        figure=first_time(),
                        style={'width': '800px', 'height': '600px'}
                    )
                ], 
                style={'border':'1px solid black', 'display': 'inline-block', 'verticalAlign': 'top','margin':'10px'}
            ),
        
        #Control painel and logos    
        html.Div([   
     
            #Control painel
            html.Div([
                html.Div(
                    id='dataUser_variablesFather',style={'display': 'block'},
                    children=[
                        html.Div(id='dataUser_variables')
                    ]
                ),
                    
                #Para os dois inputs Cluster e iterações
                dbc.Row([
                    dbc.Col([
                        html.H3('K'),
                        dcc.Input(
                            id='input-cluster',
                            type='number',
                            min=1,
                            max=20,
                            value=1,
                            style={'width': '50px'}
                        )
                    ], width=4, className='btn-lg', style={'marginTop': '10px', 'marginLeft': 'auto', 'marginLeft': '87px',}),
                    
                    dbc.Col([
                        html.H3('Iterations'),
                        dcc.Input(
                            id='input-iterations',
                            type='number',
                            min=1,
                            max=10,
                            value=1,
                            style={'width': '50px'}
                        )
                    ], width=4,className='btn-lg', style={'marginTop': '10px', 'marginLeft': 'auto', 'marginRight': '87px'})
                ]),
                
                #Buttons Back, Play, Forward, Edit and Reset
                html.Div(
                    children=[
                        html.Div(
                            dbc.Row(
                                children=[
                                    dbc.Col(
                                        html.Button('Back', id='step_back', className='btn-lg', n_clicks=0, style={'width': '100px', 'marginTop':'40px', 'marginLeft':'75px'})
                                    ),
                                    dbc.Col(
                                        html.Button('PLAY', id='play-button', className='btn-lg', n_clicks=0, style={'width': '100px', 'marginTop':'40px'})
                                    ),
                                    dbc.Col(
                                        html.Button('Forward', id='step_foward',className='btn-lg', n_clicks=0, style={'width': '100px', 'marginTop':'40px','marginRight':'75px'})
                                    ),
                                ],
                                justify="center",
                                style={'marginBottom': '50px'}
                            ),
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    html.Button('Edit', id='open-button', className='btn btn-lg', style={'color': 'white', 'fontWeight': 'bold', 'background': 'rgb(21, 115, 71)', 'marginLeft':'100px'})
                                ),
                                dbc.Col(
                                    html.Button('Reset', id='reset-button', n_clicks=0, className='btn btn-lg', style={'color': 'white', 'fontWeight': 'bold', 'background': 'rgb(201, 76, 76)', 'marginRight':'100px'})
                                ),
                            ],
                            justify="center",
                            style={'marginBottom': '25px'}
                        )
                    ]
                )
            
            ], style={'border': '1px solid black', 'width': '500px', 'display': 'inline-block', 'verticalAlign': 'top','margin':'10px' ,'marginLeft': '10px', 'textAlign': 'center'}),
            
            #Logo Images 
            dbc.Row([
                dbc.Col([
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image1), style={'width': '100px', 'height': '100px'}),
                ], className='', style={'display': 'inline', 'textAlign':'center', 'margin':'10px'}),
                    
                dbc.Col([
                    html.Img(src='data:image/png;base64,{}'.format(encoded_image2), style={'width': '100px', 'height': '100px'})
                ], className='', style={'display': 'inline', 'textAlign':'center','margin':'10px'})
                
            ], style={'border':'1px solid black','display': 'inline-flex','verticalAlign': 'top','margin':'10px' ,'marginLeft': '10px', 'textAlign': 'center'}),
        ], style={'display':'inline-flex','flexDirection': 'column'}),
        
        #Modal html
        dbc.Modal([
            dbc.ModalBody([
                html.Strong("K-means", style={"color": "black", "fontSize": "30px"}),
                dbc.Button("X", id='close-button', className="close", style={"position": "absolute", "top": "10px", "right": "10px"}),
            ], id="title-kmeans", className="text-center"),
            html.Hr(),
            
            #rounded-pill - para arredondado
            dbc.Button("Example", id='example-button',size="lg", className="pill text-center mx-auto", style={'color':'black','width': '20%', 'height': '10%', 'marginTop':'10px'}),
            dbc.ModalBody([
                html.Strong("or", style={"color": "grey",'fontSize': '20px'}),
            ], className="text-center"),  
            
            #Botão invisivel para não dar erro no código, creado após o upload
            html.Div(
                children=[
                    html.Button(id='submit-button',style={'display': 'none'}),  
                ]     
            ), 
                
            #Div para upload
            html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Div([
                        'Drag and drop or ',
                        html.A('Select Files')
                    ]),
                    html.Div(
                        'Only .xlsx and .csv files with a maximum size of 2 MB',
                        style={'marginTop': '10px', 'color':'grey','fontSize':'14px'}
                    )
                ]),
                style={
                    'width': '100%',
                    'height': '80px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'justifyContent': 'center',
                    'alignItems': 'center',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=False,
                accept='.xlsx, .csv',
                max_size=2097152,  # Tamanho máximo de 2 MB (em bytes)
            ),
            ], className="rounded-pill text-center mx-auto",
            #style={'width': '500px', 'verticalAlign': 'right', 'textAlign': 'center', 'marginTop':'10px', 'fontWeight': 'bold'}
            style={"width": "calc(60% - 20px)", "verticalAlign": "right", "textAlign": "center", "marginTop": "10px", "marginBottom": "50px", "fontWeight": "bold"}  # Ajustado o estilo do componente dcc.Upload
            ),
            html.Div(id='output-datatable'),
            html.Div([
                html.P('* For a better processing use tables with properly named columns',
                       style={'textAlign': 'center', 'fontSize': '12px', 'fontWeight':'bold'}),
                html.P('* The columns used in the creation of the graphic will only be of the numerical type.',
                       style={'textAlign': 'center', 'fontSize': '12px', 'fontWeight':'bold', 'marginTop': '-10px'})
            ]),
            
        ], id='modal', is_open=True, size="xl", keyboard=False),
        # Definir is_open=True para abrir o pop-up por padrão
        
])
##############--Fim do Dash--################

#Callback para o input de arquivos e criação da tabela no modal
def parse_contents(contents, filename, date):
    global dataFrame
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        
        all_strings = all(isinstance(element, str) for element in df.columns)

        if not all_strings:
            # Substituir os títulos das colunas por "coluna 1", "coluna 2", ...
            first_line = df.columns.copy()
            df.columns = [f"column_{i}" for i in range(1, len(df.columns) + 1)]
            first_line = first_line.to_numpy()
            df = df.shift(1)
            df.iloc[0] = first_line
                    
    
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
        
    dataFrame = df
    return html.Div([
    html.Hr(),
    
    dbc.Button("Create Graph",
        id="submit-button",
        color="success",
        #className="rounded-pill text-center mx-auto",
        className="pill text-center mx-auto",
        style={"display": "block", "width": "200px", 'marginTop':'10px'}
    ),
    html.Hr(),

    html.Div(
        style={'overflowX': 'auto', 'margin': '0 25px'},
        children=[
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_size=15
            )
        ]
    ),
    dcc.Store(id='stored-data', data=df.to_dict('records')),
    html.Hr(), 
])
@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        return parse_contents(list_of_contents, list_of_names, list_of_dates)

#Callback para fechar o pop up(modal) e tratar os botões
@app.callback(
    Output('modal', 'is_open'),
    Output('dataUser_variables', 'children'),
    Output('submit-button', 'n_clicks'),
    Output('example-button', 'n_clicks'),
    Output('open-button', 'n_clicks'),
    Output('close-button', 'n_clicks'),
    [Input('open-button', 'n_clicks'),
     Input('close-button', 'n_clicks'),
     Input('example-button', 'n_clicks'),
     Input('submit-button','n_clicks')],
    [State('modal', 'is_open')]
)
def toggle_modal(open_clicks, close_clicks, example_button, submit_clicks,is_open):
    #print(f'Example:{example_button}, User: {submit_clicks}, Open: {open_clicks}, Close: {close_clicks}')
    global exampleChoosed, dataUserChoosed 
    
    #Just open the modal    
    if open_clicks:
        open_clicks = None
        return not is_open, dash.no_update, submit_clicks, example_button, open_clicks, close_clicks
    
    if close_clicks:
        
        if close_clicks and dataUserChoosed == 1 or close_clicks and exampleChoosed == 1:
            close_clicks = None
            return not is_open, dash.no_update, submit_clicks, example_button, open_clicks, close_clicks 
        
        if close_clicks and dataUserChoosed == 0:
            close_clicks = None
            return not is_open, html.Div([
                                    html.Div(
                                    style={'overflowX': 'auto', 'margin': '0 25px'},
                                        children=[html.H1('Select parameters')]
                                    ),
                                    html.Hr(), 
                                ]), submit_clicks, example_button, open_clicks, close_clicks

    #Example Data
    if example_button:
        exampleChoosed = 1
        dataUserChoosed = 0
        submit_clicks = None
        example_button = None

        global centers, cluster_std
        global scaled_features
        
        # Gera um valor aleatório
        cluster_std = random.uniform(1, 10)
        centers = round(random.uniform(2, 10))
        #Conjunto de dados exemplo
        features, true_labels = make_blobs(
            n_samples=300,
            centers=centers,
            cluster_std=cluster_std,
            random_state=42
        )

        # Padronizar as features dos dados
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        return not is_open, html.Div([
                                html.Div(
                                style={'overflowX': 'auto', 'margin': '0 25px'},
                                    children=[html.H1('Select parameters')]
                                ),
                                html.Hr(), 
                            ]), submit_clicks, example_button, open_clicks, close_clicks
        
    #User Data
    if submit_clicks:
        exampleChoosed = 0
        dataUserChoosed = 1
        global dataFrame, dataFrameNum
        global x_aux, y_aux, z_aux

        #Filtrar só as linhas com numeros
        dataFrameNum = dataFrame.iloc[1:]
        num_columns = dataFrameNum.select_dtypes(include='number').columns
        dataFrameNum = dataFrameNum[num_columns]
        submit_clicks = None
        example_button = None
        return not is_open, html.Div([
                                html.Div(
                                style={'overflowX': 'auto', 'margin': '0 25px'},
                                    children=[html.H2('Select the axis')]
                                ),
                                html.Div(
                                    children=[
                                        dbc.Row([
                                            dbc.Col([
                                                html.P([html.Strong("X", style={"fontWeight": "bold"}), " axis data"]),
                                                dcc.Dropdown(id='xaxis-data', 
                                                            options=[
                                                                {'label': x, 'value': x} for x in dataFrameNum.columns
                                                                if x not in [y_aux, z_aux]
                                                            ],
                                                            value=None,  placeholder="Select")
                                            ], width={"size": 3}, style={"margin": "0 1px"}),
                                            dbc.Col([
                                                html.P([html.Strong("Y", style={"fontWeight": "bold"}), " axis data"]),
                                                dcc.Dropdown(id='yaxis-data', 
                                                            options=[
                                                                {'label': x, 'value': x} for x in dataFrameNum.columns
                                                                if x not in [x_aux, z_aux]
                                                            ],
                                                            value=None, placeholder="Select",)
                                            ], width=3, style={"margin": "0 1px"}),
                                            dbc.Col([
                                                html.P([html.Strong("Z", style={"fontWeight": "bold"}), " axis data"]),
                                                dcc.Dropdown(id='zaxis-data', 
                                                            options=[
                                                                {'label': x, 'value': x} for x in dataFrameNum.columns
                                                                if x not in [y_aux, x_aux]
                                                            ],
                                                            value=None, placeholder="Select",)
                                            ], width=3, style={"margin": "0 1px"}),
                                        ], justify="center", style={'margin':'20px'}),
                                        
                                        dcc.Store(id='stored-data', data=dataFrameNum.to_dict('records')),  # Store para armazenar os valores selecionados
                                        
                                        html.Div(id='selected-values')  # Div para exibir os valores selecionados
                                    ]
                                )
                            ]), submit_clicks, example_button, open_clicks, close_clicks 
        
    return is_open, None, submit_clicks, example_button, open_clicks, close_clicks

#Callback para atualizar o valor do X, Y e Z enquanto o usuário clica
@app.callback(
    Output('stored-data', 'data'),
    Output('xaxis-data', 'options'),
    Output('yaxis-data', 'options'),
    Output('zaxis-data', 'options'),

    [Input('xaxis-data', 'value'),
     Input('yaxis-data', 'value'),
     Input('zaxis-data', 'value')]
)
def store_selected_values(x_value, y_value, z_value):
    global x_aux, y_aux, z_aux
    x_aux, y_aux, z_aux = x_value, y_value, z_value 
    #print(f'x: {x_aux}, Y: {y_aux}, Z: {z_aux}')
    
    #Elimin the selected options in X axis, Y and Z.
    optionsX=[
        {'label': x, 'value': x} for x in dataFrameNum.columns
        if x not in [y_aux, z_aux]
    ]
    optionsY=[
        {'label': x, 'value': x} for x in dataFrameNum.columns
        if x not in [x_aux, z_aux]
    ]
    optionsZ=[
        {'label': x, 'value': x} for x in dataFrameNum.columns
        if x not in [x_aux, y_aux]
    ]
    #print(f'Opcoes:{optionsY}')


    return {'x': x_value, 'y': y_value, 'z': z_value}, optionsX, optionsY, optionsZ

#Função para criar o gráfico do usuário
def make_graphs(x_data= None, y_data=None, z_data=None):
        global dataFrame 
        global x_axis, y_axis, z_axis
        df = (dataFrame)
        
        if x_data is not None and y_data is not None and z_data is not None:
            colunas_desejadas = [x_data, y_data, z_data]
            
        else:
            if x_data is not None and y_data is not None and z_data is None:
                colunas_desejadas = [x_data, y_data]
            elif x_data is not None and y_data is None and z_data is not None:
                colunas_desejadas = [x_data, z_data]
            elif x_data is None and y_data is not None and z_data is not None:
                colunas_desejadas = [y_data, z_data]
       
        df_filtrado = df.filter(items=colunas_desejadas)
               
        try:
            pd.to_numeric(df_filtrado[x_data].iloc[1:])
            #Todos eixos
            if x_data is not None and y_data is not None and z_data is not None:
                x_axis = x_data
                y_axis = y_data
                z_axis = z_data
            #Dois eixos    
            elif (x_data is not None and y_data is not None) or (x_data is not None and z_data is not None) or (y_data is not None and z_data is not None):
                
                # Verifica quais são os dois valores que não são None
                if x_data is not None and y_data is not None:
                    x_axis = x_data
                    y_axis = y_data
                elif x_data is not None and z_data is not None:
                    x_axis = x_data
                    y_axis = z_data
                else:
                    x_axis = y_data
                    y_axis = z_data
 
            global scaled_features 
            scaled_features = scaler.fit_transform(df_filtrado)
        except ValueError:
            print()

@app.callback(
    Output('selected-values', 'children'),
    [Input('stored-data', 'data')],
)
def display_selected_values(selected_values):
    x_value = selected_values['x']
    y_value = selected_values['y']
    z_value = selected_values['z']
    
    if x_value is not None and y_value is not None and z_value is not None:
         make_graphs(x_value, y_value, z_value)
    elif x_value is not None and y_value is not None:
         make_graphs(x_value, y_value)
    elif x_value is not None and z_value is not None:
         make_graphs(x_value, z_value)
    elif y_value is not None and z_value is not None:
         make_graphs(y_value, z_value)
 
    #values_text = f" X={x_value}, Y={y_value}, Z={z_value}"
    #return html.P(values_text)
    return html.P("")

#Chamada das funções de k-means
@app.callback(
    [Output('k-means-first', 'figure'),                # Controla a Div que contém o gráfico principal 
    Output('input-cluster', 'disabled'),               # Adiciona a saída para desabilitar o slider n cluster
    Output('input-iterations', 'disabled'),            # Adiciona a saída para desabilitar o slider n interações
    Output('play-button', 'disabled'),                 # Adiciona a saída para desabilitar o botão play
    Output('step_back', 'disabled'),                   # Adiciona a saída para desabilitar o botão back 
    Output('step_foward', 'disabled'),                 # Adiciona a saída para desabilitar o botão forward   
    Output('step_foward', 'n_clicks'),
    Output('play-button', 'n_clicks'),
    Output('reset-button', 'n_clicks'),
    Output('step_back', 'n_clicks'),
    Output('input-cluster', 'value'),
    Output('input-iterations', 'value'),
    Output('dataUser_variablesFather','style')],
    
    [Input('step_foward', 'n_clicks'),
    Input('play-button', 'n_clicks'),
    Input('step_back', 'n_clicks'),
    Input('reset-button', 'n_clicks')],
    
    [State('input-cluster', 'value'),
    State('input-iterations', 'value'),
    State('k-means-first', 'figure'),
    State('dataUser_variablesFather','style')]
)
def update_graph(n_clicks_foward, n_clicks2, n_clicks_back, n_clicks_reset, cluster_number, iter_value, figure, divAxis):
    global indexFig, n_back_aux, configFig
    global scaled_features
    global x_axis, y_axis, z_axis
    global frames
    #Put forward button how disable=false, if no one change in this function
    status_foward = False
    #Substituir o primeiro gráfico random pelo do usuário
    if len(configFig) == 2:
        configFig[0] = first_time_noAppend()
        configFig[0].update_layout(scene=dict(xaxis_title=x_axis, yaxis_title=y_axis, zaxis_title=z_axis))

    #Botão Forward
    if n_clicks_foward > 0:
        #divAxis['visibility'] = 'hidden'
        divAxis['display'] = 'none'
        
        if n_clicks_foward == 1 :
            if cluster_number > 0:
                fig, tip1, tip2, tip3 = first_time_cluster(cluster_number, iter_value), True, True, True
               
        elif n_clicks_foward >= 2 and n_clicks_back == 0 and n_back_aux == 0:
            fig, tip1, tip2, tip3 = update_centroids(cluster_number, iter_value), True, True, True
        
        elif n_clicks_foward >= 2 and n_clicks_back == 0 and n_back_aux != 0:
            n_back_aux -=1
            indexFig += 1
            fig = configFig[indexFig]
            tip1, tip2, tip3 = True, True, True
        
    #Button step-back
    if n_clicks_back >= 1:
        n_back_aux += 1
        indexFig = len(configFig) - n_back_aux - 1
        if indexFig >= 0:
            back_figure = configFig[indexFig]
            fig = back_figure
            tip1, tip2, tip3 = True, True, True
        n_clicks_back = 0 
                  
    # Botão Reset
    elif n_clicks_reset > 0: 
        #divAxis['visibility'] = 'visible'
        divAxis['display'] = 'block'
        fig, tip1, tip2, tip3, status_backButton = first_time(), False, False, False, True
        n_clicks_foward, n_clicks2, n_clicks_reset, n_clicks_back, cluster_number, iter_value = 0,0,0,0,1,1
        configFig = configFig[:1]
        indexFig, n_back_aux = 0, 0
        x_axis = None
        y_axis = None
        z_axis = None
        
    # Botão PLAY
    elif n_clicks2 == 1:
        divAxis['display'] = 'none'
        global centroids, predicted_labels 
         
        # Configurar a escala de cores para os clusters
        custom_colorscale = [
            [0.0, "rgb(20, 8, 140)"],
            [0.2, "rgb(100, 1, 166)"],
            [0.4, "rgb(175, 41, 150)"],
            [0.6, "rgb(225, 100, 100)"],
            [0.8, "rgb(250, 160, 57)"],
            [1.0, "rgb(240, 240, 33)"],
        ] 
                    
        fig_dict = {
            "data": [],
            "layout": {},
            "frames": []
        }         
        
        kmeans = KMeans(
            init="random",
            n_clusters=cluster_number,
            n_init=1,
            max_iter=iter_value,
            random_state=42
        )
        kmeans.fit(scaled_features)
        #Labels Iniciais
        predicted_labels = kmeans.labels_
        #Centroides iniciais
        centroids = kmeans.cluster_centers_
        
        num_dimensions_Play = scaled_features.shape[1]
        if num_dimensions_Play == 2:
            for i in range(10):
                if i == 0:
                    # Criação das listas do dado
                    #Datas
                    data_dict_data = {
                        "x": list(scaled_features[:, 0]),
                        "y": list(scaled_features[:, 1]),
                        "mode": "markers",
                        "marker": {
                            "symbol": "circle",
                            "color": predicted_labels,
                            "size": 6,
                            "line": {"color": "black", "width": 0.5},
                            "opacity": 0.8,
                            "colorscale" : custom_colorscale
                        },
                        "name": f"Cluster {i+1}"  # Nome personalizado para cada conjunto de dados
                    }
                    
                    #Centroids
                    data_dict_centroids = {
                        "x": list(centroids[:, 0]),
                        "y": list(centroids[:, 1]),
                        "mode": "markers",
                        "marker": {
                            "symbol": "x",
                            "color": "red",
                            "size": 7,
                            "line": {"color": "black", "width": 0.5},
                            "opacity": 1,
                            "colorscale" : custom_colorscale
                        },
                        "name": f"Centroids {i+1}"  # Nome personalizado para cada conjunto de centroids
                    }

                    # Criação do frame
                    frame = {"data": [data_dict_data, data_dict_centroids], "name": f"Frame {i+1}"}
                    frames_list.append(frame)
                    
                else:
                    # Configurações do algoritmo de agrupamento k-means
                    kmeans = KMeans(
                        init=centroids,
                        n_clusters=cluster_number,
                        n_init=1,
                        max_iter=iter_value,
                        random_state=42
                    )
                    kmeans.fit(scaled_features)
                    
                    # Labels Iniciais
                    predicted_labels = kmeans.labels_
                    
                    # Centroides iniciais
                    centroids = kmeans.cluster_centers_

                    # Dentro do dicionário "marker" do traço do gráfico de dados
                    data_dict_data = {
                        "x": list(scaled_features[:, 0]),
                        "y": list(scaled_features[:, 1]),
                        "mode": "markers",
                        "marker": {
                            "symbol": "circle",
                            "color": predicted_labels,
                            "size": 6,
                            "line": {"color": "black", "width": 0.5},
                            "opacity": 0.8,
                            "colorscale": custom_colorscale,
                            "colorbar": {  # Configuração da colorbar
                                "thickness": 30,  # Espessura da colorbar
                                "len": 1,  # Comprimento da colorbar (valor de 0 a 1)
                                "x": 1.05,  # Posição horizontal da colorbar (valor maior que 1)
                                "xanchor": "left",  # Âncora horizontal à esquerda
                                "outlinewidth": 0,  # Largura da linha de contorno da colorbar
                                "tickvals": list(range(max(predicted_labels) + 1)),  # Valores dos ticks (numeros inteiros)
                                "ticktext": [str(int(val)) for val in range(max(predicted_labels) + 1)],  # Textos dos ticks (numeros inteiros)
                                "tickfont": {"color": "black"}  # Cor dos ticks da colorbar
                            }
                        },
                        "name": f"Cluster {i+1}"  # Nome personalizado para cada conjunto de dados
                    }
                    
                    #Criação dos centroids
                    data_dict_centroids = {
                        "x": list(centroids[:, 0]),
                        "y": list(centroids[:, 1]),
                        "mode": "markers",
                        "marker": {
                            "symbol": "x",
                            "color": "red",
                            "size": 7,
                            "line": {"color": "black", "width": 0.5},
                            "opacity": 1,
                            "colorscale" : custom_colorscale
                        },
                        "name": f"Centroids {i+1}"  # Nome personalizado para cada conjunto de centroids
                    }

                    # Criação do frame
                    frame = {"data": [data_dict_data, data_dict_centroids], "name": f"Frame {i+1}"}
                    frames_list.append(frame)
            
            #If Example data use X and Y how the labels, or the real name in user's data
            if x_axis is None:
                if y_axis is not None and z_axis is not None:
                    x_axis_title = y_axis
                    y_axis_title = z_axis
                else:
                    x_axis_title = 'x'
                    y_axis_title = 'y'
            elif y_axis is None:
                if x_axis is not None and z_axis is not None:
                    x_axis_title = x_axis
                    y_axis_title = z_axis
                else:
                    x_axis_title = 'x'
                    y_axis_title = 'y' 
            elif z_axis is None:
                if x_axis is not None and y_axis is not None:
                    x_axis_title = x_axis
                    y_axis_title = y_axis
                else:
                    x_axis_title = 'x'
                    y_axis_title = 'y'
            
            layout = {
                "title": {
                    "text": "K-means",
                    "font": {
                        "size": 50,  # Ajuste o tamanho da fonte conforme necessário
                },
                    "x": 0.5  # Centraliza o título horizontalmente
                },
                "showlegend": False,
                "xaxis": {"title": x_axis_title, "zeroline": False, "showgrid": True, "gridcolor": "lightgray", "gridwidth": 1, "tickfont": {"color": "black"}},
                "yaxis": {"title": y_axis_title, "zeroline": False, "showgrid": True, "gridcolor": "lightgray", "gridwidth": 1, "tickfont": {"color": "black"}},
                "hovermode": "closest",
                "plot_bgcolor": "white",  # Define o fundo do gráfico como branco
                "paper_bgcolor": "white",  # Define o fundo do papel como branco
                "updatemenus": [
                    {
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                                "fromcurrent": True, "transition": {"duration": 100,
                                                                                    "easing": "quadratic-in-out"}}],
                                "label": "Play",
                                "method": "animate",
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 20, "t": 5},
                        "showactive": True,
                        "type": "buttons",
                        "x": 0.5,  # Centralizar horizontalmente (valor de 0 a 1)
                        "xanchor": "center",  # Âncora horizontal no centro
                        "y": -0.15,  # Posição vertical abaixo do gráfico (valor negativo)
                        "yanchor": "top"  # Âncora vertical no topo
                    }
                ]
            }

            # Criação da figura
            fig_dict = {
                "data": [data_dict_data, data_dict_centroids],  # Adiciona os dados iniciais na figura
                "layout": layout,
                "frames": frames_list
            }

            fig = go.Figure(fig_dict)
        
        if num_dimensions_Play == 3:
            x_data = scaled_features[:, 0]
            y_data = scaled_features[:, 1]
            z_data = scaled_features[:, 2]

            # Criar figura inicial
            fig = go.Figure()

            # Criar trace para o gráfico de dispersão 3D
            scatter_trace = go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    size=6,
                    opacity=1,
                ),
                name="Pontos",
            )

            # Adicionar trace à figura
            fig.add_trace(scatter_trace)

            # Criar layout
            layout = go.Layout(
                scene=dict(
                    xaxis=dict(title=x_axis),
                    yaxis=dict(title=y_axis),
                    zaxis=dict(title=z_axis),
                ),
                title={
                    "text": "K-means",  # Use HTML tags for bold formatting
                    "font": {"size": 50},  # Adjust the font size as needed
                    "x": 0.5,  # Set the x position to 0.5 to center the title horizontally
                    "y": 0.95,  # Set the y position to 0.95 to position the title at the top
                    "xanchor": "center",  # Set the x anchor to "center" to center the title horizontally
                    "yanchor": "top"  # Set the y anchor to "top" to position the title at the top
                },
                showlegend=False,
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=True,
                        buttons=[
                            dict(
                                label='Play',
                                method='animate',
                                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')],
                            ),
                            dict(
                                label='Pause',
                                method='animate',
                                args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')],
                            )
                        ],
                        direction = 'left',
                        
                        x=0.5,  # Set the horizontal position to the middle (0.5)
                        y=-0.15,  # Set the vertical position to 0.1 (place it at the bottom)
                        xanchor='center',  # Center the buttons horizontally
                        yanchor='top',  # Anchor the buttons to the top
                    )
                ]
            )

            # Atualizar layout da figura
            fig.update_layout(layout)
           
            # Create the trace for centroids
            centroids_trace = go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode="markers",
                marker=dict(
                    symbol="x",
                    color="red",
                    size=7,
                    line=dict(color="black", width=0.5),
                    opacity=1,
                    colorscale=custom_colorscale,
                ),
                name=f"Centroids {0}"  # Name for the initial centroids
            )

            # Add the trace for centroids to the figure
            fig.add_trace(centroids_trace)

            for i in range(10):
                if i == 0:
                    x = scaled_features[:, 0]
                    y = scaled_features[:, 1]
                    z = scaled_features[:, 2]

                else:
                    kmeans = KMeans(
                        init=centroids,
                        n_clusters=cluster_number,
                        n_init=1,
                        max_iter=iter_value,
                        random_state=42,
                    )
                    kmeans.fit(scaled_features)
                    predicted_labels = kmeans.labels_
                    centroids = kmeans.cluster_centers_

                    x = scaled_features[:, 0]
                    y = scaled_features[:, 1]
                    z = scaled_features[:, 2]

                # Create a new trace for the current frame (points)
                frame_trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        color=predicted_labels,
                        size=5,
                        line=dict(color="black", width=0.1),
                        opacity=0.8,
                        colorscale=custom_colorscale,
                        colorbar={  # Colorbar configuration
                            "thickness": 30,
                            "len": 1,
                            "x": 1.05,
                            "xanchor": "left",
                            "outlinewidth": 0,
                            "tickvals": list(range(max(predicted_labels) + 1)),
                            "ticktext": [str(int(val)) for val in range(max(predicted_labels) + 1)],
                            "tickfont": {"color": "black"}
                        }
                    ),
                    name=f"Pontos Frame {i+1}",
                )
                
                # Create a new trace for the centroids in the current frame
                centroids_trace = go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    z=centroids[:, 2],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        color="red",
                        size=7,
                        line=dict(color="black", width=0.5),
                        opacity=1,
                        colorscale=custom_colorscale,
                    ),
                    name=f"Centroids {i+1}"
                )

                # Add frame traces to the frame list
                frame_traces = [frame_trace, centroids_trace]
                frames_list.append(go.Frame(data=frame_traces, name=f"Frame {i+1}"))

            # Add the frames to the figure
            fig.update(frames=frames_list)   
                          
            
                
        tip1, tip2, tip3, status_backButton, status_foward= True, True, True, True, True
        n_clicks2 = 0    
    
    # Grafico Padrão        
    elif n_clicks_foward == 0 & n_clicks2 == 0:
        return dash.no_update 
    
    #In case of first graph or first position of the graph after pressing a lot of time the back buttons
    #disable the back button
    if len(configFig) == 2 or indexFig == -1:
        status_backButton = True
        if(indexFig == -1):
            fig = configFig[0]
            indexFig += 1
            n_back_aux -= 1
            tip1, tip2, tip3 = True, True, True
    else:
        status_backButton = False
       
      
    #Debugging print bellow
    #print(f'Forward: {n_clicks}, Back:{n_clicks_back}, Tamanho: {len(configFig)} e index: {indexFig}, backAux: {n_back_aux}')    
    return fig, tip1, tip2, tip3, status_backButton, status_foward, n_clicks_foward, n_clicks2, n_clicks_reset, n_clicks_back, cluster_number, iter_value, divAxis
      
if __name__ == '__main__':
    app.run_server(debug=True)
