# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:30:48 2022

@author: giuli
"""

#--------IMPORT---------------
from dash import dcc, html, Input, Output, callback, State
import math
from numpy import linalg as LA
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

#-------PARAMS-------
nb_runs = 250
#np.random.seed(0)

text_f='*Here you can study the double descent phenomenon when approximating the function ||x-0.4| - 0.2| + x/2 - 0.1 with a polynomial regressor:\
        test error first decreases, then increases, and then decreases again with increasing model\'s polynomial degree.\
        The phenomenon occurs under the following conditions: zero noise on training samples, regularization\'s parameter Rho equal to 1e-12, 8 train samples, polynomial degree of 16.\
        You can modify the degree of the polynomial and the regularization\'s parameter Rho and see\
        how this affect training and test error trends and the prediction (select a point in MSE graph and the prediction for that polynomial degree will be plotted beside). If you click\
        \'Change dataset\' button you can modify the train dataset and play with the regressor model, to return \
        to the initial conditions click the relevant button. Click run button to apply every change. See reference https://arxiv.org/pdf/2105.14368.pdf for more details on Double Descent.*'
        
copyright_phrase = '*\u00A9 2022 Giulia Cademartori All Rights Reserved*' 
image_path_2 = 'assets/unige_logo.png'
       
colors = {'background': 'white','text': 'black', 'background_card': 'grey'} 

#-------function definition----------
def pol_value(alpha, x): #modello polinomiale P(x;alpha)
    x_pow = x.reshape(-1, 1) ** np.arange(alpha.shape[0]).reshape(1, -1)
    return x_pow @ alpha #scalar product

def fit_alpha(x, y, D, rho, a = 0, b = 1): #trovo alpha ottimi
    M = x.reshape(-1, 1) ** np.arange(D + 1).reshape(1, -1)
    B = y

    if D >= 2:
        q = np.arange(2, D + 1, dtype = x.dtype).reshape(1, -1)
        r = q.reshape(-1,  1)
        beta = np.zeros((D + 1, D + 1))
        beta[2:, 2:] = (q-1) * q * (r-1) * r * (b**(q+r-3) - a**(q+r-3))/(q+r-3)
        l, U = LA.eig(beta)
        Q = U @ np.diag(np.clip(l,a_min=0,a_max=100) ** 0.5)
        B = np.concatenate((B, np.zeros((Q.shape[0]))), 0)
        M = np.concatenate((M, math.sqrt(rho) * Q.transpose()), 0)         
    return LA.lstsq(M,B, rcond=None)[0] #soluzione ai minimi quadrati

def phi(x): # la vera funzione da cui ottengo y_true
    return np.abs(np.abs(x - 0.4) - 0.2) + x/2 - 0.1

def compute_mse(nb_train_samples,rho, D_max,train_noise_std,seed=8):
    np.random.seed(seed)
    mse_train = np.zeros((nb_runs, D_max + 1))
    mse_test = np.zeros((nb_runs, D_max + 1))
    predictions = np.zeros((100, D_max + 1))

    for k in range(nb_runs): #nb_runs: num di volte su cui medio errore
        x_train = np.random.rand(nb_train_samples)
        y_train = phi(x_train)
        if train_noise_std > 0:
            y_train = y_train + np.random.standard_normal(size=y_train.shape)*train_noise_std
        x_test = np.linspace(0, 1, 100, dtype = x_train.dtype)#num test set 100 punti
        y_test = phi(x_test)

        for D in range(D_max + 1):
            alpha_d = fit_alpha(x_train, y_train, D, rho)
            mse_train[k, D] = ((pol_value(alpha_d, x_train) - y_train)**2).mean()
            mse_test[k, D] = ((pol_value(alpha_d, x_test) - y_test)**2).mean()
            if(k==0):
               print(pol_value(alpha_d, x_test).shape)
               predictions[:,D] =  pol_value(alpha_d, x_test)

    return np.median(mse_train, axis=0), np.median(mse_test, axis=0), predictions

def rhoQ(alpha,D,rho):
    tmp = 0.
    for q in range(2,D+1):
        for r in range(2,D+1):
            beta = ((q-1.0)*q*(r-1.0)*r)/(q+r-3.0)
            tmp += beta*alpha[q]*alpha[r]
    return tmp*rho

#--------graphs---------
#graph prediction vs true
def pred_graph (rho, D_max, train_noise_std=0, nb_train_samples=8, seed=8):
    np.random.seed(seed)
    x_train = np.random.rand(nb_train_samples)
    y_train = phi(x_train)
    if train_noise_std > 0:
        y_train = y_train + np.random.standard_normal(size=y_train.shape)*train_noise_std
    x_test = np.linspace(0, 1, 100, dtype = x_train.dtype)
    y_test = phi(x_test)
    
    layout = go.Layout(
        title_text='<i>Degree {}</i>'.format( D_max), 
        title_x=0.5,
        title_y=0.92,
        font = dict(color=colors['text'], size=10),
        xaxis_title="y",
        yaxis_title="x",
        yaxis_range=[-0.1, 1.1],
        legend=dict(
        yanchor="top",
        y=-0.15,
        xanchor="left",
        x=0,
        orientation='h'),
        margin=dict(l=60, r=40, t=65, b=0),
    )
    
    fig = go.Figure(layout=layout)
    
    fig.add_trace(go.Scatter(
        x=x_train,
        y=y_train,
        name="Train samples",
        mode ='markers',
        marker = {'color' : 'cornflowerblue'}     
    ))
    
    fig.add_trace(go.Scatter(
        x=x_test,
        y=y_test,
        name="Test values",
        mode="lines",
        line=go.scatter.Line(color="black"),
    ))
    
    alpha = fit_alpha(x_train, y_train, D_max, rho)
    fig.add_trace(go.Scatter(
        x=x_test,
        y=pol_value(alpha, x_test),#predictions[:,D_max],
        name="Fitted polynomial",
        mode="lines",
        line=go.scatter.Line(color="red"),
    ))
    
    fig.update_layout(
        title_text='<i>Degree {}</i>'.format( D_max), 
        title_x=0.5,
        title_y=0.92,
        font = dict(color=colors['text'], size=10),
        xaxis_title="y",
        yaxis_title="x",
        yaxis_range=[-0.1, 1.1],
        legend=dict(
        yanchor="top",
        y=-0.15,
        xanchor="left",
        x=0,
        orientation='h'),
        margin=dict(l=60, r=40, t=65, b=0),
    )
    return fig

#graph mse on test and train
def mse_graph (rho, D_max, train_noise_std=0,nb_train_samples=8, seed=8):
    print(seed)
    #MSE graph
    mse_train, mse_test, predictions = compute_mse(nb_train_samples,rho, D_max,train_noise_std,seed)
    #print(mse_train.shape)
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(D_max + 1),
        y=mse_train,
        name="Train MSE"
        #marker = {'color' : 'blue'}     
    ))
    
    fig.add_trace(go.Scatter(
        x=np.arange(D_max + 1),
        y=mse_test,
        name="Test MSE",
        marker = {'color' : 'red'}
    ))
    
    fig.add_trace(go.Scatter(
        x = [nb_train_samples - 1, nb_train_samples - 1],
        y = [0,1000],
        mode='lines',
        line = dict(shape = 'linear',color='grey', dash = 'dash'),
        name="Nb. params = nb. samples",
    ))
    
    fig.update_yaxes(title_text="MSE (log scale)", type="log", range=[-5,2])
    fig.update_layout(
        title_text='<i>Train and Test Error: N_train={}, Noise={}</i>'.format(nb_train_samples,train_noise_std), 
        font = dict(color=colors['text'], size=10),
        title_x=0.5,
        title_y=0.92,
        xaxis_title="Polynomial degree",
        annotations=[dict(x=nb_train_samples + 0.005, y=-1, yanchor='top',showarrow=False, text= 'Nb. params = nb. samples',textangle=90),],
        legend=dict(
        yanchor="top",
        y=-0.15,
        xanchor="left",
        x=0,
        orientation='h'),
        margin=dict(l=60, r=30, t=65, b=0),
    )   
    
    return fig, predictions


layout = html.Div([
    # first row
    html.Div(children=[
      #dcc.Markdown('**Double Descent**',style={'width': '97%', 'align': 'justify', 'font-size':16}),
      dcc.Markdown(text_f,style={'width': '98.5%', 'text-align': 'justify', 'font-size':13})
    ],style={'margin-top': 15,'margin-bottom': 0, 'margin-left': 0,'color': colors['text'],'align': 'justify'}, className='row'),
    #second row
    html.Div(children=[
        html.Div(children=[dcc.Graph(id="graph_mse",clickData={'points': [{'x': 16}]})],style={'width': '45%','height': '100%','display': 'inline-block','padding-left': 40,'padding-right':0, 'margin-left': 0, 'margin-right':0},className='col-md-6'),
        html.Div(children=[dcc.Graph(id="graph_pred")],style={'width': '40%','height': '100%','display': 'inline-block','padding-left': 10,'padding-right':0, 'margin-left': 70, 'margin-right':10},className='col-md-6'),
    ],style={'width': '100%','margin-right':0,'margin-top':-10,'margin-bottom':20},className='row'),
    html.Div(children=[
        dcc.Input(id='boolean-switch', type='number', min=0, max=500, step=1, value=0)
    ],style={'width': '10%','display': 'none',},className='row'),
    #third row
    html.Div(children=[
        dbc.Card( 
        dbc.CardBody([
                #params
                html.H4("Parameters setting", className="card-title", style={'margin-bottom': 5, 'margin-left':5,}),
                #select polynomial degree 
                html.Div(children=[
                    dcc.Markdown(children='Select maximum polynomial degree:', style={'margin-left': -5}),#
                    dmc.NumberInput(id='pol_degree', size='sm', min=1, max=19, step=1, value=16, style={'margin-left':0,'color':'#323232','width': '40%'}),
                ],style={"verticalAlign": "top",'width': '25%', 'display': 'inline-block','margin-top':15,'margin-bottom':30,'margin-right': 0,'margin-left': 0},className='row'),
                #change dataset
                html.Div(children=[
                    dbc.Button('Change dataset', id='datasetButton', outline=True, color="dark")
                ],style={"verticalAlign": "top",'width': '18%', 'display': 'inline-block','margin-left': 15,'margin-right': 30,'margin-top':40,'margin-bottom':0},className="me-1"),
                #select train n_samples
                html.Div(children=[ 
                    dcc.Markdown(children ='Enter number of training samples:', id='train_title'),
                    dmc.NumberInput(id='sliderSamples_dd', size='sm', min=1, max=100, step=1, value=8, style={'color':'#323232','width': '50%'}),
                ],style={"verticalAlign": "top",'width': '25%','display':'inline-block','margin-left': 0,'margin-right': 15,'margin-top':10,'margin-bottom':30},className='row'),
                #select noise 
                html.Div(children=[ 
                    dcc.Markdown(children='Enter noise value on training set:', id='noise_title'),
                    dmc.NumberInput(id='sliderNoise_dd', size='sm', precision=1, min=0, max=1, step=0.1, value=0, style={'color':'#323232','width': '50%'}),
                ],style={"verticalAlign": "top",'width': '25%', 'display':'inline-block','margin-top':10,'margin-bottom':30,'margin-right': 15,'margin-left': 0},className='row'),
                #select reg parameter rho
                html.Div(children=[ 
                    dcc.Markdown(children='Select regularization parameter Rho:', style={'margin-left': -15}),
                    dcc.Slider(min=-12,max=3,id='sliderR', step=None, marks={i: '{}'.format(pow(10,i)) for i in range(-12,4)},value=-12,updatemode='drag'),
                ],style={'width': '67%', 'display': 'inline-block','margin-left': 10,'margin-right': 0,'margin-bottom':10},className='row'),#,'margin-left': -10
                #back original dataset
                html.Div(children=[ 
                    dbc.Button('Go back to initial conditions', id='backButton', outline=True, color="dark")
                ],style={'width': '30%', 'display': 'block','margin-left': 5,'margin-top':20,'margin-bottom':30},className='row'),
                #button
                html.Div(children=[ 
                    dbc.Button("Run", outline=True, color="success", id='runButton_dd',style={'backgroundColor': 'lightgreen'})
                ],style={'margin-left': 5,'display': 'block'},className="me-1")
            ]),color=colors['background']),
    ], style={'margin-left': 20, 'margin-right': 40,'margin-top': 20,'margin-bottom': 10,'width': '97%'},className='row'),
    #copyright
    html.Div(children=[
        html.Div([
            html.A([
                html.Img(src=image_path_2, style={'height':'80%', 'width':'7%','align': 'justify', 'margin-left': -30})
            ], href='https://unige.it/en', target="_blank")
        ]),
        html.Div(children=[dcc.Markdown(copyright_phrase)], style={'width': '97%', 'align': 'justify', 'font-size':13,'display': 'block','margin-top': 0}), 
    ],style={'margin-top': 20,'margin-bottom': 0, 'margin-left': 0,'color': colors['text'],'textAlign': 'center'}, className='row'),
], style={'width': '100%', 'max-width':'none', 'padding':0,'backgroundColor':colors['background']},className='container')


#--- CALLBACKS ---   
#Update graphs
@callback([Output('graph_pred', 'figure'),Output('graph_mse', 'figure'),Output('runButton_dd', 'n_clicks'),Output('boolean-switch', 'value')], 
          [Input('graph_mse', 'clickData'),
           Input('runButton_dd', 'n_clicks'),
           State('datasetButton', 'n_clicks'),
           State('sliderR', 'value'),
           State('sliderSamples_dd', 'value'),
           State('sliderNoise_dd', 'value'),
           State('pol_degree', 'value'),
           State('boolean-switch', 'value'),
           State('graph_mse', 'figure')])
def click_update(clickdata, runbutton, datasetbutton,rho_val,nb_train_samples,train_noise_std,D_max,random_seed, figure):
    selected_degree = clickdata['points'][0]['x']
    rho = pow(10,rho_val) 
    if(random_seed==0):
        fig1, predictions = mse_graph(rho, D_max)
        fig2 = pred_graph(rho, D_max)
        #fig2.update_layout(legend=dict(yanchor="top",y=-0.15,xanchor="left",x=0,orientation='h'))        
    #fig1, predictions = mse_graph(rho, D_max)
    if(runbutton != None):
        print('run pressed')
        # print(runbutton)
        if (datasetbutton == None):
            fig1, predictions = mse_graph(rho, D_max)
            fig2 = pred_graph(rho, D_max)
        else:  
            fig1, predictions = mse_graph(rho, D_max, train_noise_std, nb_train_samples,random_seed)
            fig2 = pred_graph(rho, D_max, train_noise_std, nb_train_samples,random_seed)
            random_seed = random_seed+1
    else:
        if(random_seed!=0):
            fig1=figure
        else:
            random_seed=1
        print('run is none')
        #fig1=figure
        if (datasetbutton == None): 
            #fig1, predictions = mse_graph(rho, D_max)
            fig2 = pred_graph(rho, selected_degree)
            #fig2.update_layout(margin=dict(b=20,))
        else:
            #fig1, predictions = mse_graph(rho, D_max, train_noise_std, nb_train_samples,random_seed)
            fig2 = pred_graph(rho, selected_degree, train_noise_std, nb_train_samples,random_seed)
    print('random_seed is', random_seed)
    
    return fig2,fig1,None,random_seed

    
#disable 
@callback([Output('sliderSamples_dd','style'),Output('train_title','style')], 
           Input('datasetButton', 'n_clicks'))
def disable_samples(n):
    if(n==None):
        return [{'display':'none'},{'display':'none'}]
    else:
        return [{'display':'inline-block','color':'#323232','margin-left': 10,'width': '40%'},{'margin-left': 10,'display':'inline-block'}]

@callback([Output('sliderNoise_dd','style'),Output('noise_title','style')], 
           Input('datasetButton', 'n_clicks'))
def disable_noise(n):
    if(n==None):
        return [{'display':'none'},{'display':'none'}]
    else:
        return [{'display':'inline-block','color':'#323232','width': '40%'},{'display':'inline-block'}]

@callback(Output('backButton','style'), 
          Input('datasetButton', 'n_clicks'))
def disable_backbutton(n):
    if(n==None):
        return {'display':'none'}
    else:
        return {'display':'inline-block','margin-left':0,'margin-top':10,'margin-bottom':20,'color':'#323232','width': '50%'}
      
@callback(Output('datasetButton', 'n_clicks'), 
           Input('backButton', 'n_clicks'))
def rest_datasetbutton(n):
    if(n!=None):
        return None

@callback(Output('backButton', 'n_clicks'), 
           Input('backButton', 'n_clicks'))
def rest_backbutton(n):
    if(n!=None):
        return None
    
@callback([Output('sliderR','value'),Output('pol_degree','value')], 
            Input('backButton', 'n_clicks'))
def reset_conditions1(n):
        return [-12,16]

@callback(Output('pol_degree', 'error'), 
          [Input('pol_degree', 'value'),
           State('pol_degree', 'min'),
           State('pol_degree', 'max')])
def errorMessage1_3(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message

@callback(Output('sliderSamples_dd', 'error'), 
          [Input('sliderSamples_dd', 'value'),
           State('sliderSamples_dd', 'min'),
           State('sliderSamples_dd', 'max')])
def errorMessage2_3(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message

@callback(Output('sliderNoise_dd', 'error'), 
          [Input('sliderNoise_dd', 'value'),
           State('sliderNoise_dd', 'min'),
           State('sliderNoise_dd', 'max')])
def errorMessage3_3(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message