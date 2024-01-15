
"""
Created on Thu May 19 10:36:06 2022

@author: giuli
"""
#--------IMPORT---------------
from dash import dcc, html, Input, Output, callback, State
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.svm import SVC
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import dash_mantine_components as dmc
import circles_dataset


# --------- DEFINITION FUNCTION ------------
#generate dataset
def generate_data(dataset,n_samples,noise,class_distance, class_num):
    if dataset == 'Moons':
        moons = datasets.make_moons(n_samples=n_samples, noise=noise,random_state=0)  
        return moons

    elif dataset == 'Circles':     
        circles = circles_dataset.innested_circles(n_samples=n_samples, noise=noise, dist=class_distance, random_state=1,num_class=class_num)
        return circles

    elif dataset == 'Linearly separable':
        a=class_distance
        if(class_num == 2):
            centers = [[a, a], [-a, -a]]
        if(class_num == 3):
            centers = [[a, a], [-a, -a], [a, -a]]
        if(class_num == 4):
            centers = [[a, a], [-a, -a], [a, -a], [-a, a]]
        if(class_num == 5):
            centers = [[a, a], [-a, -a], [a, -a], [-a, a], [0, 0]]
        X, y = datasets.make_blobs(n_samples=n_samples, centers=centers, random_state=2, cluster_std=0.3)
        rng = np.random.RandomState(2)
        X_n = X + (noise) * rng.normal(0,1,X.shape)
        data = [X_n,y]
        return data

    else:
        raise ValueError('Data type incorrectly specified. Please choose an existing dataset.')

#Scatter plot
def scatterplot(kernel,C,gamma,dataset,n_samples,noise,multi_class,class_distance,class_num):
    #GENERATE DATA FOR PLOT
    X,y = generate_data(dataset,n_samples,noise,class_distance,class_num)
    print(np.unique(y))
    #prepare data for the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    scalerX = MinMaxScaler()
    XL = scalerX.fit_transform(X_train)
    YL = y_train
    XT = scalerX.transform(X_test)
    YT = y_test
    #CREATE MODEL
    if(kernel=='Linear'):
        kerneltype = 'linear'  
    if(kernel=='Radial Basis Function'):
        kerneltype = 'rbf'
    ALG = SVC(kernel=kerneltype, C=C, gamma=gamma,probability=True,decision_function_shape=multi_class)
    M = ALG.fit(XL,YL)
    YP = M.predict(XT)
    X_tot = np.concatenate((XL,XT),axis=0)
    y_tot = np.concatenate((YL,YT),axis=0)
    
    #CONTOUR
    h=0.02    
    bright_cscale = [[0, '#66FFFF'], [1, '#FFE800']] # blue#66FFFF   yellow#FFE800  #07dddd
    if (class_num > 3 and dataset!='Moons'):
        showline=True
    else:
        showline=False
    x_min, x_max = X_tot[:, 0].min()-.15, X_tot[:, 0].max()+.15 
    y_min, y_max = X_tot[:, 1].min()-.2, X_tot[:, 1].max()+.2 
    x1, x2 = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    #z = M.predict_proba(np.c_[x1.ravel(), x2.ravel()])
    #p_class = M.predict(np.c_[x1.ravel(), x2.ravel()])
    z= M.predict(np.c_[x1.ravel(), x2.ravel()])
    trace1 = go.Contour(
        x=np.arange(x1.min(), x1.max(),h),
        y=np.arange(x2.min(), x2.max(),h),
        z=z.reshape(x1.shape), 
        #z=z[:,0].reshape(x1.shape), 
        #text = p_class.reshape(x1.shape),
        name=' ',
        showscale=False,
        hovertemplate = '<i>Predicted class</i>:%{z}',
        colorscale=bright_cscale,
        contours_showlines=showline,
        opacity=0.9)

    #SCATTERPLOT
    #raggruppo per classi i dati per gli scatter
    X_0 = np.zeros((1,2))
    X_1 = np.zeros((1,2))
    X_2 = np.zeros((1,2))
    X_3 = np.zeros((1,2))
    X_4 = np.zeros((1,2))

    for k in range(XL.shape[0]):
        if(YL[k]==0):
            X_0 = np.concatenate((X_0,XL[k,:].reshape(1,-1)),axis=0)
        if(YL[k]==2):
            X_2 = np.concatenate((X_2,XL[k,:].reshape(1,-1)),axis=0)
        if(YL[k]==1):
            X_1 = np.concatenate((X_1,XL[k,:].reshape(1,-1)),axis=0)
        if(YL[k]==3):
            X_3 = np.concatenate((X_3,XL[k,:].reshape(1,-1)),axis=0)
        if(YL[k]==4):
            X_4 = np.concatenate((X_4,XL[k,:].reshape(1,-1)),axis=0)

    X_0 = np.delete(X_0,0,0)
    X_1 = np.delete(X_1,0,0)
    X_2 = np.delete(X_2,0,0)
    X_3 = np.delete(X_3,0,0)
    X_4 = np.delete(X_4,0,0)
    
    X_0_t = np.zeros((1,2))
    X_1_t = np.zeros((1,2))
    X_2_t = np.zeros((1,2))
    X_3_t = np.zeros((1,2))
    X_4_t = np.zeros((1,2))

    for k in range(XT.shape[0]):
        if(YT[k]==0):
            X_0_t = np.concatenate((X_0_t,XT[k,:].reshape(1,-1)),axis=0)
        if(YT[k]==2):
            X_2_t = np.concatenate((X_2_t,XT[k,:].reshape(1,-1)),axis=0)
        if(YT[k]==1):
            X_1_t = np.concatenate((X_1_t,XT[k,:].reshape(1,-1)),axis=0)
        if(YT[k]==3):
            X_3_t = np.concatenate((X_3_t,XT[k,:].reshape(1,-1)),axis=0)
        if(YT[k]==4):
            X_4_t = np.concatenate((X_4_t,XT[k,:].reshape(1,-1)),axis=0)

    X_0_t = np.delete(X_0_t,0,0)
    X_1_t = np.delete(X_1_t,0,0)
    X_2_t = np.delete(X_2_t,0,0)
    X_3_t = np.delete(X_3_t,0,0)
    X_4_t = np.delete(X_4_t,0,0)

    #scatter
    if(class_num == 2 or dataset=='Moons'):
        color_2 = '#A3BF45'
        color_1 = '#FFCD01'#giallo
        color_0 = '#33CCFF'#azzurro
        color_3 = '#FF0000'#rosso
        color_4 = '#8F00FF'#viola
    if(class_num == 3 and dataset!='Moons'):
        color_1 = '#A3BF45'#verde chiaro #3C8046
        color_2 = '#FFCD01'
        color_0 = '#33CCFF'
        color_3 = '#FF0000'
        color_4 = '#8F00FF'
    if(class_num == 4 and dataset!='Moons'):
        color_1 = '#2e856e'#verde scuro
        color_2 = '#A3BF45'
        color_0 = '#33CCFF'
        color_3 = '#FFCD01'
        color_4 = '#8F00FF'
    if(class_num == 5 and dataset!='Moons'):
        color_1 = '#2e856e'
        color_2 = '#A3BF45'
        color_0 = '#33CCFF'
        color_3 = '#C5E90B'#verde brillante
        color_4 = '#FFCD01'
                
    #TRAINING SET
    trace4 = go.Scatter(x=X_2[:,0],
                        y=X_2[:,1],
                        mode='markers',
                        name='Training Data class 2',
                        marker=dict(
                                    size=10,
                                    color=color_2,
                                    line=dict(width=1)
                                    ),                   
                        hovertemplate = '<i>(%{x},%{y})</i>')
    trace2 = go.Scatter(x=X_0[:,0],
                        y=X_0[:,1],
                        mode='markers',
                        name='Training Data class 0',
                        marker=dict(
                                    size=10,
                                    color=color_0,#BLUE
                                    line=dict(width=1)
                                    ),                   
                        hovertemplate = '<i>(%{x},%{y})</i>')

    trace3 = go.Scatter(x=X_1[:,0],
                        y=X_1[:,1],
                        mode='markers',
                        name='Training Data class 1',
                        marker=dict(
                                    size=10,
                                    color=color_1,
                                    line=dict(width=1)
                                    ),                   
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    trace5 = go.Scatter(x=X_3[:,0],
                        y=X_3[:,1],
                        mode='markers',
                        name='Training Data class 3',
                        marker=dict(
                                    size=10,
                                    color=color_3,
                                    line=dict(width=1)
                                    ),                   
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    trace6 = go.Scatter(x=X_4[:,0],
                        y=X_4[:,1],
                        mode='markers',
                        name='Training Data class 4',
                        marker=dict(
                                    size=10,
                                    color=color_4,
                                    line=dict(width=1)
                                    ),                   
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    #TEST SET
    trace9 = go.Scatter(x=X_2_t[:, 0],
                        y=X_2_t[:, 1],
                        mode='markers',
                        name='Test Data class 2',
                        marker=dict(
                                size=10,
                                symbol='triangle-up',
                                color=color_2,
                                line=dict(width=1)
                                ),
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    trace7 = go.Scatter(x=X_0_t[:, 0],
                        y=X_0_t[:, 1],
                        mode='markers',
                        name='Test Data class 0',
                        marker=dict(
                                size=10,
                                symbol='triangle-up',
                                color='#33CCFF', #BLUE
                                line=dict(width=1)
                                ),
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    trace8 = go.Scatter(x=X_1_t[:, 0],
                        y=X_1_t[:, 1],
                        mode='markers',
                        name='Test Data class 1',
                        marker=dict(
                                size=10,
                                symbol='triangle-up',
                                color=color_1,
                                line=dict(width=1)
                                ),
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    trace10 = go.Scatter(x=X_3_t[:, 0],
                        y=X_3_t[:, 1],
                        mode='markers',
                        name='Test Data class 3',
                        marker=dict(
                                size=10,
                                symbol='triangle-up',
                                color=color_3,
                                line=dict(width=1)
                                ),
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    trace11 = go.Scatter(x=X_4_t[:, 0],
                        y=X_4_t[:, 1],
                        mode='markers',
                        name='Test Data class 4',
                        marker=dict(
                                size=10,
                                symbol='triangle-up',
                                color=color_4,
                                line=dict(width=1)
                                ),
                        hovertemplate = '<i>(%{x},%{y})</i>')
    
    data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8, trace9,trace10,trace11]
    layout = go.Layout(legend=dict(
        yanchor="top",
        y=-0.15,
        xanchor="left",
        x=0,
        orientation='h'),
        autosize=True,
        #width=600,
        #height=450,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        xaxis_title="x0",
        yaxis_title="x1",
        font = {'color': colors['text'], 'size':11},
        margin=dict(t=40, b=0, l=0, r=30,),
        )
    
    fig = go.Figure(data=data, layout=layout)
    
    fig.update_layout(title_text='<i>Data plot with decision boundary</i>', title_x=0.08, font = dict(color=colors['text'], size=11))
 
    return fig

#plot Confusion Matrix
def plot_cm(kernel,C,gamma,dataset,n_samples,noise,multi_class,class_distance,class_num):
    #GENERATE DATA FOR PLOT
    X,y = generate_data(dataset,n_samples,noise,class_distance,class_num)
    #prepare data for the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    scalerX = MinMaxScaler()
    XL = scalerX.fit_transform(X_train)
    YL = y_train
    XT = scalerX.transform(X_test)
    YT = y_test
    #create model
    if(kernel=='Linear'):
        kerneltype = 'linear'       
    else:
        kerneltype = 'rbf'
    ALG = SVC(kernel=kerneltype, C=C, gamma=gamma, probability=True,decision_function_shape=multi_class)
    M = ALG.fit(XL,YL)
    YP = M.predict(XT)
    
    matrix = confusion_matrix(y_true=YT, y_pred=YP)
    
    if(class_num==2 or dataset=='Moons'):
        tn, fp, fn, tp = matrix.ravel()
        values = [tp, fn, fp, tn]
        label_text = ["True Positive",
                      "False Negative",
                      "False Positive",
                      "True Negative"]
        labels = ["TP", "FN", "FP", "TN"]
        palette = {"TP": "#0064FF","FN":"#FF8FDF", "FP": "#FFC300", "TN": "#FFC300"}
        palette = ["#339966","#FF8FDF", "#CC0000","#FF9933"]
        trace0 = go.Pie(
                labels=label_text,
                values=values,
                hoverinfo='label+value+percent',
                textinfo='text+value',
                text=labels,
                sort=False,
                domain=dict(x=[0,1]),
                marker=dict(
                    colors=palette)
                )
        
        layout = go.Layout(
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor='rgba(255,255,255,0)',
                orientation='h',
                y=-0.15),
            #width=300,
            #height=380,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            title=go.layout.Title(text='<i>Confusion matrix</i>'),
            font = dict(color=colors['text'], size=11)            
            )
        
        fig = go.Figure(data=trace0, layout=layout) 
        fig.update_layout(title_text='<i>Confusion matrix</i>')
        
    if (class_num>2 and dataset!='Moons'):
        if(class_num==3):
            x = ['class0', 'class1', 'class2']
            y = ['class0', 'class1', 'class2']
        if(class_num==4):
            x = ['class0', 'class1', 'class2', 'class3']
            y = ['class0', 'class1', 'class2', 'class3']   
        if(class_num==5):
            x = ['class0', 'class1', 'class2', 'class3', 'class4']
            y = ['class0', 'class1', 'class2', 'class3', 'class4']
        fig = ff.create_annotated_heatmap(z=matrix, x=x, y=y,colorscale='Viridis')#annotation_text=z_text,
        fig.add_annotation(dict(font=dict(color="black",size=11),
                                x=0.5,
                                y=-0.18,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        fig.add_annotation(dict(font=dict(color="black",size=11),
                                x=1.13,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        
        # layout = go.Layout(            
        #     margin=dict(l=0, r=0, t=0, b=0), 
        #     plot_bgcolor=colors['background'],
        #     paper_bgcolor=colors['background'],
        #     font = dict(color=colors['text'], size=10))     
        fig.update_layout(margin=dict(l=20, r=60, t=70, b=50), 
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font = dict(color=colors['text'], size=11),
            title_text='<i>Confusion matrix</i>')
    
    return fig

#------------LAYOUT PAGE SVM----------------
#colors = {'background': '#303030','text': '#E0E0E0', 'background_card': '#D3D3D3'} 
colors = {'background': 'white','text': 'black', 'background_card': 'grey'}  
text_f = '*Support vector machine (SVM) is a supervised learning method used for both classification and regression. \
Here you can test the algorithm on classification tasks: you can choose between three different datasets (two classes Moons dataset, \
multi class circles and linearly separable datasets) and set noise, number of samples \
and distance between classes, but most of all you can tune the main model’s hyperparameters: kernel type, kernel parameter γ (which regulates the \
non-linearity of the solution and it is not available in case of linear kernel), regularization parameter C (which \
regulates the complexity of the solution), and the approach to adopt in muti-class case (one-vs-one, one-vs-rest). See reference https://dl.acm.org/doi/abs/10.1145/130385.130401 for more details on the algorithm.*'

copyright_phrase = '*\u00A9 2022 Giulia Cademartori All Rights Reserved*' 
image_path_2 = 'assets/unige_logo.png'
                                                                                                    
layout = html.Div([
    # first row
    html.Div(children=[
      dcc.Markdown(text_f,style={'width': '98.5%', 'text-align': 'justify', 'font-size':13, 'color': 'black'})
    ],style={'margin-top': 15,'margin-bottom': 0, 'margin-left': 0,'color': colors['text'],'align': 'justify'}, className='row'),
    #second row
    html.Div(children=[      
        # first column of second row
        html.Div(children=[  
            dcc.Graph(id="graph_scatter")],style={'width': '45%','height': '100%','backgroundColor': colors['background'],'display': 'inline-block','padding-left': 0,'padding-right':0, 'margin-left': 0, 'margin-right':0},className='col-md-6'),
        html.Div(children=[  
            dcc.Graph(id="cm")],style={'backgroundColor': colors['background'],'display': 'inline-block','padding-left': 120,'padding-top': 35,'padding-bottom': 0, 'padding-right':0, 'margin-left': 0, 'margin-right':0},className='col-md-6'),
        ],id="graphrow",style={'width': 'auto', 'height': 'auto', 'margin-left': 30, 'margin-right':0,'margin-top':-5,'margin-bottom':10},className='row'),
    #third row
    html.Div(children=[  
        dbc.Card( 
        dbc.CardBody([
            html.H4("Parameters setting", className="card-title", style={'margin-bottom': 5}),
            #select dataset
            html.Div(children=[
                dcc.Markdown(children='Select Dataset:'),
                dcc.Dropdown(['Moons','Linearly separable','Circles'],'Moons',id='datasetDropdown'),
            ],style={'width': '23%', 'display': 'inline-block','margin-top': 10,'margin-bottom': 20,'color':'#323232'},className='row'),
            #select noise
            html.Div(children=[
                dcc.Markdown(children='Enter dataset\'s noise value:', style={'margin-left': 10}),
                dmc.NumberInput(id='sliderNoise', size='sm', precision=1, min=0, max=1, step=0.1, value=0.1, style={'margin-left':10,'color':'#323232','width': '60%'}),
            ],style={"verticalAlign": "top",'width': '20%', 'display': 'inline-block','margin-left': 60,'margin-right': 12,'margin-top':10,'margin-bottom':20},className='row'),
            #select n_samples
            html.Div(children=[
                dcc.Markdown(children='Enter number of samples:', style={'margin-left': 10}),
                dmc.NumberInput(id='sliderSamples', size='sm', min=1, max=5000, step=1, value=500, style={'margin-left':10,'color':'#323232','width': '75%'}),
            ],style={"verticalAlign": "top",'width': '15%', 'display': 'inline-block','margin-top':10,'margin-bottom':20,'margin-right': 20,'margin-left': 0},className='row'),
            #select class_distance
            html.Div(children=[
                dcc.Markdown(id='classdistance_title', children='Enter classes\'s distance:', style={'margin-left': 10}),
                dmc.NumberInput(id='classDistance', size='sm', precision=1, min=0, max=2, step=0.1, value=0.8, style={'margin-left':10,'color':'#323232','width': '60%'}),
            ],style={"verticalAlign": "top",'width': '15%', 'display': 'inline-block','margin-top':10,'margin-bottom':20,'margin-right': 20,'margin-left': 30},className='row'),
            #select number of classes
            html.Div(children=[ 
                dcc.Markdown(id='number_class_title', children='Select number of classes:', style={'margin-left': 10}),
                dmc.NumberInput(id='number_class', size='sm', min=2, max=5, step=1, value=2, style={'margin-left':10,'color':'#323232','width': '70%'}),
            ],style={"verticalAlign": "top",'width': '15%', 'display': 'inline-block','margin-top':10,'margin-bottom':20,'margin-right': 30,'margin-left': 0},className='row'),
            #select kernel
            html.Div(children=[
                dcc.Markdown(children='Select Kernel:'),
                dcc.Dropdown(['Radial Basis Function','Linear'],'Radial Basis Function',id='kernelDropdown'),             
            ],style={'width': '28%', 'display': 'inline-block','margin-top': 10,'margin-bottom': 20,'color':'#323232'},className='row'),
            #select kernel par
            html.Div(children=[ 
                dcc.Markdown(id='gamma_title', children='Select Kernel parameter γ:'),
                dcc.Slider(min=-3,max=4,id='sliderGamma', step=None, marks={i: '{}'.format(pow(10,i)) for i in range(-3,5)},value=2,updatemode='drag'),
            ],style={"verticalAlign": "top",'width': '40%', 'display': 'inline-block','margin-top': 10,'margin-left': 60,'margin-right': 20,'margin-bottom':20},className='row'),
            #select reg parameter
            html.Div(children=[ 
                dcc.Markdown(children='Select regularization parameter C:'),
                dcc.Slider(min=-3,max=4,id='sliderC', step=None, marks={i: '{}'.format(pow(10,i)) for i in range(-3,5)},value=0,updatemode='drag'),
            ],style={'width': '40%', 'display': 'inline-block','margin-top': 10,'margin-bottom':20},className='row'),
            #select class strategy
            html.Div(children=[ 
                dcc.Markdown(id='class_strategy_title', children='Select multi-class strategy:'),
                dcc.Dropdown(['ovo','ovr'],'ovr',id='classDropdown'), 
            ],style={"verticalAlign": "top",'width': '35%', 'display': 'inline-block','margin-left': 50,'margin-bottom':20},className='row'),
            #select buttom
            html.Div(children=[ 
                dbc.Button("Run", outline=True, color="success", id='runButton',style={'backgroundColor': 'lightgreen'})
            ],style={'display': 'block'},className="me-1")
        ]),color=colors['background']),
    ], style={'margin-left': 20,'margin-top': 10,'width': '95%'},className='row'), 
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
@callback([Output('graph_scatter', 'figure'),Output('cm', 'figure'),Output('cm','style')],
          [Input('runButton', 'n_clicks'),
            State('datasetDropdown', 'value'),
            State('kernelDropdown', 'value'),
            State('sliderC', 'value'),
            State('sliderGamma', 'value'),
            State('sliderNoise', 'value'),
            State('sliderSamples', 'value'),
            State('classDropdown', 'value'),
            State('classDistance', 'value'),
            State('number_class', 'value')
            ])
def update_graphs(runbutton,dataset,kernel,Cvalue,gammavalue,noise,n_samples,multi_class,class_distance, class_num):
    C = pow(10,Cvalue)
    gamma = pow(10,gammavalue)    
    
    fig1 = scatterplot(kernel,C,gamma,dataset,n_samples,noise,multi_class,class_distance, class_num)
    fig2 = plot_cm(kernel,C,gamma,dataset,n_samples,noise,multi_class,class_distance, class_num)
    
    if(class_num>2 and dataset!='Moons'):
        style={'width': '70%','height': '75%'}
    else:
        style={'width': '90%','height': '85%'}
      
    # fig1.update_layout()
    fig2.update_layout(title_text='<i>Confusion matrix</i>', title_y=0.99, title_x=0.1, margin=dict(t=80,))
    fig2.update_yaxes(automargin=True)
    fig1.update_layout(title_text='<i>Data plot with decision boundary</i>',title_y=0.94, title_x=0.08, margin=dict(t=60,))
    
    return [fig1,fig2,style]

#disable gamma slider
@callback([Output('sliderGamma','style'),Output('gamma_title','style')], 
              Input('kernelDropdown', 'value'))
def disable_gamma(kernel):
    if(kernel=='Linear'):
        return [{'display':'none'},{'display':'none'}]
    else:
        return [{'display':'block'},{'display':'block'}]

@callback([Output('classDropdown','style'),Output('class_strategy_title','style')], 
              Input('datasetDropdown', 'value'))
def disable_multiclass(dataset):
    if(dataset == 'Linearly separable'):
        return [{'display':'block'},{'display':'block'}]
    else:
        return [{'display':'none'},{'display':'none'}]
    
@callback([Output('number_class','style'),Output('number_class_title','style')], 
              Input('datasetDropdown', 'value'))
def disable_numclass(dataset):
    if(dataset == 'Linearly separable' or dataset == 'Circles'):
        return [{'display':'block','margin-left':0,'color':'#323232','width': '60%'},{'display':'block'}]
    else:
        return [{'display':'none'},{'display':'none'}]
 
@callback([Output('classDistance','style'),Output('classdistance_title','style')], 
              Input('datasetDropdown', 'value'))
def disable_distanceclass(dataset):
    if(dataset == 'Linearly separable' or dataset == 'Circles'):
        return [{'display':'block','margin-left':0,'color':'#323232','width': '70%'},{'display':'block'}]
    else:
        return [{'display':'none'},{'display':'none'}]
    
@callback(Output('sliderNoise', 'error'), 
          [Input('sliderNoise', 'value'),
           State('sliderNoise', 'min'),
           State('sliderNoise', 'max')])
def errorMessage1(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message

@callback(Output('sliderSamples', 'error'), 
          [Input('sliderSamples', 'value'),
           State('sliderSamples', 'min'),
           State('sliderSamples', 'max')])
def errorMessage2(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message

@callback(Output('classDistance', 'error'), 
          [Input('classDistance', 'value'),
           State('classDistance', 'min'),
           State('classDistance', 'max')])
def errorMessage3(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message
    
@callback(Output('number_class', 'error'), 
          [Input('number_class', 'value'),
           State('number_class', 'min'),
           State('number_class', 'max')])
def errorMessage4(value, minimum, maximum):
    print(minimum)
    print(value)
    if (value==None or value<minimum or value>maximum):
        message = 'Invalid value! Enter number between {}-{}'.format(minimum,maximum)
        return message