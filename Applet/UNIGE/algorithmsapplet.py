# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:45:19 2022

@author: giuli
"""

from dash import Dash, dcc, html, Input, Output
import appletPage1corretto, appletPage2corretto, applet_tab3
import dash_bootstrap_components as dbc
import waitress


#-------APP---------
app = Dash(__name__, suppress_callback_exceptions=True,external_stylesheets = [dbc.themes.BOOTSTRAP])
application = app.server

intro_text1 = 'This is an interactive applet developped within the IMath project. This applet has the scope of exploring and giving you better understanding of\
    Machine Learning (ML) algorithms and how hyperparameters impact on training and test error\'s trend. In the first two tabs, you could test two basic ML algorithms, Random Forest (RF) and Support Vector Machine (SVM),\
        on a classification problem on three different datasets: you will be able to modify the alogorithm\'s parameters and see how their changes affect respectively classification\'s\
    accuracy and decision boundary. In the third tab, instead, you can experiment the test error\'s double descent phenomenon and study how data and model\'s\
    hyperparameters affect error\'s trends.'
#TABS STYLES
tabs_styles = {
    'height': '53px',
    'borderTop': 0
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

image_path = 'assets/imath_logo.png'

app.layout = html.Div([
    html.Div(children=[
      dcc.Markdown(children='**Understanding Machine Learning algorithms and the impact of hyperparameters on their performance**')
    ],style={'margin-top': 10,'margin-bottom': 0, 'margin-left': 0,'color': 'black', 'font-size':22}, className='row'),
    
    html.Div(children=[
        html.Div(children=[dcc.Markdown(children=intro_text1)],style={'display': 'inline-block','color': 'black', 'width': '87%','text-align': 'justify', 'font-size':14,'padding-left': 0,'padding-right':0, 'margin-left': 10, 'margin-right':0},className='col-md-11'),
        html.Div(
            html.A([
                html.Img(src=image_path, style={'height':'85%', 'width':'60%'})
            ], href='https://imath.pixel-online.org/', target="_blank")
        ,style={'display': 'inline-block','padding-left': 0,'padding-right':0, 'margin-left': 40, 'margin-right':0}, className='col-md-1'),        
    ],style={'width': '100%','margin-left': 0, 'margin-right':0,'margin-top':0,'margin-bottom':5}, className='row'),
    
    #110 83% 85% 70%
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Understanding algorithms: SVM', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Understanding algorithms: RF', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Understanding hyperparameters: Double Descent', value='tab-3', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return appletPage1corretto.layout
    if tab == 'tab-2':
        return appletPage2corretto.layout
    if tab == 'tab-3':
        return applet_tab3.layout



app.debug=False
if __name__ == '__main__':   
    #waitress.serve(application, host='127.0.0.1', port=8050)
    app.run_server(debug=True,host='127.0.0.1', port=8050,use_reloader=False)