# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:59:59 2022

@author: giuli

Last update on Tue May 09
@author: andrej.kosir@fe.uni-lj.si

"""

# Installs
# python.exe -m pip install -r requirements.txt
# pip install dash
# pip install dash-bootstrap-components
# pip install waitress
# cd 'C:\Users\Andrej Košir\Lucami Dropbox\Lucami Team Folder\10-iMath\00-ProgrammingCode\02-OptLearnAndLearningIndicators\partner_code_Ljubljana'

#%%
#import
from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import waitress
import random
import pygsheets
import time
import importlib
import networkx as nx

#%%
#-------APP---------
app = Dash(__name__, suppress_callback_exceptions=True,external_stylesheets = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css'])
application = app.server

image_path ='assets/imath_logo.png'

gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('Imath_prototype_data')
wks = sh[0]
df=wks.get_as_df(has_header=True)

user_input=4 
evaluation_qstion=4
n_qstion=5
init_data=['-1'] * (n_qstion*2+user_input+evaluation_qstion)
skip_message ='**You have unanswered questions, please go back and answer all the question before submitting the test**!'

# Interface to Lucami code
global qIDs_set, keys_to_qs_dc #, imt 
global q_data, student_data, s_hist, start_conM, kws_conM, mrg_conM, qrec_conf 
global setupQ 
setupQ = True
print ('setupQ in:', setupQ)
if setupQ:

    import pandas as pd 
    import iMath_tools as imt
    import conceptMapGenerator_v02 as cmg
    import QRecommender_in as qrec
    import manualConceptMaps as mcm
    import build_test_CM_tools as btcm   
    #importlib.reload(imt)


    # Concept map generator 
    X_keywords_df = sh[7].get_as_df(has_header=True) # pd.read_excel(data_fn, sheet_name='Data Keywords') # Keywords
    X_qs_df = sh[0].get_as_df(has_header=True).iloc[0:40] #pd.read_excel(data_fn, sheet_name='Data_input', nrows=41) # Questions
    #print ('X_qs_df: ', X_qs_df)
    #X_qs_df = pd.read_excel(data_fn, sheet_name='Data_input', nrows=40) # Questions
    X_stud_hist_df = sh[1].get_as_df(has_header=True) # pd.read_excel(data_fn, sheet_name='Data_output') # Student history
    qIDs_set = imt.get_question_IDs(X_qs_df['question'])
    kw_dc = dict(zip(X_keywords_df.id, X_keywords_df.name))
    qs_kwsIDs_dc = imt.get_qs_to_kwsIDs_dc(X_qs_df) # Get relational data
    codeIn = 'FirstTwo'
    kws_to_qs_dc = imt.get_kws_to_qs_dc(kw_dc, qs_kwsIDs_dc, code=codeIn) # All, FirstOnly, FirstTwo, FirstThree
    keys_to_qs_dc = imt.get_keys_to_qs_from_keywords(kws_to_qs_dc)
    set_of_qs_lst = list(np.unique(sum([list(keys_to_qs_dc[key]) for key in keys_to_qs_dc], [])))

    #qs_users_grade_mat_df = imt.get_qs_users_grade_mat(X_qs_df, X_stud_hist_df) # Get relevance and confidence matrices
    #qq_confidence_mat_df = imt.get_qq_confidence_mat(qs_users_grade_mat_df)
    #qs_concepts_relevance_mat_df = imt.get_qs_concepts_relevance_mat(kws_to_qs_dc, qs_kwsIDs_dc, qq_confidence_mat_df)
    #concept_concept_relevance_mat_df = imt.get_concept_concept_relevance_mat(qs_concepts_relevance_mat_df, qq_confidence_mat_df, est_code='q_conf')

    # QRecomender
    #[q_data, student_data] = qrec.load_data(data_fn)
    [q_data, student_data] = qrec.load_data_from_gs(sh)
    # Calculate history of all students from loaded data
    s_hist = qrec.calc_history(student_data, q_data)

    # Compute current raw independent conM
    conc_map_conf = {}
    qrec_conf = {'conc_compl_T':1.0,
                'conc_succ_T':0.8,
                'tot_compl_T':0.9,
                'no_q_code': -1}
    #alg_code = 'rel_shift'
    #kws_conM = cmg.get_conceptMap_from_keyQs(qIDs_set, keys_to_qs_dc, alg_code, pars=[0.0])
    #alg_code = 'group_qs'
    #start_conM = cmg.get_startup_conceptMap(set_of_qs_lst, keys_to_qs_dc, alg_code)
    concept_map_path = '01-ConceptMaps/'
    topic_nm = 'mat_comp_merged'
    mrg_conM = btcm.load_conM(concept_map_path, topic_nm)
    
    # print ('Done setup')
    setupQ = False


# ---
app.layout = html.Div([

    html.Div(children=[
      dcc.Markdown(children='**Self-Assessment Test**')
    ],style={'margin-top': 10,'margin-bottom': 30, 'textAlign': 'center','margin-left': 0,'color': 'black', 'font-size':26}, className='row'),
    
    html.Div(children=[
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(children=[
                        dcc.Markdown(children='Question 1', id='card_title', className="card-title", style={'margin-bottom': 10,'font-size':22}),
                        #question
                        html.Div(children=[
                            dcc.Markdown(children=df.iloc[0,0], id='question',style={'font-size':20,'margin-right': 20}),
                            dcc.RadioItems([{"label": df.iloc[0,1],"value": "0"},{"label": df.iloc[0,2],"value": "1"},{"label": df.iloc[0,3],"value": "2"},{"label": df.iloc[0,4],"value": "3"},{"label": df.iloc[0,5],"value": "4"}],None,id='question_answer', inline=False, style={'font-size':18}),
                        ],id='radio_item_div',style={'display':'none'},className='row'),
                        html.Div(children=[dbc.Row([
                        #back button
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                        ],style={'margin-bottom': 10},className="me-1"),
                        width={"size": 2, "order": 1,"offset": 3}),
                        #next button
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'none'})
                        ],style={'margin-bottom': 10},className="me-1"),
                        width={"size": 2, "order": 3}),
                        #skip button
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                        ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                        ])]),
                        html.Div(children=[dbc.Row([
                        #button new test
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                        ],className="me-1"),                    
                        width={"offset": 3}),
                        #exit button
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                        ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                        ])]),
                        #begin evaluation test button
                        dbc.Row(
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                        ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                        ),
                        #name
                        html.Div(children=[
                            dbc.Label("Name", size="md"),
                            dbc.Input(type="text",id='name',)
                        ],style=dict(display='flex', justifyContent='center')),
                        #surname
                        html.Div(children=[
                            dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                            dbc.Input(type="text", id='surname', style={'margin-bottom': 20})
                        ],style=dict(display='flex', justifyContent='center', width='50%')),
                        #university
                        html.Div(children=[
                            dbc.Label("University:", size="md", style={'margin-right': 20}),
                            dbc.Input(type="text", id='university', style={'margin-bottom': 20})
                        ],style=dict(display='flex', justifyContent='center', width='50%')),
                        #email
                        html.Div(children=[
                            dbc.Label("Email:", size="md", style={'margin-right': 20}),
                            dbc.Input(type="text", id='email', style={'margin-bottom': 40})
                        ],style=dict(display='flex', justifyContent='center', width='50%')),
                        #start test button
                        dbc.Row(
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("START TEST", size="lg", outline=True, color="success", id='start_button',style={'backgroundColor': 'lightgreen','font-size':16})
                        ],style={'margin-bottom': 10},className="me-1"),
                        width={"size": 3, "offset": 5}),    
                        ), 
                        #saving data
                        html.Div(children=[
                            dcc.Markdown(children='**Saving data...**')
                        ],id='loading_page', style={'display':'none'}, className='row'),
                        #error message
                        html.Div(children=[
                            dcc.Markdown(children=skip_message)
                        ],id='error_message', style={'display':'none'}, className='row'),
                        #submit button
                        dbc.Row(
                        dbc.Col(
                        html.Div(children=[ 
                            dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                        ],style={'margin-bottom': 10},className="me-1"),
                        width={"size": 3, "offset": 5}),    
                        ),
                    ], id='card_body'),color='white'),
                width={"size": 15, "offset": 7}
            ),
        ),
    ], style={'margin-top': 10,'width': '50%'},className='row'),
    
    html.Div(children=[
            dbc.Row(
                dbc.Col(
                html.A([
                    html.Img(src=image_path, style={'height':'60%', 'width':'5%'})
                ], href='https://imath.pixel-online.org/', target="_blank"),
                width={"size": 10, "offset": 6}),
            align='center'),
            dbc.Row(dbc.Col(
                width={"size": 10, "offset": 6}))
        ],style={'margin-top': 170,'margin-bottom': 0, 'margin-left': 0}),
    
    html.Div(children=[dcc.Store(id="counter", data=0)]),
    
    html.Div(children=[dcc.Store(id="index_previous_question", data=0)]),
    
    html.Div(children=[dcc.Store(id="seed", data=0)]),
    
    html.Div(children=[dcc.Store(id="starting_time", data=0)]),
    
    html.Div(children=[dcc.Store(id="list_index_question", data=[])]),
    
    html.Div(children=[dcc.Store(id="list_answers", data=[None,None,None,None,None])]),
    
    html.Div(children=[dcc.Store(id="backward", data=0)]),
    
    html.Div(children=[dcc.Store(id="count_df", data=0)]),
    
    html.Div(children=[dcc.Store(id="count_df_2", data=0)]),
    
    html.Div(children=[dcc.Store(id="next_question", data=-1)]),   

    html.Div(children=[dcc.Store(id="count_eq ", data=41)]),   
    
    html.Div(children=[dcc.Store(id="df_output_1", data=init_data)]), 
    
    html.Div(children=[dcc.Store(id="df_output_2", data=[])]), 
    
    html.Div(children=[dcc.Store(id="df_output_3", data=[])]),
    
    html.Div(id='trigger',children=0, style=dict(display='none')),
    
    html.Div(id='trigger_2',children=0, style=dict(display='none')),
    
    html.Div(children=[
      dcc.Markdown(children='Test completed!', id='end')
    ],style={'margin-top': 10,'margin-bottom': 30, 'textAlign': 'center','margin-left': 0,'color': 'black', 'font-size':26, 'display':'none'}, className='row'),
    
])
            

#python function to calculate next question
def algorithm_for_new_question(output_row):

    global qIDs_set, keys_to_qs_dc, qrec_conf # , cmg
    #global q_data, student_data # , qrec
    global start_conM, kws_conM, mrg_conM
    
    # Concept map
    #print ('@algorithm_for_new_question: get_conceptMap_from_hist')
    #conM = cmg.get_conceptMap_from_hist(output_row, qIDs_set, keys_to_qs_dc)
    conM = mrg_conM


    # Q recommender
    # user_id = output_row[3] #'a@b.c' users_data
    [q_data, student_data] = qrec.load_data_from_gs(sh)
    s_hist = qrec.calc_history(student_data, q_data)
    #qrec.update_user_history(output_row, s_hist, q_data, debug=False)
    result = qrec.calc_next_question(output_row, s_hist, q_data, conM, qrec_conf, debug = True)


    #print('User: ', users[selected_id], ' Next q: ', result[0], '  concept ', result[1])
    next_question = result[0]
    #print ('next_question from us: ', next_question)

    # next_question=3#remove this line inserting your body function!! Only here to return something and avoid error! The function's call is now commented.
    
    #INSERT BODY OF YOUR FUNCTION!!!!!

    #print ('\n output_row row', output_row)
    
    return next_question


l1 = list(range(0,40)) #l1=[0,1,2,3,4,5,6,7,8,9]
answers=[None,None,None,None,None]
def generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3): 
    
    print('j:{}'.format(j))
    print('backward {}'.format(backward))
    print('clicked answer {}'.format(clickedAnswer))
    print(list_answers)     
        
    if(j!=n_qstion+1):
        df_output[(j-1)*2+user_input+evaluation_qstion]= index_previous_question       
        if(isinstance(optionList[0], dict) or clickedAnswer==None): 
            df_output[(j-1)*2+user_input+evaluation_qstion+1]=(clickedAnswer)
            list_answers[j-1]=(clickedAnswer)
        else:
            df_output[(j-1)*2+user_input+evaluation_qstion+1]=str(optionList.index(clickedAnswer))#insert index not value
            list_answers[j-1]=(clickedAnswer)
            
    if(backward==0): 
        print('j:{}'.format(j))
        if(j==n_qstion): #if j==5 last question has been already chosen
            print(index_previous_question) 
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question)
            next_question=list_index_question[-1] #load last question and relevant user's answer
            clickedAnswer=list_answers[-1]
        elif(j>=len(list_index_question)):
            print('1')
            #if index_previous_question not in list_index_question:
            list_index_question.append(index_previous_question) 
            list_questions = [x for x in l1 if x not in list_index_question]
            ### --- NOW CHOSEN RANDOMLY THEN IDENTIFY BY THE ALGORITHM ----
            next_question_t = random.sample(list_questions,1)
            next_question=next_question_t[0] 
            output_row = df_output + df_output_2 + df_output_3
            print ('\n output_row:', output_row)
            next_question=algorithm_for_new_question(output_row)
            clickedAnswer=None
        else:#if j<len it means I came back and if I click next the past question and answer has to be loaded
            print('2')
            next_question=list_index_question[j]
            clickedAnswer=list_answers[j]
    else:#back button pressed
        if(j==n_qstion+1): # load last question and relevant user's answer
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question)
            next_question=list_index_question[-1] 
            clickedAnswer=list_answers[-1]
        else:
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question)
            next_question=list_index_question[j-2]
            clickedAnswer=list_answers[j-2]

    new_question = df.iloc[next_question,0]#text
    new_options = [df.iloc[next_question,1],df.iloc[next_question,2],df.iloc[next_question,3],df.iloc[next_question,4],df.iloc[next_question,5]]
    print(list_index_question)
    return list_index_question,new_options,new_question,next_question,list_answers,clickedAnswer


def save_on_google_sheets(list1,list2,list3):
    gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
    sh = gc.open('Imath_prototype_data')
    wks = sh[4]
    wks.add_rows(1)
    final_list= list1 + list2 + list3
    final_row = pd.DataFrame([final_list])
    print('final_row')
    print(final_row)
    wks.set_dataframe(final_row,(wks.rows,1),copy_head=False)


#--- CALLBACKS ---
#update question
@app.callback([Output('card_body', 'children'),Output('counter','data'),Output('index_previous_question','data'),
               Output('starting_time','data'), Output('name','value'), Output('surname','value'),Output('university','value'), 
               Output('email','value'), Output('seed','data'), Output('list_index_question','data'), Output('backward','data'),
               Output('list_answers','data'), Output('next_question','data'), Output('question_answer', 'value'),Output('count_eq ','data'),
               Output('df_output_1','data'), ],
              [Input('nextButton', 'n_clicks'),
               Input('testButton', 'n_clicks'),
               Input('start_button', 'n_clicks'),
               Input('exitButton', 'n_clicks'),
               Input('submitButton', 'n_clicks'),
               Input('trigger','children'),
               Input('skipButton', 'n_clicks'),
               Input('previuosButton', 'n_clicks'),
               Input('evalButton', 'n_clicks'),
               State('question_answer', 'value'),#clicked value
               State('question_answer', 'options'),#list of options
               State('counter','data'),
               State('index_previous_question','data'),
               State('starting_time','data'),
               State('name','value'),
               State('surname','value'),
               State('university','value'),
               State('email','value'),
               State('seed','data'),
               State('list_index_question','data'),
               State('backward','data'),
               State('list_answers','data'),
               Input('trigger_2','children'),
               State('next_question','data'),
               State('count_eq ','data'),
               State('df_output_1','data'), 
               State('df_output_2','data'),
               State('df_output_3','data'),
              ])
def update_card_body(nextbutton,testButton,start_button,exitButton,submitbutton,trigger,skipbutton,previuosbutton,evalButton,clickedAnswer,optionList,j,index_previous_question,start,name,surname,university,email,seed,list_index_question,backward,list_answers,trigger_2,next_question,count_eq,df_output,df_output_2,df_output_3):
    print('evalButton')
    print(evalButton)
    print('count_eq')
    print(count_eq)
    if(start_button==None or exitButton!=None):
        if(ctx.triggered_id !='exitButton'):
            #re-initialize
            df_output_2 = []
            df_output_3 = []
            df_output=['-1'] * (n_qstion*2+user_input+evaluation_qstion)
            print(df_output)
            login=int(time.time()*1000)
            value_string_login = str(-2) +'#'+ 'login' +'#'+ str(login)
            df_output.append(value_string_login) #take login timestamp
            print(df_output)
            
            
        print('phase 1')
        name=''
        surname=''
        university=''
        email =''
        options= [{"label": df.iloc[0,1],"value": "0"},{"label": df.iloc[0,2],"value": "1"},{"label": df.iloc[0,3],"value": "2"},{"label": df.iloc[0,4],"value": "3"},{"label": df.iloc[0,5],"value": "4"}]
        children=[
                    dcc.Markdown(children='Question 1', id='card_title', className="card-title", style={'display':'none'}),
                    #question
                    html.Div(children=[
                        dcc.Markdown(children=df.iloc[0,0], id='question',style={'display':'none'}),
                        dcc.RadioItems(options,id='question_answer', inline=False, style={'font-size':18}),
                    ],id='radio_item_div',style={'display':'none'},className='row'),
                    html.Div(children=[dbc.Row([
                    #back button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'inline-block','backgroundColor': '#99cfe0','font-size':16})
                    ],style={'display':'none'},className="me-1"),
                    width={"size": 2, "order": 1,"offset": 3}),
                    #next button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'inline-block','backgroundColor': 'lightgreen','font-size':16})
                    ],style={'display':'none'},className="me-1"),
                    width={"size": 2, "order": 3}),
                    #skip button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'inline-block','backgroundColor': '#FF7377','font-size':16})
                    ],style={'display':'none'},className="me-1"),width={"size": 3, "order": 4})
                    ])]),
                    html.Div(children=[dbc.Row([
                    #button new test
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                    ],className="me-1"),                    
                    width={"offset": 3}),
                    #exit button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                    ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                    ])]),
                    #begin evaluation test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                    ),
                    #name
                    html.Div(children=[
                        dbc.Label("Name:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='name',style={'margin-bottom': 20})
                    ],style=dict(display='flex', justifyContent='center', width='50%')),
                    #surname
                    html.Div(children=[
                        dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                    ],style=dict(display='flex', justifyContent='center', width='50%')),
                    #university
                    html.Div(children=[
                        dbc.Label("University:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                    ],style=dict(display='flex', justifyContent='center', width='50%')),
                    #email
                    html.Div(children=[
                        dbc.Label("Email:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                    ],style=dict(display='flex', justifyContent='center', width='50%')),
                    #start test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START TEST", size="lg", outline=True, color="success", id='start_button',style={'backgroundColor': 'lightgreen','font-size':16})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),  
                    ),
                    #saving data
                    html.Div(children=[
                        dcc.Markdown(children='**Saving data...**')
                    ],id='loading_page', style={'display':'none'}, className='row'),
                    #submit button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                ] 
        next_question=-1

    elif((start_button!=None and j==0) or testButton!=None or (previuosbutton!=None and j==2)):
        print('phase 2')
        if(j!=2):#first question, not going back
            starting_question = random.sample(l1,1)[0]
            #output_row = df_output + df_output_2 + df_output_3
            #starting_question = algorithm_for_new_question(output_row)
            new_question=df.iloc[starting_question,0] 
            new_options=[df.iloc[starting_question,1],df.iloc[starting_question,2],df.iloc[starting_question,3],df.iloc[starting_question,4],df.iloc[starting_question,5]]
            index_previous_question=starting_question
        else:
            if(clickedAnswer==0):#if user doesn't answer, clickedAnswer None
                clickedAnswer = None
            backward=1
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3)
            j=0
            next_question=index_previous_question
            clickedAnswer=list_answers[0]
        
        #LOAD IMAGES
        size='auto'
        #verify if options are images an load them as images
        if(isinstance(new_options[0], str)) and ('assets' in new_options[0].split('/')):         
            options0 = {"label": html.Img(src=new_options[0],style={'height':size, 'width':size}),"value": "0"}
        else:
            options0 = {"label": html.Div(new_options[0], style={'display':'inline-block','font-size':18}),"value": "0"}
            
        if(isinstance(new_options[1], str)) and ('assets' in new_options[1].split('/')):         
            options1 = {"label": html.Img(src=new_options[1],style={'height':size, 'width':size}),"value": "1"}
        else:
            options1 = {"label": html.Div(new_options[1], style={'display':'inline-block','font-size':18}),"value": "1"}
            
        if(isinstance(new_options[2], str)) and ('assets' in new_options[2].split('/')):         
            options2 = {"label": html.Img(src=new_options[2],style={'height':size, 'width':size}),"value": "2"}
        else:
            options2 = {"label": html.Div(new_options[2], style={'display':'inline-block','font-size':18}),"value": "2"}              
        
        if(isinstance(new_options[3], str)) and ('assets' in new_options[3].split('/')):         
            options3 = {"label": html.Img(src=new_options[3],style={'height':size, 'width':size}),"value": "3"}
        else:
            options3 = {"label": html.Div(new_options[3], style={'display':'inline-block','font-size':18}),"value": "3"}
        
        options4 = {"label": html.Div(new_options[4], style={'display':'inline-block','font-size':18}),"value": "4"}
        
        options = [options0,options1,options2,options3,options4]
        r_item = dcc.RadioItems(options,clickedAnswer, id='question_answer', inline=False)            
        
        #verify if question is an image an load it as an image   
        if(isinstance(new_question, str)): 
            if('assets' in new_question.split('/')): 
                question_item = html.Img(src=new_question,style={'height':'auto', 'width':'auto'})                  
            else:
                question_item = dcc.Markdown(children=new_question, id='question',style={'font-size':18,'margin-right': 40})  
        else:
            question_item = dcc.Markdown(children=new_question, id='question',style={'font-size':18,'margin-right': 40})
           
        total_children = [question_item,r_item]
        
        children=[
                    dcc.Markdown(children='Question 1', id='card_title', className="card-title", style={'margin-bottom': 10,'font-size':22}),
                    #question
                    html.Div(children=total_children,id='radio_item_div',style={'width': '100%','margin-top': 10,'margin-bottom': 30,'margin-left': 20,'margin-right': 20,'color':'#323232'} ,className='row'),
                    html.Div(children=[dbc.Row([
                    #back button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 1,"offset": 3}),
                    #next button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'inline-block','backgroundColor': 'lightgreen','font-size':16})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 3}),
                    #skip button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'inline-block','backgroundColor': '#FF7377','font-size':16})
                    ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                    ])]),
                    html.Div(children=[dbc.Row([
                    #button new test
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                    ],className="me-1"),                    
                    width={"offset": 3}),
                    #exit button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                    ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                    ])]),
                    #begin evaluation test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                    ),
                    #name
                    html.Div(children=[
                        dbc.Label("Name", size="md"),
                        dbc.Input(id="name", type="text")
                    ],style={'display':'none'}),
                    #surname
                    html.Div(children=[
                        dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #university
                    html.Div(children=[
                        dbc.Label("University:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #email
                    html.Div(children=[
                        dbc.Label("Email:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='email',style={'margin-bottom': 50})
                    ],style={'display':'none'}),
                    #start test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                    #saving data
                    html.Div(children=[
                        dcc.Markdown(children='**Saving data...**')
                    ],id='loading_page', style={'display':'none'}, className='row'),
                    #submit button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                ] 
        #save user data
        df_output[0]= name
        df_output[1] = surname
        df_output[2] = university
        df_output[3] = email
        
        print('==============================================================================')
        print('User data reloaded:')
        global q_data, student_data, s_hist
        #[q_data, student_data] = qrec.load_data(data_fn)
        [q_data, student_data] = qrec.load_data_from_gs(sh)
        # Calculate history of all students from loaded data
        s_hist = qrec.calc_history(student_data, q_data)
            
        
        j=j+1
    
    elif(start_button!=None and ((nextbutton!=None and 0<j<n_qstion) or (skipbutton!=None and 0<j<n_qstion) or (previuosbutton!=None and 2<j<=n_qstion+1))):#if clicked go to next question
        print('phase 3')
        if(ctx.triggered_id=='skipButton'):
            clickedAnswer = None
            title='Question '+str(j+1)
            backward=0
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3)
            next_question=index_previous_question
            j=j+1
        if(ctx.triggered_id=='nextButton'):
            if(clickedAnswer==0):
                clickedAnswer = None
            backward=0
            title='Question '+str(j+1)
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3)
            next_question=index_previous_question
            print('clickedAnswer:{}'.format(clickedAnswer))
            j=j+1                      
        if(ctx.triggered_id=='previuosButton'):
            print('qui!')
            if(clickedAnswer==0):
                clickedAnswer = None
            title='Question '+str(j-1)
            backward=1
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3)
            next_question=index_previous_question
            j=j-1   
        print('clickedAnswer:{}'.format(clickedAnswer))
        
        #LOAD IMAGES
        size='auto'
        #verify if options are images an load them as images
        if(isinstance(new_options[0], str)) and ('assets' in new_options[0].split('/')):         
            options0 = {"label": html.Img(src=new_options[0],style={'height':size, 'width':size}),"value": "0"}
        else:
            options0 = {"label": html.Div(new_options[0], style={'display':'inline-block','font-size':18}),"value": "0"}
            
        if(isinstance(new_options[1], str)) and ('assets' in new_options[1].split('/')):         
            options1 = {"label": html.Img(src=new_options[1],style={'height':size, 'width':size}),"value": "1"}
        else:
            options1 = {"label": html.Div(new_options[1], style={'display':'inline-block','font-size':18}),"value": "1"}
            
        if(isinstance(new_options[2], str)) and ('assets' in new_options[2].split('/')):         
            options2 = {"label": html.Img(src=new_options[2],style={'height':size, 'width':size}),"value": "2"}
        else:
            options2 = {"label": html.Div(new_options[2], style={'display':'inline-block','font-size':18}),"value": "2"}              
        
        if(isinstance(new_options[3], str)) and ('assets' in new_options[3].split('/')):         
            options3 = {"label": html.Img(src=new_options[3],style={'height':size, 'width':size}),"value": "3"}
        else:
            options3 = {"label": html.Div(new_options[3], style={'display':'inline-block','font-size':18}),"value": "3"}
        
        options4 = {"label": html.Div(new_options[4], style={'display':'inline-block','font-size':18}),"value": "4"}
        
        options = [options0,options1,options2,options3,options4]
        r_item = dcc.RadioItems(options,clickedAnswer, id='question_answer', inline=False) 
        
        #verify if question is an image an load it as an image   
        if(isinstance(new_question, str)): 
            if('assets' in new_question.split('/')): 
                question_item = html.Img(src=new_question,style={'height':'auto', 'width':'auto'}) #60%                 
            else:
                question_item = dcc.Markdown(children=new_question, id='question',style={'font-size':18,'margin-right': 40})  
        else:
            question_item = dcc.Markdown(children=new_question, id='question',style={'font-size':18,'margin-right': 40})
           
        total_children = [question_item,r_item]
        
        children=[
                    dcc.Markdown(children=title, id='card_title', className="card-title", style={'margin-bottom': 10,'font-size':22}),
                    #question
                    html.Div(children=total_children,
                    id='radio_item_div',style={'width': '100%','margin-top': 10,'margin-bottom': 30,'margin-left': 20,'margin-right': 20,'color':'#323232'} ,className='row'),
                    html.Div(children=[dbc.Row([
                    #back button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'inline-block','backgroundColor': '#99cfe0','font-size':16})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 1,"offset": 3}),
                    #next button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'inline-block','backgroundColor': 'lightgreen','font-size':16})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 3}),
                    #skip button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'inline-block','backgroundColor': '#FF7377','font-size':16})
                    ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                    ])]),
                    html.Div(children=[dbc.Row([
                    #button new test
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                    ],className="me-1"),                    
                    width={"offset": 3}),
                    #exit button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                    ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                    ])]),
                    #begin evaluation test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                    ),
                    #name
                    html.Div(children=[
                        dbc.Label("Name", size="md"),
                        dbc.Input(id="name", type="text")
                    ],style={'display':'none'}),
                    #surname
                    html.Div(children=[
                        dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #university
                    html.Div(children=[
                        dbc.Label("University:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #email
                    html.Div(children=[
                        dbc.Label("Email:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                    ],style={'display':'none'}),
                    #start test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                    #saving data
                    html.Div(children=[
                        dcc.Markdown(children='**Saving data...**')
                    ],id='loading_page', style={'display':'none'}, className='row'),
                    #submit button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                ]         
 
    elif(start_button!=None and (count_eq==41) and ((nextbutton!=None and j==n_qstion) or (skipbutton!=None and j==n_qstion))): #submit button
        if(ctx.triggered_id=='skipButton'):
            clickedAnswer = None
        if(ctx.triggered_id=='nextButton' and clickedAnswer==0):
            clickedAnswer = None
        backward=0
        title=0
        list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3)
        next_question=n_qstion+1
        children=[
                    dcc.Markdown(children=title, id='card_title', className="card-title", style={'display':'none'}),
                    #question
                    html.Div(children=[
                        dcc.Markdown(children=new_question, id='question',style={'font-size':18,'margin-right': 40}),
                        dcc.RadioItems([{"label": new_options[0],"value": "0"},{"label": new_options[1],"value": "1"},{"label": new_options[2],"value": "2"},{"label": new_options[3],"value": "3"},{"label": new_options[4],"value": "4"}],id='question_answer', inline=False, style={'font-size':18}),
                    ],id='radio_item_div',style={'display':'none'} ,className='row'),
                    html.Div(children=[dbc.Row([
                    #back button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 1,"offset": 3}),
                    #next button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 3}),
                    #skip button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                    ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                    ])]),
                    html.Div(children=[dbc.Row([
                    #button new test
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                    ],className="me-1"),                    
                    width={"offset": 3}),
                    #exit button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                    ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                    ])]),
                    #begin evaluation test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                    ),
                    #name
                    html.Div(children=[
                        dbc.Label("Name", size="md"),
                        dbc.Input(id="name", type="text")
                    ],style={'display':'none'}),
                    #surname
                    html.Div(children=[
                        dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #university
                    html.Div(children=[
                        dbc.Label("University:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #email
                    html.Div(children=[
                        dbc.Label("Email:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                    ],style={'display':'none'}),
                    #start test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                    #submit button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success",id='submitButton',style={'backgroundColor': 'lightgreen','font-size':16})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                    #saving 
                    html.Div(children=[
                        dcc.Markdown(children='**Saving data...**')
                    ],id='loading_page', style={'display':'none'}, className='row'),
                ]
        clickedAnswer = None
        
    elif(start_button!=None and (submitbutton!=None) and (None in list_answers)): #if you don't answer all questions  
        #print('Some value is None')
        next_question=n_qstion+2
        children=[
                    dcc.Markdown(children=0, id='card_title', className="card-title", style={'display':'none'}),
                    #question
                    html.Div(children=[
                    dcc.Markdown(children=df.iloc[0,0], id='question',style={'font-size':18,'margin-right': 40}),
                    dcc.RadioItems([{"label": df.iloc[0,1],"value": "0"},{"label": df.iloc[0,2],"value": "1"},{"label": df.iloc[0,3],"value": "2"},{"label": df.iloc[0,4],"value": "3"},{"label": df.iloc[0,5],"value": "4"}],id='question_answer', inline=False, style={'font-size':18}),
                    ],id='radio_item_div',style={'display':'none'} ,className='row'),
                    html.Div(children=[dbc.Row([
                    #back button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'inline-block','backgroundColor': '#99cfe0','font-size':16})
                    ],style={'margin-bottom': 10, 'margin-top': 30},className="me-1"),
                    width={"size": 2, "order": 1,"offset": 5}),
                    #next button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 2, "order": 3}),
                    #skip button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                    ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                    ])]),
                    html.Div(children=[dbc.Row([
                    #button new test
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                    ],className="me-1"),                    
                    width={"offset": 3}),
                    #exit button
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                    ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                    ])]),
                    #begin evaluation test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1,style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                    ),
                    #name
                    html.Div(children=[
                        dbc.Label("Name", size="md"),
                        dbc.Input(id="name", type="text")
                    ],style={'display':'none'}),
                    #surname
                    html.Div(children=[
                        dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #university
                    html.Div(children=[
                        dbc.Label("University:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                    ],style={'display':'none'}),
                    #email
                    html.Div(children=[
                        dbc.Label("Email:", size="md", style={'margin-right': 20}),
                        dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                    ],style={'display':'none'}),
                    #start test button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                    #saving 
                    html.Div(children=[
                        dcc.Markdown(children='**Saving data...**')
                    ],id='loading_page', style={'display':'none'}, className='row'),
                    #error message
                    html.Div(children=[
                        dcc.Markdown(children=skip_message)
                    ],id='error_message', style={'color':'red','margin-top': 10,'margin-right': 20,'margin-left': 20,'width': '90%','font-size':15}, className='row'),
                    #submit button
                    dbc.Row(
                    dbc.Col(
                    html.Div(children=[ 
                        dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                    ],style={'margin-bottom': 10},className="me-1"),
                    width={"size": 3, "offset": 5}),    
                    ),
                ]
        
        j=j+1

    elif(start_button!=None and (submitbutton!=None) and (None not in list_answers)):#ha risposto a tutte le domande
        count=0 
        next_question=n_qstion+3
        for i in range(0,n_qstion):
            if(str(df.iloc[list_index_question[i],6])==df_output[(i)*2+user_input+evaluation_qstion+1]):
                count=count+1 
        children=[
                dcc.Markdown(children='Question 1', id='card_title', className="card-title", style={'display':'none'}),
                #question
                html.Div(children=[
                    dcc.Markdown(children=df.iloc[0,0], id='question',style={'font-size':18,'margin-right': 40}),
                    dcc.RadioItems([{"label": df.iloc[0,1],"value": "0"},{"label": df.iloc[0,2],"value": "1"},{"label": df.iloc[0,3],"value": "2"},{"label": df.iloc[0,4],"value": "3"},{"label": df.iloc[0,5],"value": "4"}],id='question_answer', inline=False, style={'font-size':18}),
                ],id='radio_item_div',style={'display':'none'} ,className='row'),
                #test finished
                html.Div(children=[
                dcc.Markdown(children='Test completed!  \nYour score is {}/{}'.format(count,n_qstion))
                ],style={'margin-top': 10,'margin-bottom': 30, 'textAlign': 'center','margin-left': 0,'color': 'black', 'font-size':20}, className='row'),
                html.Div(children=[dbc.Row([
                #back button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 1,"offset": 3}),
                #next button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 3}),
                #skip button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                ])]),
                html.Div(children=[dbc.Row([
                #button new test
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                ],className="me-1"),                    
                width={"offset": 3}),
                #exit button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                ])]),
                #begin evaluation test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1,style={'display':'inline-block','margin-bottom': 10,'backgroundColor': 'lightgreen','font-size':16})
                ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 4})
                ),
                #name
                html.Div(children=[
                    dbc.Label("Name", size="md"),
                    dbc.Input(id="name", type="text")
                ],style={'display':'none'}),
                #surname
                html.Div(children=[
                    dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #university
                html.Div(children=[
                    dbc.Label("University:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #email
                html.Div(children=[
                    dbc.Label("Email:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                ],style={'display':'none'}),
                #start test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #submit button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #saving data
                html.Div(children=[
                    dcc.Markdown(children='**Saving data...**')
                ],id='loading_page', style={'display':'none'}, className='row'),
              ] 
        
    elif((evalButton>1) and (nextbutton==None)):#first eval question
        print('1 domanda eval test!')
        clickedAnswer=None #df.iloc[count_eq,1]
        children=[
                dcc.Markdown(children='Evaluation Question 1', id='card_title', className="card-title", style={'margin-bottom': 10,'font-size':22}),
                #question
                html.Div(children=[
                    dcc.Markdown(children=df.iloc[41,0], id='question',style={'font-size':18,'margin-right': 40}),
                    dcc.RadioItems(options=[df.iloc[41,1],df.iloc[41,2],df.iloc[41,3],df.iloc[41,4],df.iloc[41,5]],id='question_answer', inline=False, style={'font-size':18}),
                ],id='radio_item_div',style={'width': '100%','margin-top': 10,'margin-bottom': 30,'margin-left': 20,'margin-right': 20,'color':'#323232'} ,className='row'),
                #test finished
                html.Div(children=[
                dcc.Markdown(children='Test completed!  \nYour score is {}/{}'.format(0,n_qstion))
                ],style={'display':'none'}, className='row'),
                html.Div(children=[dbc.Row([
                #back button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 1,"offset": 3}),
                #next button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'inline-block','backgroundColor': 'lightgreen','font-size':16})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 3}),
                #skip button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                ])]),
                html.Div(children=[dbc.Row([
                #button new test
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                ],className="me-1"),                    
                width={"offset": 3}),
                #exit button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                ])]),
                #begin evaluation test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                ),
                #name
                html.Div(children=[
                    dbc.Label("Name", size="md"),
                    dbc.Input(id="name", type="text")
                ],style={'display':'none'}),
                #surname
                html.Div(children=[
                    dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #university
                html.Div(children=[
                    dbc.Label("University:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #email
                html.Div(children=[
                    dbc.Label("Email:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                ],style={'display':'none'}),
                #start test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #submit button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #saving data
                html.Div(children=[
                    dcc.Markdown(children='**Saving data...**')
                ],id='loading_page', style={'display':'none'}, className='row'),
              ] 
        count_eq=count_eq+1
    elif(nextbutton!=None and count_eq!=45):#2-4 eval qstion
        print('2 domanda eval test!')
        #save survey data
        df_output[count_eq-42+user_input]=str(optionList.index(clickedAnswer))#insert index not value
        clickedAnswer=None #df.iloc[count_eq,1]
        children=[
                dcc.Markdown(children='Evaluation Question {}'.format(count_eq-40), id='card_title', className="card-title", style={'margin-bottom': 10,'font-size':22}),
                #question
                html.Div(children=[
                    dcc.Markdown(children=df.iloc[count_eq,0], id='question',style={'font-size':18,'margin-right': 40}),
                    dcc.RadioItems(options=[df.iloc[count_eq,1],df.iloc[count_eq,2],df.iloc[count_eq,3],df.iloc[count_eq,4],df.iloc[count_eq,5]],id='question_answer', inline=False, style={'font-size':18}),
                ],id='radio_item_div',style={'width': '100%','margin-top': 10,'margin-bottom': 30,'margin-left': 20,'margin-right': 20,'color':'#323232'} ,className='row'),
                #test finished
                html.Div(children=[
                dcc.Markdown(children='Test completed!  \nYour score is {}/{}'.format(0,n_qstion))
                ],style={'display':'none'}, className='row'),
                html.Div(children=[dbc.Row([
                #back button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 1,"offset": 3}),
                #next button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'inline-block','backgroundColor': 'lightgreen','font-size':16})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 3}),
                #skip button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                ])]),
                html.Div(children=[dbc.Row([
                #button new test
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'none'})
                ],className="me-1"),                    
                width={"offset": 3}),
                #exit button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'none'})
                ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                ])]),
                #begin evaluation test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                ),
                #name
                html.Div(children=[
                    dbc.Label("Name", size="md"),
                    dbc.Input(id="name", type="text")
                ],style={'display':'none'}),
                #surname
                html.Div(children=[
                    dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #university
                html.Div(children=[
                    dbc.Label("University:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #email
                html.Div(children=[
                    dbc.Label("Email:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                ],style={'display':'none'}),
                #start test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #submit button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #saving data
                html.Div(children=[
                    dcc.Markdown(children='**Saving data...**')
                ],id='loading_page', style={'display':'none'}, className='row'),
              ] 
        count_eq=count_eq+1
        
    elif(nextbutton!=None and count_eq==45 and trigger==1):#survey finished   
        #save survey data
        df_output[count_eq-42+user_input]=str(optionList.index(clickedAnswer))#insert index not value
        children=[
                dcc.Markdown(children='Question 1', id='card_title', className="card-title", style={'display':'none'}),
                #question
                html.Div(children=[
                    dcc.Markdown(children=df.iloc[0,0], id='question',style={'font-size':18,'margin-right': 40}),
                    dcc.RadioItems([{"label": df.iloc[0,1],"value": "0"},{"label": df.iloc[0,2],"value": "1"},{"label": df.iloc[0,3],"value": "2"},{"label": df.iloc[0,4],"value": "3"},{"label": df.iloc[0,5],"value": "4"}],id='question_answer', inline=False, style={'font-size':18}),
                ],id='radio_item_div',style={'display':'none'} ,className='row'),
                #test finished
                html.Div(children=[
                dcc.Markdown(children='Satisfaction survey completed and test data successfully saved!')
                ],style={'margin-top': 10,'margin-bottom': 30, 'textAlign': 'center','margin-left': 0,'color': 'black', 'font-size':20}, className='row'),
                html.Div(children=[dbc.Row([
                #back button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("BACK", size="lg", outline=True, color="primary", id='previuosButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 1,"offset": 3}),
                #next button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("CONFIRM", size="lg", outline=True, color="success", id='nextButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 2, "order": 3}),
                #skip button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SKIP", size="lg", outline=True, color="danger", id='skipButton',style={'display':'none'})
                ],style={'margin-right': 40,'margin-bottom': 10},className="me-1"),width={"size": 3, "order": 4})
                ])]),
                html.Div(children=[dbc.Row([
                #button new test
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("TAKE A NEW TEST", size="lg", outline=True, color="success", id='testButton',style={'display':'inline-block','margin-bottom': 10,'backgroundColor': 'lightgreen','font-size':16})
                ],className="me-1"),                    
                width={"offset": 3}),
                #exit button
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("EXIT", size="lg", outline=True, color="success", id='exitButton',style={'display':'inline-block','backgroundColor': 'lightgreen','font-size':16})
                ],style={'margin-right': 30,'margin-bottom': 10},className="me-1"),)
                ])]),
                #begin evaluation test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START SATISFACTION SURVEY", size="lg", outline=True, color="success", id='evalButton',n_clicks=1, style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),width={"size": 3, "offset": 5})
                ),
                #name
                html.Div(children=[
                    dbc.Label("Name", size="md"),
                    dbc.Input(id="name", type="text")
                ],style={'display':'none'}),
                #surname
                html.Div(children=[
                    dbc.Label("Surname:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='surname',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #university
                html.Div(children=[
                    dbc.Label("University:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='university',style={'margin-bottom': 20})
                ],style={'display':'none'}),
                #email
                html.Div(children=[
                    dbc.Label("Email:", size="md", style={'margin-right': 20}),
                    dbc.Input(type="text", id='email',style={'margin-bottom': 40})
                ],style={'display':'none'}),
                #start test button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("START TEST", size="lg", outline=True, color="success", n_clicks=1, id='start_button',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #submit button
                dbc.Row(
                dbc.Col(
                html.Div(children=[ 
                    dbc.Button("SUBMIT TEST", size="lg", outline=True, color="success", id='submitButton',style={'display':'none'})
                ],style={'margin-bottom': 10},className="me-1"),
                width={"size": 3, "offset": 5}),    
                ),
                #saving data
                html.Div(children=[
                    dcc.Markdown(children='**Saving data...**')
                ],id='loading_page', style={'display':'none'}, className='row'),
              ]  
        
        #saving data on google sheet
        save_on_google_sheets(df_output,df_output_2,df_output_3)
        #re-initialize  
        df_output=['-1'] * (n_qstion*2+user_input+evaluation_qstion)
        print(df_output)
        
        seed=seed+1
        list_index_question=[]
        j=0
        list_answers = [None,None,None,None,None]
        backward=0
        clickedAnswer=None
        count_eq=41
    return [children,j,index_previous_question,start,name,surname,university,email,seed,list_index_question,backward,list_answers,next_question,clickedAnswer,count_eq,df_output]
    
#update next button
@callback(Output('testButton', 'n_clicks'), 
          Input('testButton','n_clicks'))
def reset_backbutton(n):
    if(n!=None):
        return None

#trigger disabling submit button and showing loading sentence
@app.callback([Output('trigger','children'),Output('loading_page','style'), Output('nextButton', 'disabled')],
              [Input('nextButton', 'n_clicks'),
               State('count_eq ','data'),])
def trigger_function(nextButton,count_eq):
    trigger = 0
    button=False
    loading_page_style={'display':'none'}
    if (nextButton!=None and (count_eq==45)):
        print('Saving') 
        trigger = 1
        button=True
        loading_page_style={'textAlign': 'center','margin-top': 10,'margin-bottom': 20,'margin-left': 200,'width': '50%','font-size':18}
        
    return [trigger,loading_page_style,button]

# collect button time-stamps -> ms since 1970
@app.callback([Output('count_df_2','data'),Output('trigger_2','children'),Output('df_output_2','data')],
              [Input('start_button', 'n_clicks_timestamp'),
                Input('previuosButton', 'n_clicks_timestamp'),
                Input('nextButton', 'n_clicks_timestamp'),
                Input('skipButton', 'n_clicks_timestamp'),
                Input('submitButton', 'n_clicks_timestamp'),
                Input('testButton', 'n_clicks_timestamp'),
                Input('exitButton', 'n_clicks_timestamp'),
                Input('evalButton', 'n_clicks_timestamp'),
                State('count_df_2','data'),
                State('list_index_question','data'),
                State('next_question','data'),
                State('counter','data'),
                State('list_answers','data'),
                State('count_eq ','data'),
                State('df_output_2','data'),
              ])

def time_stamps_function(startbutton,previuosbutton,nextbutton,skipbutton,submitbutton,testbutton,exitbutton,evalbutton,count_df_2,list_index_question,next_question,j,list_answers,count_eq,df_output):
    trigger_2=0
    values = []
    if(startbutton!=None and ctx.triggered_id=='start_button'):
        print('start button press') 
        values.append(0)
        #values.append(next_question)
        values.append('start_button')
        values.append(startbutton)
        #print('count_df_2:{}'.format(count_df_2))
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) 
        df_output.append(values_string)
    
    if(startbutton!=None and ctx.triggered_id=='start_button' and not df_output):
        df_output = []
        
    if(previuosbutton!=None and ctx.triggered_id=='previuosButton'):
        print('previous button press') 
        values.append(j)
        #values.append(next_question)
        values.append('previuosButton')
        values.append(previuosbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) #+'#'+ str(values[3])
        df_output.append(values_string)
        
    if(nextbutton!=None and ctx.triggered_id=='nextButton'):
        print('next button press')
        # print('time:{}'.format(nextbutton)) 
        if(count_eq==41):
            values.append(j)
        else:
            values.append('e'+str(count_eq-41))
        #values.append(next_question)
        values.append('nextButton')
        values.append(nextbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        #print('count_df_2:{}'.format(count_df_2))
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) #+'#'+ str(values[3])
        df_output.append(values_string)
        if(count_eq==45):#re-inizialize
            count_df_2=0
        
    if(skipbutton!=None and ctx.triggered_id=='skipButton'):
        print('skip button press') 
        values.append(j)
        #values.append(next_question)
        values.append('skipButton')
        values.append(skipbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) #+'#'+ str(values[3])
        df_output.append(values_string)
        
    if(submitbutton!=None and ctx.triggered_id=='submitButton'):
        print('submit button press')  
        values.append(n_qstion+1)
        values.append('submitButton')
        values.append(submitbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        df_output.append(values_string)
            
    if(testbutton!=None and ctx.triggered_id=='testButton'):
        print('test button press') 
        df_output = []
        values.append(0)
        #values.append(next_question)
        values.append('start_button')
        values.append(testbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        df_output.append(values_string)        
        
    if(exitbutton!=None and ctx.triggered_id=='exitButton'):
        print('exit button press') 
        df_output = []
        values.append(-2)
        values.append('login')
        values.append(exitbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        df_output.append(values_string)
        
    if(evalbutton!=None and ctx.triggered_id=='evalButton'):
        print('eval button press') 
        values.append(n_qstion+2)
        values.append('evalbutton')
        values.append(evalbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        df_output.append(values_string)
              
    return [count_df_2,trigger_2,df_output]   

#click on answers
@app.callback([Output('count_df','data'),Output('df_output_3','data')],
              [Input('question_answer', 'value'),
                State('counter','data'),
                State('count_df','data'),
                State('list_answers','data'),
                State('count_eq ','data'),
                State('df_output_3','data'),
                Input('testButton', 'n_clicks'),
                Input('exitButton', 'n_clicks'),
                Input('start_button', 'n_clicks'),
              ])
def answer_clicks_function(clickedAnswer,j,count_df,list_answers,count_eq,df_output,testbutton,exitbutton,start_button):
    if(ctx.triggered_id=='question_answer'):
        values = []
        if(clickedAnswer!=0 and clickedAnswer!=None):
            ms = int(time.time()*1000) 
            if(count_eq==41):
                values.append(j)
            else:
                values.append('e'+str(count_eq-41))
            values.append('answer')
            values.append(ms)
            print('saving timestamp')
            values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
            count_df=count_df+1
            df_output.append(values_string) 
        if(j==0 and (list_answers==[None,None,None,None,None])):#re-inizialize count
            print('riaggiorno count!')
            count_df=0
    else:
        df_output = []
    return [count_df,df_output]
    
    
app.debug=False
if __name__ == '__main__':   
    #waitress.serve(application, host='127.0.0.1', port=8050)
    app.run_server(debug=True,host='0.0.0.0', port=8050,use_reloader=False)
    #app.run_server(debug=True,host='127.0.0.1', port=8050,use_reloader=False)
# %%
