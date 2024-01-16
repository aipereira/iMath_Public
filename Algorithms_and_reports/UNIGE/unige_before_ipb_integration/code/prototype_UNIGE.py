# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 17:59:59 2022

@author: giuli
"""

#import
from dash import Dash, dcc, html, Input, Output, State, callback, ctx
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import waitress
import random
import pygsheets
import time
import pickle as pk

#-------APP---------
app = Dash(__name__, suppress_callback_exceptions=True,external_stylesheets = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css'])
application = app.server

#GLOBAL VARIABLES
image_path ='assets/imath_logo.png'

#input data
gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('Imath_prototype_data')
#file where list of the 40 possible questions are saved
wks = sh[0]
df=wks.get_as_df(has_header=True,include_tailing_empty=False)
#file where data of new tests are saved
wks = sh[9]
past_history = wks.get_as_df(has_header=True,include_tailing_empty=False)
#print(past_history.shape)
past_history = past_history.iloc[:,0:-1]#remove random number saved

user_input=4#data from the login phase
evaluation_qstion=4 #final satisfaction questions
n_qstion=5#number of questions per test
init_data=['-1'] * (n_qstion*2+user_input+evaluation_qstion)#initialize data with -1
skip_message ='**You have unanswered questions, please go back and answer all the question before submitting the test**!'

#upload average values computed on training data
sh = gc.open('averages')
wks = sh[1]
avg  = wks.get_as_df(has_header=False, include_tailing_empty=False) 
wks = sh[0]
user_tests_df = wks.get_as_df(has_header=False, include_tailing_empty=False) 
user_tests_df = user_tests_df.iloc[1:,:]#percentage of correct answers for each student
#upload user features (data from informative questionnaire) already processed
with open('user_features_df.pickle', 'rb') as f:
   user_features_df = pk.load(f)

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
                        #processing new question question message
                        html.Div(children=[
                            dcc.Markdown(children='**Processing question...**')
                        ],id='processing_message', style={'display':'none'}, className='row'),
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
    
    html.Div(children=[dcc.Markdown(children='**v.U.**', style={'margin-left': 885})]),
    
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
    
    html.Div(children=[dcc.Store(id="ifMaxCorrectQst_past", data=0)]),
    
    html.Div(children=[dcc.Store(id="df_output_1", data=init_data)]), 
    
    html.Div(children=[dcc.Store(id="df_output_2", data=[])]), 
    
    html.Div(children=[dcc.Store(id="df_output_3", data=[])]),
    
    html.Div(id='trigger',children=0, style=dict(display='none')),
    
    html.Div(id='trigger_2',children=0, style=dict(display='none')),
    
    html.Div(id='trigger_3',children=0, style=dict(display='none')),
    
    html.Div(children=[
      dcc.Markdown(children='Test completed!', id='end')
    ],style={'margin-top': 10,'margin-bottom': 30, 'textAlign': 'center','margin-left': 0,'color': 'black', 'font-size':26, 'display':'none'}, className='row'),
    
])

#FUNCTIONs NECESSARY TO CALCULATE NEW QUESTION
#compute time spent on a question (nb_qstion is number between 1 and 5)
def time_question(data_session,nb_qstion): #input: timestamps vector, number of questions(1-5)
    times=[]
    print(data_session)
    for i in range(0,len(data_session)):
        if(data_session.iloc[i].split('#')[0]==str(nb_qstion) and data_session.iloc[i].split('#')[1] != 'answer'):
            end_time= int(data_session.iloc[i].split('#')[2])
            for j in range(i-1,-1,-1):#go back till the begin
                if(data_session.iloc[j].split('#')[1] != 'answer'):
                    start_time= int(data_session.iloc[j].split('#')[2])
                    times.append(end_time-start_time)
                    break      
        if(data_session.iloc[i].split('#')[0]=='6' and data_session.iloc[i].split('#')[1]=='previousButton' and nb_qstion==5):
            end_time= int(data_session.iloc[i].split('#')[2])
            for j in range(i-1,-1,-1):#go back till the begin
                if(data_session.iloc[j].split('#')[1] != 'answer'):
                    start_time=int(data_session.iloc[j].split('#')[2])
                    times.append(end_time-start_time)
                    break
    time_ms=sum(times)
    time=time_ms/1000
    return time

#compute number of clicks on a question's answers (nb_qstion is number between 1 and 5)
def nclick_question(data_session,nb_qstion):#input: timestamps vector, number of questions(1-5)
    count=0
    for i in range(0,len(data_session)):
        if(data_session.iloc[i].split('#')[0]==str(nb_qstion) and data_session.iloc[i].split('#')[1] == 'answer'):
            count=count+1
    if(count == 0):
         count=1       
    return count

#the question has been correctly answered?
def correct_answer(df_output,nb_qstion):#input: complete row vector, number of questions(1-5)
    quest_answ = df_output.iloc[8:18]
    question_index = quest_answ.iloc[(nb_qstion*2)-2]
    if(quest_answ.iloc[(nb_qstion*2)-1] == str(df.iloc[question_index,6])):#anche se è None va bene tanto sarà false e domanda sbagliata
        return 1
    else:
        return 0 
    
#the question has been skipped?
def skipped_question(data_session,nb_qstion):#input: complete row vector, number of questions(1-5)
    count=0
    for i in range(0,len(data_session)):
        if(data_session.iloc[i].split('#')[0]==str(nb_qstion) and data_session.iloc[i].split('#')[1] == 'skipButton'):
            count=count+1
    return count

#STATISTICS
#Compute averaged values   
#compute the percentage of times the users have correctly answered the question, difficulty of the question on a statistics basis
def perc_correct_answers(question_index,avg):#input: index of the question(0-39), vector of questions and answers
    arr = np.zeros((1,2))
    if(avg.iloc[question_index,1] != 0): #se è stata già rx
        perc_correct = avg.iloc[question_index,0]
        qstion_done = 1
    else:#se non è stata ancora risposta 
        qstion_done = 0 
        df_search = avg.loc[avg.iloc[:,1]!=0] 
        if(df_search.empty):#sono tutte a 0
            perc_correct = 0
        else:
            perc_correct = round(df_search.mean()[0])
    # print(perc_correct)
    arr[0,0] = perc_correct 
    arr[0,1] = qstion_done
    return arr

#update the dataframe containing the average values with the new data from the current test
def update_perc_cr_answers(current_row,avg):#current test row(pd.Series), dataframe with the averaged values
    qst_answ = current_row.iloc[8:18]
    #print(qst_answ)
    for column in range(0,10,2):
        index = qst_answ.iloc[column] 
        value = correct_answer(current_row,(column//2)+1)
        updated_value = round((avg.iloc[index,0]/100)*avg.iloc[index,1]) + value
        avg.iloc[index,0] = round((updated_value/(avg.iloc[index,1]+1))*100)
        avg.iloc[index,1] = avg.iloc[index,1]+1
    return avg

#average time spent on a question
def average_time(avg,question_index):#input: dataframe containing all the input data, index of the question(0-39)
    arr = np.zeros((1,2))
    if(avg.iloc[question_index,3] != 0): 
        time = avg.iloc[question_index,2]
        qstion_done = 1
    else:
        qstion_done = 0
        df_search = avg.loc[avg.iloc[:,3]!=0] 
        if(df_search.empty):
            time = 0
        else:
            time = round(df_search.mean()[2],1)
    # print(perc_correct)
    arr[0,0] = time
    arr[0,1] = qstion_done
    return arr

#update the dataframe containing the average values with the new data from the current test
def update_average_time(current_row,avg):#current test row(pd.Series), dataframe with the averaged values
    qst_answ = current_row.iloc[8:18]
    timestamp = current_row.iloc[18:]
    for column in range(0,10,2):
        index = qst_answ.iloc[column] 
        value = time_question(timestamp,(column//2)+1)
        updated_value = (avg.iloc[index,2]*avg.iloc[index,3]) + value
        avg.iloc[index,2] = round((updated_value/(avg.iloc[index,3]+1)),1)
        avg.iloc[index,3] = avg.iloc[index,3]+1
    return avg
        
#average click on an answer
def average_click_answer(avg,question_index):#input: dataframe containing all the input data, index of the question(0-39)
    arr = np.zeros((1,2))
    if(avg.iloc[question_index,5] != 0): 
        click_answer = avg.iloc[question_index,4]
        qstion_done = 1
    else:
        qstion_done = 0
        df_search = avg.loc[avg.iloc[:,5]!=0] 
        if(df_search.empty):
            click_answer = 0
        else:
            click_answer = round(df_search.mean()[4],1)
    arr[0,0] = click_answer 
    arr[0,1] = qstion_done
    return arr

#update the dataframe containing the average values with the new data from the current test
def update_average_click_answer(current_row,avg):#current test row(pd.Series), dataframe with the averaged values
    qst_answ = current_row.iloc[8:18]
    timestamp = current_row.iloc[18:]
    for column in range(0,10,2):
        index = qst_answ.iloc[column]
        value = nclick_question(timestamp,(column//2)+1)
        updated_value = (avg.iloc[index,4]*avg.iloc[index,5]) + value
        avg.iloc[index,4] = round((updated_value/(avg.iloc[index,5]+1)),2)
        avg.iloc[index,5] = avg.iloc[index,5]+1
    return avg
        
#average number of skips on a answer
def average_skip_answer(avg,question_index):#input: dataframe containing all the input data, index of the question(0-39)
    arr = np.zeros((1,2))
    if(avg.iloc[question_index,7] != 0): 
        skip_answer = avg.iloc[question_index,6]
        qstion_done = 1
    else:
        qstion_done = 0
        df_search = avg.loc[avg.iloc[:,7]!=0] 
        if(df_search.empty):
            skip_answer = 0
        else:
            skip_answer = round(df_search.mean()[6])
    arr[0,0] = skip_answer 
    arr[0,1] = qstion_done
    return arr

#update the dataframe containing the average values with the new data from the current test
def update_average_skip_answer(current_row,avg):#current test row(pd.Series), dataframe with the averaged values
    qst_answ = current_row.iloc[8:18]
    timestamp = current_row.iloc[18:]
    for column in range(0,10,2):
        index = qst_answ.iloc[column] 
        value = skipped_question(timestamp,(column//2)+1)
        updated_value = round((avg.iloc[index,6]/100)*avg.iloc[index,7]) + value
        avg.iloc[index,6] = round((updated_value/(avg.iloc[index,7]+1))*100)
        avg.iloc[index,7] = avg.iloc[index,7]+1
    return avg
        
#compute number of correct answers for a specific user + bol value if it is his/her test or not
def correct_answer_user(mail,user_tests_df):
    arr = np.zeros((1,2))
    if(user_tests_df.empty == False):
        df_search = user_tests_df.loc[user_tests_df.iloc[:,0]==mail] 
        if(df_search.empty):#if first test
           qstion_done = 0 
           user_correct = round(user_tests_df.iloc[:,1:3].mean()[1])#insert average
        else:
           qstion_done = 1
           user_correct = df_search.iloc[0,1]
    else:
        user_correct = 0
        qstion_done = 0
    arr[0,0] = user_correct 
    arr[0,1] = qstion_done
    return arr

#compute the total number of correct answer of the current test
def correct_answer_cr_row(current_row):#current test row(pd.Series)
    cr_asw = 0
    for column in range(0,10,2):
        value = correct_answer(current_row,(column//2)+1)
        cr_asw = cr_asw + value
    return cr_asw

#update the dataframe containing the average values with the new data from the current test
def update_correct_answer_user(current_row,wks,user_tests_df):#current test row(pd.Series), GoogleSheet where saving the data, dataframe with the averaged values
    mail = current_row.iloc[3]
    if(user_tests_df.empty == False):
        count=0
        for i in range(0,user_tests_df.shape[0]):
            if(user_tests_df.iloc[i,0] == mail):
                count=1
                old_value = round((user_tests_df.iloc[i,1]/100)*user_tests_df.iloc[i,2])
                user_tests_df.iloc[i,1] = round(((old_value + correct_answer_cr_row(current_row))/(user_tests_df.iloc[i,2]+5))*100)#+number correct answer relevant to current test
                user_tests_df.iloc[i,2] = user_tests_df.iloc[i,2] + 5
                break
        if(count == 0):#if first test
            wks.add_rows(1)
            new_value = correct_answer_cr_row(current_row)
            final_value = round((new_value/5)*100)
            tot_q = 5 #1 test has 5 questions
            new_row = pd.Series(data=[mail,final_value,tot_q], index=np.arange(3)).to_frame().T
            new_row.index = np.arange(len(user_tests_df)+user_tests_df.index[0],len(user_tests_df)+user_tests_df.index[0]+1)
            user_tests_df = pd.concat([user_tests_df, new_row])#, ignore_index=True)
        
    else:#if first user
        wks.add_rows(1)
        new_value = correct_answer_cr_row(current_row)#numero rx corrette della riga
        final_value = round((new_value/5)*100)
        tot_q = 5 #1 test ha 5 domande
        new_row = pd.Series(data=[mail,final_value,tot_q], index=np.arange(3)).to_frame().T
        new_row.index = np.arange(len(user_tests_df)+user_tests_df.index[0],len(user_tests_df)+user_tests_df.index[0]+1)
        user_tests_df = pd.concat([user_tests_df, new_row], ignore_index=True)
    return user_tests_df

#prepare data
#input:the number of past previous answered questions of the test, vector of current test data, dataframe containg averaged values,
#dataframe containg the data relevant to the informative questionnaire
def data_t(nb_qstion,row_output,avg,user_tests_df):
    data_previous_question = np.zeros((1,7*nb_qstion), dtype=float)#data relevant to past questions and answers
    quest_answ_t = row_output.iloc[8:18]#questions and answers data
    timestamp_t = row_output.iloc[18:]#timestamps of events
    name_t = row_output[0] #user's name
    mail_t = row_output.iloc[3]#user's mail
    user_data_temp = user_features_df.loc[user_features_df.iloc[:,0] == mail_t]
    if(user_data_temp.empty):#if I don't find the mail, look for the name
        user_data_temp = user_features_df.loc[user_features_df.iloc[:,-1] == name_t]
        if(user_data_temp.empty):#no mail nor name found
            user_data_t = user_features_df.iloc[0,1:-1].to_numpy() #default for unknown users
        else:
            user_data_t = user_data_temp.iloc[0,1:-1].to_numpy()
    else:
        user_data_t = user_data_temp.iloc[0,1:-1].to_numpy()
    past_average_score = correct_answer_user(mail_t,user_tests_df)#percentage of correct answers on past tests
    for i in range(1,nb_qstion+1):
        data_previous_question[0,0+(7*(i-1))] = correct_answer(row_output,i) #previous question is correct?
        data_previous_question[0,1+(7*(i-1))] = skipped_question(timestamp_t, i) #previous question has been skipped?
        data_previous_question[0,2+(7*(i-1))] = round(time_question(timestamp_t,i),1) #time spent on previous question
        question_index = quest_answ_t.iloc[(i*2)-2]
        if(df.iloc[question_index,7] == 'advanced'):#difficulty previous question: 1:advanced 0:basic
            data_previous_question[0,3+(7*(i-1))] = 1
        else:
            data_previous_question[0,3+(7*(i-1))] = 0
        data_previous_question[0,4+(7*(i-1))] = nclick_question(timestamp_t,i)#number of clicks on previous question
        perc_cr_answers = perc_correct_answers(question_index,avg)#statistics difficulty of previous question   
        data_previous_question[0,5+(7*(i-1))] = perc_cr_answers[0,0] 
        data_previous_question[0,6+(7*(i-1))] = perc_cr_answers[0,1]#if the question has really been answered or not          
 
    x = np.concatenate((user_data_t.reshape(1,-1),past_average_score.reshape(1,-1),data_previous_question),axis=1)
    return x

#data on the next question
#question done:if the question has really been answered or not
def futureQuestionData(question_index_next,avg):
    data_next_question = np.zeros((1,9), dtype=float)
    if(df.iloc[question_index_next,7] == 'advanced'):#question difficulty: 1:advanced 0:basic   
        data_next_question[0,0] = 1
    else:
        data_next_question[0,0] = 0
    perc_cr_answers = perc_correct_answers(question_index_next,avg)#statistics difficulty next question
    data_next_question[0,1] = perc_cr_answers[0,0] 
    data_next_question[0,2] = perc_cr_answers[0,1]#question done 
    average_time_answers = average_time(avg,question_index_next)#average time spent on next question
    data_next_question[0,3] = average_time_answers[0,0] 
    data_next_question[0,4] = average_time_answers[0,1] #question done
    average_skip_answers = average_skip_answer(avg,question_index_next)#average number of skips on next question
    data_next_question[0,5] = average_skip_answers[0,0]
    data_next_question[0,6] = average_skip_answers[0,1]#question done
    average_click_answers = average_click_answer(avg,question_index_next)#average number of skips on next question
    data_next_question[0,7] = average_click_answers[0,0]
    data_next_question[0,8] = average_click_answers[0,1]#question done
    
    return data_next_question

            
#python function to calculate next question
#input: vector of current test data
#output: the index of the next question(0-39)
def algorithm_for_new_question(output_row,ifMaxCorrectQst_past):
    #output_row è lista cambiare in dataframe:
    row_output = pd.Series(data=output_row, index=np.arange(len(output_row)))
    list_questions = list(range(0,39))
    row_qst = row_output.iloc[8:18]
    list_index_question=[]
    mail_t = row_output.iloc[3]
    #re-read excel with new data
    sh = gc.open('Imath_prototype_data')
    wks = sh[9]
    past = wks.get_as_df(has_header=True,include_tailing_empty=False)
    user_past_tests = past.loc[past.iloc[:,3] == mail_t]
    #remove past questions from the list of possible next questions
    for i in range(0,row_qst.shape[0],2):
        if(row_qst.iloc[i] != '-1'):
            list_index_question.append(row_qst.iloc[i])  
    if(user_past_tests.empty == False):
        for test in range(0,len(user_past_tests)):
            test_qst = user_past_tests.iloc[test,8:18]
            for i in range(0,test_qst.shape[0],2):
                list_index_question.append(test_qst.iloc[i]) 
    print('list')
    print(list_index_question)
    list_questions = [x for x in list_questions if x not in list_index_question]
    #print(list_questions)
    #obtain nb_question
    print(row_qst)
    for i in range(1,row_qst.shape[0],2): 
        if(row_qst.iloc[i] == '-1'):
            nb_qstion = ((i//2)+1)-1#first question equal to -1
            break
    t=nb_qstion+1
    if(t==2):
        #possibility to to choose if max or min the probability of a correct answer according to the value of a random number
        #fix for the current session and changing for each session
        #now disabled
        pox = [0,1]
        ifMaxCorrectQst = random.sample(pox,1)[0] 
        print('random')
        print(ifMaxCorrectQst)
        ifMaxCorrectQst_past = ifMaxCorrectQst
    else:
        ifMaxCorrectQst = ifMaxCorrectQst_past
    preds = []
    indexes = []
    #latest values
    sh = gc.open('averages')
    wks = sh[1]
    avg = wks.get_as_df(has_header=False,include_tailing_empty=False)
    wks = sh[0]
    user_tests_df = wks.get_as_df(has_header=False, include_tailing_empty=False) 
    user_tests_df = user_tests_df.iloc[1:,:]#percentage of correct answers for each student
    #print(user_tests_df)
    #past data
    x = data_t(nb_qstion,row_output,avg,user_tests_df)
    #load scalers and encoders
    with open('encoder_t{}.pickle'.format(t),'rb') as f:
        enc = pk.load(f)
    with open('scaler_t{}.pickle'.format(t), 'rb') as f:
        scaler = pk.load(f) 
    #divide numerical and categorical variables and scale
    numerical_indexes = []
    categorical_indexes = []
    for i in range(x.shape[1]):
        if(isinstance(x[0,i], str)):    
           categorical_indexes.append(i)
        else:
           numerical_indexes.append(i) 
    x_c= np.delete(x,numerical_indexes,1) 
    x_c_encoded = enc.transform(x_c)
    x_n_temp = np.delete(x,categorical_indexes,1)
    #load model and features to cancel
    with open('rf_t{}.pickle'.format(t), 'rb') as f:
            rf = pk.load(f)
    print(rf.classes_)    
    with open('featuresToCancel_t{}.pickle'.format(t), 'rb') as f:
            feat_eliminate_train = pk.load(f)  
    for question_index_next in list_questions:
        #add data on the next question
        x_nextq = futureQuestionData(question_index_next,avg)
        x_n = np.concatenate((x_n_temp,x_nextq),axis=1) 
        #scale numerical data
        x_n_scaled = scaler.transform(x_n) 
        x_new = np.concatenate((x_n_scaled,x_c_encoded),axis=1)   
        #delete less important features
        x_short = np.delete(x_new,feat_eliminate_train, 1)  
        #predict probability that the user answer correctly 
        y = rf.predict_proba(x_short)#classes_ [0 1] #1:correctly answer
        preds.append(y[0,1])
        indexes.append(question_index_next)
    #if(ifMaxCorrectQst == 0):#maximize the number of correct answer
        print(max(preds))
        print(preds.index(max(preds)))
        next_question = indexes[preds.index(max(preds))] 
    # else:#minimize the number of correct answer
    #     print(min(preds))
    #     print(preds.index(min(preds)))
    #     next_question = indexes[preds.index(min(preds))] 
    return next_question,ifMaxCorrectQst_past


l1=list(range(0,40)) #l1=[0,1,2,3,4,5,6,7,8,9]
answers=[None,None,None,None,None]
def generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3,ifMaxCorrectQst_past):         
    if(j!=n_qstion+1):
        df_output[(j-1)*2+user_input+evaluation_qstion]= index_previous_question       
        if(isinstance(optionList[0], dict) or clickedAnswer==None): 
            df_output[(j-1)*2+user_input+evaluation_qstion+1]=(clickedAnswer)
            list_answers[j-1]=(clickedAnswer)
        else:
            df_output[(j-1)*2+user_input+evaluation_qstion+1]=str(optionList.index(clickedAnswer))#insert index not value
            list_answers[j-1]=(clickedAnswer)
    print('j')
    print(df_output)
    print(j)
    print(len(list_index_question))       
    if(backward==0): 
        if(j==n_qstion): #if j==5 last question has been already chosen
            #print(index_previous_question) 
            #if index_previous_question not in list_index_question:
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question)
            next_question=list_index_question[-1] #load last question and relevant user's answer
            clickedAnswer=list_answers[-1]
        elif(j>=len(list_index_question)):
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question) 
            ### --- TO CHOOSE RANDOMLY ----
            # next_question_t = random.sample(list_questions,1)
            # next_question=next_question_t[0] 
            output_row = df_output + df_output_2 + df_output_3
            start = time.time()
            next_question,ifMaxCorrectQst_past=algorithm_for_new_question(output_row,ifMaxCorrectQst_past)
            end = time.time()
            print('timeTotFunc:{}'.format(end-start)) 
            clickedAnswer=None
        else:#if j<len it means I came back and if I click next the past question and answer has to be loaded
        #I have already computed next question, no call RF
            print('now!')
            next_question=list_index_question[j]
            clickedAnswer=list_answers[j]
    else:#back button pressed
        print('here!')
        print(list_index_question)
        print(j)
        print(n_qstion)
        if(j==n_qstion+1): # load last question and relevant user's answer
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question)
            print('l')
            print(list_index_question)
            next_question=list_index_question[-1] 
            clickedAnswer=list_answers[-1]
        else:
            if index_previous_question not in list_index_question:
                list_index_question.append(index_previous_question)
            print('r')
            print(list_index_question)
            next_question=list_index_question[j-2]
            clickedAnswer=list_answers[j-2]

    new_question = df.iloc[next_question,0]#text
    new_options = [df.iloc[next_question,1],df.iloc[next_question,2],df.iloc[next_question,3],df.iloc[next_question,4],df.iloc[next_question,5]]
    
    return list_index_question,new_options,new_question,next_question,list_answers,clickedAnswer,ifMaxCorrectQst_past

def save_on_google_sheets(list1,list2,list3,avg,user_tests_df,random_nb):
    gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
    sh = gc.open('Imath_prototype_data')
    wks = sh[9]
    wks.add_rows(1)
    final_list= list1 + list2 + list3 + list(str(random_nb))#save also random number
    final_row = pd.DataFrame([final_list])
    wks.set_dataframe(final_row,(wks.rows,1),copy_head=False)
    #update avg
    final_list = final_list[0:-1]#rimuovo random number prima di chiamare funzioni x aggiornamento avg
    #print(final_list)
    final_row = pd.Series(data=final_list, index=np.arange(len(final_list)))
    #take latest value of avg
    sh = gc.open('averages')
    wks = sh[1]
    avg = wks.get_as_df(has_header=False,include_tailing_empty=False)
    #update
    avg = update_perc_cr_answers(final_row,avg)
    avg = update_average_time(final_row,avg)
    avg = update_average_click_answer(final_row,avg)
    avg = update_average_skip_answer(final_row,avg)
    wks.set_dataframe(avg,(1,1),copy_head=False) 
    #update user correct answers
    wks = sh[0]
    user_tests_df = wks.get_as_df(has_header=False, include_tailing_empty=False) 
    user_tests_df = user_tests_df.iloc[1:,:]#percentage of correct answers for each student
    #print(user_tests_df)
    user_tests_df = update_correct_answer_user(final_row,wks,user_tests_df)
    wks.set_dataframe(user_tests_df,(2,1),copy_head=False) 


#--- CALLBACKS ---
#update question
@app.callback([Output('card_body', 'children'),Output('counter','data'),Output('index_previous_question','data'),
               Output('starting_time','data'), Output('name','value'), Output('surname','value'),Output('university','value'), 
               Output('email','value'), Output('seed','data'), Output('list_index_question','data'), Output('backward','data'),
               Output('list_answers','data'), Output('next_question','data'), Output('question_answer', 'value'),Output('count_eq ','data'),
               Output('df_output_1','data'), Output('ifMaxCorrectQst_past','data'),],
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
               Input('trigger_3','children'),
               State('ifMaxCorrectQst_past','data'),
              ])
def update_card_body(nextbutton,testButton,start_button,exitButton,submitbutton,trigger,skipbutton,previuosbutton,evalButton,clickedAnswer,optionList,j,index_previous_question,start,name,surname,university,email,seed,list_index_question,backward,list_answers,trigger_2,next_question,count_eq,df_output,df_output_2,df_output_3,trigger_3,ifMaxCorrectQst_past):
    if(start_button==None or exitButton!=None):
        if(ctx.triggered_id !='exitButton'):
            #re-initialize
            df_output_2 = []
            df_output_3 = []
            df_output=['-1'] * (n_qstion*2+user_input+evaluation_qstion)
            #print(df_output)
            login=int(time.time()*1000)
            value_string_login = str(-2) +'#'+ 'login' +'#'+ str(login)
            df_output.append(value_string_login) #take login timestamp
            #print(df_output)
            
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
                    #processing new question question message
                    html.Div(children=[
                        dcc.Markdown(children='**Processing question...**')
                    ],id='processing_message', style={'display':'none'}, className='row'),
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
        #save user data
        df_output[0]= name
        df_output[1] = surname
        df_output[2] = university
        df_output[3] = email
        
        if(j!=2):#first question, not going back
            output_row = df_output + df_output_2 + df_output_3
            mail_t = output_row[3]
            sh = gc.open('Imath_prototype_data')
            wks = sh[9]
            past = wks.get_as_df(has_header=True,include_tailing_empty=False)#rileggo excel con nuovi dati
            user_past_tests = past.loc[past.iloc[:,3] == mail_t]
            l2=[]
            #remove questions from past tests
            if(user_past_tests.empty == False):
                for test in range(0,len(user_past_tests)):
                    test_qst = user_past_tests.iloc[test, 8:18]
                    for i in range(0,test_qst.shape[0],2):
                        l2.append(test_qst.iloc[i]) 
            l1_updated = [x for x in l1 if x not in l2]
            #print(l1_updated)
            starting_question = random.sample(l1_updated,1)[0]
            new_question=df.iloc[starting_question,0] 
            new_options=[df.iloc[starting_question,1],df.iloc[starting_question,2],df.iloc[starting_question,3],df.iloc[starting_question,4],df.iloc[starting_question,5]]
            index_previous_question=starting_question
        else:
            if(clickedAnswer==0):#if user doesn't answer, clickedAnswer None
                clickedAnswer = None
            backward=1
            print('here1')
            print(list_index_question)
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer,ifMaxCorrectQst_past = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3,ifMaxCorrectQst_past)
            print(list_index_question)
            j=0
            #print(index_previous_question)
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
                    #processing new question question message
                    html.Div(children=[
                        dcc.Markdown(children='**Processing question...**')
                    ],id='processing_message', style={'display':'none'}, className='row'),
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
    
    elif(start_button!=None and ((nextbutton!=None and 0<j<n_qstion) or (skipbutton!=None and 0<j<n_qstion) or (previuosbutton!=None and 2<j<=n_qstion+1))):#if clicked go to next question
        print('phase 3')
        if(ctx.triggered_id=='skipButton' and trigger_3 == 1):
            clickedAnswer = None
            title='Question '+str(j+1)
            backward=0
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer,ifMaxCorrectQst_past = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3,ifMaxCorrectQst_past)
            next_question=index_previous_question
            j=j+1
        if(ctx.triggered_id=='nextButton' and trigger_3 == 1):
            if(clickedAnswer==0):
                clickedAnswer = None
            backward=0
            title='Question '+str(j+1)
            #print(list_index_question)
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer,ifMaxCorrectQst_past = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3,ifMaxCorrectQst_past)
            next_question=index_previous_question
            #print('clickedAnswer:{}'.format(clickedAnswer))
            j=j+1                      
        if(ctx.triggered_id=='previuosButton'):
            if(clickedAnswer==0):
                clickedAnswer = None
            title='Question '+str(j-1)
            backward=1
            print('here2')
            print(list_index_question)
            list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer,ifMaxCorrectQst_past = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3,ifMaxCorrectQst_past)
            print(list_index_question)
            next_question=index_previous_question
            j=j-1   
        #print('clickedAnswer:{}'.format(clickedAnswer))
        
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
                    #processing new question question message
                    html.Div(children=[
                        dcc.Markdown(children='**Processing question...**')
                    ],id='processing_message', style={'display':'none'}, className='row'),
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
        list_index_question,new_options,new_question,index_previous_question,list_answers,clickedAnswer,ifMaxCorrectQst_past = generate_next_question(clickedAnswer,optionList,j,index_previous_question,seed,list_index_question,backward,list_answers,df_output,df_output_2,df_output_3,ifMaxCorrectQst_past)
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
                    #processing new question question message
                    html.Div(children=[
                        dcc.Markdown(children='**Processing question...**')
                    ],id='processing_message', style={'display':'none'}, className='row'),
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
                    #processing new question question message
                    html.Div(children=[
                        dcc.Markdown(children='**Processing question...**')
                    ],id='processing_message', style={'display':'none'}, className='row'),
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
                #processing new question question message
                html.Div(children=[
                    dcc.Markdown(children='**Processing question...**')
                ],id='processing_message', style={'display':'none'}, className='row'),
              ] 
        
    elif((evalButton>1) and (nextbutton==None)):#first eval question
        clickedAnswer=None 
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
                #processing new question question message
                html.Div(children=[
                    dcc.Markdown(children='**Processing question...**')
                ],id='processing_message', style={'display':'none'}, className='row'),
              ] 
        count_eq=count_eq+1
    elif(nextbutton!=None and count_eq!=45):#2-4 eval qstion
        #save survey data
        df_output[count_eq-42+user_input]=str(optionList.index(clickedAnswer))#insert index not value
        clickedAnswer=None 
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
                #processing new question question message
                html.Div(children=[
                    dcc.Markdown(children='**Processing question...**')
                ],id='processing_message', style={'display':'none'}, className='row'),
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
                #processing new question question message
                html.Div(children=[
                    dcc.Markdown(children='**Processing question...**')
                ],id='processing_message', style={'display':'none'}, className='row'),
              ]  
        
        #saving data on google sheet
        save_on_google_sheets(df_output,df_output_2,df_output_3,avg,user_tests_df,ifMaxCorrectQst_past)
        #re-initialize  
        df_output=['-1'] * (n_qstion*2+user_input+evaluation_qstion)     
        seed=seed+1
        list_index_question=[]
        j=0
        list_answers = [None,None,None,None,None]
        backward=0
        clickedAnswer=None
        count_eq=41
    return [children,j,index_previous_question,start,name,surname,university,email,seed,list_index_question,backward,list_answers,next_question,clickedAnswer,count_eq,df_output,ifMaxCorrectQst_past]
    
#update next button
@callback(Output('testButton', 'n_clicks'), 
          Input('testButton','n_clicks'))
def reset_backbutton(n):
    if(n!=None):
        return None

#trigger disabling submit button and showing loading sentence
#trigger disabling next button when calculating new question
@app.callback([Output('trigger','children'),Output('trigger_3','children'),Output('processing_message','style'), Output('loading_page','style'),
               Output('nextButton', 'disabled'),Output('skipButton', 'disabled'), Output('previuosButton', 'disabled')],
              [Input('nextButton', 'n_clicks'),
               Input('skipButton', 'n_clicks'),
               Input('start_button', 'n_clicks'),
               State('counter','data'),
               State('count_eq ','data')])
def trigger_funct_2(nextbutton,skipbutton,start_button,j,count_eq):
    trigger_1 = 0
    trigger_2 = 0
    button_next=False
    button_skip=False
    button_prev=False
    processing_message_style={'display':'none'}
    loading_page_style={'display':'none'}
    if (start_button!=None and ((nextbutton!=None and 0<j<n_qstion) or (skipbutton!=None and 0<j<n_qstion))):#MODIFICARE E VEDERE DOVE CHIAMARE!
        trigger_1 = 1
        button_next=True
        button_skip=True
        button_prev=True
        processing_message_style={'textAlign': 'center','margin-top': 10,'margin-bottom': 20,'margin-left': 200,'width': '50%','font-size':18}    
    if (nextbutton!=None and (count_eq==45)):
        trigger_2 = 1
        button_next=True
        loading_page_style={'textAlign': 'center','margin-top': 10,'margin-bottom': 20,'margin-left': 200,'width': '50%','font-size':18}
    return [trigger_2,trigger_1,processing_message_style,loading_page_style,button_next,button_skip,button_prev]

# collect buttons' time-stamps -> ms since 1970
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
        #print('start button press') 
        values.append(0)
        #values.append(next_question)
        values.append('start_button')
        values.append(startbutton)
        #print('count_df_2:{}'.format(count_df_2))
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) 
        if not df_output:            
            df_output.append(values_string)
        else:
            if(df_output[-1][0] != str(values[0])):
                df_output.append(values_string)
                
    if(startbutton!=None and ctx.triggered_id=='start_button' and not df_output):
        df_output = []
        
    if(previuosbutton!=None and ctx.triggered_id=='previuosButton'):
        #print('previous button press') 
        values.append(j)
        #values.append(next_question)
        values.append('previuosButton')
        values.append(previuosbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) #+'#'+ str(values[3])
        if(df_output[-1][0] != str(values[0])):
                df_output.append(values_string)
        
    if(nextbutton!=None and ctx.triggered_id=='nextButton'):
        #print('next button press')
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
        if(df_output[-1][0] != str(values[0])):
            if(df_output[-1][0] == 'e'):
                if(df_output[-1][0:2] != str(values[0])):
                    df_output.append(values_string)
            else:
                df_output.append(values_string)
        if(count_eq==45):#re-inizialize
            count_df_2=0
        
    if(skipbutton!=None and ctx.triggered_id=='skipButton'):
        #print('skip button press') 
        values.append(j)
        #values.append(next_question)
        values.append('skipButton')
        values.append(skipbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2]) #+'#'+ str(values[3])
        if(df_output[-1][0] != str(values[0])):
                df_output.append(values_string)
        
    if(submitbutton!=None and ctx.triggered_id=='submitButton'):
        #print('submit button press')  
        values.append(n_qstion+1)
        values.append('submitButton')
        values.append(submitbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        if(df_output[-1][0] != str(values[0])):
                df_output.append(values_string)
            
    if(testbutton!=None and ctx.triggered_id=='testButton'):
        #print('test button press') 
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
        #print('exit button press') 
        df_output = []
        values.append(-2)
        values.append('login')
        values.append(exitbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        df_output.append(values_string)
        
    if(evalbutton!=None and ctx.triggered_id=='evalButton'):
        #print('eval button press') 
        values.append(n_qstion+2)
        values.append('evalbutton')
        values.append(evalbutton)
        #save values on google sheet
        count_df_2=count_df_2+1
        values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
        if(df_output[-1][0] != str(values[0])):
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
            #print('saving timestamp')
            values_string = str(values[0]) +'#'+ str(values[1]) +'#'+ str(values[2])
            count_df=count_df+1
            df_output.append(values_string) 
        if(j==0 and (list_answers==[None,None,None,None,None])):#re-inizialize count
            count_df=0
    else:
        df_output = []
    return [count_df,df_output]
    
    
app.debug=False
if __name__ == '__main__':   
    #waitress.serve(application, host='127.0.0.1', port=8050)
    app.run_server(debug=True,host='127.0.0.1', port=8050,use_reloader=False)