# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:46:17 2023

@author: giuli
"""
import numpy as np
import pandas as pd
import pygsheets
import pickle as pk


#upload user features (data from informative questionnaire) already processed
# with open('user_features_df.pickle', 'rb') as f:
#    user_features_df = pk.load(f)
   
#file where list of the 40 possible questions are saved 
gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('Imath_prototype_data')
wks = sh[0]
df=wks.get_as_df(has_header=True,include_tailing_empty=False)

def skipped_question(data_session,nb_qstion):#input: complete row vector, number of questions(1-5)
    count=0
    for i in range(0,len(data_session)):
        if(data_session.iloc[i].split('#')[0]==str(nb_qstion) and data_session.iloc[i].split('#')[1] == 'skipButton'):
            count=count+1
    return count

#the question has been correctly answered?
def correct_answer(df_output,nb_qstion):#input: complete row vector, number of questions(1-5)
    quest_answ = df_output.iloc[8:18]
    question_index = quest_answ.iloc[(nb_qstion*2)-2]
    if(quest_answ.iloc[(nb_qstion*2)-1] == str(df.iloc[question_index,6])):#anche se è None va bene tanto sarà false e domanda sbagliata
        return 1
    else:
        return 0
    
#compute number of correct answers for a specific user + bol value if it is his/her test or not
def correct_answer_user(mail,user_tests_df):
    arr = np.zeros((1,2))
    #print(user_tests_df)
    if(user_tests_df.empty == False):
        df_search = user_tests_df.loc[user_tests_df.iloc[:,0]==mail] 
        #print(df_search)
        if(df_search.empty):#if first test
           qstion_done = 0 
           user_correct = round(user_tests_df.iloc[:,1:3].mean()[1])#insert average
        else:
           qstion_done = 1
           user_correct = df_search.iloc[0,1]
    else:
        user_correct = 0
        qstion_done = 0
    # print(perc_correct)
    arr[0,0] = user_correct 
    arr[0,1] = qstion_done
    return arr

def time_question(data_session,nb_qstion): #input: timestamps vector, number of questions(1-5)
    times=[]
    #print(data_session)
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

def process_user_data(user_features,userDf):
    #differentiate categorical and numerical features
    fraction_indexes = [5,16,17,18,19,20,21,22]
    numerical_indexes = [2,7,13,14,15]
    categorical_indexes = [1,3,4,6,8,9,10,11,12]
    exam_done = np.zeros((1,7), dtype=int)
    #get to all the grades into a scale 0-100
    #decimal numbers have to be expressed with dots, no commas
    print('here!')
    print(user_features)
    for column in fraction_indexes:
        if (type(user_features.iloc[0,column]) != float): #if nan/stringa vuota
          try:
              value1 = float(user_features.iloc[0,column].split('/')[0])
              value2 = int(user_features.iloc[0,column].split('/')[1])
              user_features.iloc[0,column] = int((value1/value2)*100)
              if(column != 5):#put the boolean variable to one to indicate that the student has effectively done an exam
                  exam_done[0,column-16] = 1 
          except ValueError: 
              user_features.iloc[0,column] = np.nan#if the input is not in the correct format, we put a nan (missing value)
          except IndexError:
              user_features.iloc[0,column] = np.nan#if the input is not in the correct format, we put a nan (missing value)
    #if college year == other put -1
    if(user_features.iloc[0,7] == 'other'):   
         user_features.iloc[0,7] = -1
        
    #convert numerical features from string to int
    for column in numerical_indexes:
        if (type(user_features.iloc[0,column]) != float): #if not empty string
            try:
                value = int(user_features.iloc[0,column])
            except ValueError:#if the input is not in the correct format, we put a nan (missing value)
                value = np.nan
            user_features.iloc[0,column] = value
          
    #replace all nan values, i.e. all the missing values, with the mode of the respective column
    for column in range(user_features.shape[1]):  
      if (type(user_features.iloc[0,column]) == float):#if empty string  
          user_features.iloc[0,column] = userDf.iloc[:,column].value_counts().idxmax()
          #print(userDf.iloc[:,column].value_counts().idxmax())
    #print(user_features)
    #transform mode values if needed
    for column in fraction_indexes:
        if isinstance(user_features.iloc[0,column], str):
            try:
                value1 = float(user_features.iloc[0,column].split('/')[0])
                value2 = int(user_features.iloc[0,column].split('/')[1])
                user_features.iloc[0,column] = int((value1/value2)*100)
            except ValueError:
                user_features.iloc[0,column] = 60
            except IndexError:
              user_features.iloc[0,column] = 60
    for column in numerical_indexes:
        if isinstance(user_features.iloc[0,column], str):
            user_features.iloc[0,column] = int(user_features.iloc[0,column])
    #add to the data the boolean column that indicates if he do/don't the exams  
    user_features_np = user_features.iloc[0,1:].to_numpy().reshape(1,-1) 
    user_features_np = np.concatenate((user_features_np,exam_done),axis=1)#add names as last 
    print(user_features_np)
    return user_features_np

def data_t(nb_qstion,row_output,avg,user_tests_df):
    data_previous_question = np.zeros((1,7*nb_qstion), dtype=float)#data relevant to past questions and answers
    quest_answ_t = row_output.iloc[8:18]#questions and answers data
    timestamp_t = row_output.iloc[18:]#timestamps of events
    name_t = row_output[0] #user's name
    mail_t = row_output.iloc[3]#user's mail
    #data from background questionnaire
    userDf =  pd.read_csv('new_user_data17-04.txt', sep=';', header=None)
    userDf = userDf.iloc[1:,:] #remove header
    userDf = userDf.iloc[:,:len(userDf.columns)-1] #remove columns wih comments
    userDf_cut = userDf.drop(userDf.columns[[0,2]], axis=1) #delete names and timestamps
    #print(userDf)
    #print(userDf_cut)
    user_data_temp = userDf_cut.loc[userDf_cut.iloc[:,0] == mail_t]
    print(user_data_temp)
    if(user_data_temp.empty):#if I don't find the mail, look for the name
        user_data_temp = userDf.loc[userDf.iloc[:,2] == name_t]
        if(user_data_temp.empty):#no mail nor name found
            user_data_temp = userDf_cut.loc[userDf_cut.iloc[:,0] == 'alfredstephen77@gmail.com'] #default for unknown users 
            user_data_t = process_user_data(user_data_temp,userDf_cut) 
            #user_data_t = user_features_df.iloc[0,1:-1].to_numpy() #default for unknown users
        else:
            user_data_temp = user_data_temp.drop(user_data_temp.columns[[0,2]], axis=1) #delete names and timestamps
            user_data_t = process_user_data(user_data_temp,userDf_cut)   
            #user_data_t = user_data_temp.iloc[0,1:-1].to_numpy()
    else:
        user_data_t = process_user_data(user_data_temp,userDf_cut)
        #user_data_t = user_data_temp.iloc[0,1:-1].to_numpy()
    print(user_data_t)#user data, those from the informative questionnaire, remove mail and name
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
    # print(skip_answer)
    arr[0,0] = skip_answer 
    arr[0,1] = qstion_done
    return arr

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
    # print(perc_correct)
    arr[0,0] = click_answer 
    arr[0,1] = qstion_done
    return arr

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

#update
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

def chooseSheet(topic):
    if(topic=='prototype'):
        n_sheet = 1
    if(topic=='Analytic Geometry'):
        n_sheet = 2
    elif(topic=='Complex Numbers'):
        n_sheet = 3
    elif(topic=='Differential Equations'):
        n_sheet = 4
    elif(topic=='Differentiation'):
        n_sheet = 5
    elif(topic=='Descrete Mathematics'):
        n_sheet = 6
    elif(topic=='Fundamentals Mathematics'):
        n_sheet = 7
    elif(topic=='Graph Theory'):
        n_sheet = 8
    elif(topic=='Integration'):
        n_sheet = 9
    elif(topic=='Linear Algebra'):
        n_sheet = 10
    elif(topic=='Numerical Methods'):
        n_sheet = 11
    elif(topic=='Optimization'):
        n_sheet = 12
    elif(topic=='Probability'):
        n_sheet = 13
    elif(topic=='Real Functions of a single variable'):
        n_sheet = 14
    elif(topic=='Real Functions of several variables'):
        n_sheet = 15
    elif(topic=='Set Theory'):
        n_sheet = 16
    else:
        n_sheet = 17
    return n_sheet


def RF_model(list_questions,t,row_output,topic):    
    preds = []
    indexes = []
    #get statistics of the questions and the users
    gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
    sh = gc.open('averages')
    n_sheet = chooseSheet(topic)
    wks = sh[n_sheet]#topic's questions
    avg = wks.get_as_df(has_header=False,include_tailing_empty=False)
    wks = sh[0]#user
    user_tests_df = wks.get_as_df(has_header=False, include_tailing_empty=False) 
    user_tests_df = user_tests_df.iloc[1:,:]#percentage of correct answers for each student
    #PREPARE DATA
    nb_qstion=t-1
    x = data_t(nb_qstion,row_output,avg,user_tests_df)
    print(x)
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
    #print(rf.classes_)    
    with open('featuresToCancel_t{}.pickle'.format(t), 'rb') as f:
            feat_eliminate_train = pk.load(f) 
    print(list_questions)
    for question_index_next in list_questions:#for each question in the graph
        #add data on the next question
        x_nextq = futureQuestionData(question_index_next,avg)
        x_n = np.concatenate((x_n_temp,x_nextq),axis=1) 
        #scale numerical data
        x_n_scaled = scaler.transform(x_n) 
        x_new = np.concatenate((x_n_scaled,x_c_encoded),axis=1)   
        #delete less important features
        x_short = np.delete(x_new,feat_eliminate_train, 1) 
        print(x_short)
        #predict probability that the user answer correctly 
        y = rf.predict_proba(x_short)#classes_ [0 1] #1:correctly answer
        print(y)
        preds.append(y[0,1])#probabilities the student will correctly answer the questions
        indexes.append(question_index_next)#id questions
    return preds,indexes

