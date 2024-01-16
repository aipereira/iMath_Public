# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:04:30 2022

@author: giuli
"""

#library
import pandas as pd
import numpy as np
import pygsheets
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pickle as pk
from sklearn.inspection import permutation_importance
import time as tm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print('The scikit-learn version is {}.'.format(sklearn.__version__))

#DATA FROM HISTORICAL FILE
userDf =  pd.read_csv('new_user_data17-04.txt', sep=';', header=None)
userDf = userDf.iloc[1:,:] #remove header 
print(userDf)

user_features = userDf.iloc[:,:len(userDf.columns)-1] #remove columns wih comments
user_features = user_features.drop(user_features.columns[[0,2]], axis=1) #delete columns with timestamps
print(user_features)

#differentiate categorical and numerical features
fraction_indexes = [5,16,17,18,19,20,21,22]
numerical_indexes = [2,7,13,14,15]
categorical_indexes = [1,3,4,6,8,9,10,11,12]
exam_done = np.zeros((len(user_features),7), dtype=int)

#function get to all the grades into a scale 0-100
#decimal numbers have to be expressed with dots, no commas
for row in range(len(user_features)):
    for column in fraction_indexes:
        if (type(user_features.iloc[row,column]) != float): #if nan/stringa vuota
          try:
              value1 = float(user_features.iloc[row,column].split('/')[0])
              value2 = int(user_features.iloc[row,column].split('/')[1])
              user_features.iloc[row,column] = int((value1/value2)*100)
              if(column != 5):#put the boolean variable to one to indicate that the student has effectively done an exam
                  exam_done[row,column-16] = 1 
          except ValueError:
              user_features.iloc[row,column] = np.nan#if the input is not in the correct format, we put a nan (missing value)

#if college year == other put -1
for row in range(user_features.shape[0]):  
   if(user_features.iloc[row,7] == 'other'):   
        user_features.iloc[row,7] = -1
        
#convert numerical features from string to int
for row in range(len(user_features)):
    for column in numerical_indexes:
        if (type(user_features.iloc[row,column]) != float):
            try:
                value = int(user_features.iloc[row,column])
            except ValueError:#if the input is not in the correct format, we put a nan (missing value)
                value = np.nan
            user_features.iloc[row,column] = value
          
#replace all nan values, i.e. all the missing values, with the mode of the respective column
for row in range(user_features.shape[0]):
  for column in range(user_features.shape[1]):  
    if (type(user_features.iloc[row,column]) == float):#se lasciato vuoto    
        user_features.iloc[row,column] = user_features.iloc[:,column].value_counts().idxmax()
        
#add to the data the boolean column that indicates if he do/don't the exams  
user_features_np = user_features.to_numpy() 
names = userDf.iloc[:,2].to_numpy().reshape(-1,1)
user_features_np = np.concatenate((user_features_np,exam_done,names),axis=1)#add names as last     
user_features_df = pd.DataFrame(data=user_features_np)   

print(user_features_df.iloc[7,:]) 
     
#save user_features to use then in the prototype
with open('user_features_df.pickle', 'wb') as f:
    pk.dump(user_features_df, f)

#DATA FROM THE PAST TESTS
#read excel output     
gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('Imath_prototype_data')
wks = sh[8]
df_test = wks.get_as_df(has_header=False, include_tailing_empty=False) 
df_test = df_test.iloc[1:,:]#delete header
quest_answ = df_test.iloc[:,8:18] #questions and answers vector
timestamp = df_test.iloc[:,18:]  #timestamps vector

#read excel input 
gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('Imath_prototype_data')
wks = sh[0]
df=wks.get_as_df(has_header=True)

#compute time spent on a question (nb_qstion is number between 1 and 5)
def time_question(data_session,nb_qstion): #input: timestamps vector, number of questions(1-5)
    times=[]
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
time = round(time_question(timestamp.iloc[0,:],1),1) #time in seconds
#print(time)

#compute number of clicks on a question's answers (nb_qstion is number between 1 and 5)
def nclick_question(data_session,nb_qstion):#input: timestamps vector, number of questions(1-5)
    count=0
    for i in range(0,len(data_session)):
        if(data_session.iloc[i].split('#')[0]==str(nb_qstion) and data_session.iloc[i].split('#')[1] == 'answer'):
            count=count+1
    return count
n_click = nclick_question(timestamp.iloc[0,:],2) #n_click
#print(n_click)

#the question has been correctly answered?
def correct_answer(df_output,nb_qstion):#input: complete row vector, number of questions(1-5)
    quest_answ = df_output.iloc[8:18]
    question_index = quest_answ.iloc[(nb_qstion*2)-2]
    if(quest_answ.iloc[(nb_qstion*2)-1] == df.iloc[question_index,6]):
        return 1
    else:
        return 0 
c_a = correct_answer(df_test.iloc[0,:],1) #n_click
#print(c_a)

#the question has been skipped?
def skipped_question(data_session,nb_qstion):#input: timestamps vector, number of questions(1-5)
    count=0
    for i in range(0,len(data_session)):
        if(data_session.iloc[i].split('#')[0]==str(nb_qstion) and data_session.iloc[i].split('#')[1] == 'skipButton'):
            count=count+1
    return count
n_skip = skipped_question(df_test.iloc[22,18:],2) #n_click
#print(n_skip)

#STATISTICS
#Compute averaged values
avg = np.zeros((40,8))#vector for average values
#For each quantity to retrieve, there are two functions: one which allow to compute the quantity online, the other one wich
#computes the values for all the questions, for all the input data, to save them in a dataframe. These last values will be used in
#the prototype when new tests are perfomed: when a user perform a test these tables will be used as questions' statistics in the 
#prediction and each time a new test is finished, the statistics are updated with the info of the new test, to be ready for
#the next user, reducing the latency when generating a question for the user.

#compute number of correct answers of a specific user + bool value if the user has performed at least one test or not
def cr_answ(user_tests):#input: group of tests
    tot_cr=0
    for i in range(0,len(user_tests)):
        for nb_qstion in range(1,6):
            cr = correct_answer(user_tests.iloc[i,:],nb_qstion)
            tot_cr = tot_cr + cr
    tot_cr = (tot_cr / (len(user_tests)*5))*100
    return tot_cr

def correct_aswer_user(mail,j):#input: student's mail, number of row in the input data
    tot_cr=0
    test_done = np.zeros((1,2), dtype=int)
    df_search = df_test.iloc[0:j,:]
    if(df_search.empty == False):
        user_tests = df_search.loc[df_search.iloc[:,3] == mail]
        if(user_tests.empty):
            tot_cr = df_test.groupby(df_test.iloc[:,3]).apply(lambda x: cr_answ(x)).mean()#average of all the other users'results
            test_done[0,1] = 0
        else:
            for i in range(0,len(user_tests)):
                for nb_qstion in range(1,6):
                    cr = correct_answer(user_tests.iloc[i,:],nb_qstion)
                    tot_cr = tot_cr + cr
            tot_cr = (tot_cr / (len(user_tests)*5))*100
            test_done[0,1] = 1
    else:
        tot_cr = df_test.groupby(df_test.iloc[:,3]).apply(lambda x: cr_answ(x)).mean()#average of all the other users'results
        test_done[0,1] = 0
    test_done[0,0] = tot_cr
    return test_done

#compute number of correct answers per question, for each question(from 0 to 39)
def f1(x,nb_qstion):#input: dataframe containing group of tests, number of questions(1-5)
    tot_cr = 0
    arr=np.zeros((1,2))
    for i in range(0,x.shape[0]):#for all the rows
        cr = correct_answer(x.iloc[i,:],nb_qstion)
        tot_cr = tot_cr + cr
    arr[0,0] = tot_cr
    arr[0,1] = x.shape[0]
    return arr

def average_cr_answ_all_qstion(df_test):#input: dataframe containing all the input data
    for column in range(0,10,2):
        arr1=np.zeros((1,2))
        list1 = df_test.groupby(df_test.iloc[:,column+8]).apply(lambda x: f1(x,(column//2)+1))
        #gather together rows which have the same question as question 1,2, etc. and calculate the number of correct answers
        #print(list1)
        for element in list1:
            arr1=np.concatenate((arr1,element),axis=0)
        arr1=arr1[1:,:]
        #print(arr1)
        list2 = df_test.iloc[:,column+8].unique()
        list2.sort()
        #print(list2)
        arr2 = np.asarray(list2).reshape(-1,1)
        if(column==0):
            arr = np.concatenate((arr2,arr1),axis=1)
        else:
            for i in range(0,arr2.shape[0]):
                for j in range(0,arr.shape[0]):                   
                    if(arr[j,0] == arr2[i,0]):
                        arr[j,1] = arr[j,1] + arr1[i,0]
                        arr[j,2] = arr[j,2] + arr1[i,1]
                        break
                    else:
                        if(j == arr.shape[0]-1):
                            arr_temp = np.concatenate((arr2[i].reshape(1,-1),arr1[i,:].reshape(1,-1)),axis=1)
                            arr = np.concatenate((arr,arr_temp),axis=0)
    #print(arr.shape)
    for i in range(0,arr.shape[0]):
      arr[i,1] = round((arr[i,1] / arr[i,2])*100)
      #print(arr[i,1])
    avg = round(arr[:,1].mean()) 
    return avg,arr

avg_cr_answ_all_qstion,arr = average_cr_answ_all_qstion(df_test)
#print(avg_cr_answ_all_qstion)
for i in range(0,arr.shape[0]):
    avg[arr[i,0],0:2] = arr[i,1:3] #saved computed values in this matrix

#compute the percentage of times the users have correctly answered the question, difficulty of the question on a statistics basis
def perc_correct_ansewrs(question_index,quest_answ):#input: index of the question(0-39), vector of questions and answers
    tot_correct = 0
    tot_qst = 0
    qstion_done = 0
    arr = np.zeros((1,2))
    for column in range(0,10,2):
        qst = quest_answ.loc[quest_answ.iloc[:,column] == question_index] #identify rows with question_index as question    
        correct_per_column = qst.loc[qst.iloc[:,column+1] == df.iloc[question_index,6]].shape[0]
        tot_correct = tot_correct + correct_per_column    
        tot_qst = tot_qst + qst.shape[0]
    # print(tot_correct) 
    if(tot_qst != 0):#if the question has been answered by someone
        perc_correct = round((tot_correct/tot_qst)*100)
        qstion_done = 1
    else:#if the question has never been chosen
        perc_correct = avg_cr_answ_all_qstion
        qstion_done = 0
    #print(perc_correct)
    arr[0,0] = perc_correct
    arr[0,1] = qstion_done
    return arr

perc_correct = perc_correct_ansewrs(2,quest_answ)
#print(perc_correct)    

#AVERAGE time spent on each question by all users, for all the questions
def f2(x,nb_qstion):#input: dataframe containing group of tests, number of questions(1-5)
    tot_cr = 0
    arr=np.zeros((1,2))
    for i in range(0,x.shape[0]):#for each row
        #print(nb_qstion)
        #print(x.iloc[i,18:])
        cr = time_question(x.iloc[i,18:],nb_qstion)
        #print(cr)
        tot_cr = tot_cr + cr
    arr[0,0] = tot_cr
    arr[0,1] = x.shape[0]
    return arr

def average_time_all_qstion(df_test):#input: dataframe containing all the input data
    for column in range(0,10,2):
        arr1=np.zeros((1,2))
        list1 = df_test.groupby(df_test.iloc[:,column+8]).apply(lambda x: f2(x,(column//2)+1))
        #gather together rows which have the same question as question 1,2, etc. and apply f2
        #print(list1)
        for element in list1:
            arr1=np.concatenate((arr1,element),axis=0)
        arr1=arr1[1:,:]
        # print(arr1)
        list2 = df_test.iloc[:,column+8].unique()
        list2.sort()
        #print(list2)
        arr2 = np.asarray(list2).reshape(-1,1)
        if(column==0):
            arr = np.concatenate((arr2,arr1),axis=1)
        else:
            for i in range(0,arr2.shape[0]):
                for j in range(0,arr.shape[0]):                   
                    if(arr[j,0] == arr2[i,0]):
                        #print('ok!') #ok!
                        arr[j,1] = arr[j,1] + arr1[i,0]
                        arr[j,2] = arr[j,2] + arr1[i,1]
                        break
                    else:
                        if(j == arr.shape[0]-1):
                            #print('no')
                            arr_temp = np.concatenate((arr2[i].reshape(1,-1),arr1[i,:].reshape(1,-1)),axis=1)
                            arr = np.concatenate((arr,arr_temp),axis=0)
    #print(arr.shape)
    for i in range(0,arr.shape[0]):
      arr[i,1] = round((arr[i,1]/arr[i,2]),1)
      #print(arr[i,1])
    avg = round(arr[:,1].mean(),1) 
    return avg,arr

avg_time_all_qstion,arr = average_time_all_qstion(df_test)
#print(avg_time_all_qstion)
for i in range(0,arr.shape[0]):
    avg[arr[i,0],2:4] = arr[i,1:3]
#print(avg[0,:])
    
#average time spent on a question
def average_time(df_test,question_index):#input: dataframe containing all the input data, index of the question(0-39)
    time_tot = 0
    times = 0
    qstion_done = 0
    arr = np.zeros((1,2))
    data_session = df_test.iloc[:,8:]
    for column in range(0,10,2):
        qst = data_session.loc[data_session.iloc[:,column] == question_index] 
        times = times + qst.shape[0]
        for row in range(0,qst.shape[0]):
            time_q = time_question(qst.iloc[row,10:],(column//2)+1) 
            #print(time_q)
            time_tot = time_tot + time_q
    if(times != 0):       
        average_time = round(time_tot/times,1)
        qstion_done = 1
    else:
        average_time = avg_time_all_qstion
        qstion_done = 0
    arr[0,0] = average_time
    arr[0,1] = qstion_done
    return arr
avg_time = average_time(df_test,2)  
#print(avg_time) 

#average number of clicks on each question by all users, for all the questions
def f3(x,nb_qstion):#input: dataframe containing group of tests, number of questions(1-5)
    tot_cr = 0
    arr=np.zeros((1,2))
    for i in range(0,x.shape[0]):
        #print(nb_qstion)
        #print(x.iloc[i,18:])
        cr = nclick_question(x.iloc[i,18:],nb_qstion)
        tot_cr = tot_cr + cr
    arr[0,0] = tot_cr
    arr[0,1] = x.shape[0]
    return arr

def average_click_all_qstion(df_test):#input: dataframe containing allthe input data
    for column in range(0,10,2):
        arr1=np.zeros((1,2))
        list1 = df_test.groupby(df_test.iloc[:,column+8]).apply(lambda x: f3(x,(column//2)+1))#raggruppo per index ogni domanda e applico f2
        #print(list1)
        for element in list1:
            arr1=np.concatenate((arr1,element),axis=0)
        arr1=arr1[1:,:]
        #print(arr1)
        list2 = df_test.iloc[:,column+8].unique()
        list2.sort()
        #print(list2)
        arr2 = np.asarray(list2).reshape(-1,1)
        if(column==0):
            arr = np.concatenate((arr2,arr1),axis=1)
        else:
            for i in range(0,arr2.shape[0]):
                for j in range(0,arr.shape[0]):                   
                    if(arr[j,0] == arr2[i,0]):
                        arr[j,1] = arr[j,1] + arr1[i,0]
                        arr[j,2] = arr[j,2] + arr1[i,1]
                        break
                    else:
                        if(j == arr.shape[0]-1):
                            arr_temp = np.concatenate((arr2[i].reshape(1,-1),arr1[i,:].reshape(1,-1)),axis=1)
                            arr = np.concatenate((arr,arr_temp),axis=0)
    #print(arr.shape)
    for i in range(0,arr.shape[0]):
      arr[i,1] = round((arr[i,1]/arr[i,2]),1)
      #print(arr[i,1])     
    avg = round(arr[:,1].mean(),1) 
    return avg,arr

avg_click_all_qstion,arr = average_click_all_qstion(df_test)
#print(avg_click_all_qstion)
for i in range(0,arr.shape[0]):
    avg[arr[i,0],4:6] = arr[i,1:3]    
#print(avg)

#average click on an answer
def average_click_answer(df_test,question_index):#input: dataframe containing all the input data, index of the question(0-39)
    nclick_tot = 0
    times = 0
    arr = np.zeros((1,2))
    data_session = df_test.iloc[:,8:]
    for column in range(0,10,2):
        qst = data_session.loc[data_session.iloc[:,column] == question_index] #identifico righe che hanno question_index come domanda
        times = times + qst.shape[0]
        for row in range(0,qst.shape[0]):
            nclick_q = nclick_question(qst.iloc[row,10:],(column//2)+1) 
            #print(nclick_q)
            nclick_tot = nclick_tot + nclick_q
    if(times != 0): 
        average_click = round(nclick_tot/times,1)
        #print(average_click)
        qstion_done = 1
    else:
        average_click = avg_click_all_qstion
        qstion_done = 0
    arr[0,0] = average_click
    arr[0,1] = qstion_done
    return arr
avg_nclick = average_click_answer(df_test,2)  
#print(avg_nclick)    

#average number of skips on each question by all users, for all the questions
def f4(x,nb_qstion):#input: dataframe containing group of tests, number of questions(1-5)
    tot_cr = 0
    arr=np.zeros((1,2))
    for i in range(0,x.shape[0]):#per tutte le righe
        #print(nb_qstion)
        #print(x.iloc[i,18:])
        cr = skipped_question(x.iloc[i,18:],nb_qstion)
        tot_cr = tot_cr + cr
    arr[0,0] = tot_cr
    arr[0,1] = x.shape[0]
    return arr

def average_skip_all_qstion(df_test):#input: dataframe containing allthe input data
    for column in range(0,10,2):
        arr1=np.zeros((1,2))
        list1 = df_test.groupby(df_test.iloc[:,column+8]).apply(lambda x: f4(x,(column//2)+1))
        #print(list1)
        for element in list1:
            arr1=np.concatenate((arr1,element),axis=0)
        arr1=arr1[1:,:]
        #print(arr1)
        list2 = df_test.iloc[:,column+8].unique()
        list2.sort()
        #print(list2)
        arr2 = np.asarray(list2).reshape(-1,1)
        if(column==0):
            arr = np.concatenate((arr2,arr1),axis=1)
        else:
            for i in range(0,arr2.shape[0]):
                for j in range(0,arr.shape[0]):                   
                    if(arr[j,0] == arr2[i,0]):
                        arr[j,1] = arr[j,1] + arr1[i,0]
                        arr[j,2] = arr[j,2] + arr1[i,1]
                        break
                    else:
                        if(j == arr.shape[0]-1):
                            arr_temp = np.concatenate((arr2[i].reshape(1,-1),arr1[i,:].reshape(1,-1)),axis=1)
                            arr = np.concatenate((arr,arr_temp),axis=0)
    #print(arr)
    for i in range(0,arr.shape[0]):
      arr[i,1] = round((arr[i,1]/arr[i,2])*100)
      #print(arr[i,1])
    avg = round(arr[:,1].mean()) 
    return avg,arr

avg_skip_all_qstion,arr = average_skip_all_qstion(df_test)
#print(avg_skip_all_qstion)
for i in range(0,arr.shape[0]):
    avg[arr[i,0],6:8] = arr[i,1:3]    
#print(avg[0,:])

#average number of skips on a answer
def average_skip_answer(df_test,question_index):#input: dataframe containing all the input data, index of the question(0-39)
    skip_tot = 0
    times = 0
    arr = np.zeros((1,2))
    data_session = df_test.iloc[:,8:] 
    for column in range(0,10,2):
        qst = data_session.loc[data_session.iloc[:,column] == question_index] #identifico righe che hanno question_index come domanda
        times = times + qst.shape[0]
        for row in range(0,qst.shape[0]):
            skip_q = skipped_question(qst.iloc[row,10:],(column//2)+1) 
            #print(skip_q)
            skip_tot = skip_tot + skip_q
    if(times != 0): 
        average_skip = round((skip_tot/times)*100)
        #print(average_click)
        qstion_done = 1
    else:
        average_skip = avg_skip_all_qstion
        qstion_done = 0
    arr[0,0] = average_skip
    arr[0,1] = qstion_done
    return arr
avg_skip = average_skip_answer(df_test,2)  
#print(avg_skip)   

#prepare data
def data_t(nb_qstion):#nb_qstion is the number of past previous answered questions of the test
    data_next_question = np.zeros((1,9), dtype=float)#data relevant to the next question
    data_previous_question = np.zeros((1,7*nb_qstion), dtype=float)#data relevant to past questions and answers
    n_features = (user_features_np.shape[1]-2) + data_previous_question.shape[1] + data_next_question.shape[1] + 2#total number of features
    #print(n_features)
    #-2 because i don't use name and mail as features
    X = np.zeros((1,n_features), dtype=float)
    Y = np.zeros((df_test.shape[0],1), dtype=int)
    for j in range(0,df_test.shape[0]):#for each row of the dataframe
       #print(j)
       test = df_test.iloc[j,:] 
       quest_answ_t = df_test.iloc[j,8:18]
       timestamp_t = df_test.iloc[j,18:]
       mail_t = df_test.iloc[j,3]
       #print(mail_t)
       user_data_temp = user_features_df.loc[user_features_df.iloc[:,0] == mail_t]
       print(mail_t)
       user_data_t = user_data_temp.iloc[:,1:-1].to_numpy() #user data, those from the informative questionnaire 
       #print(user_data_t.shape)
       past_average_score = correct_aswer_user(mail_t,j)
       for i in range(1,nb_qstion+1):
           data_previous_question[0,0+(7*(i-1))] = correct_answer(test,i) #previous question is correct?
           data_previous_question[0,1+(7*(i-1))] = skipped_question(timestamp_t, i) #previous question has been skipped?
           data_previous_question[0,2+(7*(i-1))] = round(time_question(timestamp_t,i),1) #time spent on previous question
           question_index = quest_answ_t.iloc[(i*2)-2]
           if(df.iloc[question_index,7] == 'advanced'):
               data_previous_question[0,3+(7*(i-1))] = 1#difficulty previous question: 1:advanced 0:basic
           else:
               data_previous_question[0,3+(7*(i-1))] = 0
           data_previous_question[0,4+(7*(i-1))] = nclick_question(timestamp_t,i)#number of clicks on previous question
           perc_cr_answers = perc_correct_ansewrs(question_index,quest_answ)#statistics difficulty of previous question 
           data_previous_question[0,5+(7*(i-1))] = perc_cr_answers[0,0] 
           data_previous_question[0,6+(7*(i-1))] = perc_cr_answers[0,1]#if the question has really been answered or not

       question_index_next = quest_answ_t.iloc[(nb_qstion*2)]#next question
       #question done:if the question has really been answered or not
       if(df.iloc[question_index_next,7] == 'advanced'):    
           data_next_question[0,0] = 1 #difficulty question: 1:advanced 0:basic
       else:
           data_next_question[0,0] = 0 
       perc_cr_answers = perc_correct_ansewrs(question_index_next,quest_answ)#statistics difficulty next question
       #print(perc_cr_answers)
       data_next_question[0,1] = perc_cr_answers[0,0] 
       data_next_question[0,2] = perc_cr_answers[0,1]#question done 
       average_time_answers = average_time(df_test,question_index_next)#average time spent on next question
       #print(average_time_answers)
       data_next_question[0,3] = average_time_answers[0,0] 
       data_next_question[0,4] = average_time_answers[0,1] #question done
       average_skip_answers = average_skip_answer(df_test,question_index_next)#average number of skips on next question
       #print(average_skip_answers)
       data_next_question[0,5] = average_skip_answers[0,0]
       data_next_question[0,6] = average_skip_answers[0,1]#question done
       average_click_answers = average_click_answer(df_test,question_index_next)#average number of skips on next question
       #print(average_click_answers)
       data_next_question[0,7] = average_click_answers[0,0]
       data_next_question[0,8] = average_click_answers[0,1]#question done
    
       x = np.concatenate((user_data_t[0,:].reshape(1,-1),past_average_score.reshape(1,-1),data_previous_question,data_next_question),axis=1)
       # print(x.shape)
       # print(user_data_t[0,:].reshape(1,-1).shape)
       # print(past_average_score.reshape(1,-1).shape)
       # print(data_previous_question.shape)
       # print(data_next_question.shape)
       
       y = correct_answer(test,t) #the target values: if the answer to next answer is correct(1) or not(0)
       X = np.concatenate((X,x),axis=0)
       Y[j] = y
    
    X = X[1:,:]
    
    #print(X.shape)
    #print(Y.shape)
    
    return X,Y

#Save the average values for each question (time,clicks,skips,percentage of correct answers) on a dataframe and then save it on a GoogleSheet called averages
#these values are used then in the online phase (prototype.py)
gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('averages')
avg_df = pd.DataFrame(data=avg, index=np.arange(avg.shape[0]), columns=np.arange(avg.shape[1]))#df for average values 
wks = sh[1]
wks.set_dataframe(avg_df,(1,1),copy_head=False)      
#print(avg)

#compute percentage of correct answers for each student, for all the students and save the dataframe again on a GoogleSheet
#these values are used then in the online phase (prototype.py)
user_answer = pd.DataFrame(0.0, index=np.arange(df_test.iloc[:,3].unique().shape[0]), columns=np.arange(3))
#print(user_answer.shape)
count=0
for mail in df_test.iloc[:,3].unique():
    user_answer.iloc[count,0] = mail
    v = correct_aswer_user(mail,df_test.shape[0])#number of correct answers for that mail on all the data input file
    user_answer.iloc[count,1] = v[0,0]
    count_mail=0
    for row in range(0,df_test.shape[0]):
        if(df_test.iloc[row,3] == mail):
            count_mail = count_mail+1
    user_answer.iloc[count,2] = count_mail*5 #total number of questions answered by the user
    count = count+1
wks = sh[0]
wks.set_dataframe(user_answer,(2,1),copy_head=False) 
#print(user_answer)
  
#TRAIN AND TEST OF THE MODEL
t=5 #number of the question to predict(2-5), question number 1 is chosen randomly because there's no past history of the current test
nb_qstion = t-1 #number of past answered questions
random_state=0
cancel_percentage=60
n_trees=900
X,Y = data_t(nb_qstion)#prepare the data
print(X.shape)
#print(X[0,:])
#divide data into train(90%) and test(10%)
X_l, X_test, y_l, y_test = train_test_split(X, Y, test_size=0.10, random_state=random_state)
#print(X_l[0,:])

#scale and encode data
numerical_indexes = []
categorical_indexes = []
for i in range(X_l.shape[1]):
    if(isinstance(X_l[0,i], str)):  
       categorical_indexes.append(i)
    else:
       numerical_indexes.append(i)  
#print(len(categorical_indexes))
# print(X_l[0,:])
#numerical data train(N_l) and test(N_t)
N_l = np.delete(X_l,categorical_indexes,1)
N_t = np.delete(X_test,categorical_indexes,1)
#categorical data train(C_l) and test(C_t)
C_l = np.delete(X_l,numerical_indexes,1)
C_t = np.delete(X_test,numerical_indexes,1)

#Encode categorical data: onehot econding 
enc = OneHotEncoder(categories='auto',sparse=False,handle_unknown='ignore')#mette none se non ha visto valore nel fit
C_l_encoded = enc.fit_transform(C_l)
with open('encoder_t{}.pickle'.format(t), 'wb') as f:#save the encoder
    pk.dump(enc, f)
C_t_encoded = enc.transform(C_t)
print(enc.categories_)
#print(C_l_encoded.shape)#each element of the one-hot vector is now a feature, with de-encoding you come bacj to the orignal values
# A = enc.inverse_transform(C_l_encoded) 
# print(A[0,:])
# print(C_l[0,:])

#Scale numerical data: MinMaxScaler between 0 and 1 
scaler = MinMaxScaler(feature_range=(0,1))
N_l_scaled = scaler.fit_transform(N_l)
with open('scaler_t{}.pickle'.format(t), 'wb') as f:
    pk.dump(scaler, f)
N_t_scaled = scaler.transform(N_t)
#print(N_t_scaled.shape)
#B = scaler.inverse_transform(N_t_scaled) #vedi che fa giusto! anche sotto per numeriche
# print(B[0,:])
# print(N_l[0,:])

#Final data
X_L = np.concatenate((N_l_scaled,C_l_encoded),axis=1)
X_T = np.concatenate((N_t_scaled,C_t_encoded),axis=1)
print(N_l_scaled.shape)
print(C_l_encoded.shape)

#RF model 
model = RandomForestClassifier(n_estimators = n_trees, criterion='entropy', random_state=random_state,max_features='sqrt')
#Compute features importance to delete the features which less influence the prediction
#divide train and validation set
X_tr, X_validation, y_tr, y_validation = train_test_split(X_L, y_l, test_size=0.15, random_state=random_state)
rf_fitted = model.fit(X_tr,y_tr.ravel())#train model
#compute importance on validation set
features_importance = permutation_importance(rf_fitted, X_validation,y_validation, n_repeats=4, random_state=0,scoring='accuracy')
importance = features_importance.importances_mean
feat_ordered = importance.argsort()[::-1]#features ordered according to importance score
print('Features:')
print(feat_ordered)
print('Importance:')
print(importance[feat_ordered])
feat_eliminate_train = feat_ordered[-round((feat_ordered.shape[0]/100)*cancel_percentage):]#delete the cancel_percentage percentage of the less important values 
#save the features to eliminate
with open('featuresToCancel_t{}.pickle'.format(t), 'wb') as f:
    pk.dump(feat_eliminate_train, f) 
X_L_new = np.delete(X_L, feat_eliminate_train, 1)#delete the non-important features in the complete train set(including validation)
#print(X_L_new.shape)
X_T_new = np.delete(X_T, feat_eliminate_train, 1)#delete the non-important features in the test set
#train the model on the new train dataset with the less important features removed
H = model.fit(X_L_new,y_l.ravel())
start = tm.time()
yp = model.predict(X_T_new)
end = tm.time()
#print(end-start)#time for prediction of the test set
acc = accuracy_score(y_test, yp) #accuracy classification score
print('Accuracy:{}'.format(acc))
#Save trained model
with open('rf_t{}.pickle'.format(t), 'wb') as f:
    pk.dump(H, f)

#calculate confusion matrix
CM = confusion_matrix(y_test, yp)
TN = CM[0][0] #true negatives
FN = CM[1][0] #false negatives
TP = CM[1][1] #true positives
FP = CM[0][1] #false positives
FPR = FP/(FP+TN) # Fall out or false positive rate
TPR = TP/(TP+FN) # Sensitivity, hit rate, recall, or true positive rate
ACC = (TP+TN)/(TP+TN+FP+FN)
print(FPR)
print(TPR)
print(ACC)
classNames = np.arange(0,2)
disp = ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=classNames)
disp.plot()
disp.figure_.savefig('C:/Users/giuli/Desktop/progettoEuropeo/algorithm/final_code_with_comments/cm_graph_t{}_r{}.png'.format(t,random_state), bbox_inches='tight')
disp.figure_.clf()
