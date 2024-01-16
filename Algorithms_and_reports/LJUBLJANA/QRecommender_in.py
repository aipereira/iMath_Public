# Methods of student question recommender

# NEW VERSION 3.5.2023

import numpy as np
import pandas as pd
import networkx as nx
import conceptMapGenerator_v02 as cmg



#%% Lodad data from Excel

# @brief Load data from the excel file: student history (all users), question data
# @par data_fn : filename
# @ret q_data : Pandas dataframe with question data
# @ret student_data : Pandas dataframe with students history (answers)



def load_data(data_fn = 'Imath_prototype_data.xlsx'):
    """
    load_data: Load data from the excel file: student history (all users), question data

    :param data_fn: filename, default is 'Imath_prototype_data.xlsx'
    :return: list [q_data, student_data] : Pandas dataframes with question data, student history
    """


    # Questions
    q_data = pd.read_excel(data_fn, sheet_name='Data_input', nrows=41)

    # Student history data, without column ids
    student_data = pd.read_excel(data_fn, sheet_name='Data_output')

    return [q_data, student_data]

# @brief Load data from the Google sheets
# @par sh : Google sheet
# @ret q_data : Pandas dataframe with question data
# @ret student_data : Pandas dataframe with students history (answers)
def load_data_from_gs(sh):
    """
    load_data_from_gs: Load data from Google sheets

    :param sh: Google sheet
    :return: list [q_data, student_data] : Pandas dataframes with question data, student history
    """

    data_input_wks = sh[0] # Sheet = 'Data_input'
    data_input_df=data_input_wks.get_as_df(has_header=True)
    q_data = data_input_df.iloc[:41, :]
    #q_data = pd.read_excel(data_fn, sheet_name='Data_input', nrows=41)

    # Student history data, without column ids
    data_output_wks = sh[4] # Sheet = 'partner_output_Ljubljana'
    student_data=data_output_wks.get_as_df(has_header=True)

    return [q_data, student_data]


# %% Test loading
#import pandas as pd
#[q_data, student_data] = load_data_from_gs()


# %%
# NOVA VERZIJA PARSER
# Spremenljivka, kateri ID je bil vprasanja v predhnodnem tokenu. Ce je -1, je bil predhodni ID odgovora in se nastavi vprasanje
# currentQId = -1
#

# @brief Parse text describing click action of the student in the web browser, to identify question Id and answer Id
def parse_token_2(token, currentQuestId):
    """
    parse_token_2: Parse text describing click action of the student in the web browser, to identify question Id and answer Id

    :param token: text info about the click (from system)
    :param currentQuestId: global variable current question id
    :return: cQId, ('qAnsw', questId, answId): new current question id, 'qAnsw' string, question ID and answer ID (student's answer)
    """

    cQId = currentQuestId

    #  preverja če je integer, prejšnja ni delovala s podatki
    if isinstance(token, (int, np.integer)):
        value = int(token)
        # Ali je ta token vprasanje
        if cQId < 0:
            cQId = value
            return cQId, ('qId', cQId)
        else :
            # to je answer id
            answId = value
            questId = cQId
            # Nastavi cQId nazaj na -1
            cQId = -1
            # Vrni podatek vprasanje, odgovor
            return cQId, ('qAnsw', questId, answId)


    if isinstance(token, str):
        token_lst = token.split('#')
        if len(token_lst) == 3:
            return cQId, ('tstamp', token_lst[0], token_lst[1], token_lst[2])
        if len(token_lst) == 1:
            if token_lst[0].isnumeric():
                value = int(token_lst[0])
                # Ali je ta token vprasanje
                if cQId < 0:
                    cQId = value
                    return cQId, ('qId', cQId)
                else : 
                    # to je answer id
                    answId = value
                    questId = cQId
                    # Nastavi cQId nazaj na -1
                    cQId = -1
                    # Vrni podatek vprasanje, odgovor
                    return cQId, ('qAnsw', questId, answId)
            else:
                return cQId, ('-1', -1)

    return cQId, -1

#%%
# NOV PARSER ENE VRSTICE UPORABNIKOVIH PODATKOV

# @brief Parse row of studnet data to obtain questions and answers
# @par @par output_row : only from question 1 forward
#
def parse_student_row(output_row, question_data, convert_tstamp = True):
    """
    parse_student_row: Parse text describing click action of the student in the web browser, to identify question Id and answer Id

    :param output_row: one row of data from the student history dataframe (for one student)
    :param question_data: Dataframe with questions and answers
    :param convert_tstamp: boolean, default is True, to convert timestamps into seconds
    :return: answ_list, answer_ts: list of answers data, list of answers timings. Answer list contains: [question index, answer index, correct anwer index, 
      correctnes of the answer (boolean)]. Example: [[0, 0, 3.0, False], [1, 3, 0.0, False]]
    """
    #print(output_row)
    answ_list = []
    answer_ts = []

    # Dictionary, key is integer corresponding to question
    startQ = dict()
    endQ = dict()

    currQId = -1

    # Scan raw
    laasQ_cr = '0'
    for curr_tk in output_row:

        #curr_tk_lst = parse_token(curr_tk)
        # Parse token
        currQId, curr_tk_lst = parse_token_2(curr_tk, currQId)

        # print(curr_tk_lst)
        # Error checking
        if curr_tk_lst == -1:
            #print('ERROR in ', curr_tk)
            continue

        if curr_tk_lst[0] == 'tstamp':
            # tip vprasanja
            ts_type = curr_tk_lst[1]
            # print(ts_type)
            t_sec = float(curr_tk_lst[3])
            if convert_tstamp == True:
                t_sec = float(curr_tk_lst[3])/1000.0
            # is integer
            #if isinstance(int(ts_type), (int, np.integer)):
            if ts_type.isnumeric():
                # print('   integer: ', int(ts_type))
                ts_value = int(ts_type)

                if curr_tk_lst[2] == 'login':
                    login_t = t_sec
                    #startQ[0] = t_sec
                if curr_tk_lst[2] == 'start_button':
                    start_t = t_sec
                    #startQ.append(t_sec)
                    startQ[0] = t_sec
                if curr_tk_lst[2] == 'nextButton':
                    start_t = t_sec
                    #startQ.append(t_sec)
                    startQ[ts_value] = t_sec
                if curr_tk_lst[2] == 'answer':
                    #print('answer', t_sec - start_t)
                    #answer_ts.append(t_sec - start_t)
                    #endQ.append(t_sec)
                    endQ[ts_value-1] = t_sec

        # Preveri pravilnost odgovora
        if curr_tk_lst[0] == 'qAnsw':

            #print(curr_tk_lst)
            q_ind = curr_tk_lst[1]
            answ_ind = curr_tk_lst[2]
            correct_ind = question_data['correct answer'][q_ind]
            #print('Correct : ', correct_ind, correct_ind==answ_ind)

            # output for each question
            q_out = [q_ind, answ_ind, correct_ind, correct_ind==answ_ind]

            answ_list.append(q_out)

    #print(startQ)
    #print(endQ)

    # Calculate answer times (delays)
    for i in range(min(len(endQ), len(startQ))):
        if len(endQ)>0 & len(startQ) > 0:
            answer_ts.append(endQ[i]-startQ[i])

    # Update all answers with answer times (delays)
    for answ_i in range(len(answ_list)):
        if answ_i < len(answer_ts):
            answ_list[answ_i].append(answer_ts[answ_i])

    return  answ_list, answer_ts



#%%

#%%
# ANALYSE STUDENT DATA

# @brief Create history of students answers from input data
def calc_history(input_data, question_data):
    """
    calc_history: Create history of students answers (all students) from input data

    :param input_data: Pandas dataframe containing student data (history of answers)
    :param question_data: Pandas dataframe with question data (answers, correct answer..)
    :return: hist, dictionary with email as user id, and list of answered question data
    """
    # Result dictionary
    hist = dict()

    for ind in input_data.index:

        # Id of the user in current row : email
        curr_id = input_data.iloc[ind, 3]

        # current row only answers
        curr_answ = input_data.iloc[ind][8:].to_list()

        # Parse answers
        answ_list, answer_delays = parse_student_row(curr_answ, question_data)


        # Find user Id (email) in output list
        if curr_id in hist:
            #print('Found')

            for answ in answ_list:
                hist[curr_id].append(answ)

        else:
            # new user, start a new list of answers
            hist[curr_id] = answ_list

    return hist




#%%
# Calculate completness score 
def calc_score(user_id, users_data, knowl_gr, debug = False):
    """
    calc_score: Calculate success scores for the user for each concept

    :param user_id: email of the student
    :param users_data: dictionary with history data of all users (generated by calc_history)
    :param knowl_gr: concept map (dictionary)
    :return: u_score dictionary, key is concept, value is [c_compl, c_corr, Qunt, Quns, Qsuc] : % answered questions of the concept, % of correct answers,
    number of not answered, number of incorrectly answered (unsuccessful attempts), number of successfully answered questions
    """

    # user history
    user_h  = []
    if user_id in users_data:
        user_h = users_data[user_id]


    # For each node in knowledge graph, calculate score
    u_score = dict()

    # Dictionary: concepts graph, corresponding question ids
    Cg = nx.get_node_attributes(knowl_gr, "Qs")
    # Concept sequence from concept map
    cSeq = cmg.get_conceptMap_sequence(knowl_gr, c_s=0, alg_code='weighted_BFS')

    conc_index = 0
    concList = list(cSeq.keys())
    conc = concList[conc_index]

    # Over all concepts:
    #for conc in Cg:
    while conc != None:

        Qs = Cg[conc]
        if debug==True:
            print('Concept ',conc, ' Qs:', Qs)
        Qtaken = []
        Qscore = []
        Quns = []   # unsuccessful questions (wrong answer)
        Qunt = []   # untaken questions
        Qsuc = []   # taken, successfully solved

        # Check all questions of this concept, if user has solved
        for q in Qs:
            # find in user data
            q_in_hist = False
            for ans in user_h:
                if ans[0]==q:
                    # question in user history
                    q_in_hist = True
                    Qtaken.append(q)
                    Qscore.append(ans[3])
                    if ans[3]==False:
                        Quns.append(q)
                    else:
                        Qsuc.append(q)
                #else:
                #    # question not found in user history
                #    if  Qunt.count(q)==0:
                #       Qunt.append(q)
            # question not yet answered
            if q_in_hist == False:
                Qunt.append(q)

        # array correct answers: 1/0 
        Qcorrect = np.int16(np.array(Qscore))

        if debug==True:
            print(Qcorrect)
        #print(Qs)

        # Indicators for concept: completed, correctness
        c_compl = 0.0
        if len(Qtaken) > 0:
            c_compl = len(Qtaken)/len(Qs)

        c_corr = 0.0
        if len(Qtaken) > 0:
            c_corr = np.sum(Qcorrect)/len(Qtaken)

        # user result for concept:
        uQunt = list(np.unique(Qunt))
        uQuns = list(np.unique(Quns))
        uQsuc = list(np.unique(Qsuc))
        u_conc = [c_compl, c_corr, uQunt, uQuns, uQsuc]
        u_score[conc] = u_conc

        if debug==True:
            print("Concept score: ", c_compl, ' ', c_corr )
        
        # find next concept
        if conc in cSeq.keys():
            conc = cSeq[conc]
        else:
            conc_index+=1
            if len(concList) > conc_index:
                conc = concList[conc_index]
            else:
                conc = None

    return u_score

# %%
# @brief Update users_data with new answers from output_row
# @par output_row trace from the system
# @ret 
def update_user_history(output_row, users_data, question_data, debug = False):
    """
    update_user_history: Updates users history object with new question answer data

    :param output_row: data of new clicks of the user
    :param users_data: dictionary with history data of all users (generated by calc_history)
    :param question_data: questions and answers
    :return: nothing
    """
    # user
    user_id = output_row[3]

    # user history
    user_h  = []
    if user_id in users_data:
        user_h = users_data[user_id]    

    if debug==True:
        print(user_h)

    # current row only answers
    curr_answ = output_row[8:]

    # Parse answers
    answ_list, answer_delays = parse_student_row(curr_answ, question_data)

    if debug==True:
        print("Parsed answers: ", answ_list)

    for answ in answ_list:
        # check if already in history 
        if answ not in user_h:
            user_h.append(answ)
    
    if debug==True:
        print("New history: ", user_h)

    # Update global user data
    users_data[user_id] = user_h


# @brief: get questions of this session
def get_this_sess_qs(output_row):
    sess_qs = []
    q_ind = 0
    while float(str(output_row[8+2*q_ind]).strip() or 0) >= 0:
        sess_qs.append(output_row[8+2*q_ind])
        q_ind += 1
    return sess_qs
    



#%%

# Recommendation function
# calculates completness score for all concepts in a map, from user history
# This includess list of  questions taken, and successfully solved, for each concept
# Current algorithm: If concept was completed less than 50% and Q success 
# was less than 50%, continue with this concept, with next unaswered Qs. Otherwise
# proceed to next concept. 


# @brief Define algorithm_for_new_question
# @par output_row trace from the system
# @ret next question id
def calc_next_question(output_row, users_data, question_data, knowl_graph, qrec_conf, debug = False):
    """
    calc_next_question: Main method to generate next recommended question for the student, within current session (output_row)
 Calculates completness score for all concepts in a map, from user history. This includess list of  questions taken, and successfully solved, for each concept
 Current algorithm: If concept was completed less than 50% and Q success was less than 50%, continue with this concept, with next unaswered Qs. Otherwise
 proceed to next concept. 

    :param output_row: data of clicks of the user
    :param users_data: dictionary with history data of all users (generated by calc_history)
    :param question_data: questions and answers
    :param knowl_graph: concept map dictionary
    :return: [q_next, conc] : next question id, concept

    qrec_conf['conc_compl_T'], qrec_conf['conc_succ_T'], qrec_conf['tot_compl_T'], qrec_conf['no_q_code']

    """
    # Update users history
    if len(output_row) > 0:
        update_user_history(output_row, users_data, question_data, debug)

    user_id = output_row[3]
    sess_qs = get_this_sess_qs(output_row) # Get this session qs

    # user score for each concept
    u_score = calc_score(user_id, users_data, knowl_graph)

    # Concept sequence from concept map
    cSeq = cmg.get_conceptMap_sequence(knowl_graph, c_s=0, alg_code='weighted_BFS')
    Cg = nx.get_node_attributes(knowl_graph, "Qs")

    if debug==True:
        print('calc_next_question user: ', user_id)
        print(u_score)

    # next question ID
    q_next = -1

    # last concept in user score
    last_conc = -1


    # ------------------------------------------------------------------------------
    # Att 0: if no history found, pose the first question
    #print(users_data[user_id])
    if len(users_data[user_id]) == 0:
        cList = list(cSeq.keys())
        first_conc = cList[0]
        if len(Cg[first_conc]) > 0:
            q_next = list(Cg[first_conc])[0]
            return [q_next, first_conc]



    # ----------------------------------------------------------------------------------
    # Att1:  check all already visite concepts in u_score for untaken Qs to find question
    for conc in u_score:

        last_conc = conc
        # data of previous attempts
        c_hist = u_score[conc]
        c_compl = c_hist[0]
        c_succ = c_hist[1]

        # conditions : concept is not completed, then find new Q in this concept
        if c_compl < qrec_conf['conc_compl_T'] or c_succ < qrec_conf['conc_succ_T']:
            # find next question from untaken questions
            if len(c_hist[2]) > 0:
                q_next = c_hist[2][0]
            else:
                # ALL Qs were taken for this concept
                # Version 2: Continue with next concept
                continue

                # FIRST VERSION: all Qs were taken
                # next q from unsuccessful attempts
                #if len(c_hist[3]) > 0:
                    # first from unsuccesfull
                #    q_next = c_hist[3][0]
                #else:
                #    if debug==True:
                #        print('Warning no next Q found')

            # if found, stop
            #if q_next >= 0:
            #    break
        else:
            # go to next concept
            if debug==True:
                print('Concept: ', conc, ' success: ', c_succ, ' completed: ', c_compl)

        # if next Q found, stop
        if q_next >= 0:
            #break
            return [q_next, conc]

    

    # ------------------------------------------------------------------------------
    # Att 2: If no new Q found for already visited concepts in user history, 
    # then 
    #   go to the next concept
    #   get first question from it
    if q_next < 0:
        # Find next concept from map
        cList = list(cSeq.keys())
        next_conc = -1
        # if there was last concept
        if last_conc in cList:
            next_conc = cSeq[last_conc]
            if next_conc != None:
                if len(Cg[next_conc]) > 0:
                    t_q_next = list(Cg[next_conc])[0]
                    if not t_q_next in sess_qs:
                        q_next = t_q_next
                        conc = next_conc
    
        if q_next >= 0: # Question found, so exit
            #break
            return [q_next, conc]

    if debug==True:
        print('Next Q found: ', q_next, ' for concept: ', conc)


    # ------------------------------------------------------------------------------
    # Att 3: If no question among unmastered questions found, 
    # then
    #   go to unsucessfull again if not posed in this session 
    for conc in u_score:

        last_conc = conc
        # data of previous attempts
        c_hist = u_score[conc]
        c_compl = c_hist[0]
        c_succ = c_hist[1]

        # conditions : concept is not completed, then find new Q in this concept
        if c_compl < qrec_conf['conc_compl_T'] or c_succ < qrec_conf['conc_succ_T']:
            # find next question from untaken questions
            if len(c_hist[2]) > 0:
                t_q_next = c_hist[3][0]
                if not t_q_next in sess_qs:
                    q_next = t_q_next

        if q_next >= 0: # Question found, so exit
            break
            


    # ------------------------------------------------------------------------------
    # Att 4: If no question 

    '''
    # if no concept and question found, no history of user
    # what is the first question ?
    if q_next < 0:
        q_next = 0
        conc = ''
        # get first concept from list
        cList = list(cSeq.keys())
        if len(cList)>0:
            conc = cList[0]
            # get first Q for this concept

            if len(Cg[conc]) > 0:
                q_next = Cg[conc][0]
            else:
                q_next = -1

        if debug==True:
            print("** ADDING Q: ", q_next, ', ', conc)

        #concepts = list(u_score.keys())
        #if len(concepts)> 0:
        #    conc = concepts[0]

    '''

    if q_next < 0: # No question available at this settingd
        q_next = qrec_conf['no_q_code']

    # return next question ID
    return [q_next, conc]

