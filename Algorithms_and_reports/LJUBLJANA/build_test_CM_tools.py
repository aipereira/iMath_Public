import numpy as np
import pandas as pd
import pickle
import networkx as nx

import pygsheets
import iMath_tools as imt
import QRecommender_in as qrec



# @brief load student history, questions, etc
def load_initial_state():

    gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
    sh = gc.open('Imath_prototype_data')
    

    # Student data
    student_wks = sh[4] # Sheet = 'partner_output_Ljubljana'
    student_data = student_wks.get_as_df(has_header=True)

    # Questions 
    questions_wks = sh[0] # Sheet = 'Data_input'
    q_data = questions_wks.get_as_df(has_header=True)[0:40]



    # Keywords and qs
    #df=wks.get_as_df(has_header=True)
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

    return q_data, student_data, qIDs_set, keys_to_qs_dc, sh





#%% Get mini profile
def get_mini_profile(user_id, student_data, q_data):

    all_course_qs = list(range(40))

    s_hist = qrec.calc_history(student_data, q_data)
    c_s_hist = s_hist[user_id]

    true_qs, false_qs, all_qs = [], [], []
    for q_hist in c_s_hist:
        if q_hist[3]:
            true_qs.append(q_hist[0])
        else:
            false_qs.append(q_hist[0])
        all_qs.append(q_hist[0])

    rest_qs = list(set(all_course_qs).difference(set(all_qs)))

    if len(all_qs) > 0:
        corr_q = len(true_qs) / len(all_qs) 
        cover_q = len(all_qs) / len(all_course_qs)

    return true_qs, false_qs, all_qs, rest_qs, corr_q, cover_q




# @brief: get repeated ids inside session
def get_repeat_inside_sess(student_data, printQ):
    scores_lst = []
    rep_inses_uIDs = []
    for ind, row in student_data.iterrows():
        uID = row[3]
        curr_qs = [row[8], row[10], row[12], row[14], row[16]]
        curr_unique_qs = np.unique(curr_qs)
        if len(curr_unique_qs) < 5:
            rep_inses_uIDs.append(uID)
            if printQ:
                print (uID, curr_qs)
    rep_inses_uIDs = list(np.unique(rep_inses_uIDs))

    return rep_inses_uIDs


# @brief print repeatences between sessions
def get_repeat_between_sess(student_data, printQ):

    uID_qs_dc = {}
    for ind, row in student_data.iterrows():
        uID = row[3]
        uID_qs_dc[uID] = []

    scores_lst = []
    for ind, row in student_data.iterrows():
        uID = row[3]
        curr_qs = [row[8], row[10], row[12], row[14], row[16]]
        uID_qs_dc[uID].extend(curr_qs)


    print ('\nPrint uIDs having repated qs:')
    rep_betses_uIDs = []
    for key in uID_qs_dc:
        full_qs_lst = uID_qs_dc[key]
        duplicates_d = [q for q in full_qs_lst if full_qs_lst.count(q) > 1]
        duplicates = list(np.unique(duplicates_d))
        unique_qs_lst = list(np.unique(full_qs_lst))
        if len(unique_qs_lst) < len(full_qs_lst):
            rep_betses_uIDs.append(key)
            if printQ:
                print (key, ':', duplicates)
    rep_betses_uIDs = list(np.unique(rep_betses_uIDs))  

    return rep_betses_uIDs



def load_conM(concept_map_path, topic_nm):
    ''' Load concept map
        Parameters
        Return
    '''
    conM = pickle.load(open(concept_map_path + topic_nm + '_conM.pickle', 'rb'))

    return conM
# Run
#concept_map_path = '01-ConceptMaps/'
#topic_nm = 'mat_comp'
#conM = load_conM(concept_map_path, topic_nm)



def save_conM(conM, concept_map_path, topic_nm):
    ''' Save concept map
        Parameters
        Return
    '''
    pickle.dump(conM, open(concept_map_path + topic_nm + '_conM.pickle', 'wb'))

    return 0
# Run
#concept_map_path = '01-ConceptMaps/'
#topic_nm = 'mat_comp'
#save_conM(conM, concept_map_path, topic_nm)


def print_conM(conM):
    conc_qs_dc = nx.get_node_attributes(conM, 'Qs')
    all_qs = []
    for c in conc_qs_dc:
        all_qs.extend(list(conc_qs_dc[c]))
        print (c, conc_qs_dc[c])
    all_unique_qs = list(np.unique(all_qs))

    return



# Generate anwser randomly
# @brief for q_id in 0 .. 39 get correct anwser 
def get_rand_anwser(q_id, corr_qansws_dc, p_corr):

    anws_lst = [0, 1, 2, 3]
    corr_qan = corr_qansws_dc[q_id]
    probabs = [p_corr if al == corr_qan else (1-p_corr)/3.0 for al in anws_lst]
    answ = np.random.choice(anws_lst, 1, p=probabs)[0]
    
    return answ

# register student
def register_student(inst_id, email_id):

    return


# @brief generate student raw
def get_start_output_row(student_data, student_id):

    last_output_row = list(student_data[student_data.iloc[:, 3]==student_id].iloc[-1, :]) # ['Marko', 'Medved', 'Ljubljana', 'mm6985@student.uni-lj.si', '2', '2', '1', '1', '31', '2', '2', '2', '25', '2', '33', '3', '32', '1', '-2#login#1671890495929', '0#start_button#1671890537407', '1#skipButton#1671890817657', '2#nextButton#1671891007983', '3#nextButton#1671891151453', '4#nextButton#1671891206302', '5#nextButton#1671891238891', '6#submitButton#1671891243714', '6#previuosButton#1671891258720', '5#previuosButton#1671891260826', '4#previuosButton#1671891261512', '3#previuosButton#1671891262255', '2#previuosButton#1671891263623', '1#nextButton#1671891748290', '2#nextButton#1671891749370', '3#nextButton#1671891750066', '4#nextButton#1671891750747', '5#nextButton#1671891751629', '6#submitButton#1671891752754', '7#evalbutton#1671891757878', 'e1#nextButton#1671891762714', 'e2#nextButton#1671891768684', 'e3#nextButton#1671891773339', 'e4#nextButton#1671891779635', '2#answer#1671891007387', '3#answer#1671891150619', '4#answer#1671891204098', '5#answer#1671891227148', '5#answer#1671891259137', '4#answer#1671891261165', '3#answer#1671891261798', '2#answer#1671891262579', '1#answer#1671891501488', '1#answer#1671891745175', '2#answer#1671891748640', '3#answer#1671891749695', '4#answer#1671891750389', '5#answer#1671891751021', 'e1#answer#1671891762356', 'e2#answer#1671891768237', 'e3#answer#1671891771755', 'e4#answer#1671891779265'] # Get last output_row
    output_row = last_output_row[0:8] + 10*[-1]
    return output_row


# @brief Update output row
def update_output_row(sess_q_id, output_row, next_question, anws):
    updated_output_row = output_row
    updated_output_row[2*sess_q_id+8] = next_question
    updated_output_row[2*sess_q_id+9] = anws 

    return updated_output_row


# @brief Save to google sheet as a last row
def save_to_google_sheets(output_row):
    gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
    sh = gc.open('Imath_prototype_data')
    wks = sh[4]
    wks.add_rows(1)
    final_row = pd.DataFrame([output_row])
    #print('final_row')
    #print(final_row)
    wks.set_dataframe(final_row,(wks.rows,1),copy_head=False)


# @brief Delete last num_of_raws rows
def clean_gdrive_rows(num_of_raws=-1):

    gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
    sh = gc.open('Imath_prototype_data')
    wks = sh[4]
    rows_n = wks.rows

    if num_of_raws == -1:    
        wks.delete_rows(rows_n)
        
    if num_of_raws > 0:
        wks.delete_rows(rows_n-num_of_raws+1, num_of_raws)

    return 0