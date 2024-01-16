# iMath tools
'''
iMath tools: functions supporting automatic concept map generation including evaluation tools such as 
    visualisation. 
Author: andrej.kosir@fe.uni-lj.si
'''
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 




# -----------------------------------------------------------------------------------------
def plot_conceptMap(conM, label_code='all'):
    '''
    Produces a plot of a concept map with sized concepts (nodes) and listed questions.
        Parameters: 
            conM: conceptual map as a directed networkx graph labeled by questions.
            label_code: how to label nodes
                'all': plot all question idnexes 
                'keys_only': plot only concept keys if available
        Returns:
            ax: axis handle of the plot
            the plot itself 
    '''

    qs_attrs = nx.get_node_attributes(conM, 'Qs') 
    if label_code == 'all':
        labels = {l: (l, qs_attrs[l]) for l in qs_attrs}
    elif label_code == 'keys_only':    
        labels = {l: l for l in qs_attrs}
    
    
    CM_node_attrs = nx.get_node_attributes(conM, 'Qs')
    node_sizes = [1000*len(CM_node_attrs[u]) for u in CM_node_attrs]

    nx.draw(conM, labels=labels, node_size=node_sizes, pos=nx.spring_layout(conM), with_labels=True, verticalalignment='top')
    #nx.draw_networkx_edge_labels(conM, pos=nx.spring_layout(conM))
    ax = plt.gca()
    ax.set_title(conM.name)
    plt.show()

    return ax


# -----------------------------------------------------------------------------------------
def get_question_IDs(qs_df):
    '''
    Returns a generated question ids, by default a set of natural numbers starting by 1
        Parameters: 
            qs_df: pandas dataframe of questions
        Returns:
            question ids as a Python set
    '''

    n = len(qs_df)

    return set(range(0, n))


# -----------------------------------------------------------------------------------------
def get_qs_to_kwsIDs_dc(X_qs_df):
    '''
    Returns dictionary of concept keywords keyed by question indexes
        Parameters: 
            X_qs_df: pandas dataframe of questions and their keys
        Returns:
            qs_kwsIDs_dc: dictionary of concept kewords keyed by questions
    '''
    
    qs_kwsIDs_dc = {}
    qIDs = get_question_IDs(X_qs_df['question'])
    for qID in qIDs:
        c_df = X_qs_df.loc[qID][12:].dropna()
        # print ('c_df', c_df)
        qs_kwsIDs_dc[qID] = set(c_df) #np.delete(c_df, np.where(c_df == '')))

    return qs_kwsIDs_dc



# -----------------------------------------------------------------------------------------
def get_kws_to_qs_dc(kw_dc, qs_kwsIDs_dc, code='All'):
    '''
    Returns a dictionary of questions keyed by concept keywords - questions assigned to keywords. 
        Parameters: 
            kw_dc: dictionary of concept keywords
            qs_kwsIDs_dc: dictionary of questions to concpet keywords
            code: 
                'All': use all keywords
                'FirstOnly': use first keyword only
                'FirstTwo': use frist two keywords
                'FirstThree': use frist thre keywords
        Returns:
            
    '''

    kws_qs_dc = {}

    if code == 'All':     
        for kws in kw_dc.keys():
            kws_qs_dc[kws] = {q for q in qs_kwsIDs_dc if kws in qs_kwsIDs_dc[q]}

    if code == 'FirstOnly': 
        qs_kwsIDs_reduced_dc = {q:{list(qs_kwsIDs_dc[q])[0]} for q in qs_kwsIDs_dc}
        for kws in kw_dc.keys():
            kws_qs_dc[kws] = {q for q in qs_kwsIDs_reduced_dc if kws in qs_kwsIDs_reduced_dc[q]}

    if code == 'FirstTwo': 
        qs_kwsIDs_reduced_dc = {q:set(list(qs_kwsIDs_dc[q])[0:2]) for q in qs_kwsIDs_dc}
        for kws in kw_dc.keys():
            kws_qs_dc[kws] = {q for q in qs_kwsIDs_reduced_dc if kws in qs_kwsIDs_reduced_dc[q]}

    if code == 'FirstThree': 
        qs_kwsIDs_reduced_dc = {q:set(list(qs_kwsIDs_dc[q])[0:3]) for q in qs_kwsIDs_dc}
        for kws in kw_dc.keys():
            kws_qs_dc[kws] = {q for q in qs_kwsIDs_reduced_dc if kws in qs_kwsIDs_reduced_dc[q]}
    
    return kws_qs_dc
# kws_qs_dc = get_kws_to_qs_dc(kw_dc, qs_kws_dc):
# print(kws_qs_dc)


# -----------------------------------------------------------------------------------------
def get_keys_to_qs_from_keywords(kws_to_qs_dc):
    '''
    Returns disjoint nodes from key-qs relationship. This may be from keywords or from anyhewe. 
        Parameters: 
            kws_to_qs_dc: kewords to questions dictionary
        Returns:
            keys to questions dictionary 

    Assumptions
        1. Options for grouping:
            1.1. Questions having at least one keyword in common 
            1.2. Questions having first keyword in comon
            1.3. Questions having at least qT keywords in common, for instance qT = 2
    '''

    # Key containing the first question
    min_qs_np = np.array([min(kws_to_qs_dc[kw]) if not len(kws_to_qs_dc[kw])==0 else -1 for kw in kws_to_qs_dc])
    min_q_ind = min(min_qs_np[min_qs_np>0])
    keys_with_first_q_lst = [kw if min_q_ind in kws_to_qs_dc[kw] else -1 for kw in kws_to_qs_dc]
    first_q_key = keys_with_first_q_lst[np.argmax(keys_with_first_q_lst)]
    
    # Scan all keys and progressively do nodes
    keys_to_qs = {}
    the_rest_qs = set.union(*[kws_to_qs_dc[kw] for kw in kws_to_qs_dc]) # All at this stage
    #the_rest_qs_qs = set.union(*[kws_to_qs_dc[kw] for kw in kws_to_qs_dc if not (kw == key)]) 
    already_assigned_qs = set({})
    for key in kws_to_qs_dc.keys():
        # print (key)

        # This key questions
        this_key_qs = kws_to_qs_dc[key] - already_assigned_qs 
        
        # What to assign here and what in the rest
        this_key_assigned_qs = this_key_qs

        # Update the rest
        the_rest_qs = the_rest_qs - this_key_assigned_qs
        already_assigned_qs = set.union(already_assigned_qs, this_key_assigned_qs)
        #print ('this_key_qs: ', this_key_qs)
        #print ('the_rest_qs', the_rest_qs)

        #print ('len(this_diff_key_qs): ', this_diff_key_qs, len(this_diff_key_qs))
        if len(this_key_assigned_qs) > 0:
            keys_to_qs[key] = this_key_assigned_qs
            #node_labels[key] = {"Qs": {key: this_diff_key_qs}}
        #else: # Do not add empty concepts
        #    keys_to_qs[key] = {} 

    return keys_to_qs



# -----------------------------------------------------------------------------------------
#% 2010 chen concept map
# @brief Get questions users matrix
def get_qs_users_grade_mat(X_qs_df, X_stud_hist_df):
    '''
    Returns users grade matrix according to the 2010 Chen and 2013 Chen papers
        Parameters: 
            X_qs_df: dataframe of questions
            X_stud_hist_df: student history in terms of anwsered questions
        Returns:
            qs_users_grade_mat_df: user grade matrix as a pandas dataframe
    '''

    # df of anwsers
    id_anwsers_cols = [3, 16, 17]
    anws_X_df = X_stud_hist_df.iloc[:, id_anwsers_cols]
    anws_X_df.columns = ['uID', 'qID', 'aID']
    anws_X_df.set_index('uID', inplace=True)
    
    # df of questions
    X_qIDs_df = pd.DataFrame(columns=['qID', 'correct answer'])
    X_qIDs_df['correct_answer'] = X_qs_df['correct answer'][0:40]
    X_qIDs_df['qID'] = range(1, 41)
    X_qIDs_df.set_index('qID')
    X_qIDs_dc = dict(zip(X_qIDs_df.qID, X_qIDs_df.correct_answer))
    #X_qIDs_dc = X_qIDs_df.to_dict()
    
    # Compose matrix
    unique_uIDs = np.unique(anws_X_df.index)
    unique_qIDs = np.unique(list(X_qIDs_dc.keys()))
    
    qs_users_grade_mat_df = pd.DataFrame(index=unique_qIDs, columns=unique_uIDs)
    qs_users_grade_mat_df[:] = 0.0
    for uID in unique_uIDs:
        curr_anwsr_np = np.array(anws_X_df[anws_X_df.index == uID])
        for row in curr_anwsr_np:
            qID, aID = row[0], row[1]
            if qID > 0:
                if X_qIDs_dc[qID]==aID: 
                    qs_users_grade_mat_df.at[qID, uID] = 1.0
        
    return qs_users_grade_mat_df



# -----------------------------------------------------------------------------------------
def get_qq_confidence_mat(qs_users_grade_mat_df):
    '''
    Returns qq matrix according to the 2010 Chen and 2013 Chen papers
        Parameters: 
            qs_users_grade_mat_df: user grade matrix as a pandas dataframe
        Returns:
            qq_confidence_mat: qq confidence matrix as a pandas dataframe
    '''

    qq_confidence_mat_df = pd.DataFrame(index=qs_users_grade_mat_df.index, columns=qs_users_grade_mat_df.index)
    qq_confidence_mat_df[:] = 0.0
    sup_qi = qs_users_grade_mat_df.sum(axis=1)
    for qID_i in qs_users_grade_mat_df.index:
        if sup_qi[qID_i] > 0:
            for qID_j in qs_users_grade_mat_df.index:
                #print (qID_i, qID_j)
                sup_qij = sum(qs_users_grade_mat_df.loc[qID_i] * qs_users_grade_mat_df.loc[qID_j])        
                qq_confidence_mat_df.at[qID_i, qID_j] = sup_qij / sup_qi[qID_i]

    return qq_confidence_mat_df


# -----------------------------------------------------------------------------------------
# @brief 
# We assume:
# - concepts are keys. At first, keys are keywords
# - relevance is based on TF.IDF principle:
#   - TF = # times in givem concept this q appears  / # all appearances: ussually 0 or 1
#   - IDF = 1 / # how many concepts this q appears
def get_qs_concepts_relevance_mat(kws_to_qs_dc, qs_kwsIDs_dc, qq_confidence_mat_df):
    '''
    Returns questions concepts relevance matrix according to the 2010 Chen and 2013 Chen papers
        Parameters: 
            kws_to_qs_dc: kewords to questions dictionary
            qs_kwsIDs_dc: questions to kewords dictionary 
            qq_confidence_mat_df: qq confidence matrix as a pandas dataframe
        Returns:
            questions concepts relevance matrix
    '''

    concepts_lst = kws_to_qs_dc.keys()   # Concepts are keys (keywords at first)
    questions_lst = qq_confidence_mat_df.index # Qestion codes

    qs_concepts_relevance_mat_df = pd.DataFrame(index=questions_lst, columns=concepts_lst)
    N = len(concepts_lst) # Number of documents = number of concepts
    for qID in questions_lst:
        num_of_appears = sum([qID in kws_to_qs_dc[cIDi] for cIDi in concepts_lst])
        if num_of_appears > 0:
            TF = 1.0 / num_of_appears # In each concept only a single appearance
        else:
            TF = 0.0
        for cID in concepts_lst:
            if qID in kws_to_qs_dc[cID]:
                if num_of_appears > 0:
                    IDF = np.log(N / num_of_appears)
                else:
                    IDF = 0
            #print (qID, cID)


            qs_concepts_relevance_mat_df.at[qID, cID] = TF*IDF

    return qs_concepts_relevance_mat_df


# -----------------------------------------------------------------------------------------
# @brief concept to concept relevance matrix
def get_concept_concept_relevance_mat(qs_concepts_relevance_mat_df, qq_confidence_mat_df, est_code='simple'):
    '''
    Returns concept concept relevance matrix according to the 2010 Chen and 2013 Chen papers
        Parameters: 
            qs_concepts_relevance_mat_df: 
            qq_confidence_mat_df: 
            est_code:
                'simple': simple multiplication
                'q_conf': based on question confidence
        Returns:
            concept concept relevance matrix as a pandas dataframe
    '''

    concept_concept_relevance_mat_df = pd.DataFrame(index=qs_concepts_relevance_mat_df.columns, columns=qs_concepts_relevance_mat_df.columns)

    if est_code == 'simple':
        concept_concept_relevance_mat_df = qs_concepts_relevance_mat_df.T @ qs_concepts_relevance_mat_df
    
    if est_code == 'q_conf':
        for cID_i in concept_concept_relevance_mat_df:
            for cID_j in concept_concept_relevance_mat_df:    
                ss = 0
                for qi in qq_confidence_mat_df:
                    for qj in qq_confidence_mat_df:
                        ss += min(qs_concepts_relevance_mat_df.at[qi, cID_i], qs_concepts_relevance_mat_df.at[qj, cID_j]) * qq_confidence_mat_df.at[qi, qj]
                concept_concept_relevance_mat_df.at[cID_i, cID_j] = ss

    return concept_concept_relevance_mat_df


# -----------------------------------------------------------------------------------------
# @brief get graph from relation matrix by thresholding adjacency matrix
# @par rel_mat_df
# @par p_T threshold
# @par keys_to_qs_dc, 
# @par lb_up_code 'lb' to get OK to OK and and 'ub' to get wrong to wrong
def get_conceptMap_from_relMat(rel_mat_df, keys_to_qs_dc, p_T, lb_ub_code='lb'):
    '''
    Returns graph from relation matrix by thresholding adjacency matrix
        Parameters: 
            rel_mat_df: relevance matrix
            p_T: threshold
            keys_to_qs_dc: keywords to questions dictionary
            lb_up_code: 
                'lb': to get OK to OK 
                'ub': to get wrong to wrong
        Returns:
            
    '''

    # Remove diagonals
    np.fill_diagonal(rel_mat_df.values, 0)


    # Threshold it
    if lb_ub_code == 'lb':
        max_w = np.max(rel_mat_df.values)
        lb_T = p_T * max_w
        rel_mat_T_df = rel_mat_df.mask(rel_mat_df < lb_T , 0).astype(float)
    if lb_ub_code == 'ub':
        min_w = np.min(rel_mat_df.values)
        ub_T = (1.0+p_T) * min_w
        rel_mat_T_df = rel_mat_df.mask(rel_mat_df < ub_T , 0).astype(float)


    # Create graph
    conM = nx.from_pandas_adjacency(rel_mat_T_df, create_using=nx.MultiDiGraph)
    conM.name = 'From adjac. matrix'
    #print ('conM 1', conM.nodes)

    # Set questions
    node_attrs = {u: {'Qs':keys_to_qs_dc[u]} for u in keys_to_qs_dc}
    nx.set_node_attributes(conM, node_attrs)
    #print (node_attrs)

    #print ('conM 2', conM.nodes)

    # Set weights
    #edge_attrs = {(u,v,0): {'w': rel_mat_T_df.at[u,v]} for u,v in rel_mat_T_df} # First edge
    edge_attrs = {}
    for u in rel_mat_T_df.index:
        for v in rel_mat_T_df.columns:
            edge_attrs[(u,v,0)] = {'w': rel_mat_T_df.at[u,v]}
    nx.set_edge_attributes(conM, edge_attrs)
    #print ('conM 3', conM.nodes)

    return conM



# -----------------------------------------------------------------------------------------
def get_merged_concept_qs(c_qs, c_a_qs, alg_code='union'):
    '''
    Returns new node from two candidate concept nodes.
        Parameters: 
            c_qs: questions of the original concept map
            c_a_qs: questions of the added concept map
            alg_code:
                'union': union
                'primary_only': primary only
            
        Returns:
    Note: a sequence of nodes should be implemented
    '''

    if alg_code=='union':
        m_cqs = c_qs.union(c_a_qs)
    elif alg_code=='primary_only':
        m_cqs = c_qs
    
    return m_cqs