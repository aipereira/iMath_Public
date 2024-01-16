#%% 
import pandas as pd
import numpy as np
import random
import time
import importlib
import networkx as nx
import pygsheets

import iMath_tools as imt
import conceptMapGenerator_v02 as cmg
import QRecommender_in as qrec

import build_test_CM_tools as btcm  

gc = pygsheets.authorize(service_file='keyGoogleSheet.json')
sh = gc.open('Imath_prototype_data')


#%% Rebuild concept map ================================================================================

importlib.reload(imt)
importlib.reload(btcm)

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



# %%
alg_code = 'rel_shift'
kws_conM = cmg.get_conceptMap_from_keyQs(qIDs_set, keys_to_qs_dc, alg_code, pars=[0.0])
conM = kws_conM

if True:
    concept_map_path = '01-ConceptMaps/'
    topic_nm = 'mat_comp'
    btcm.save_conM(conM, concept_map_path, topic_nm)



# %% Test concept map: 

# Load concept map
concept_map_path = '01-ConceptMaps/'
topic_nm = 'mat_comp'
conM = btcm.load_conM(concept_map_path, topic_nm)



conc_qs_dc = nx.get_node_attributes(conM, 'Qs')
covered_qs = []
for c in conc_qs_dc:
    covered_qs.extend(list(conc_qs_dc[c]))
    print (conc_qs_dc[c])
covered_unique_qs = list(np.unique(covered_qs))

print('Covered:', len(covered_qs), len(covered_unique_qs))


# %% Test concept map sequence
importlib.reload(cmg)

# Generate sequence
cSeq = cmg.get_conceptMap_sequence(conM, c_s=0, alg_code='weighted_BFS')
c0 = list(cSeq.keys())[0]

# Test sequence
conc_qs_dc = nx.get_node_attributes(conM, 'Qs')
covered_qs = []
ii = 0
curr_c = c0
while ii < len(cSeq.keys()):
    
    if curr_c != None:
        covered_qs.extend(list(conc_qs_dc[curr_c]))
        print (curr_c, conc_qs_dc[curr_c])
        curr_c = cSeq[curr_c]
    ii += 1

if curr_c != None:
    covered_qs.extend(list(conc_qs_dc[curr_c]))

covered_unique_qs = list(np.unique(covered_qs))

print('Covered:', len(covered_qs), len(covered_unique_qs))

# Walk and test



# %%
