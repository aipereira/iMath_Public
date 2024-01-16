# Algorithm principles and outline
# 1. Generate digraph of keywords with natural order used;
# 
# 
# 

#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import networkx as nx
import iMath_tools as imt
import pickle







# -----------------------------------------------------------------------------------------
def get_conceptMap_from_keyQs(qIDs_set, keys_to_qs_dc, alg_code, pars=[]):
    '''
    Returns concept map from keys (keywords or not)
        Parameters: 
            qIDs_set: set of question ids
            keys_to_qs_dc: kewords to questions dictionary
            alg_code: which algorithm to use
                'min_q': miminal intersection
                'rel_shift': how far two question sets are

            pars=[]: any parameter used in the concept map generation. pars[0] is a lower bound of the link to be added 
        Returns:
            
    '''

    # Create graph
    conM = nx.MultiDiGraph(name='From Keywords')
    
    
    # Add nodes 
    conM.add_nodes_from(keys_to_qs_dc)
    node_attrs = {v: {'Qs': keys_to_qs_dc[v]} for v in keys_to_qs_dc} # Creat node attributes
    nx.set_node_attributes(conM, node_attrs)

    # Add edges
    if alg_code == 'min_q':
        w_def = 0.6
        min_q_inds = np.argsort([min(keys_to_qs_dc[v]) for v in keys_to_qs_dc])
        keys_ordered_np = np.array(list(keys_to_qs_dc.keys()))[min_q_inds]
        edges = [(keys_ordered_np[ii], keys_ordered_np[ii+1], w_def) for ii in range(len(keys_ordered_np)-1)]
        conM.add_edges_from((u, v, {"w": w}) for u, v, w in edges)

    if alg_code == 'rel_shift':
        weight_T = pars[0]
        qs_node_attrs = nx.get_node_attributes(conM, 'Qs')
        n_n = len(qs_node_attrs)

        # Get limits
        u_v_lims = {}
        for u in qs_node_attrs:
            for v in qs_node_attrs:
                if len(qs_node_attrs[u]) > 0:
                    u_m, u_M = min(qs_node_attrs[u]), max(qs_node_attrs[u])
                else: 
                    u_m, u_M = 0, 0
                if len(qs_node_attrs[v]) > 0:
                    v_m, v_M = min(qs_node_attrs[v]), max(qs_node_attrs[v])
                else: 
                    v_m, v_M = 0, 0
                u_v_lims[(u,v)] = [u_m, u_M, v_m, v_M]

        # add links 
        edges_dc = {}
        for (u,v) in u_v_lims:
            mu, Mu, mv, Mv = u_v_lims[u, v]
            w_uv = 0.0 # Default is 0
            if Mu < mv: # all u before v
                w_uv = 1.0    
            if mu > Mv: # all u after v
                w_uv = 0.0
            if (mu > mv) and (Mu < Mv): # u included in v
                w_uv = 1.0
            if (mu < mv) and (Mu < Mv): # v is shifted u and comes after
                w_uv = 1.0 - max(0, Mu-mv)/(Mv-mu)

            if w_uv > weight_T:
                edges_dc[(u,v)] = w_uv



        weighted_edges_lst = [(u,v,edges_dc[(u,v)]) for u,v in edges_dc]
        conM.add_weighted_edges_from(weighted_edges_lst)
        
        edge_attrs = {(u,v,0): {'w': edges_dc[(u,v)]} for u,v in edges_dc} # First edge
        nx.set_edge_attributes(conM, edge_attrs)

    return conM # , edge_attrs #, nodes, edges, node_labels









# -----------------------------------------------------------------------------------------
# @brief Get trivial startup concept map 
def get_startup_conceptMap(set_of_qs_lst, keys_to_qs_dc, alg_code):
    '''
    Returns trivial startup concept map 
        Parameters: 
            set_of_qs_lst: list of questions
            keys_to_qs_dc: keywords to questions dictionary
            alg_code:
        Returns:
            'group_qs': simply group questions by 5 each group
            'from_keys': use keys to group them
    '''

    if alg_code == 'group_qs':
        
        # Compose names and labels
        con_names = {}
        node_qs = {}
        conc_core = 'C_'
        num_of_qq_in_con = 5
        num_of_qs = len(set_of_qs_lst)
        n_of_cons = int(num_of_qs/num_of_qq_in_con)
        for ii in range(n_of_cons):
            con_names[ii] = conc_core + str(ii)
            node_qs[con_names[ii]] = set_of_qs_lst[ii*num_of_qq_in_con:ii*num_of_qq_in_con+5]
        last_qs = set_of_qs_lst[n_of_cons*num_of_qq_in_con:]
        if len(last_qs) > 0:
            con_names[n_of_cons] = conc_core + str(n_of_cons)
            node_qs[con_names[n_of_cons]] = last_qs

        # Node attributes 
        node_attrs = {v: {'Qs': set(node_qs[v])} for v in node_qs}

        # Edge pairs and attributes
        concepts_lst = list(node_qs.keys())
        edge_pairs = []
        edge_attrs = {}
        for ii in range(len(concepts_lst)-1):
            cID_i, cID_j = concepts_lst[ii], concepts_lst[ii+1]
            edge_attrs[(cID_i,cID_j,0)] = {'w': 1.0}
            edge_pairs.append((cID_i, cID_j))

        conM = nx.MultiDiGraph(incoming_graph_data=edge_pairs, name='Startup')
        nx.set_node_attributes(conM, node_attrs)
        nx.set_edge_attributes(conM, edge_attrs)

    elif alg_code == 'from_keys':
        concepts_lst = list(keys_to_qs_dc.keys())

        # node attributes
        node_attrs = {v: {'Qs': keys_to_qs_dc[v]} for v in keys_to_qs_dc}

        # Edge pairs and attributes
        edge_pairs = []
        edge_attrs = {}
        for ii in range(len(concepts_lst)-1):
            cID_i, cID_j = concepts_lst[ii], concepts_lst[ii+1]
            edge_attrs[(cID_i,cID_j,0)] = {'w': 1.0}
            edge_pairs.append((cID_i, cID_j))

        conM = nx.MultiDiGraph(incoming_graph_data=edge_pairs, name='Startup')
        nx.set_node_attributes(conM, node_attrs)
        nx.set_edge_attributes(conM, edge_attrs)
    else:
        conM = nx.MultiDiGraph(name='Startup')

    return conM






# -----------------------------------------------------------------------------------------
#% Merge function
# @brief merge two concept maps
# @par conMa_w merging weight
def merge_conceptMaps(conM, conM_a, merge_w):
    '''
    Returns merged concept maps
        Parameters: 
            conM: first concept map
            conM_a: second concept map
            merge_w: merging weight: small resulst in mainly conM, close to 1 equal weight
        Returns:
            merged concept map
    '''

    # Create merging concept
    merged_conM = nx.MultiDiGraph(name='Merged: ' + conM.name + ' and ' + conM_a.name + '.')

    conM_qs = nx.get_node_attributes(conM, 'Qs')
    conM_ws = nx.get_edge_attributes(conM, 'w')
    conM_a_qs = nx.get_node_attributes(conM_a, 'Qs')
    conM_a_ws = nx.get_edge_attributes(conM_a, 'w')

    #print (conM_qs)
    #print (conM_a_qs)

    # Set 'visited' property to False
    nx.set_node_attributes(conM, False, 'v')
    nx.set_edge_attributes(conM, False, 'v')
    nx.set_node_attributes(conM_a, False, 'v')
    nx.set_edge_attributes(conM_a, False, 'v')

    # Concept naming
    conc_core = 'C_'
    conc_count = 0
    
    for c in conM.nodes: # Scan original cM
        if not nx.get_node_attributes(conM, 'v')[c]:

            # Add if isolated node
            if len(list(conM.neighbors(c))) == 0:
                m_c_name = conc_core + str(conc_count)
                conc_count += 1
                merged_conM.add_node(m_c_name)
                m_cqs = conM_qs[c]
                nx.set_node_attributes(merged_conM, {m_c_name: {'Qs': m_cqs}})

            for c_a in conM_a.nodes: # Scan merging cM for intersections
                if not nx.get_node_attributes(conM_a, 'v')[c_a]:
                    #print ('conM_qs:', conM_qs.keys(), ',  conM_a_qs: ', conM_a_qs.keys())
                    c_qs, c_a_qs = conM_qs[c], conM_a_qs[c_a] # intersection candidates
                    if c_qs.intersection(c_a_qs): # If not empty intersection

                        # Label as visited
                        nx.set_node_attributes(conM, {c:{'v': True}})
                        nx.set_node_attributes(conM_a, {c_a:{'v': True}})

                        # Get and add merged concept
                        m_c_name = conc_core + str(conc_count)
                        conc_count += 1
                        m_cqs = imt.get_merged_concept_qs(c_qs, c_a_qs)
                        merged_conM.add_node(m_c_name)
                        nx.set_node_attributes(merged_conM, {m_c_name: {'Qs': m_cqs}})
                        
                        for nc in conM.neighbors(c): # Scan neighbours of c
                            for nc_a in conM_a.neighbors(c_a): # Scan neighbors of c_a
                                
                                nc_qs, nc_a_qs = conM_qs[nc], conM_a_qs[nc_a] # intersection candidates
                                if nc_qs.intersection(nc_a_qs): # If not empty intersection
                                    
                                    # Create and add a new node
                                    m_nc_name = conc_core + str(conc_count)
                                    conc_count += 1
                                    m_ncqs = imt.get_merged_concept_qs(nc_qs, nc_a_qs)
                                    merged_conM.add_node(m_nc_name)
                                    nx.set_node_attributes(merged_conM, {m_nc_name: {'Qs': m_ncqs}})

                                    # Create and add new edge
                                    merged_conM.add_edge(m_c_name, m_nc_name)

                                    # Label edges as used
                                    nx.set_edge_attributes(conM, {(c,nc,0): {'v': True}})
                                    nx.set_edge_attributes(conM_a, {(c_a,nc_a,0): {'v': True}})

                                    # Compute and set weight
                                    w, w_a = conM_ws[(c,nc,0)], conM_a_ws[(c_a,nc_a,0)]
                                    m_w = merge_w*np.sqrt(w*w_a) # Harmonic mean
                                    nx.set_edge_attributes(merged_conM, {(m_c_name, m_nc_name,0):{'w': m_w}})


    return merged_conM








# -----------------------------------------------------------------------------------------
def get_conceptMap_sequence(conM, c_s=0, alg_code='weighted_BFS'):
    '''
    Returns best sequence of concepts starting in concept s_c
        Parameters: 
            conM: input concept map
            c_s: stating concept s_c
            alg_code: Hamilton = ignore weights and get Hamiltonian path, weighted_BFS = go to highest weight
        Returns:
            Sequence of concepts as a walk along concept map.
    '''

    # Read graph
    nodes_lst = list(conM.nodes)
    edge_w = nx.get_edge_attributes(conM, 'w')
    seq_dc = {}

    # Get starting node
    if c_s == 0:
        # Get best node to start
        best_deg = -len(nodes_lst)
        c_s = nodes_lst[0]
        for c in conM.nodes:
            c_deg = conM.out_degree(c) - conM.in_degree(c)
            if c_deg > best_deg:
                best_deg = c_deg
                c_s = c


    if alg_code == 'Hamilton':

        # Get path ignoring weights
        conM_S = nx.DiGraph(conM)
        path = nx.algorithms.tournament.hamiltonian_path(conM_S)

        # Create sequence
        for ii_c in range(len(path)-1):
            seq_dc[path[ii_c]] = path[ii_c+1]
            

    if alg_code == 'weighted_BFS':
        # Do BFS on the graph
        visited_lst = []
        visited_nexts_lst = []
        queue_lst = []
        visited_lst.append(c_s)
        queue_lst.append(c_s)

        allCompQ = False
        while not allCompQ: # Scan all components
            #print ('seq_dc: ', seq_dc)
            # Scan this component
            while queue_lst:
                c = queue_lst.pop(0) 
                # print (c) 

                # Get those with biggest weight
                curr_w = [] # List of weights
                curr_cons = [] # List of nodes
                for cn in conM.neighbors(c): # Collect weights of non-visited
                    curr_w.append(edge_w[(c, cn,0)])
                    curr_cons.append(cn)
                    if (cn not in queue_lst) and (cn not in visited_lst):
                        queue_lst.append(cn)

                # Strategy: add concept with highest weight
                ins_lst = np.flip(np.argsort(curr_w)) # Sort them in descending order
                
                for ii in ins_lst: # insert by decreasing weights - nexts
                    curr_c = curr_cons[ii]
                    if curr_c not in visited_lst:
                        visited_lst.append(curr_c)
                        

                if len(curr_cons) > 0: # Insert into the sequence - first one
                    insertedQ = False
                    for cn in curr_cons:
                        if not cn in visited_nexts_lst and cn != c:
                            seq_dc[c] = cn
                            visited_nexts_lst.append(cn)
                            insertedQ = True
                            break
                    if not insertedQ:
                        seq_dc[c] = None
                             
            #    else:
            #        seq_dc[c] = None
            # Next component
            #print ('visited_lst:', visited_lst)
            #print ('nodes_lst:', nodes_lst)
            if len(visited_lst) < len(nodes_lst): # Not all covered
                remided_nodes_lst = list(set(nodes_lst) - set(visited_lst))
                next_c = remided_nodes_lst[0]
                if next_c not in visited_lst:
                    visited_lst.append(next_c) 
                    queue_lst.append(next_c)
            else: 
                allCompQ = True # All components done
    
    # Connect sequence
    no_next_conc_lst = list(set(conM.nodes) - set(seq_dc.keys())) 
    no_prev_conc_lst = list(set(seq_dc.keys()) - set(seq_dc.values()) - set([c_s]))
    for c in no_next_conc_lst:
        if len(no_prev_conc_lst) > 0:
            c_n = no_prev_conc_lst.pop()
            seq_dc[c] = c_n
        else:
            seq_dc[c] = None    

    return seq_dc 


# -----------------------------------------------------------------------------------------
# Simplify function
# @brief it removes multiple edges and combine weights
def simplify_conceptMap(conM, pars=[]):
    '''
    Returns simplified concept map
        Parameters: 
            conM: concept map
            pars=[]: any input parameter. pars[0] is link weight cut threshold
        Returns:
            simplified concept map
    '''

    conM_qs = nx.get_node_attributes(conM, 'Qs')
    conM_ws = nx.get_edge_attributes(conM, 'w')

    # Create graph and copy nodes
    smpl_conM = nx.MultiDiGraph(name='Simplified ' + conM.name)
    for u in conM.nodes:
        smpl_conM.add_node(u)
        nx.set_node_attributes(smpl_conM, {u: {'Qs': conM_qs[u]}})

    # Set cut weight
    if len(pars) > 0:
        lb_w_T = pars[0]
    else:
        lb_w_T = 0.2

    # Add and simplify double weights, threshold them: substract oposite directions
    for e in conM.edges:
        u,v,w = e[0],e[1],e[2] 
        if w == 0: # Single directed link only
            e_forw_w = conM_ws[e]
            if (e[1],e[0],0) in conM_ws:
                e_back_w = conM_ws[(e[1],e[0],0)] 
                e_forw_w = e_forw_w - e_back_w
        #else: # Implement parallel weights
        if e_forw_w > lb_w_T:
            smpl_conM.add_edge(e[0],e[1])
            nx.set_edge_attributes(smpl_conM, {e: {'w': e_forw_w}})

    #print ('smpl_conM before')
    #print (nx.get_edge_attributes(smpl_conM, 'w'))

    # Remove empty concepts
    for c in conM.nodes:
        if not conM_qs[c]:
            #print (c)
            # Reconnect if any  
            for c_u in conM.predecessors(c):
                for c_v in conM.successors(c):
                    c_u_w = conM_ws[(c_u, c, 0)]
                    c_v_w = conM_ws[(c, c_v, 0)]
                    e = (c_u, c_v,0)
                    smpl_conM.add_edge(c_u, c_v)
                    nx.set_edge_attributes(smpl_conM, {e: {'w': np.sqrt(c_u_w*c_v_w)}})
            
            smpl_conM.remove_node(c) 
            conM_qs = nx.get_node_attributes(conM, 'Qs')
            conM_ws = nx.get_edge_attributes(conM, 'w')

    # Remove repeated questions

    return smpl_conM


# -----------------------------------------------------------------------------------------
# Test API to the server ===================================================================================================
def get_conceptMap_from_hist(output_row, qIDs_set, keys_to_qs_dc):
    '''
    Returns combined concept map.
        Parameters: 
            
        Returns:
            
    '''

    alg_code = 'rel_shift' # 'min_q'
    keyw_conM = get_conceptMap_from_keyQs(qIDs_set, keys_to_qs_dc, alg_code, pars=[0.2])

    return keyw_conM



