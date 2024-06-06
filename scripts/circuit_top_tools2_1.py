import os
import numpy as np

from Bio.PDB import MMCIFParser,Selection, PDBParser, Selection, NeighborSearch
from Bio import BiopythonWarning
import warnings
from collections import Counter

from scipy.spatial.distance import pdist, squareform

"""
Created on Mon May 24 17:00:09 2021

@author: DuaneM

Function creating a topological relationship matrix for a Residue contact map, 
using either a single chain or a whole model.
"""

def get_matrix(index,protid):
    if index.shape == (0,):
        print('Error - index empty')
        mat = np.zeros((len(index), len(index)),dtype = 'int')
        psc = [protid,0,0,0]
        return mat,psc
    #Determines whether index came from model or single chain
    if np.shape(index)[1] == 2:

        #create a numerical and character matrix based on the amount of nonzero values found in the previous function
        mat = np.zeros((len(index), len(index)),dtype = 'int')

        #Change the values based on the type of connection
        

        P = 0
        S = 0
        X = 0

        for x in range(0,len(index)):
            i = index[x,0]
            j = index[x,1]
            for y in range(x+1,len(index)):
                k = index[y,0]
                l = index[y,1]
                #series
                if (j < k):
                    S=S+1
                    mat[x, y]=1
                    mat[y, x]=1

                #parallel    
                elif (i>k and j<l):
                    P=P+1
                    mat[x, y]=2
                    mat[y, x]=3
                
                #5: CP
                #6: CP-1    
                elif (i==k and j<l):
                    mat[x, y]=5
                    mat[y, x]=6
                    P += 1
            
                elif (i==k and l<j):
                    mat[x, y]=6
                    mat[y, x]=5
                    P += 1
                
                elif (k>i and j==l):
                    mat[x,y]=6
                    mat[y,x]=5
                    P += 1
                    
                elif(i>k and l==j):
                    mat[x,y]=5
                    mat[y,x]=6
                    P += 1
                #inverse parallel
                elif (k>i and l<j):
                    P += 1
                    mat[x, y]=3
                    mat[y, x]=2
                #CS
                elif (j ==k):
                    mat[x,y]=7
                    mat[y,x]=7
                    S += 1
                #Cross
                if (k>i and k<j and j<l):
                    X += 1
                    mat[x, y]=4
                    mat[y, x]=4
                elif (i>k and i< l and j> l):
                    X += 1
                    mat[x, y]=4
                    mat[y, x]=4
        total = sum([P,S,X])
        psc = [protid,P,S,X,round(P/total,3),round(S/total,3),round(X/total,3)]

        return mat,psc

    elif np.shape(index)[1] == 4:
        
        chainids = np.unique(index[:,2:])
        chainstats = {}
        for i in chainids:
            chainstats[i] = {'p':0,'s':0,'x':0,'i2':0,'i3':0,'i4':0,'t2':0,'t3':0,'l':0}
            
        mat = np.zeros((len(index), len(index)),dtype = 'int')
    
        P = 0
        S = 0
        X = 0
        I2 = 0
        I3 = 0
        I4 = 0
        T2 = 0
        T3 = 0
        L = 0
        
        
        for x in range(len(index)):
            chain1 = False
            
            i = index[x][0]
            j = index[x][1]
            chaini = index[x][2]
            chainj = index[x][3]
            
            if chaini == chainj:
                chain1 = True
                
            for y in range(x+1,len(index)):
                chain2 = False
                
                k = index[y][0]
                l = index[y][1]
                chaink = index[y][2]
                chainl = index[y][3]
                
                set1 = set([chaini,chainj])
                set2 = set([chaink,chainl])

                if chaink == chainl:
                    chain2 = True
                    
                if chain1 and chain2:
                    if chaini == chaink:
                        #series
                        if j < k:
                            S += 1
                            mat[x,y] = 2
                            mat[y,x] = 2
                            chainstats[chaink]['s'] += 1
                            
                        #parallel
                        elif k < i and j < l:
                            P += 1
                            mat[x,y] = 1
                            mat[y,x] = 1
                            chainstats[chaink]['p'] += 1
                                
                        elif i < k and l < j:
                            P += 1
                            mat[x,y] = 1
                            mat[y,x] = 1
                            chainstats[chaink]['p'] += 1
                            
                        elif (i==k and j<l):
                            mat[x, y]=1
                            mat[y, x]=1
                            P += 1
                            chainstats[chaink]['p'] += 1
                            
                        elif (i==k and l<j):
                            mat[x, y]=1
                            mat[y, x]=1
                            P += 1
                            chainstats[chaink]['p'] += 1
                            
                        elif (k>i and j==l):
                            mat[x,y]=1
                            mat[y,x]=1
                            P += 1
                            chainstats[chaink]['p'] += 1
                            
                        elif(i>k and l==j):
                            mat[x,y]=1
                            mat[y,x]=1
                            P += 1
                            chainstats[chaink]['p'] += 1
                        #CS
                        elif j == k:
                            S += 1
                            mat[x,y] = 2
                            mat[y,x] = 2
                            chainstats[chaink]['s'] += 1
                            
                        #Cross
                        if (k>i and k<j and j<l):
                            X += 1
                            mat[x, y]=3
                            mat[y, x]=3
                            chainstats[chaink]['x'] += 1
                            
                        elif (i>k and i< l and j> l):
                            X += 1
                            mat[x, y]=3
                            mat[y, x]=3
                            chainstats[chaink]['x'] += 1
                    #Independent
                    else:
                        I2 += 1
                        mat[x,y] = 4
                        mat[y,x] = 4
                        chainstats[chaink]['i2'] += 1
                        chainstats[chainj]['i2'] += 1

                elif chain1 and not set1.intersection(set2):
                    I3 += 1
                    mat[x,y] = 4
                    mat[y,x] = 4
                    chainstats[chaini]['i3'] += 1
                    chainstats[chaink]['i3'] += 1
                    chainstats[chainl]['i3'] += 1
                
                elif chain2 and not set1.intersection(set2):
                    I3 += 1
                    mat[x,y] = 4
                    mat[y,x] = 4
                    chainstats[chaini]['i3'] += 1
                    chainstats[chaink]['i3'] += 1
                    chainstats[chainl]['i3'] += 1

                #I - multiple chains
                elif not set1.intersection(set2):
                    I4 += 1
                    mat[x,y] = 4
                    mat[y,x] = 4
                    chainstats[chaini]['i4'] += 1
                    chainstats[chainj]['i4'] += 1
                    chainstats[chaink]['i4'] += 1
                    chainstats[chainl]['i4'] += 1
                
                #T
                elif chain1 and set1.intersection(set2) :
                    T2 += 1
                    mat[x,y] = 5
                    mat[y,x] = 5
                    chainstats[chaini]['t2'] += 1
                    chainstats[list(set2-set1)[0]]['t2'] += 1
                    
                elif chain2 and set1.intersection(set2):
                    T2 += 1
                    mat[x,y] = 5
                    mat[y,x] = 5
                    chainstats[chaink]['t2'] += 1
                    chainstats[list(set1-set2)[0]]['t2'] += 1
                elif ~chain1 and ~chain2 and len(set1.intersection(set2)) == 1:
                    T3 += 1
                    mat[x,y] = 5
                    mat[y,x] = 5
                    chainstats[list(set1.intersection(set2))[0]]['t3'] += 1
                    chainstats[list(set2 - set1)[0]]['t3'] += 1
                    chainstats[list(set1 - set2)[0]]['t3'] += 1
                #L
                elif ~chain1 and ~chain2 and set1 == set2:
                    L += 1
                    mat[x,y] = 6
                    mat[y,x] = 6
                    chainstats[list(set1.intersection(set2))[0]]['l'] += 1
                    chainstats[list(set1.intersection(set2))[1]]['l'] += 1
                else:
                    print('error - ',i,chaini,j,chainj,k,chaink,l,chainl)
                
        stats = [protid,P,S,X,I2,I3,I4,T2,T3,L]
        return mat,stats,chainstats    

"""
Created on Mon May 24 17:00:09 2021

@author: DuaneM

Function for creating a Residue-Residue based contact map for either a single chain or a whole model.
Note! this does not produce a contact map but a matrix of the non-zero values in that contact map. 
"""

def get_cmap(
            chain,
            level = 'chain',
            cutoff_distance = 4.5,
            cutoff_numcontacts = 0,        
            exclude_neighbour = 0):
    
    if level == 'chain':

        #Unpack chain object into atoms and residues
        atom_list = Selection.unfold_entities(chain,'A')
        res_list = Selection.unfold_entities(chain,'R')
        
        res_names, numbering = [], []
        for res in res_list:
            res_names.append(res.get_resname())
            numbering.append(res.get_id()[1])
        
        #search for neighbouring atoms within specified distance
        ns = NeighborSearch(atom_list)
        all_neighbours = ns.search_all(cutoff_distance,'A')
        
        numbering = np.array(numbering)
        segment = np.array(range(len(numbering)))

        #transform atom contacts into residue contacts
        index_list = []
        for atompair in all_neighbours:
            res1 = segment[numbering == atompair[0].get_parent().id[1]][0]
            res2 = segment[numbering == atompair[1].get_parent().id[1]][0]

            if abs(res1-res2) > exclude_neighbour:
                index_list.append((res1,res2))

        #sort residue contacts and check if they occur more than the minimum set in cutoff_numcontacts
        index_list.sort()
        count = Counter(index_list)

        index = [values for values in count if count[values] >= cutoff_numcontacts]
        protid = chain.get_parent().get_parent().id + '_' + chain.id
        return np.array(index),numbering, protid,res_names
        
    #same as single chain analysis but unpacks whole model instead of single chain
    if level == 'model':

        atom_list_model = Selection.unfold_entities(chain.get_parent(),'A')
        res_list_model = Selection.unfold_entities(chain.get_parent(),'R')

        ns = NeighborSearch(atom_list_model)
        all_neighbours = ns.search_all(cutoff_distance,'A')

        res_names, numbering = [], []
        for res in res_list_model:
            res_names.append(res.get_resname())
            numbering.append([res.get_full_id()[2],res.get_full_id()[3][1]])

        numbering = np.array(numbering)

        segment = np.array(range(len(numbering)))
            
        index_list = []
        for atompair in all_neighbours:
            res1 = segment[(numbering == [atompair[0].get_parent().get_full_id()[2],str(atompair[0].get_parent().get_full_id()[3][1])]).all(axis=1)][0]
            res2 = segment[(numbering == [atompair[1].get_parent().get_full_id()[2],str(atompair[1].get_parent().get_full_id()[3][1])]).all(axis=1)][0]
            chain1 = atompair[0].get_parent().get_full_id()[2]
            chain2 = atompair[1].get_parent().get_full_id()[2]

            if abs(res1-res2) > exclude_neighbour:
                index_list.append((res1,res2,chain1,chain2))
                
        index_list.sort()
        index_list.sort(key= lambda x : x[2:4])
        
        count = Counter(index_list)

        index = [values for values in count if count[values] >= cutoff_numcontacts]
        protid = chain.get_parent().get_parent().id
        
        return np.array(index),numbering,protid,res_names

"""
Created on Mon March 28 17:00:09 2022

@author: DuaneM

Function for creating a Residue-Residue based contact map for either a single chain or a whole model.
Contacts are based on the Centre of Geometry of the residues.

Note! this does not produce a contact map but a matrix of the non-zero values in that contact map. 
"""
def get_cmap_cog(chain,cutoff_distance = 4.5,exclude_neighbour=3):

    atom_list = Selection.unfold_entities(chain,"A")

    #Make list of the atom information
    residue_number = np.zeros(len(atom_list),dtype='int')
    coords = np.zeros([len(atom_list),3])
    name = []


    for num, atom in enumerate(atom_list):
        residue_number[num] = atom.get_parent().get_id()[1]
        coords[num] = atom.get_coord()
        name.append(atom.get_name())

    numbering = list(range(residue_number[0],residue_number[-1]+1))
    residue_number = residue_number - numbering[0] 

    duplicate = np.zeros(len(atom_list),dtype='int')
    for i in range(1,len(duplicate)):
        if residue_number[i] == residue_number[i-1] and name[i] == name[i-1]:
            duplicate[i] = 1


    residue_number = residue_number[np.where(duplicate != 1)]
    coords = coords[np.where(duplicate != 1)]

    cog = []
    for i in set(residue_number):
        cog.append(list(np.mean(coords[residue_number == i],axis=0)))

    cmap = squareform(pdist(cog))
    cmap = (cmap < cutoff_distance) * 1

    index1 = np.transpose(np.triu(cmap).nonzero())

    index = []
    for i in index1:
        if abs(i[0]-i[1]) > exclude_neighbour:
            index.append(list(i))

    return np.array(index),numbering

"""
Created on Mon May 24 17:00:09 2021

@author: DuaneM

Function for creating the chain object used in Bio.PDB and all the functions. Can specify which chain

"""
def retrieve_chain(input_file,chainid = 0):
    #determines which format is used
    if input_file.endswith('cif'):
        
        input_filepath= input_file
        #Supress harmless warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BiopythonWarning)
            #Import the protein data
            structure = MMCIFParser().get_structure(input_file.replace('.cif',''),input_filepath)
   
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BiopythonWarning)
        
            input_filepath= input_file
            #import protein data
            structure = PDBParser(PERMISSIVE=1).get_structure(input_file.replace('.pdb',''),input_filepath)

    model = structure[0]
    chainlist = model.get_list()
    #removes heteroresidues from protein
    residue_to_remove = []
    chain_to_remove = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != ' ':
                residue_to_remove.append((chain.id, residue.id))
                #REMOVES DNA MOLECULES AND UNKNOWN RESIDUES FROM MODEL
            elif residue.get_resname() in ['DG','DA','DT','DC','DU','UNK','A','G','C','T']:
                residue_to_remove.append((chain.id, residue.id))

    for residue in residue_to_remove:
        model[residue[0]].detach_child(residue[1])

    for chain in model:
        if len(chain.get_list()) == 0:
            chain_to_remove.append(chain.id)

    for chain in chain_to_remove:
        model.detach_child(chain)

    if type(chainid) == int:
        chain = model.get_list()[chainid]
    elif type(chainid) == str:
        chain = model[chainid]
    else:
        raise TypeError

    protid = structure.id+ '_' + chain.id

    return chain,protid


# ----------------------------------------------------
# Michelle Garcia (Dartmouth College) Functions 

def create_combos_map(prot_len):
    """
    create all pair combinations and map to an index 
    """
    combinations = []
    for i in range(prot_len):
        for j in range(i+2, prot_len): 
            combinations.append((i,j))
    combos = np.array(combinations)
    
    map_dict = {}
    for num, k in enumerate(combos):
        map_dict[tuple(k)] = num

    return map_dict

def compute_circuit_top_model(trj, fname, prot_len=38, cutoff_distance = 10, cutoff_numcontacts = 1, exclude_neighbour = 1): 
    
    map_dict = create_combos_map(prot_len)

    dim_m = np.sum(np.arange(len(map_dict)))
    data_type = np.uint8 # reduce the number of bytes 
    
    # memory mapped model and topology numpy array 
    circuit_model = np.memmap(fname, dtype=data_type, mode='w+', shape=(trj.n_frames, dim_m))
    topologies = np.zeros((trj.n_frames, 3))

    # create a dictionary that maps the correct index for each pair which is precomputed. 
    pairs_dict = {}
    count = 0
    for j in range(len(map_dict)): 
        for i in range(j+1, len(map_dict)):
            pairs_dict[(j, i)] = count
            count += 1 

    # make a temporary directory 
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # compute the circuit_top_model for the frame 
    for i in range(trj.n_frames): 
        # compute this for each one, could vectorize later
        # create an md traj object from the trajectory that you want 
        trj.slice([i]).save_pdb("./temp/slice.pdb")
        temp_file = "./temp/slice.pdb"

        # Creates a chain object from a CIF/PDB file
        chain, protid = retrieve_chain(temp_file)
        #Step 1 - Draw a residue-residue based contact map 
        index,numbering,protid,res_names = get_cmap(
                                                    chain,
                                                    cutoff_distance = cutoff_distance,
                                                    cutoff_numcontacts = cutoff_numcontacts,        
                                                    exclude_neighbour = exclude_neighbour)

        #Step 3 - Draw a circuit topology relations matrix
        mat, psc = get_matrix(index,protid)

        # encode circuit model
        temp = np.zeros((dim_m,))
        for k, e in enumerate(index): 
            for l, j in enumerate(index[k+1:]): 
                val = mat[k,l] 
                x = map_dict[tuple(e)]
                y = map_dict[tuple(j)]
                temp[pairs_dict[(x,y)]] = val
        circuit_model[i, :] = temp
        topologies[i,:] = psc[-3:]
        del temp 
        del mat 

        # delete the object 
        os.remove(temp_file)

    # remove the temporary directory 
    os.rmdir("temp")
    return circuit_model 