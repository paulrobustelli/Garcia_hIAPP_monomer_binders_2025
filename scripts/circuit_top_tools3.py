import numpy as np
import mdtraj as md 

def get_topology_info(index): 
    """
    This is only a blurb of the following code which only considers the single chain model
    Created on Mon May 24 17:00:09 2021

    @author: DuaneM

    Function creating a topological relationship matrix for a Residue contact map, 
    using either a single chain or a whole model.
    """
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
    psc = [P,S,X,round(P/total,3),round(S/total,3),round(X/total,3)]

    return mat,psc

# create a contact map from a trajectory 
def get_indices(trj, cutoff_distance, exclude_neighbour, res_num=38):
    """
    get indices that matter 
    trj : trajectory 
    res_num: length of trajectory/ number of residues 
    cutoff_distance: the cutoff distance for defineing a contact 
    exclude_neighbour: the number of neighbors to exclude so as to remove any noisey calculations 
    returns: a list of numpy arrays with indices 
    """

    contacts = np.zeros(shape=(res_num,res_num,trj.n_frames))

    for i in range(0, res_num):
        for j in range(i+exclude_neighbour+1, res_num): # do not double count the indices that you find 
            contacts[i][j] = np.where(md.compute_contacts(trj,[[i,j]], scheme="closest-heavy")[0] < cutoff_distance/10, 1, 0).reshape(trj.n_frames)

    # broadcast arrays?
    indices = np.triu_indices(res_num, k=exclude_neighbour)
    reshaped_array = contacts.reshape(-1,trj.n_frames)

    values_along_axis_2 = contacts[indices[0], indices[1], :]

    indices_trj = []
    
    # O (n_frames) bleh 
    for f in range(trj.n_frames): 
        idxs_equal_to_1 = np.where(values_along_axis_2[:,f] == 1)[0]
        # convert indices to a list of lists 
        idxs = []
        for idx in np.nditer(idxs_equal_to_1): 
            idxs.append((indices[0][idx], indices[1][idx]))
        indices_trj.append(np.array(idxs))

    return indices_trj 

def create_combos_map(prot_len, exclude_neighbour=1):
    """
    create all pair combinations and map to an index 
    """
    combinations = []
    for i in range(prot_len):
        for j in range(i+1+exclude_neighbour, prot_len): 
            combinations.append((i,j))
    combos = np.array(combinations)
    
    map_dict = {}
    for num, k in enumerate(combos):
        map_dict[tuple(k)] = num

    return map_dict

def compute_circuit_top_model(trj, res_num, cutoff_distance, exclude_neighbour, fname):
    # get the indices that matter
    indices_trj = get_indices(trj, cutoff_distance, exclude_neighbour, res_num)
    map_dict = create_combos_map(res_num)
    
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

    # compute topology for each frame (also bleh O n_frames)
    for f in range(trj.n_frames): 
        if f%1000 == 0: 
            print(f)
        
        index = indices_trj[f]
        mat, psc = get_topology_info(index)

        temp = np.zeros((dim_m,))
        for k, e in enumerate(index): 
            for l, j in enumerate(index[k+1:]): 
                val = mat[k,l] 
                x = map_dict[tuple(e)]
                y = map_dict[tuple(j)]
                temp[pairs_dict[(x,y)]] = val
        circuit_model[f, :] = temp
        topologies[f,:] = psc[3:]
        del temp 
        del mat 

    # Explicitly flush changes to disk before closing
    circuit_model.flush()
    del circuit_model

    return topologies