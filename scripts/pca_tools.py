# general imports 
import os
import sys
import time
import numpy as np

# plotting imports 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib import colors, cm

# computing inputs 
from numpy import log2, zeros, mean, var, sum, loadtxt, arange, \
                  array, cumsum, dot, transpose, diagonal, floor
import itertools
import mdtraj as md

# pca and Kmeans function from a python package sklearn 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# other stuff to compute 
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# define the font sizes and weights 
from matplotlib import rc 
font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 14}

rc('font', **font)

# kmeans 
####                                        computing                                               ###
def block(x):
    """
    This is a piece of code that we often use in the Robustelli lab to calculate the error bars/ stats 
    """
    # preliminaries
    d = log2(len(x))
    if (d - floor(d) != 0):
    #    print("Warning: Data size = %g, is not a power of 2." % floor(2**d))
    #    print("Truncating data to %g." % 2**floor(d) )
        x = x[:2**int(floor(d))]
    d = int(floor(d))
    n = 2**d
    s, gamma = zeros(d), zeros(d)
    mu = mean(x)
    # estimate the auto-covariance and variances 
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q =array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
              16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
              24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
              31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
              38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
              45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print("Warning: Use more data")
    
    print(k)
    print(d)

    return (s[k]/2**(d-k))

def make_contact_distance_map(trj, res_num):
    """
    All elements to create a contact map of distances
    :trj: (mdtraj object) trajectory 
    :res_num: (int) the number of residues
    returns: (np.array) model input for PCA, contact average, and total contacts 
    """
    contacts = np.zeros(shape=(res_num,res_num,trj.n_frames))
    contact_avg = np.zeros(shape=(res_num,res_num))
    
    dim = int((res_num)*(res_num - 1)/2)
    model_input = np.zeros(shape=(dim, trj.n_frames))
    
    count = 0 
    for i in range(0, res_num):
        for j in range(i+1, res_num):
            contacts[i][j] = md.compute_contacts(trj,[[i,j]])[0].reshape(trj.n_frames,)
            contacts[j][i] = contacts[i][j]
             
            contact_avg[i][j] = np.average(contacts[i][j])
            contact_avg[j][i] = contact_avg[i][j] 
            
            model_input[count] = contacts[i][j]
            count += 1
    
    # transpose
    model_input = np.transpose(model_input)
    return model_input, contacts, contact_avg

def make_contact_map(trj, res_num, cutoff=1.2):
    """
    All elements to create a contact map one-hot encoded 
    :trj: (mdtraj object) trajectory 
    :res_num: (int) the number of residues 
    returns: (np.array) model input for PCA, contact average, and total contacts 
    """
    contacts = np.zeros(shape=(res_num,res_num,trj.n_frames))
    contact_avg = np.zeros(shape=(res_num,res_num))
    
    dim = int((res_num)*(res_num - 1)/2)
    model_input = np.zeros(shape=(dim, trj.n_frames))
    
    count = 0 
    for i in range(0, res_num):
        for j in range(i+1, res_num):
            contacts[i][j] = np.where(md.compute_contacts(trj,[[i,j]])[0] < cutoff, 1, 0).reshape(trj.n_frames,)
            contacts[j][i] = contacts[i][j]
             
            contact_avg[i][j] = np.average(contacts[i][j])
            contact_avg[j][i] = contact_avg[i][j] 
            
            model_input[count] = contacts[i][j]
            count += 1
    
    # transpose
    model_input = np.transpose(model_input)
    return model_input, contacts, contact_avg



def gen_CA(trj, CAatoms):
    """
    Generate an array of alpha carbons  
    input: CAatoms 
    output: numpy array of CA labels 
    """
    
    CAlabel=[]    
    for i in range(0,len(CAatoms)):
        CAlabel.append(trj.topology.atom(CAatoms[i]).residue.resSeq)
    CAlabel =np.array(CAlabel).astype(int)

    return CAlabel

 # define centroid 
def get_centroid(traj_basin, ind_list): 
    """
    Get the centroid 
    :traj_basin: (mdtraj object) the trajectory of the basin 
    :ind_list: the list of indices that made that traj 
    """
    atom_indices = [a.index for a in traj_basin.topology.atoms if a.element.symbol != 'H']
    distances = np.empty((traj_basin.n_frames, traj_basin.n_frames))
    
    for i in range(traj_basin.n_frames):
        distances[i] = md.rmsd(traj_basin, traj_basin, i, atom_indices=atom_indices)

    beta = 1
    index = np.exp(-beta*distances / distances.std()).sum(axis=1).argmax()

    return int(index), ind_list[index]

# calculate the backbone secondary structure, so that you can get the 
def calc_BB_dssp(traj):
    """
    Calculate the BB dssp based on a trajectory.
    :traj: (mdtraj object) the trajectory  
    """
    # create a deep copy of the trajectory 
    trjBB = traj.slice(np.arange(0,traj.n_frames))
    # calculate dssp based on backbone 
    BB=traj.top.select("backbone")
    trjBB.restrict_atoms(BB)
    trjBB.center_coordinates()
    x = [res.resSeq for res in trjBB.top.residues]
    dssp = md.compute_dssp(trjBB,simplified=True)
    return dssp 

def dssp_convert_nose(traj):
    """
    get secondary structure values for helix and sheet 
    :traj: (mdtraj object) the trajectory
    """
    dssp = calc_BB_dssp(traj)
    dsspH=np.copy(dssp)
    dsspE=np.copy(dssp)
    
    dsspH[dsspH=='H']=1 # helix                                                                                  
    dsspH[dsspH=='E']=0 # extended strand                                                                              
    dsspH[dsspH=='C']=0 # coil                                                                                  
    dsspH[dsspH=='NA']=0 # don't know 
    dsspH=dsspH.astype(int) # cast to int 
    TotalH=np.sum(dsspH,axis=1) 
    Hprop=np.sum(dsspH,axis=0).astype(float)/len(dsspH) 

    dsspE[dsspE=='H']=0
    dsspE[dsspE=='E']=1
    dsspE[dsspE=='C']=0
    dsspE[dsspE=='NA']=0
    dsspE=dsspE.astype(int)
    TotalE=np.sum(dsspE,axis=1) 
    Eprop=np.sum(dsspE,axis=0).astype(float)/len(dsspE) 

    return Hprop, Eprop


def contact_map_avg(trj, prot_len, cutoff = 1.2):
    """
    This is almost the same as the above function but returns less
    create average contact maps and distance maps for entire trajectory one-hot encoded 
    :trj: (mdtraj object) trajectory 
    :prot_len: (int) the number of residues 
    returns: a signle contact map
    """     
    contact_maps = []
    contact_distances = []
    
    for i in range(0,prot_len):
        contact_map = []
        contact_distance = []

        for j in range(i,prot_len):
            if i == j:
                contacts = 0
            else:
                dist = md.compute_contacts(trj, [[i, j]])
                array = np.asarray(dist[0]).astype(float)
                distance = np.average(array)
                contact_distance.append(distance)
                contact = np.where(array < cutoff, 1, 0)
                contacts = np.average(contact)
            contact_map.append(contacts)
        contact_maps.append(contact_map)
        contact_distances.append(contact_distance)
    final_map = np.asarray(contact_maps).astype(float)
    final_distance = np.asarray(contact_distances).astype(float)

    return final_map, final_distance

####                                        plotting                                               ###

# these are some functions that I made so that plotting can be made easier 

def plt_avg_matrx(matrx, title="Average Matrix", count=0): 
    """
    Plots a heatmap with the matplotlib interface 
    :Returns: an axis object 
    """
    f, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrx,cmap='jet')
    ax.set_xlabel("segment 1", size = 20)
    ax.set_ylabel("segment 2", size = 20)
    ax.set_title(title, size = 20)

    # set your colorbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')

    return ax 

    
def plt_project_time(PC1, PC2, num_frames, time=50, title= "Timecourse PCA"):    
    """ 
    :PC1: (numpy array) the first principal component axis
    :PC2: (numpy array) the second principal component axis 
    :num_frames: (int) the number of frames 
    :time: (int) the amount of total time of the simulation 
    :title: (str) the title of the plot 

    :returns: an axis object 
    """
    # create a time array 
    time= np.linspace(0,time, num_frames)
    f, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(PC1, PC2, marker='x', c=time)
    ax.set_xlabel('PC1',size=20)
    ax.set_ylabel('PC2',size=20) 
    ax.set_title(title, size=20)
    
    # set your colorbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(sc, cax=cax, orientation='vertical')
    cax.set_ylabel('Time (ns)')

    return ax 
    
def plt_project_rg(PC1, PC2, trj, title= "Rg PCA"): 
    """
    plots the radius of gyration over the first two pc axis
    :PC1: (numpy array) the first principal component axis
    :PC2: (numpy array) the second principal component axis 
    :trj: (mdtraj object) the loaded trajectory
    :title: (str) the title of the plot 

    :returns: an axis object 
    """

    # Projecting radius of gyration on principal components 
    f, ax = plt.subplots(figsize=(10, 8))

    # compute the radius of gyration based on the trajectory 
    # https://mdtraj.org/1.6.2/api/generated/mdtraj.compute_rg.html
    rg = md.compute_rg(trj, masses=None)

    sc = ax.scatter(PC1, PC2, marker='x', c=rg)
    ax.set_xlabel('PC1', size=20)
    ax.set_ylabel('PC2', size=20) 
    ax.set_title(title, size=20)
    
    # set your colorbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(sc, cax=cax, orientation='vertical')
    cax.set_ylabel('Rg (nm)')

    return ax


def free_energy(a, b, T, y0, ymax, x0, xmax):
    """
    Computing the free energy 
    :a: (numpy array) data vector
    :b: (numpy array) data vector 
    :T: (int) temperature in Kelvin
    The rest of the arguments define the bins for free energy calculation s
    returns the free energy 
    """
    free_energy, xedges, yedges = np.histogram2d(a, b, 30, [[y0, ymax], [x0, xmax]], density=True)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.001987*T)*free_energy
    return free_energy, xedges, yedges

def plt_free_energy(PC1, PC2, col_map, centers=None, title="Free Energy, Writhe", xmin=-3, xmax=3, ymin=-3, ymax=3):
    """
    plots the free energy 
    given the limits for the x-axis and y axis, this could be changed 
    :PC1: (numpy array) the first principal component axis
    :PC2: (numpy array) the second principal component axis 
    :col_map: colormap object matplotlib 
    :centers: (numpy array) size is (num_clusters, 2)
    """
    
    # set limits and overide with arguments, automatically includes all data 
    xmin = np.amin(PC1) if np.amin(PC1) < xmin else xmin
    xmax = np.amax(PC1) if np.amax(PC1) > xmax else xmax 
    ymin = np.amin(PC2) if np.amin(PC2) < ymin else ymin
    ymax = np.amax(PC2) if np.amax(PC2) > ymax else ymax 
    
    dG,xedges,yedges=free_energy(PC2, PC1, 300, ymin, ymax, xmin, xmax)

    f, ax = plt.subplots(figsize=(10, 8))
    
    # set color 
    im = ax.imshow(dG, interpolation='gaussian', extent=[
                    yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap=col_map, aspect='auto', vmin=0.01, vmax=8.0)

    # set your colorbar 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    cbar_ticks = [i for i in range(1,9)]
    f.colorbar(im, cax=cax, orientation='vertical', ticks=cbar_ticks, format=('% .1f'), aspect=10)
    cax.set_ylabel("kcal/mol", size=20) 

    
    
    ax.set_ylabel("PC2", size=20, labelpad=15)
    ax.set_xlabel("PC1", size=20, labelpad=15)
    ax.set_title(title, size = 20, loc="left")
    
    if centers != None: 
        # insert centers 
        scat = np.split(centers, 2, axis=1)
        ax.scatter(scat[0], scat[1], c="white", marker = "o", s=100)

    return ax 
    
    
def kmeans_cluster(PC1, PC2, clusters=3, title = "Kmeans Clustering"): 
    """
    Runs the kmeans clustering algorithm
    :PC1: (numpy array) the first principal component axis
    :PC2: (numpy array) the second principal component axis 
    :clusters: (int) number of clusters 
    """
    # do the kmeans clustering 
    concat = np.column_stack((PC1, PC2))
    X = concat
    kmeans = KMeans(n_clusters=clusters).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # make a plot and save it 
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    f, ax = plt.subplots(figsize=(10,8))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask] #  & core_samples_mask for dbscan
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
            label= "cluster" + str(k)
        )

    ax.set_ylabel("PC2", size=20, labelpad=15)
    ax.set_xlabel("PC1", size=20, labelpad=15)
    ax.set_title(title, size = 20, loc="center")
    return ax, labels, centers

def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

    return 

def plt_rmsd(traj_path, pdb_path, time=50): 
    """
    Calculate the RMSD given a trajectory path and pdb path 
    :traj_path: (str)
    :pdb_path: (str)
    :time: (int) the amount of time of the simulation in ns 
    """
    trjCA = md.load(traj_path, top= pdb_path)
    trj = md.load(traj_path, top= pdb_path)
    
    # compute
    CA = trjCA.top.select("name CA")
    trjCA.restrict_atoms(CA)
    trjCA.center_coordinates()

    CA_trj_sel = trj.topology.select("name CA")
    rmsd = md.rmsd(trjCA, trjCA[0])
    
    # plot 
    f, ax = plt.subplots(1,2,figsize=(10,6))
    t = np.linspace(0, time,trj.n_frames)
    ax[0].plot(t,rmsd)
    ax[0].set_xlabel("Time (ns)",size=20)
    ax[0].set_ylabel("$\\alpha$C RMSD ($\AA$)", size=20)
    ax[1].hist(rmsd,histtype='step',bins=300)
    ax[1].set_xlabel("rmsd", size=20)
    ax[1].set_ylabel("count", size=20)
    return ax 

def plt_dssp(traj, residue_renum = None, title = "secondary structure propensity"):
    """
    plot the secondary structure propensity 
    :residue_num: (list) new labels if you have a different residue numbering
    """ 
    Hprop,Eprop=dssp_convert_nose(traj)

    f, ax = plt.subplots(figsize=(8,6))

    if residue_renum == None: 
        residue_renum = [residue.index for residue in traj.topology.residues]
    ax.plot(residue_renum, Hprop,c='r',label='helix')
    ax.plot(residue_renum, Eprop,c='b',label='sheet')
    ax.legend(loc="upper right")
    ax.set_xlabel('Residue', size=20)
    ax.set_ylabel('Secondary Strcture Fraction', size=20)
    ax.set_title(title)
    return ax 
