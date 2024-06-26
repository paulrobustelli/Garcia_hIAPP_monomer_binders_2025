import numpy as np
import mdtraj as md
from numpy import log2, zeros, mean, var, sum, arange, \
    array, cumsum, floor
from scipy.interpolate import make_interp_spline
import pyblock
import math
from time import time 

# plotting imports 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from matplotlib import colors, cm

# pca and Kmeans function from a python package sklearn 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# other stuff to compute 
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def dssp_convert(dssp):
    dsspH = np.copy(dssp)
    dsspE = np.copy(dssp)
    dsspH[dsspH == 'H'] = 1
    dsspH[dsspH == 'E'] = 0
    dsspH[dsspH == 'C'] = 0
    dsspH[dsspH == 'NA'] = 0
    dsspH = dsspH.astype(int)
    TotalH = np.sum(dsspH, axis=1)
    SE_H = np.zeros((len(dssp[0]), 2))

    for i in range(0, len(dssp[0])):
        data = dsspH[:, i].astype(float)
        if(np.mean(data) > 0):
            SE_H[i] = [np.mean(data), (block(data))**.5]

    dsspE[dsspE == 'H'] = 0
    dsspE[dsspE == 'E'] = 1
    dsspE[dsspE == 'C'] = 0
    dsspE[dsspE == 'NA'] = 0
    dsspE = dsspE.astype(int)
    TotalE = np.sum(dsspE, axis=1)
    Eprop = np.sum(dsspE, axis=0).astype(float)/len(dsspE)
    SE_E = np.zeros((len(dssp[0]), 2))

    for i in range(0, len(dssp[0])):
        data = dsspE[:, i].astype(float)
        if(np.mean(data) > 0):
            SE_E[i] = [np.mean(data), (block(data))**.5]
    return SE_H, SE_E

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

        for j in range(0,prot_len):
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

# block function from pyblock package -  https://github.com/jsspencer/pyblock
def block(x):
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
    for i in arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum((x[0:(n-1)]-mu)*(x[1:n]-mu))
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum(((gamma/s)**2*2**arange(1, d+1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
              16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
              24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
              31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
              38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
              45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0, d):
        if(M[k] < q[k]):
            break
    # if (k >= d-1):
    #     print("Warning: Use more data")

    return (s[k]/2**(d-k))

def Kd_calc(bound, conc):
    return((1-bound)*conc/bound)


def get_blockerrors(Data, bound_frac):
    n_data = len(Data[0])
    block_errors = []
    ave = []
    for i in range(0, n_data):
        data = Data[:, i]
        average = np.average(data)
        be = block(data)**.5
        ave.append(np.average(data))
        block_errors.append(be)
    ave_bf = np.asarray(ave)/bound_frac
    be_bf = np.asarray(block_errors)/bound_frac

    return ave_bf, be_bf


def get_blockerrors_pyblock(Data, bound_frac):
    n_data = len(Data[0])
    block_errors = []
    ave = []
    for i in range(0, n_data):
        data = Data[:, i]
        average = np.average(data)
        if (average != 0) and (average != 1):
            reblock_data = pyblock.blocking.reblock(data)
            opt = pyblock.blocking.find_optimal_block(
                len(data), reblock_data)[0]
            opt_block = reblock_data[opt]
            be = opt_block[4]
        else:
            be = 0
        ave.append(average)
        block_errors.append(be)

    ave_bf = np.asarray(ave)/bound_frac
    be_bf = np.asarray(block_errors)/bound_frac
    return ave_bf, be_bf


def get_blockerror(Data):
    data = Data
    average = np.average(data)
    be = block(data)**.5
    return average, be


def get_blockerror_pyblock(Data):
    average = np.average(Data)
    if (average != 0) and (average != 1):
        reblock_data = pyblock.blocking.reblock(Data)
        opt = pyblock.blocking.find_optimal_block(len(Data), reblock_data)[0]
        be = reblock_data[opt][4]
    else:
        be = 0
    return average, float(be)


def get_blockerror_pyblock_nanskip(Data):
    average = np.average(Data)
    # print(average,Data,len(Data))
    if (average != 0) and (average != 1):
        reblock_data = pyblock.blocking.reblock(Data)
        opt = pyblock.blocking.find_optimal_block(len(Data), reblock_data)[0]
        # print(opt)
        # print(math.isnan(opt))
        if(math.isnan(opt)):
            be_max = 0
            for i in range(0, len(reblock_data)):
                be = reblock_data[i][4]
                if(be > be_max):
                    be_max = be
        else:
            be = reblock_data[opt][4]
    else:
        be = 0
    return average, float(be)



def calc_Sa(traj, helixBB):
    trjBB = traj
    BB = trjBB.topology.select("name CA")
    HBB = helixBB.topology.select("name CA")

    trjBB.restrict_atoms(BB)
    helixBB.restrict_atoms(HBB)
    trjBB.center_coordinates()
    helixBB.center_coordinates()

    RMS_start = 1
    RMS_stop = 31
    RMS = []
    for i in range(RMS_start, RMS_stop):
        sel = helixBB.topology.select(
            "residue %s to %s and backbone" % (i, i+6))
        rmsd = md.rmsd(trjBB, helixBB, atom_indices=sel)
        RMS.append(rmsd)
    RMS = np.asarray(RMS)

    Sa = (1.0-(RMS/0.10)**8)/(1-(RMS/0.10)**12)
    Sa_total = np.sum(Sa, axis=0)

    return Sa_total


def calc_rg(traj):
    mass = []
    for at in traj.topology.atoms:
        mass.append(at.element.mass)
    mass_CA = len(mass)*[0.0]
    # put the CA entries equal to 1.0
    for i in traj.topology.select("name CA"):
        mass_CA[i] = 1.0
    # calculate CA radius of gyration
    rg_CA = md.compute_rg(traj, masses=np.array(mass_CA))

    return rg_CA


def make_smooth(x, y):
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(xnew)
    return xnew, y_smooth


def get_hist(array):
    histo = np.histogram(array, bins=20)
    norm = histo[0]/np.sum(histo[0])
    x = histo[1][0:-1]
    y = norm
    X1_smooth, Y1_smooth = make_smooth(x, y)
    return X1_smooth, Y1_smooth

def contact_map(traj,prot_len=38):
    contact_maps = []
    for i in range(prot_len):
        contact_map = []
        for j in range(prot_len):
            if i == j:
                contacts = 0
            else:
                dist = md.compute_contacts(traj,[[i,j]])
                array = np.asarray(dist[0]).astype(float)
                distance = np.average(array)
                contact = np.where(array < 1.2, 1, 0)
                contacts = np.average(contact)
            contact_map.append(contacts)
        contact_maps.append(contact_map)
    final_map = np.asarray(contact_maps).astype(float)
    
    return final_map

def get_Sa_rg_hist(rg_CA, Sa_total):
    a, xedges, yedges = np.histogram2d(rg_CA, Sa_total, 30, [
        [0.9, 2.5], [0, 25.0]], density=True, weights=None)
    a = np.log(np.flipud(a)+.000001)
    T = 300
    a = -(0.001987*T)*a

    return a, xedges, yedges


### plotting functions  ###

def plot_Sa_rg(rg_CA, Sa_total):
    a, xedges, yedges = np.histogram2d(rg_CA, Sa_total, 30, [
        [0.9, 2.5], [0, 25.0]], density=True, weights=None)
    a = np.log(np.flipud(a)+.000001)
    T = 300
    a = -(0.001987*T)*a

    im = plt.imshow(a, interpolation='gaussian', extent=[
                    yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet', aspect='auto')
    cbar_ticks = [0, 1, 2, 3, 4, 5]
    cb = plt.colorbar(ticks=cbar_ticks, format=('% .1f'),
                      aspect=10)  # grab the Colorbar instance
    imaxes = plt.gca()
    plt.xlim(0, 24.9)
    plt.ylabel("Radius of Gryation", size=35, labelpad=15)
    plt.xlabel(r'S$\alpha$', size=35, labelpad=15)
    plt.xticks(size='26')
    plt.yticks(size='26')
    plt.axes(cb.ax)
    plt.clim(vmin=0.1, vmax=3.0)
    plt.tight_layout()

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
            label= "cluster " + str(k + 1)
        )

    ax.set_ylabel("PC2", size=20, labelpad=15)
    ax.set_xlabel("PC1", size=20, labelpad=15)
    ax.set_title(title, size = 20, loc="center")
    ax.legend()
    return ax, labels, centers

def bench_k_means(name, data, clusters = np.arange(2,15)):
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
    clusters : ndarray of shape (n_samples,)
        clusters to test
    """
    # first check to see how many clusters you have 
    inertia = []
     # manually set the number of clusters that you care about 

    print("KMeans clustering analysis for " + name)
    print(82 * "_")
    #print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    print("n_cluster\tsilhouette score\tInertia")

    for n in clusters: 
        kmeans = KMeans(n_clusters=n).fit(data)
        inertia.append(kmeans.inertia_)
        print(str(n) + "\t \t %f"%metrics.silhouette_score(
                data,
                kmeans.labels_,
                metric="euclidean",
                sample_size=2500) + "\t\t"+ str(kmeans.inertia_) )

    print(82 * "_")
    fig,ax = plt.subplots()
    ax.plot(clusters,inertia, "bo-")
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("Inertia")
    return fig 

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