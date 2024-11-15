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
from matplotlib.cm import get_cmap
from matplotlib.patheffects import withStroke

# computing inputs 
import math 
from numpy import log2, zeros, mean, var, loadtxt, arange, \
                  array, cumsum, dot, transpose, diagonal, floor
from numpy.linalg import inv, lstsq
from scipy.interpolate import make_interp_spline
import itertools
import pyblock
import mdtraj as md

# pca and Kmeans function from a python package sklearn 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# other computations 
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

####                        Error Analysis Functions                      ###
def block(x):
    """
    block function from pyblock package -  https://github.com/jsspencer/pyblock
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
    mu = np.mean(x)
    # estimate the auto-covariance and variances 
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = np.var(x)
        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q =np.array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
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

    return (s[k]/2**(d-k))

def free_energy_1D_blockerror(a, T, x0, xmax, bins, blocks):
    histo, xedges = np.histogram(
        a, bins=bins, range=[x0, xmax], density=True, weights=None)
    max = np.max(histo)
    free_energy = -(0.001987*T)*np.log(histo+.000001)
    free_energy = free_energy-np.min(free_energy)
    xcenters = xedges[:-1] + np.diff(xedges)/2
    Ind = chunkIt(len(a), blocks)
    block_size = (Ind[0][1]-Ind[0][0])
    hist_blocks = []
    for i in range(0, len(Ind)):
        block_data = a[Ind[i][0]:Ind[i][1]]
        hist, binedges = np.histogram(block_data, bins=bins, range=[
                                      x0, xmax], density=True, weights=None)
        hist_blocks.append(hist)
    hist_blocks = np.array(hist_blocks)
    average = np.average(hist_blocks, axis=0)
    variance = np.var(hist_blocks, axis=0)
    print(variance)
    print(average)
    N = len(hist_blocks)
    error = np.sqrt(variance / N)
    ferr = -(0.001987*T)*(error / average)
    return free_energy, xcenters, ferr

def chunkIt(a, num):
    avg = a / float(num)
    out = []
    last = 0.0
    while last < a-1:
        out.append([int(last), int(last+avg)])
        last += avg
    return out

def histo_blockerror(a, x0, xmax, bins, blocks):
    histo, xedges = np.histogram(
        a, bins=bins, range=[x0, xmax], density=True, weights=None)
    xcenters = xedges[:-1] + np.diff(xedges)/2
    Ind = chunkIt(len(a), blocks)
    block_size = (Ind[0][1]-Ind[0][0])
    hist_blocks = []
    for i in range(0, len(Ind)):
        block_data = a[Ind[i][0]:Ind[i][1]]
        hist, binedges = np.histogram(block_data, bins=bins, range=[
                                      x0, xmax], density=True, weights=None)
        hist_blocks.append(hist)
    hist_blocks = np.array(hist_blocks)
    average = np.average(hist_blocks, axis=0)
    variance = np.var(hist_blocks, axis=0)
    N = len(hist_blocks)
    error = np.sqrt(variance / N)
    return average, xcenters, error

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
            opt = int(pyblock.blocking.find_optimal_block(
                len(data), reblock_data)[0])
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
        opt = int(pyblock.blocking.find_optimal_block(len(Data), reblock_data)[0])
        be = reblock_data[opt][4]
    else:
        be = 0
    return average, float(be)

def get_blockerror_pyblock_nanskip(Data):
    average = np.average(Data)
    if (average != 0) and (average != 1):
        reblock_data = pyblock.blocking.reblock(Data)
        opt = pyblock.blocking.find_optimal_block(len(Data), reblock_data)[0]
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

def get_blockerror_pyblock_nanskip_bf(Data, bound_frac):
    n_data = len(Data[0])
    block_errors = []
    ave = []
    for i in range(0, n_data):
        data = Data[:, i]
        average = np.average(data)
        if (average != 0) and (average != 1):
            reblock_data = pyblock.blocking.reblock(data)
            opt = pyblock.blocking.find_optimal_block(len(data), reblock_data)[0]
 #           opt_block = reblock_data[opt]
 #           be = opt_block[4]
            if (math.isnan(opt)):
                be_max=0
                for i in range(0, len(reblock_data)):
                    be = reblock_data[i][4]
                    if (be > be_max) :
                        be_max=be
            else:
                be = reblock_data[opt][4]
            
        else:
            be = 0
        ave.append(average)
        block_errors.append(be)

    ave_bf = np.asarray(ave)/bound_frac
    be_bf = np.asarray(block_errors)/bound_frac
    return ave_bf, be_bf
####                        Calculation  Functions                      ###
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

def get_Sa_rg_hist(rg_CA, Sa_total):
    a, xedges, yedges = np.histogram2d(rg_CA, Sa_total, 30, [
        [0.9, 2.5], [0, 25.0]], density=True, weights=None)
    a = np.log(np.flipud(a)+.000001)
    T = 300
    a = -(0.001987*T)*a

    return a, xedges, yedges


def calc_SA(trj, helix, RMS_start=1, RMS_stop=31):
    top_helix = helix.topology
    RMS = []
    for i in range(RMS_start, RMS_stop):
        sel = top_helix.select("residue %s to %s and backbone" % (i, i+6))
        rmsd = md.rmsd(trj, helix, atom_indices=sel)
        RMS.append(rmsd)
    RMS = np.asarray(RMS)

    Sa = (1.0-(RMS/0.10)**8)/(1-(RMS/0.10)**12)

    return Sa

def calc_Rg(trj):
    mass = []
    for at in trj.topology.atoms:
        mass.append(at.element.mass)
    mass_CA = len(mass)*[0.0]
    for i in trj.topology.select("name CA"):
        mass_CA[i] = 1.0
    rg_CA = md.compute_rg(trj, masses=np.array(mass_CA))
    return rg_CA

def free_energy(a, b, T, y0, ymax, x0, xmax):
    free_energy, xedges, yedges = np.histogram2d(
        a, b, 30, [[y0, ymax], [x0, xmax]], normed=True, weights=None)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.001987*T)*free_energy
    return free_energy, xedges, yedges

def free_energy_1D(a, T, x0, xmax, bins):
    free_energy, xedges = np.histogram(
        a, bins=bins, range=[x0, xmax], density=True, weights=None)
    max = np.max(free_energy)
    free_energy = np.log(free_energy+.0000001)
    free_energy = -(0.001987*T)*(free_energy-np.log(max+.0000001))
    xcenters = xedges[:-1] + np.diff(xedges)/2
    return free_energy, xcenters

def alphabeta_rmsd(phi, psi, phi_ref, psi_ref):
    alphabetarmsd = np.sum(0.5*(1+np.cos(psi-psi_ref)),
                           axis=1)+np.sum(0.5*(1+np.cos(phi-phi_ref)), axis=1)
    return alphabetarmsd

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


def calc_phipsi(trj):
    indices_phi, phis = md.compute_phi(trj)
    indices_psi, psis = md.compute_psi(trj)
    phi_label = []
    for i_phi in range(0, indices_phi.shape[0]):
        resindex = trj.topology.atom(indices_phi[i_phi][2]).residue.resSeq
        phi_label.append(resindex)
    phi_label = np.array(phi_label)
    psi_label = []
    for i_psi in range(0, indices_psi.shape[0]):
        resindex = trj.topology.atom(indices_psi[i_psi][2]).residue.resSeq
        psi_label.append(resindex)
    psi_label = np.array(psi_label)
    phipsi = []
    for i in range(0, len(phi_label)-1):
        current_phipsi = np.column_stack((phis[:, i+1], psis[:, i]))
        phipsi.append(current_phipsi)
    phipsi_array = np.array(phipsi)
    return(phipsi_array, psi_label, phi_label)

def Kd_calc(bound, conc):
    return((1-bound)*conc/bound)

def contact_map_ligand_(trj,residues, ligand_res_index,cutoff=0.6):

    contact_maps = []
    for i in range(0, residues):
        # print(i)
        contact_map = []
        for j in range(0, residues):
            dist1 = md.compute_contacts(trj, [[i, ligand_res_index]], scheme='closest-heavy')
            dist2 = md.compute_contacts(trj, [[j, ligand_res_index]], scheme='closest-heavy')
            array1 = np.asarray(dist1[0]).astype(float)
            array2 = np.asarray(dist2[0]).astype(float)
            contact1 = np.where(array1 < cutoff, 1, 0)
            contact2 = np.where(array2 < cutoff, 1, 0)
            sum = contact1 + contact2

            contacts = np.average(np.where(sum == 2, 1, 0))
            contact_map.append(contacts)
        contact_maps.append(contact_map)
    return np.asarray(contact_maps).astype(float)

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

#### Calculations for Intermolecular Interactions ####

# Calculating Aromatic Interactions 
def find_plane_normal(points):

    N = points.shape[0]
    A = np.concatenate((points[:, 0:2], np.ones((N, 1))), axis=1)
    B = points[:, 2]
    out = np.linalg.lstsq(A, B, rcond=-1)
    na_c, nb_c, d_c = out[0]
    if d_c != 0.0:
        cu = 1./d_c
        bu = -nb_c*cu
        au = -na_c*cu
    else:
        cu = 1.0
        bu = -nb_c
        au = -na_c
    normal = np.asarray([au, bu, cu])
    normal /= math.sqrt(dot(normal, normal))
    return normal

def find_plane_normal2(positions):
    v1 = positions[0]-positions[1]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[2]-positions[1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)
    return normal

def find_plane_normal2_assign_atomid(positions, id1, id2, id3):
    # Alternate approach used to check sign
    v1 = positions[id1]-positions[id2]
    v1 /= np.sqrt(np.sum(v1**2))
    v2 = positions[id3]-positions[id1]
    v2 /= np.sqrt(np.sum(v2**2))
    normal = np.cross(v1, v2)
    return normal

def get_ring_center_normal_assign_atomid(positions, id1, id2, id3):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2_assign_atomid(positions, id1, id2, id3)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal
    return center, normal

def get_ring_center_normal_(positions):
    center = np.mean(positions, axis=0)
    normal = find_plane_normal(positions)
    normal2 = find_plane_normal2(positions)
    # check direction of normal using dot product convention
    comp = np.dot(normal, normal2)
    if comp < 0:
        normal = -normal
    return center, normal

def normvector_connect(point1, point2):
    vec = point1-point2
    vec = vec/np.sqrt(np.dot(vec, vec))
    return vec

def angle(v1, v2):
    return np.arccos(np.dot(v1, v2)/(np.sqrt(np.dot(v1, v1))*np.sqrt(np.dot(v2, v2))))


def get_ring_center_normal_trj_assign_atomid(position_array, id1, id2, id3):
    length = len(position_array)
    centers = np.zeros((length, 3))
    normals = np.zeros((length, 3))
    centers_normals = np.zeros((length, 2, 3))
    print(np.shape(length), np.shape(centers), np.shape(normals))
    for i in range(0, len(position_array)):
        center, normal = get_ring_center_normal_assign_atomid(
            position_array[i], id1, id2, id3)
        centers_normals[i][0] = center
        centers_normals[i][1] = normal
    return centers_normals


# MDtraj Functions to Calculate Hydrogen Bonds with custom selections of donors and acceptors

def _get_bond_triplets_print(topology, lig_donors, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        # print("get_donors e0 e1:",e0,e1)
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
                 if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')

    nh_donors = get_donors('N', 'H')
    print("nh_donors", nh_donors)
    for i in nh_donors:
        print(topology.atom(i[0]), topology.atom(i[1]))
    oh_donors = get_donors('O', 'H')
    print("oh_donors", oh_donors)
    for i in oh_donors:
        print(topology.atom(i[0]), topology.atom(i[1]))
    sh_donors = get_donors('S', 'H')
    print("sh_donors", sh_donors)
    for i in sh_donors:
        print(topology.atom(i[0]), topology.atom(i[1]))
    for i in lig_donors:
        print(topology.atom(i[0]), topology.atom(i[1]))
    # ADD IN ADDITIONAL SPECIFIED LIGAND DONORS
    xh_donors = np.array(nh_donors + oh_donors + sh_donors+lig_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
                 if a.element.symbol in acceptor_elements and can_participate(a)]
    print("acceptors")
    for i in acceptors:
        print(topology.atom(i))
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]


def _get_bond_triplets(topology, lig_donors, exclude_water=True, sidechain_only=False):
    def can_participate(atom):
        # Filter waters
        if exclude_water and atom.residue.is_water:
            return False
        # Filter non-sidechain atoms
        if sidechain_only and not atom.is_sidechain:
            return False
        # Otherwise, accept it
        return True

    def get_donors(e0, e1):
        # Find all matching bonds
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        # Filter non-participating atoms
        atoms = [atom for atom in atoms
                 if can_participate(atom[0]) and can_participate(atom[1])]
        # Get indices for the remaining atoms
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            # make sure to get the pair in the right order, so that the index
            # for e0 comes before e1
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)

        return indices

    # Check that there are bonds in topology
    nbonds = 0
    for _bond in topology.bonds:
        nbonds += 1
        break  # Only need to find one hit for this check (not robust)
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try using '
                         'traj._topology.create_standard_bonds() to create bonds '
                         'using our PDB standard bond definitions.')

    nh_donors = get_donors('N', 'H')
    oh_donors = get_donors('O', 'H')
    sh_donors = get_donors('S', 'H')
    xh_donors = np.array(nh_donors + oh_donors + sh_donors+lig_donors)

    if len(xh_donors) == 0:
        # if there are no hydrogens or protein in the trajectory, we get
        # no possible pairs and return nothing
        return np.zeros((0, 3), dtype=int)

    acceptor_elements = frozenset(('O', 'N', 'S'))
    acceptors = [a.index for a in topology.atoms
                 if a.element.symbol in acceptor_elements and can_participate(a)]
    # Make acceptors a 2-D numpy array
    acceptors = np.array(acceptors)[:, np.newaxis]

    # Generate the cartesian product of the donors and acceptors
    xh_donors_repeated = np.repeat(xh_donors, acceptors.shape[0], axis=0)
    acceptors_tiled = np.tile(acceptors, (xh_donors.shape[0], 1))
    bond_triplets = np.hstack((xh_donors_repeated, acceptors_tiled))

    # Filter out self-bonds
    self_bond_mask = (bond_triplets[:, 0] == bond_triplets[:, 2])
    return bond_triplets[np.logical_not(self_bond_mask), :]


def _compute_bounded_geometry(traj, triplets, distance_cutoff, distance_indices,
                              angle_indices, freq=0.0, periodic=True):
    """
    Returns a tuple include (1) the mask for triplets that fulfill the distance
    criteria frequently enough, (2) the actual distances calculated, and (3) the
    angles between the triplets specified by angle_indices.
    """
    # First we calculate the requested distances
    distances = md.compute_distances(
        traj, triplets[:, distance_indices], periodic=periodic)

    # Now we discover which triplets meet the distance cutoff often enough
    prevalence = np.mean(distances < distance_cutoff, axis=0)
    mask = prevalence > freq

    # Update data structures to ignore anything that isn't possible anymore
    triplets = triplets.compress(mask, axis=0)
    distances = distances.compress(mask, axis=1)

    # Calculate angles using the law of cosines
    abc_pairs = zip(angle_indices, angle_indices[1:] + angle_indices[:1])
    abc_distances = []

    # Calculate distances (if necessary)
    for abc_pair in abc_pairs:
        if set(abc_pair) == set(distance_indices):
            abc_distances.append(distances)
        else:
            abc_distances.append(md.compute_distances(traj, triplets[:, abc_pair],
                                                      periodic=periodic))

    # Law of cosines calculation
    a, b, c = abc_distances
    cosines = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    np.clip(cosines, -1, 1, out=cosines)  # avoid NaN error
    angles = np.arccos(cosines)
    return mask, distances, angles


def baker_hubbard2(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                   distance_cutoff=0.35, angle_cutoff=150, lig_donor_index=[]):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets

    # ADD IN LIGAND HBOND DONORS
    add_donors = lig_donor_index

    bond_triplets = _get_bond_triplets(traj.topology,
                                       exclude_water=exclude_water, lig_donors=add_donors, sidechain_only=sidechain_only)

    mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
                                                        distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)

    # Find triplets that meet the criteria
    presence = np.logical_and(
        distances < distance_cutoff, angles > angle_cutoff)
    mask[mask] = np.mean(presence, axis=0) > freq
    return bond_triplets.compress(mask, axis=0)


def print_donors_acceptors(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False,
                           distance_cutoff=0.35, angle_cutoff=150, lig_donor_index=[]):

    angle_cutoff = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard requires that traj contain topology '
                         'information')

    # Get the possible donor-hydrogen...acceptor triplets

    # ADD IN LIGAND HBOND DONORS
    # add_donors=[[296,318],[296,331]]
    # Manually tell it where to find proton donors on ligand
    # LIG58-O5 LIG58-H24
    # LIG58-O1 LIG58-H12
    # LIG58-N LIG58-H15
    # add_donors=[[768,796],[750,784],[752,787]]
    add_donors = lig_donor_index

    bond_triplets_print = _get_bond_triplets_print(traj.topology,
                                                   exclude_water=exclude_water, lig_donors=add_donors, sidechain_only=sidechain_only)

    # mask, distances, angles = _compute_bounded_geometry(traj, bond_triplets,
    #    distance_cutoff, [1, 2], [0, 1, 2], freq=freq, periodic=periodic)

    # Find triplets that meet the criteria
    # presence = np.logical_and(distances < distance_cutoff, angles > angle_cutoff)
    # mask[mask] = np.mean(presence, axis=0) > freq
    return

# calculating contacts for ligand residues 
def calc_contact(trj, ligand_residue_index, residues=38):
    contact_pairs = np.zeros((residues, 2))

    for i in range(0, residues):
        contact_pairs[i] = [i, ligand_residue_index]
    contact = md.compute_contacts(trj, contact_pairs, scheme='closest-heavy')
    contacts = np.asarray(contact[0]).astype(float)
    cutoff = 0.6
    contact_matrix = np.where(contacts < cutoff, 1, 0)

    return contact_matrix

def create_pos_contact(trj, cutoff=0.5, ligand_residue_index=38): 
    """
    # get the indices of a positive contact given the trajectory 
    """
    contact_array = np.empty([1, ligand_residue_index])
    contact = calc_contact(trj, ligand_residue_index)
    contact_array = np.append(contact_array, contact, axis=0)
    contact_array = np.delete(contact_array, 0, 0)
    avg_contact = np.mean(contact_array, axis=1)
    pos_contact = np.where(avg_contact >= cutoff)[0]
    print("this is the population of the bound frames for the given cutoff:", len(pos_contact)/trj.n_frames)
    return pos_contact

def contact_arr_avg(trj, ligand_residue_index=38, residues=38): 
    """
    # get the indices of a positive contact given the trajectory 
    """
    contact_array = np.empty([1, residues])
    contact = calc_contact(trj, ligand_residue_index)
    contact_array = np.append(contact_array, contact, axis=0)
    contact_array = np.delete(contact_array, 0, 0)
    avg_contact = np.mean(contact_array, axis=0)
    return avg_contact


####                   More  Calculations Functions                     ####
def add_color_to_string(s, values, vmin=None, vmax=None, cmap_name='viridis',fname=None):
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    cmap = get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(len(s)*0.5, 1))
    ax.axis('off')

    counter = 0 
    x = 0
    for char, value in zip(s, values):
        norm_value = (value - vmin) / (vmax - vmin)
        color = cmap(norm_value)
        
        # text = ax.text(x, 0.5, char, color=color, fontsize=12, ha='left', va='center', transform=ax.transAxes)
        # text.set_path_effects([withStroke(linewidth=0.7, foreground='black')])
        
        if counter >= 20: 
            if counter == 20: 
                x = 0
            text = ax.text(x, 0.0, char, color=color, fontsize=14, ha='left', va='center', transform=ax.transAxes)
            text.set_path_effects([withStroke(linewidth=0.7, foreground='black')])
        else: 
            text = ax.text(x, 0.5, char, color=color, fontsize=14, ha='left', va='center', transform=ax.transAxes)
            text.set_path_effects([withStroke(linewidth=0.7, foreground='black')])

        x += 0.01
        counter +=1

    plt.savefig(fname, dpi=400)
    plt.show()

def get_centroid(traj_basin): 
    """
    reference: https://mdtraj.org/1.9.3/examples/centroids.html
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

    return int(index)

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

####                 20 rep analysis functions                      ###
def contact_matrix_(trj,top,residues,ligand_residue_index,cutoff = 0.6):
    
    ligand=top.select("resid "+str(ligand_residue_index))
    protein=top.select("residue 1 to " + str(residues))
    
    ligand_atomid = []
    for atom in ligand:
        indices = []
        indices.append(atom)
        indices.append(top.atom(atom))
        ligand_atomid.append(indices)
        
    protein_atomid = []
    for atom in protein:
        indices = []
        indices.append(atom)
        indices.append(top.atom(atom))
        protein_atomid.append(indices)
    
    # subtract one because an indexing error 
    contact_pairs = np.zeros((residues, 2))
    for i in range(0, residues):
        contact_pairs[i] = [i, ligand_residue_index]
    contact = md.compute_contacts(trj, contact_pairs, scheme='closest-heavy')
    
    contacts = np.asarray(contact[0]).astype(float)
    contact_matrix = np.where(contacts < cutoff, 1, 0)
    
    return contact_matrix

def bound_frac_kd(box_len,nreps,contact_matrix=[],trj_frames=[]):
    
    Box_L = box_len
    # Convert nM to meters for Box_V in M^3
    Box_V = (Box_L*10**-9)**3
    # Convert Box_V to L
    Box_V_L = Box_V*1000
    #Concentraion in Mols/L
    Concentration = 1/(Box_V_L*(6.023*10**23))
    #print("L:", Box_L, "V:", Box_V, "Conc:", Concentration)
    OUT=[]
    for i in range(nreps):

        contact_rows = np.sum(contact_matrix[int(trj_frames[str(i)][0]):int(trj_frames[str(i)][1])], axis=1)
        bf, bf_be = get_blockerror_pyblock(np.where(contact_rows > 0, 1, 0))
    
        upper = bf+bf_be
        KD = Kd_calc(bf, Concentration)
        KD_upper = Kd_calc(upper, Concentration)
        KD_error = KD-KD_upper
    
        kd=np.round(KD*1000,4)
        kde=np.round(KD_error*1000,4)
    
        OUT.append([bf, bf_be, kd, kde])
    
    OUT=np.array(OUT)
    
    return OUT

def vec_angles(trj,atom_indices=[]):
    
    #Get xyz co-ordinates
    xyz=[]
    for  atom_idx in atom_indices :
        a=[]
        for frame_idx in range(trj.n_frames):

            a.append(trj.xyz[frame_idx, atom_idx,:].astype(float))
        xyz.append(a)    

    xyz=np.array(xyz)
    xyz.shape

    #Define vectors with 2nd atom as starting point
    V=[]
    v1=xyz[0]-xyz[1]
    v2=xyz[2]-xyz[1]
    V.append(v1)
    V.append(v2)

    #Compute angles between two vectors
    angles=[]
    for i in range(trj.n_frames):
    
        a=np.rad2deg(np.arccos(np.dot(V[0][i],V[1][i])/
                               (np.sqrt(np.dot(V[0][i],V[0][i])*np.dot(V[1][i],V[1][i])))))
        
        angles.append(a)
    
    angles=np.array(angles)
    
    return angles

def norm_weights(file_name):

    colvar=np.loadtxt(file_name,comments=['#','@'])
    num_cvs=len(colvar[0])-1

    kt=2.494339
    w=np.exp((colvar[:,num_cvs]/kt))

    max_=np.sum(w)
    w_norm=w/max_

    return num_cvs, w, w_norm

def kd_timecourse(box_len,n_reps,contact_matrix=[],trj_frames=[],sim_length=None,stride=None):
    
    Box_L = box_len
    # Convert nM to meters for Box_V in M^3
    Box_V = (Box_L*10**-9)**3
    # Convert Box_V to L
    Box_V_L = Box_V*1000
    #Concentraion in Mols/L
    Concentration = 1/(Box_V_L*(6.023*10**23))
    #print("L:", Box_L, "V:", Box_V, "Conc:", Concentration)
    OUT=[]
    for i in range(n_reps):

        contact_rows = np.sum(contact_matrix[int(trj_frames[str(i)][0]):int(trj_frames[str(i)][1])], axis=1)
        contact_binary=np.where(contact_rows > 0, 1, 0)
    
        time = np.linspace(0, sim_length, len(contact_binary))
        boundfrac_by_frame = []
        t2 = []
        err_by_frame = []
        err_upper = []
        err_lower = []
        #stride = 100
        
        for j in range(stride, len(contact_binary), stride):
            #Data = np.asarray(contact_binary[0:j])
            bf, be = get_blockerror_pyblock_nanskip(np.asarray(contact_binary[0:j]))
            boundfrac_by_frame.append(bf)
            err_by_frame.append(be)
            err_upper.append(bf-be)
            err_lower.append(bf+be)
            t2.append(time[j])

        Kd = Kd_calc(np.asarray(boundfrac_by_frame), Concentration)*1000
        Kd_upper = Kd_calc(np.asarray(err_upper), Concentration)*1000
        Kd_lower = Kd_calc(np.asarray(err_lower), Concentration)*1000
        
        OUT.append(np.column_stack((t2, Kd, Kd_upper, Kd_lower, boundfrac_by_frame)))
    
    return np.array(OUT)

def bf_timecourse(n_reps,contact_matrix=[],trj_frames=[],sim_length=None,stride=None):
    
    OUT=[]
    for i in range(n_reps):

        contact_rows = np.sum(contact_matrix[int(trj_frames[str(i)][0]):int(trj_frames[str(i)][1])], axis=1)
        contact_binary=np.where(contact_rows > 0, 1, 0)
    
        time = np.linspace(0, sim_length, len(contact_binary))
        boundfrac_by_frame = []
        t2 = []
        err_by_frame = []
        err_upper = []
        err_lower = []
        #stride = 100
        
        for j in range(stride, len(contact_binary), stride):
            #Data = np.asarray(contact_binary[0:j])
            bf, be = get_blockerror_pyblock_nanskip(np.asarray(contact_binary[0:j]))
            boundfrac_by_frame.append(bf)
            err_by_frame.append(be)
            err_upper.append(bf-be)
            err_lower.append(bf+be)
            t2.append(time[j])
        
        OUT.append(np.column_stack((t2, err_upper, err_lower, boundfrac_by_frame)))
    
    return np.array(OUT)

def charge_contacts_rw_(trj,top,residue_offset,residues=38, cutoff=0.5, weights_norm=0, Ligand_Pos_Charges=[],Ligand_Neg_Charges=[],weights=[]):
    
    def add_charge_pair(pairs,pos,neg,contact_prob):
        if pos not in pairs: 
            pairs[pos] = {} 
        if neg not in pairs[pos]:
            pairs[pos][neg] = {}
        pairs[pos][neg] = contact_prob
        
        
    Protein_Pos_Charges=top.select("(resname ARG and name CZ) or (resname LYS and name NZ) or (resname HIE and name NE2) or (resname HID and name ND1)")
    Protein_Neg_Charges=top.select("(resname ASP and name CG) or (resname GLU and name CD) or (name OXT) or (resname NASP and name CG)")
    
    neg_res=[]
    pos_res=[]
    
    for i in Protein_Neg_Charges:
        neg_res.append(top.atom(i).residue.resSeq)

    for i in Protein_Pos_Charges:
        pos_res.append(top.atom(i).residue.resSeq)
        
    charge_pairs_ligpos=[]                      
    for i in Ligand_Pos_Charges:
        for j in Protein_Neg_Charges:              
            charge_pairs_ligpos.append([i,j])
            pos=top.atom(i)
            neg=top.atom(j) 

    charge_pairs_ligneg=[]                      
    for i in Ligand_Neg_Charges:
        for j in Protein_Pos_Charges:              
            charge_pairs_ligneg.append([i,j])
            pos=top.atom(i)
            neg=top.atom(j)
            
    if len(charge_pairs_ligpos) != 0:
        contact  = md.compute_distances(trj, charge_pairs_ligpos)
        contacts = np.asarray(contact).astype(float)
#        cutoff=0.5
        neg_res_contact_frames=np.where(contacts < cutoff, 1, 0)
        
        if len(weights)>0 :
        
            #Re-weighting
            neg_res_contact_frames_re=[]
            for i in range(0,len(neg_res_contact_frames[0])):
                
                neg_res_contact_frames_re.append(np.dot(neg_res_contact_frames[:,i],weights))
#             contact_prob_ligpos_re = np.sum(neg_res_contact_frames_re,axis=0)/trj.n_frames
#             contact_prob_ligpos = np.sum(neg_res_contact_frames,axis=0)/trj.n_frames
        
    if len(charge_pairs_ligneg) != 0:
        contact  = md.compute_distances(trj, charge_pairs_ligneg)
        contacts = np.asarray(contact).astype(float)
#        cutoff=0.5
        pos_res_contact_frames=np.where(contacts < cutoff, 1, 0)
        
        if len(weights)>0 :
        
            #Re-weighting
            pos_res_contact_frames_re=[]
            for i in range(0,len(pos_res_contact_frames[0])):
                
                pos_res_contact_frames_re.append(np.dot(pos_res_contact_frames[:,i],weights_norm))
#             contact_prob_ligneg_re = np.sum(neg_res_contact_frames_re,axis=0)/trj.n_frames
#             contact_prob_ligneg = np.sum(pos_res_contact_frames,axis=0)/trj.n_frames
    
#     charge_pair_names={}
#     for i in range(0,len(charge_pairs_ligpos)):
#         pos=top.atom(charge_pairs_ligpos[i][0])
#         neg=top.atom(charge_pairs_ligpos[i][1])      
#         add_charge_pair(charge_pair_names,pos,neg,contact_prob_ligpos[i])

#     for i in range(0,len(charge_pairs_ligneg)):
#         pos=top.atom(charge_pairs_ligneg[i][1])
#         neg=top.atom(charge_pairs_ligneg[i][0])      
#         add_charge_pair(charge_pair_names,pos,neg,contact_prob_ligneg[i])
        
    neg_res_index=np.array(neg_res)-residue_offset
    
    if len(weights)>0 :
    
        Charge_Contacts_reweight=np.zeros(residues)
        for i in range(0,len(neg_res)):
            Charge_Contacts_reweight[neg_res[i]-residue_offset]=neg_res_contact_frames_re[i]
            
        return Charge_Contacts_reweight
        
    else :
        
        Charge_Contacts=np.zeros((trj.n_frames,residues))
        for i in range(0,len(neg_res)):
            Charge_Contacts[:,neg_res[i]-residue_offset]=neg_res_contact_frames[:,i]

        
        return Charge_Contacts
def hphob_contacts_rw_(trj,top, residues, ligand_residue_index, cutoff=0.4,weights=[]):
    
    def add_contact_pair(pairs, a1, a2, a1_id, a2_id, prot_res, contact_prob):
        if prot_res not in pairs:
            pairs[prot_res] = {}
        if a2 not in pairs[prot_res]:
            pairs[prot_res][a2] = {}
        if a1_id not in pairs[prot_res][a2]:
            pairs[prot_res][a2][a1_id] = contact_prob
    
    
    ligand_hphob = top.select("residue "+str(ligand_residue_index)+" and element C")
    protein_hphob = top.select("residue 1 to "+str(residues)+" and element C")
    
    ligand_hphob_atoms = []
    for atom in ligand_hphob:
        ligand_hphob_atoms.append(top.atom(atom))

    protein_hphob_atoms = []
    for atom in protein_hphob:
        protein_hphob_atoms.append(top.atom(atom))
        
    hphob_pairs = []
    for i in ligand_hphob:
        for j in protein_hphob:
            hphob_pairs.append([i, j])
            
    contact = md.compute_distances(trj, hphob_pairs)
    contacts = np.asarray(contact)

    contact_frames = np.where(contacts < cutoff, 1, 0)
    
    # Cast hydrophobic contacts as per residue in each frame
    
    Hphob_res_contacts = np.zeros((trj.n_frames, residues))
    for frame in range(trj.n_frames):
        if np.sum(contact_frames[frame]) > 0:
            contact_pairs = np.where(contact_frames[frame] == 1)
            for j in contact_pairs[0]:
                residue = top.atom(hphob_pairs[j][1]).residue.resSeq
                Hphob_res_contacts[frame][residue-1] = 1
    
    if len(weights)>0 :
    
        hphob_contact_reweight=[]
        for i in range(0,len(Hphob_res_contacts[0])):
            c=np.dot(Hphob_res_contacts[:,i],weights)
            hphob_contact_reweight.append(c)
                
        return hphob_contact_reweight
    
    else :
                
        return Hphob_res_contacts

def aro_contacts_rw_(trj,top,residue_offset, residues, ligand_rings=[],weights=[]): 
    n_rings = len(ligand_rings)
    print("Ligand Aromatics Rings:", n_rings)

    ligand_ring_params = []
    for i in range(0, n_rings):
        ring = np.array(ligand_rings[i])

        positions = trj.xyz[:, ring, :]

        ligand_centers_normals = get_ring_center_normal_trj_assign_atomid(positions, 0, 1, 2)
        ligand_ring_params.append(ligand_centers_normals)


    #Find Protein Aromatic Rings
    #Add Apropriate HIS name if there is a charged HIE OR HIP in the structure 
    prot_rings = []
    aro_residues = []
    prot_ring_name = []
    prot_ring_index = []

    aro_select = top.select("resname TYR PHE HIS TRP and name CA")
    for i in aro_select:
        atom = top.atom(i)
        resname = atom.residue.name

        if resname == "TYR":
            ring = top.select(
                "resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)

        if resname == "TRP":
            ring = top.select(
                "resid %s and name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2" % atom.residue.index)

        if resname == "HIS":
            ring = top.select("resid %s and name CG ND1 CE1 NE2 CD2" %
                              atom.residue.index)

        if resname == "PHE":
            ring = top.select(
                "resid %s and name CG CD1 CD2 CE1 CE2 CZ" % atom.residue.index)

        prot_rings.append(ring)
        prot_ring_name.append(atom.residue)
        prot_ring_index.append(atom.residue.index+residue_offset)

    prot_ring_params = []
    for i in range(0, len(prot_rings)):
        ring = np.array(prot_rings[i])
        positions = trj.xyz[:, ring, :]
        ring_centers_normals = get_ring_center_normal_trj_assign_atomid(positions, 0, 1, 2)
        prot_ring_params.append(ring_centers_normals)

#    frames = trj.n_frames
    sidechains = len(prot_rings)
    ligrings = len(ligand_rings)

    print(trj.n_frames, sidechains)

    Ringstacked_old = {}
    Ringstacked = {}
    Quadrants = {}
    Stackparams = {}
    Aro_Contacts = {}
    Pstack = {}
    Tstack = {}

    for l in range(0, ligrings):
        name = "Lig_ring.%s" % l

        Stackparams[name] = {}
        Pstack[name] = {}
        Tstack[name] = {}
        Aro_Contacts[name] = {}
        alphas = np.zeros(shape=(trj.n_frames, sidechains))
        betas = np.zeros(shape=(trj.n_frames, sidechains))
        dists = np.zeros(shape=(trj.n_frames, sidechains))
        thetas = np.zeros(shape=(trj.n_frames, sidechains))
        phis = np.zeros(shape=(trj.n_frames, sidechains))
        pstacked_old = np.zeros(shape=(trj.n_frames, sidechains))
        pstacked = np.zeros(shape=(trj.n_frames, sidechains))
        tstacked = np.zeros(shape=(trj.n_frames, sidechains))
        stacked = np.zeros(shape=(trj.n_frames, sidechains))
        aro_contacts = np.zeros(shape=(trj.n_frames, sidechains))
        quadrant=np.zeros(shape=(trj.n_frames,sidechains))

        for i in range(0, trj.n_frames):
            ligcenter = ligand_ring_params[l][i][0]
            lignormal = ligand_ring_params[l][i][1]
            for j in range(0, sidechains):
                protcenter = prot_ring_params[j][i][0]
                protnormal = prot_ring_params[j][i][1]
                dists[i, j] = np.linalg.norm(ligcenter-protcenter)
                connect = normvector_connect(protcenter, ligcenter)
                # alpha is the same as phi in gervasio/Procacci definition
                alphas[i, j] = np.rad2deg(angle(connect, protnormal))
                betas[i, j] = np.rad2deg(angle(connect, lignormal))
                theta = np.rad2deg(angle(protnormal, lignormal))
                thetas[i, j] = np.abs(theta)-2*(np.abs(theta)
                                                > 90.0)*(np.abs(theta)-90.0)
                phi = np.rad2deg(angle(protnormal, connect))
                phis[i, j] = np.abs(phi)-2*(np.abs(phi) > 90.0)*(np.abs(phi)-90.0)

        for j in range(0, sidechains):
            name2 = prot_ring_index[j]
            res2 = prot_ring_name[j]

            Ringstack = np.column_stack((dists[:, j], alphas[:, j], betas[:, j], thetas[:, j], phis[:, j]))
            stack_distance_cutoff = 0.65
            r = np.where(dists[:, j] <= stack_distance_cutoff)[0]
            aro_contacts[:, j][r] = 1

            # New Definitions
            # p-stack: r < 6.5 Å, θ < 60° and ϕ < 60°.
            # t-stack: r < 7.5 Å, 75° < θ < 90° and ϕ < 60°.
            p_stack_distance_cutoff = 0.65
            t_stack_distance_cutoff = 0.75
            r_pstrict = np.where(dists[:, j] <= p_stack_distance_cutoff)[0]
            r_tstrict = np.where(dists[:, j] <= t_stack_distance_cutoff)[0]

            a=np.where(alphas[:,j] >= 135)
            b=np.where(alphas[:,j] <= 45)
            c=np.where(betas[:,j] >= 135)
            d=np.where(betas[:,j] <= 45)
            e=np.where(dists[:,j] <= 0.5)
            q1=np.intersect1d(np.intersect1d(b,c),e)
            q2=np.intersect1d(np.intersect1d(a,c),e)
            q3=np.intersect1d(np.intersect1d(b,d),e)
            q4=np.intersect1d(np.intersect1d(a,d),e)
            stacked[:,j][q1]=1
            stacked[:,j][q2]=1
            stacked[:,j][q3]=1
            stacked[:,j][q4]=1
            quadrant[:,j][q1]=1
            quadrant[:,j][q2]=2
            quadrant[:,j][q3]=3
            quadrant[:,j][q4]=4
            total_stacked=len(q1)+len(q2)+len(q3)+len(q4)

            Stackparams[name][name2]=Ringstack

            f = np.where(thetas[:, j] <= 45)
            g = np.where(phis[:, j] <= 60)
            h = np.where(thetas[:, j] >= 75)

            pnew = np.intersect1d(np.intersect1d(f, g), r_pstrict)
            tnew = np.intersect1d(np.intersect1d(h, g), r_tstrict)
            pstacked[:, j][pnew] = 1
            tstacked[:, j][tnew] = 1
            stacked[:, j][pnew] = 1
            stacked[:, j][tnew] = 1
            total_stacked = len(pnew)+len(tnew)
            Stackparams[name][name2] = Ringstack
        Pstack[name] = pstacked
        Tstack[name] = tstacked
        Aro_Contacts[name] = aro_contacts
        Ringstacked[name] = stacked
        Quadrants[name]=quadrant
    
    
    aro_res_index = np.array(prot_ring_index)-residue_offset

    aromatic_stacking_contacts_r0 = np.zeros((trj.n_frames, residues))

    for i in range(0, len(aro_res_index)):
        aromatic_stacking_contacts_r0[:, aro_res_index[i]] = Ringstacked['Lig_ring.0'][:, i]
        
    if len(weights)>0 : 
        
        aro_stack_reweight=[]
        for i in range(0,len(aromatic_stacking_contacts_r0[0])):
            
            aro_stack_reweight.append(np.dot(aromatic_stacking_contacts_r0[:,i],weights))
            
        return aro_stack_reweight
    
    else:
        
        return aromatic_stacking_contacts_r0

def hbond_rw_(trj,top,residues,ligand_residue_index,lig_hbond_donors=[],weights=[]):

    # Select Ligand Residues
    ligand = top.select("resid "+str(ligand_residue_index))
    # Select Protein Residues
    protein = top.select("resid 0 to " + str(residues-1))


    HBond_PD = np.zeros((trj.n_frames, top.n_residues))
    HBond_LD = np.zeros((trj.n_frames, top.n_residues))
    Hbond_pairs_PD = {}
    Hbond_pairs_LD = {}


    def add_hbond_pair(donor, acceptor, hbond_pairs, donor_res):
        if donor_res not in hbond_pairs:
            hbond_pairs[donor_res] = {}
        if donor not in hbond_pairs[donor_res]:
            hbond_pairs[donor_res][donor] = {}
        if acceptor not in hbond_pairs[donor_res][donor]:
            hbond_pairs[donor_res][donor][acceptor] = 0
        hbond_pairs[donor_res][donor][acceptor] += 1

    # Donor & Acceptors Definitions from DESRES paper:
    # ligdon = mol.select('chain B and (nitrogen or oxygen or sulfur) and (withinbonds 1 of hydrogen)')
    # ligacc = mol.select('chain B and (nitrogen or oxygen or sulfur)')
    # protdon = mol.select('chain A and (nitrogen or oxygen or sulfur) and (withinbonds 1 of hydrogen)')
    # protacc = mol.select('chain A and (nitrogen or oxygen or sulfur)')


    for frame in range(trj.n_frames):
        hbonds = baker_hubbard2(trj[frame], angle_cutoff=150, distance_cutoff=0.35, 
                                lig_donor_index=lig_hbond_donors)

        for hbond in hbonds:
            if ((hbond[0] in protein) and (hbond[2] in ligand)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc = top.atom(hbond[2])
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_PD[frame][donor_res-1] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_PD, donor_res)
            if ((hbond[0] in ligand) and (hbond[2] in protein)):
                donor = top.atom(hbond[0])
                donor_id = hbond[0]
                donor_res = top.atom(hbond[0]).residue.resSeq
                acc = top.atom(hbond[2])
                acc_id = hbond[2]
                acc_res = top.atom(hbond[2]).residue.resSeq
                HBond_LD[frame][acc_res-1] = 1
                add_hbond_pair(donor, acc, Hbond_pairs_LD, acc_res)


    HB_Total = HBond_PD+HBond_LD
    
    if len(weights):
        
        HB_re=[]
        for i in range(0,len(HB_Total[0])):
            c=np.dot(HB_Total[:,i],weights)
            HB_re.append(c) 
        
        return HB_re
    
    else :
        
        return HB_Total
####                          plotting                              ###
def make_chimera_lig_contact_file(contact_probability, traj, traj_dir, psystem, c=-1, lig_resid=38): 
    # calculate the ligand centroid, with all its atoms 
    temptrj_lig = traj.atom_slice(traj.top.select("resid "+str(lig_resid)))
    lig_centroid = temptrj_lig.slice(get_centroid(temptrj_lig))
    lig_centroid.save_gro(traj_dir + "/ct_"+ psystem +"_5Acutoff_centroid_C"+str(c+1) + "_" +".gro")

    # get atom labels 
    labels = np.array([str(atom).split("-")[1] for atom in lig_centroid.top.atoms])
    file_path = outdir + psystem +".zip.lig.5Acutoff.contacts.all."+"c"+str(c+1)+ ".txt"

    # automated file writing for each atom 
    with open(file_path, "w") as file:
        file.write("attribute: contact_all \n")
        file.write("match mode: 1-to-1 \n")
        file.write("recipient: atoms \n")

        for n in range(len(labels)):
            # add residue index
            file.write("\t"+":"+str(lig_resid+1)+"@" + str(labels[n]) + "\t" + str(np.round(contact_probability[n],3)) + "\n")
    return

def ligand_contact_probability(traj, lig_resid = 38, prot_start = 0, prot_end = 37): 
    # create queries for 
    lig_atoms = traj.top.select("resid " + str(lig_resid))
    prot_atoms = traj.top.select("resid " + str(prot_start) + " to " + str(prot_end))
    lig_contact_probabilities = []

    for atom in lig_atoms: 
        lig_contacts = md.compute_neighbors(traj, cutoff=0.5, query_indices=[atom], haystack_indices=prot_atoms, periodic=True)
        lig_contacts = np.array([x.size for x in lig_contacts])
        lig_contacts = np.where(lig_contacts > 0 , 1, 0)
        lig_prob = np.sum(lig_contacts)/ traj.n_frames
        lig_contact_probabilities.append(lig_prob)

    return lig_contact_probabilities

def make_chimera_prot_contact_file(contact_probability, traj_dir, psystem, c="-1", res_num=38): 
    # calculate the ligand centroid, with all its atoms 

    # get atom labels 
    labels = np.arange(1,res_num+1)
    file_path = traj_dir + psystem +".prot.chimera.contact."+c+ ".txt"

    # automated file writing for each atom 
    with open(file_path, "w") as file:
        file.write("attribute: contact_all \n")
        file.write("match mode: 1-to-1 \n")
        file.write("recipient: residues \n")

        for n in labels:
            # add residue index
            file.write("\t"+":" +str(n) + "\t" + str(np.round(contact_probability[n-1],3)) + "\n")
    return

def plt_box(xmin,xmax,ymin,ymax):
    plt.vlines(xmin, ymin, ymax, colors="k")
    plt.vlines (xmax, ymin, ymax, colors="k")
    plt.hlines(ymin, xmin, xmax, colors="k")
    plt.hlines(ymax, xmin, xmax, colors="k")
    return

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
