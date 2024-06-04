# example us of this file for pbc corrected temperature replicas  
# python ../../scripts/analysis_demux_pbc.py ./pbc/ ../../structure_files/hiapp_s20g_apo.gro outdir_s20g_pbc 0 20 ../../structure_files/helix_s20g.pdb

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import math
import pandas as pd
from numpy import log2, zeros, mean, var, sum, loadtxt, arange, \
    array, cumsum, dot, transpose, diagonal, floor
from numpy.linalg import inv, lstsq
import pyblock
import sys 
import os 

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
sns.set_style("whitegrid")

# load in trajectory and pdb, can make this into an argument in the future
traj_path = sys.argv[1] 
pdb = sys.argv[2]
outdir = "./" + sys.argv[3] + "/"
demux = int(sys.argv[4])
nreps = int(sys.argv[5])
helixpdb = sys.argv[6]

trajectory = ""
if demux == 1: 
    trajectory = traj_path + "Demux_1.xtc"
# pbc 
else: 
    trajectory = traj_path + "pbc_1.xtc"

if not os.path.exists(outdir):
    os.makedirs(outdir)

# load the first replica
trj = md.load(trajectory, top=pdb, stride=10)
top = trj.topology

def block(x):
    # preliminaries
    d = log2(len(x))
    if (d - floor(d) != 0):
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
    q = np.array([6.634897,  9.210340,  11.344867, 13.276704, 15.086272,
              16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
              24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
              31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
              38.932173, 40.289360, 41.638398, 42.979820, 44.314105,
              45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0, d):
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

def calc_SA(trj, helix, start, stop):
    r0 = .10
    RMS_start = start
    RMS_stop = stop
    RMS = []
    for i in range(RMS_start, RMS_stop):
        sel = helix.topology.select("residue %s to %s and name CA" % (i, i+6))
        rmsd = md.rmsd(trj, helix, atom_indices=sel)
        RMS.append(rmsd)
    RMS = np.asarray(RMS)
    Sa_sum = np.zeros((trj.n_frames))
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

def free_energy_reweight(a, b, T, y0, ymax, x0, xmax, weight):
    free_energy, xedges, yedges = np.histogram2d(
        a, b, 30, [[y0, ymax], [x0, xmax]], normed=True, weights=weight)
    free_energy = np.log(np.flipud(free_energy)+.000001)
    free_energy = -(0.001987*T)*free_energy
    return free_energy, xedges, yedges

def free_energy_1D_noscale(a, T, x0, xmax, bins):
    free_energy, xedges = np.histogram(
        a, bins=bins, range=[x0, xmax], density=True, weights=None)
    free_energy = np.log(free_energy+.000001)
    free_energy = -(0.001987*T)*free_energy
    xcenters = xedges[:-1] + np.diff(xedges)/2
    return free_energy, xcenters

def free_energy_1D(a, T, x0, xmax, bins):
    free_energy, xedges = np.histogram(
        a, bins=bins, range=[x0, xmax], density=True, weights=None)
    max = np.max(free_energy)
    free_energy = np.log(free_energy+.0000001)
    free_energy = -(0.001987*T)*(free_energy-np.log(max+.0000001))
    xcenters = xedges[:-1] + np.diff(xedges)/2
    return free_energy, xcenters

def free_energy_1D_reweight(a, T, x0, xmax, bins, weight):
    free_energy, xedges = np.histogram(
        a, bins=bins, range=[x0, xmax], density=True, weights=weight)
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

def make_contacts(trj): 
    contact_maps = []
    contact_distances = []
    for i in range(1, 38):
        contact_map = []
        contact_distance = []
        for j in range(1, 38):
            if i == j:
                contacts = 0
            else:
                dist = md.compute_contacts(trj, [[i, j]])
                array = np.asarray(dist[0]).astype(float)
                distance = np.average(array)
                contact_distance.append(distance)
                contact = np.where(array < 1.2, 1, 0)
                contacts = np.average(contact)
            contact_map.append(contacts)
        contact_maps.append(contact_map)
        contact_distances.append(contact_distance)
    final_map = np.asarray(contact_maps).astype(float)
    final_distance = np.asarray(contact_distances).astype(float)
    return final_map, final_distance

# get the temperature span of replcias 
def get_temps(tmax= 500, tmin= 300, nreps = 20):
    temps = []
    for i in range(0,20): 
        t=tmin*math.exp(i*math.log(tmax/tmin)/(n-1))
        temps.append(round(t,2))
    return temps

def compute_rg(trj):
    mass = []
    for at in trj.topology.atoms:
        mass.append(at.element.mass)
    mass_CA = len(mass)*[0.0]
    # put the CA entries equal to 1.0
    for i in trj.topology.select("name CA"):
        mass_CA[i] = 1.0
    # calculate CA radius of gyration
    return md.compute_rg(trj, masses=np.array(mass_CA))

def compute_sa(trj): 
    RMS_start = 1
    RMS_stop = 31
    RMS = []
    for i in range(RMS_start, RMS_stop):
        sel = top_helix.select("residue %s to %s and backbone" % (i, i+6))
        rmsd = md.rmsd(trj, helix, atom_indices=sel)
        RMS.append(rmsd)
    RMS = np.asarray(RMS)

    Sa_sum = np.zeros((trj.n_frames))
    Sa = (1.0-(RMS/0.10)**8)/(1-(RMS/0.10)**12)
    Sa_ave = np.average(Sa, axis=1)
    Sa_total = np.sum(Sa, axis=0)
    return Sa_total

files = []
if demux == 1: 
    files = [traj_path + '/Demux_' + str(i)+ ".xtc" for i in range(1,nreps+1)]
else: 
    files = [traj_path + '/pbc_' + str(i)+ ".xtc" for i in range(1,nreps+1)]

# create sequence to plot with contact maps 
sequence = [str(residue) for residue in trj.topology.residues] [0:-1]
for i in range(len(sequence)): 
    # get rid of every other sequence 
    if i%2 == 1: 
        sequence[i] = ""

sequence_ol = []
for i in range(0, len(sequence)):
    sequence_ol.append(str(trj.topology.residue(i).code))
sequence_ol = sequence_ol[:-1]
sequence_ol.append("    NH2") # append NH2

# indices for the plot, 5 rows and 4 columns
indices = []
for j in range(1,6): 
    for k in range(1,5): 
        indices.append((j,k))

# get temperatures 
temps = []
tmax= 500
tmin= 300
for i in range(0,nreps): 
    t=tmin*math.exp(i*math.log(tmax/tmin)/(nreps-1))
    temps.append(round(t,2))

# CREATE CONTACTS 
# create colors for the reps 
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, nreps)]
fig = plt.figure(figsize=(16, 25))
rows = 5
columns = 4
grid = plt.GridSpec(rows, columns, wspace = .3, hspace = .3)

for n, file in enumerate(files): 
    trj = md.load(file, top=pdb, stride=10)
    trj.center_coordinates()  
    contact_map, distance_map = make_contacts(trj)

    exec (f"plt.subplot(grid{[n]})") 
   
    im = plt.imshow(contact_map, vmax=1.0, vmin=0.0, cmap="jet", origin="lower")
    plt.xticks(range(0, 37), sequence_ol, rotation=0, size=10)
    plt.yticks(range(0, 37), sequence_ol, rotation=0, size=10)
   
    if demux == 1: 
        plt.title("Demux " + str(n))
    else: 
        plt.title("Temperature Replica " + str(temps[n]) + "K")

# get color bar 
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.3)
#cax = plt.axes([0.85, 0.05, 0.03, 0.3])
plt.savefig(outdir+"contacts.png", dpi=400)

# CREATE SA VS RG

fig = plt.figure(figsize=(16, 25))

rows = 5
columns = 4
grid = plt.GridSpec(rows, columns, wspace = .3, hspace = .3)

# load the helix for this part, which will be the nmr determined helices
# helixpdb = "../../structure_files/helix_wt.pdb" 
helix = md.load_pdb(helixpdb)
top_helix = helix.topology

 
for n, file in enumerate(files): 
    trj = md.load(file, top=pdb, stride=10)
    trj.center_coordinates()  
    # compute anything that needs to be computed 
    rg_CA = compute_rg(trj)
    Sa_total = compute_sa(trj)
    # start plotting 
    exec (f"plt.subplot(grid{[n]})") 

    a, xedges, yedges = np.histogram2d(
    rg_CA, Sa_total, 30, [[0.5, 3.0], [0, 15.0]], normed=True, weights=None)
    a = np.log(np.flipud(a)+.000001)
    # default at the base temperature for analysis 
    T = 300
    if demux != 1: 
        T = temps[n]
    a = -(0.001987*T)*a
    im = plt.imshow(a, interpolation='gaussian', extent=[
    yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap='jet', aspect='auto', vmin=0, vmax=3.0)
     # grab the Colorbar instance
    imaxes = plt.gca()
    plt.xlim(0, 15.0)
    plt.ylim(0.5,3.0)
    if demux == 1: 
        plt.title("Demux " + str(n))
    else: 
        plt.title("Temperature Replica " + str(temps[n]) + "K")
    
    plt.savefig(outdir+ "_SavRg.png", dpi=400)

# CREATE HELICAL PROPENSITIES 
fig = plt.figure(figsize=(10, 8))
for n, file in enumerate(files):  
    trjH1 = md.load(file, top=pdb, stride=10)
    H1 = top.select("resid 1 to 37")
    trjH1.restrict_atoms(H1)
    trjH1.center_coordinates()
    frames = trjH1.n_frames

    h_res = []
    for res in trjH1.topology.residues:
        h_res.append(res.resSeq)
    h_residues = len(set(h_res))
    residue_offset = 0
    hres_renum = np.asarray(h_res)+residue_offset
    
    frames = trj.n_frames

    dsspH1 = md.compute_dssp(trjH1, simplified=True)

    H1_H, H1_E = dssp_convert(dsspH1)

    if demux == 1: 
        plt.errorbar(hres_renum, H1_H[:, 0], yerr=H1_H[:, 1], capsize=5, color=colors[n], label='Helix Demux ' + str(n))
    else: 
        plt.errorbar(hres_renum, H1_H[:, 0], yerr=H1_H[:, 1], capsize=5, color=colors[n], label='Helix ' + str(temps[n]) + "K")
    

    plt.ylim(0, 1.0)
    plt.legend(loc="upper right")
    plt.xlabel('Residue', size=18)
    plt.ylabel('Secondary Strcture Fraction', size=18)
    plt.tick_params(labelsize=18)
    plt.savefig(outdir+ "HELICAL.png", dpi=400)

# CREATE SHEET PROPENSITIES 
fig = plt.figure(figsize=(10, 8))
for n, file in enumerate(files):  
    trjH1 = md.load(file, top=pdb, stride=10)
    H1 = top.select("resid 1 to 37")
    trjH1.restrict_atoms(H1)
    trjH1.center_coordinates()
    frames = trjH1.n_frames

    h_res = []
    for res in trjH1.topology.residues:
        h_res.append(res.resSeq)
    h_residues = len(set(h_res))
    residue_offset = 0
    hres_renum = np.asarray(h_res)+residue_offset
    
    frames = trj.n_frames

    dsspH1 = md.compute_dssp(trjH1, simplified=True)

    H1_H, H1_E = dssp_convert(dsspH1)

    if demux == 1: 
        plt.errorbar(hres_renum, H1_E[:, 0], yerr=H1_E[:, 1], capsize=5, color=colors[n], label='Sheet Demux' + str(n))
    else:
        plt.errorbar(hres_renum, H1_E[:, 0], yerr=H1_E[:, 1], capsize=5, color=colors[n], label='Sheet ' + str(temps[n]) + "K")
    
    
    plt.ylim(0, 1.0)
    plt.legend(loc="upper right")
    plt.xlabel('Residue', size=18)
    plt.ylabel('Secondary Strcture Fraction', size=18)
    plt.tick_params(labelsize=18)
    plt.savefig(outdir+ "SHEET.png", dpi=400)

# CREATE OTHER PLOTS 
if demux == 1: 
    trj = md.load(traj_path + "/Demux_1.xtc", top=pdb, stride=10)
else: 
    trj = md.load(traj_path + "/pbc_1.xtc", top=pdb, stride=10)

# indices of the cys 2 and cys 7 taken from pdb 
indxs = [[24,26,28,31], [26,28,31,93], [28,31,93,90], [31,93,90,88], [93,90,88,94]]

labels = ['$\chi_1$','$\chi_2$', '$\chi_3$', '$\chi_2\'$', '$\chi_1\'$']

x2 = md.compute_dihedrals(trj,indxs)
hfont = {"fontname":"arial"}
col = ['chi_1','chi_2', 'chi_3', 'chi_2p', 'chi_1p']
df_ang = pd.DataFrame(data=x2, columns=col)

for n,l in enumerate(labels): 
    fig = plt.figure(figsize=(10,8), frameon=False, edgecolor="k")
    #df_ang["chi_2"].hist(bins=6)
    x = np.linspace(start=-3.14, stop=3.14, num=12)
    plt.bar(x,df_ang[col[n]].value_counts(bins=12, sort=False).to_numpy()/trj.n_frames, width=0.4, alpha=0.8, color="#81C469")
    x = np.linspace(start=-3.14, stop=3.14, num=72)
    y = df_ang[col[n]].value_counts(bins=72, sort=False).to_numpy()/trj.n_frames
    plt.plot(x,y, "-", color="#253685")
    plt.xlabel(l + "dihedral angle (radians)", fontweight="bold", fontsize=14, **hfont)
    plt.ylabel("Frequency", fontweight="bold", fontsize=14, **hfont)
    plt.xticks(fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)
    plt.grid(visible=False)
    #plt.axhline(linewidth=2, color="k")
    plt.axhspan(0, 0,linewidth=2, color="k")
    plt.axhspan(0.6, 0.6,linewidth=2, color="k")
    #plt.axvline(linewidth=2, color="k")
    plt.axvspan(-3.7,-3.7, linewidth=2, color="k")
    plt.axvspan(3.7,3.7, linewidth=2, color="k")
    plt.xlim((-3.7,3.7))
    plt.ylim((0, 0.6))

    plt.savefig(outdir+ "dihedrals" + "_" +col[n] +".png", dpi=400)
