"""
Creator: Michelle Garcia 
Purpose: This script computes a circuit topology model, runs PCA on the model, runs a KMEANS clustering analysis, 
and outputs a dictionary of these clusters 

Output: dictionary, images showing the KMEANS optimal number of clusters 

Notes, this script runs with circuit_top_tools.py and analysis_tools.py 
save the log file as it will ccontain the information for the KMEANS silhouette and inertia scores 
"""


# import from different python files 
from circuit_top_tools2_1 import *
from analysis_tools import * 

from sklearn.decomposition import IncrementalPCA
from scipy import sparse

import os 
import mdtraj as md 
import pickle
import sys 

# argument order is trajectory, pdb, outdir, c_cluster 
traj_path = sys.argv[1]
top_path = sys.argv[2]
res_num = int(sys.argv[3])
outdir = "./" + sys.argv[4] + "/"
c_clusters = int(sys.argv[5])
psystem = sys.argv[6] #"wt_iapp_yxi1"

# make an outdir directory 
if not os.path.exists(outdir):
    os.makedirs(outdir)


trj = md.load(traj_path, top = top_path, stride = 1) # testing it out for ~ 500 frames 
# get just the protein not the ligand 
prot = trj.topology.select("residue 1 to " + str(res_num))
trj = trj.atom_slice(prot)
top = trj.topology

### Set-up the parameters of the circuit topology stuff ###
# Format
fileformat = 'pdb'

# cutoff_distance, maximal distance (Ångström) between two atoms that will count as an atom-atom contact.
# cutoff_numcontacts, minimum number of contacts between two residues to count as a res-res contact.
# exclude_neighbour, number of neighbours that are excluded from possbile res-res contacts.

# CT variables
cutoff_distance =       10
cutoff_numcontacts =    1 # any contact counts as a contact
exclude_neighbour =     1

def main():  
    """ run the analysis """
    # here is the circuit topology analysis 
    fname = outdir + psystem + "_circuit_model.dat"
    topology_avgs = compute_circuit_top_model(trj,fname, res_num, cutoff_distance, exclude_neighbour)
    # save the topology averages 
    np.save(outdir + psystem + "_topologies" +".npy", topology_avgs, allow_pickle=False, fix_imports=False)

    # here is loading in the circuit model 
    data_type = np.uint8
    dim_m = int( (res_num * (res_num-1)) // 2 )
    circuit_model = np.memmap(fname, dtype=data_type, mode='r', shape=(trj.n_frames, dim_m))

    transformer = IncrementalPCA(n_components=2, batch_size=200)
    X_sparse = sparse.csr_matrix(circuit_model)
    X_transformed = transformer.fit_transform(X_sparse)

    PC1 = X_transformed[:,0]
    PC2 = X_transformed[:,1]

    # save the PCs
    np.save(outdir+psystem+"_PCs.npy", X_transformed, allow_pickle=False, fix_imports=False)

    # generate a KMEANS output 
    # benchmark kmeans 
    data = np.column_stack((PC1,PC2))

    plt.figure(figsize=(6,6))
    plt.plot(PC1, PC2, linestyle="None", marker="o", alpha=0.5)
    plt.ylabel("PC2")
    plt.xlabel("PC1")
    plt.savefig(outdir + "PCA.png", dpi=300)

    fig = bench_k_means(psystem, data, clusters = np.arange(2,20,1))
    fig.savefig(outdir + "kmeans_plot.png", dpi=300, bbox_inches="tight")

    n_cluster=c_clusters
    ax, kmean_labels, centers = kmeans_cluster(PC1, PC2, clusters=n_cluster, title = "Kmeans Clustering")

    # create a c_dict for each cluster  
    c_dict = {}
    for i in range(n_cluster):
        ind = np.where(kmean_labels==i)[0]
        c_dict[i] = np.array(ind)

    # save this dictionary, so that the overall stats can be saved, while still being able to index the clusters 
    with open(outdir+psystem+"_kmeans_cluster_indices.pkl", "wb") as f: 
        pickle.dump(c_dict, f)

if __name__ == "__main__":
    main()