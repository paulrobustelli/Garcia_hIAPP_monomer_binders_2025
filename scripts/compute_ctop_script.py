"""
Creator: Michelle Garcia 
Purpose: This script computes a circuit topology model, runs PCA on the model, runs a KMEANS clustering analysis, 
and outputs a dictionary of these clusters 

Output: dictionary, images showing the KMEANS optimal number of clusters 

Notes, this script runs with circuit_top_tools.py and analysis_tools.py 
save the log file as it will ccontain the information for the KMEANS silhouette and inertia scores 
"""
# import from different python files 
from circuit_top_tools import *
from tools import * 

from sklearn.decomposition import IncrementalPCA
from scipy.sparse import csr_matrix, save_npz, load_npz

import os 
import mdtraj as md 
import json
import sys 
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Function to evaluate KMeans with different cluster numbers
# Function to evaluate KMeans with different cluster numbers
def evaluate_kmeans(data, max_clusters=20):
    distortions = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_

        # Calculate distortion (sum of squared distances to centroids)
        distortions.append(kmeans.inertia_)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)

        # Calculate Calinski-Harabasz score
        ch_score = calinski_harabasz_score(data, labels)
        calinski_harabasz_scores.append(ch_score)

        # Calculate Davies-Bouldin score
        db_score = davies_bouldin_score(data, labels)
        davies_bouldin_scores.append(db_score)

    return cluster_range, distortions, silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores

def plot_kmeans(cluster_range, distortions, silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores, fname):
    # Plot the elbow method
    plt.figure(figsize=(6.6, 6))

    plt.subplot(2, 2, 1)
    plt.plot(cluster_range, distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.grid(True)

    # Plot the silhouette scores
    plt.subplot(2, 2, 2)
    plt.plot(cluster_range, silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    # Plot the Calinski-Harabasz scores
    plt.subplot(2, 2, 3)
    plt.plot(cluster_range, calinski_harabasz_scores, marker='o', color='green')
    plt.title('Calinski-Harabasz Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.grid(True)

    # Plot the Davies-Bouldin scores
    plt.subplot(2, 2, 4)
    plt.plot(cluster_range, davies_bouldin_scores, marker='o', color='red')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

    # Identify the best number of clusters based on the three metrics
    best_n_clusters_silhouette = cluster_range[np.argmax(silhouette_scores)]
    best_n_clusters_ch = cluster_range[np.argmax(calinski_harabasz_scores)]
    best_n_clusters_db = cluster_range[np.argmin(davies_bouldin_scores)]

    print(f"The best number of clusters based on Silhouette Score is: {best_n_clusters_silhouette}")
    print(f"The best number of clusters based on Calinski-Harabasz Index is: {best_n_clusters_ch}")
    print(f"The best number of clusters based on Davies-Bouldin Index is: {best_n_clusters_db}")


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)



# argument order is trajectory, pdb, outdir, c_cluster 
res_num = 38
outdir = "./ctop_outdir/"

# make an outdir directory 
if not os.path.exists(outdir):
    os.makedirs(outdir)

trj_tops = ["hiapp_wt_apo.gro", "hiapp_wt_yxi1.gro", "hiapp_wt_yxa1.gro", \
            "hiapp_s20g_apo.gro", "hiapp_s20g_yxi1.gro", "hiapp_s20g_yxa1.gro"]
trj_paths = ["hiapp_wt_apo.xtc", "hiapp_wt_yxi1.xtc", "hiapp_wt_yxa1.xtc", \
             "hiapp_s20g_apo.xtc", "hiapp_s20g_yxi1.xtc", "hiapp_s20g_yxa1.xtc"]
psystem = ["wt_hiapp_apo", "wt_hiapp_yxi1", "wt_hiapp_yxa1", \
           "s20g_hiapp_apo", "s20g_hiapp_yxi1", "s20g_hiapp_yxa1"]

# instantiate the IncrementalPCA transformer
transformer = IncrementalPCA(n_components=2, batch_size=200)

# loop through the trajectories and pdbs
counter = 0 
for trj_top, trj_path in zip(trj_tops, trj_paths):
    sys.stdout.write(f"Processing "+ psystem[counter] + "...")
    trj = md.load("./trjs/" + trj_path, top = "./structure_files/"+ trj_top, stride=1) # testing it out for ~ 500 frames
    
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

    # run analysis 
    # here is the circuit topology analysis 
    fname = outdir + psystem[counter] + "_circuit_model.dat"
    circuit_model = compute_circuit_top_model(trj,fname, res_num, cutoff_distance, exclude_neighbour)
    # move the topologies file to the right directory 
    os.rename("./topologies.npy", outdir+ psystem[counter] + "_topologies.npy")
    
    # mak sparse and save
    X_sparse = csr_matrix(circuit_model)
    save_npz(outdir+ psystem[counter] + "_ctop_model.npz", X_sparse)

    # train the IncrementalPCA transformer
    X_transformed = transformer.fit_transform(X_sparse)
    counter +=1
    
    sys.stdout.write("Finished processing "+ psystem[counter-1])

# save concatenated PCs and PCs for all systems 
pcs_all = []
for psys in psystem:
    X_sparse = load_npz(outdir + psys + "_ctop_model.npz")
    X_transformed = transformer.transform(X_sparse)
    np.save(outdir + psys + "_PCs.npy", X_transformed, allow_pickle=False)
    pcs_all.append(X_transformed)

# save the concatenated PCs
concatenated_pcs = np.concatenate(pcs_all, axis=0)
PC1 = concatenated_pcs[:,0]
PC2 = concatenated_pcs[:,1]
np.save(outdir+"all_sys_concat_incpca.npy", concatenated_pcs, allow_pickle=False)

cluster_range, distortions, silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores= evaluate_kmeans(concatenated_pcs)
plot_kmeans(cluster_range, distortions, silhouette_scores, calinski_harabasz_scores, davies_bouldin_scores, fname=outdir + "kmeans_plot.png")
