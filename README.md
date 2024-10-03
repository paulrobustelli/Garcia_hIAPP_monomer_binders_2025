# human Islet Amyloid Polypeptide Monomer in Simulation with An Accelerator and Inhibitor  
hIAPP WT and S20G with YX-I-1 and YX-A-1

Trajectories for this manuscript can be downloaded here:
https://www.dropbox.com/scl/fo/30s05ez04bliovibhufl0/AH3vOrRkNmu3ycOZsXIwPWo?rlkey=mibfhyc8cif7i5an7uwdvohom&st=a35aeyeg&dl=0

Trajectories, Gromacs Input Files, and All analysis code can be Downloaded here:

# Publication 
To learn more about analysis and insights found in this paper please check out our paper/preprint:

# Documentation 

## 1. Download the required packages 

If you plan on running on a cluster with a linux system, you should use [minforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file) as it solves for package environments faster. 

`source miniforge3/bin/activate`

`mamba init`

`mamba create -n circuit_top python`

### Package Installation

#### Mdtraj
- `mamba install mdtraj`
- `conda install -c conda-forge mdtraj`

#### Matplotlib
- `mamba install matplotlib`
- `conda install matplotlib`

#### Seaborn
- `pip3 install seaborn`

#### Scikit-Learn
- `pip3 install -U scikit-learn`

#### Scipy
- `conda install scipy`

#### Biopython
- `pip3 install biopython`

#### pyblock
- `pip install pyblock`


OR create a new environment from circuit_top_env.yml file using `conda env create -f circuit_top_env.yml`

## 2. Run Topology Analysis and Cluster data 
create a topology matrix, run incremental pca, and using the script, which requires a topology file and trajectory file. Notice that some of the functions are borrowed directly from [scalvini et al. 2023]( https://pubs.acs.org/doi/10.1021/acs.jcim.3c00391).

There are 5 variables to consider in this order: the trajectory path, the topology path, the number of residues, the out directory, the number of clusters, and the system name for file naming. 

Example usage: `python circuit_top_analysis_script.py ./2F4K.protein.s10.dcd ./2F4K.protein.pdb 34 2F4K_outdir 5 villin_2F4K`

## 3. Analyze the trajectory and Clustering
For any protein + ligand system run the {protein}_{ligand}_analysis.ipynb notebook. This notebook analyzes the ligand system. You may run circuit_top_{protein}_{ligand}_analysis.ipynb based off k-means clusters created from the topology matrix. 

This notebook requires: 
- a structure file that can be read by mdtraj (pdb, gro, etc..) file for your protein + ligand system
- an xtc or dcd file for your simulation
- a matrix file and circuit model file or a .json/.pkl file with dictionary values for assigned clusters
- a perfectly ideal helix structure for alpha-helical order parameter calculations
