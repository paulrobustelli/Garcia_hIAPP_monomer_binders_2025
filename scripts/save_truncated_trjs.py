import os 
import sys
sys.path.append(os.path.abspath("../scripts"))
from tools import * 
import mdtraj as md 

def main():
    """
    analyze the total number of replicas that exist 
    """ 
    s20g_hiapp_apo = {"ligand": False,
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_s20g_apo.gro", 
                   "out_dir" : "s20g_hiapp_apo_20rep_analyses",
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/s20g/pbc_trj_3.8us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/s20g/demux_trj_3.8us/",
                   "name" : "s20g_hiapp_apo"
                   }
    
    s20g_hiapp_yxa1 = {"ligand_rings" : [[560,561,562,563,564,565], [544,545,546,547,548,549], [553,554,555,556,557,558]], 
                   "lig_hbond_donors" : [[541,577],[552,583]],
                   "out_dir" : "s20g_hiapp_yxa1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_s20g_yxa1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/s20g_yxa1/pbc_trj_2.82us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/s20g_yxa1/demux_trj_2.82us/", 
                   "name" : "s20g_hiapp_yxa1"
                   }

    s20g_hiapp_yxi1={"ligand_rings" : [[533,534,535,536,537,538,539,540,541], [546,547,548,549,550,551]], 
                   "lig_hbond_donors" : [[545,571],[561,590]],
                   "out_dir" : "s20g_hiapp_yxi1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb", 
                   "pdb" : "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_s20g_yxi1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/s20g_714f9/2.8us/pbc_trj/", 
                   "traj_dir_demux" : '/Users/f006j60/Robustelli_Group/IAPP/s20g_714f9/2.8us/demux_trj/',
                    "name" : "s20g_hiapp_yxi1"
                   }

    wt_hiapp_apo={"ligand": False,
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_wt.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_wt_apo.gro",
                   "out_dir" : "wt_hiapp_apo_20rep_analyses",
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/wt/pbc_trj_4us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/wt/demux_trj_4us/",
                   "name" : "wt_hiapp_apo"
                   }
    
    wt_hiapp_yxi1={"ligand_rings" : [[537,538,539,540,541,542,543,544,545], [550,551,552,553,554,555]], 
                   "lig_hbond_donors" : [[549,575], [565,594]],
                   "out_dir" : "wt_hiapp_yxi1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_wt.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_wt_yxi1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/wt_714f9/pbc_trj/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/wt_714f9/demux_trj/", 
                   "name" : "wt_hiapp_yxi1"}

    wt_hiapp_yxa1={"ligand_rings" : [[564,565,566,567,568,569], [548,549,550,551,552,553], [557,558,559,560,561,562]], 
                   "lig_hbond_donors" : [[545,581],[556,587]],
                   "out_dir" : "wt_hiapp_yxa1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_wt.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_wt_yxa1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/wt_yxa1/pbc_trj_2.69us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/wt_yxa1/demux_trj_2.69us/",
                   "name" : "wt_hiapp_yxa1"}
    

    system_params_list = [s20g_hiapp_apo, s20g_hiapp_yxi1, s20g_hiapp_yxa1, wt_hiapp_apo, wt_hiapp_yxi1, wt_hiapp_yxa1]
    t_dir = "/Users/f006j60/git/hIAPP_monomer_simulations/trjs/"
    nreps = 20
    frames_to_skip = 12500 # 1us / 80ps 
    for system in system_params_list: 
        
        outdir = t_dir + system["name"] + "/"
        os.makedirs(outdir, exist_ok=True)
        
        # load and save pbc trjs 
        xtc_pbc_paths = [os.path.join(system["traj_dir_pbc"], f'pbc_{i}.xtc') for i in range(1, nreps + 1)]
        pdb_path = system["pdb"]

        pbc_outdir = t_dir + system["name"] + "/" + "pbc_trjs/"
        os.makedirs(pbc_outdir, exist_ok=True)
        for xtc_path in xtc_pbc_paths:
            trj = md.load(xtc_path, top=pdb_path)
            trj = trj.slice(np.arange(frames_to_skip+1, trj.n_frames))
            trj.save_xtc(os.path.join(pbc_outdir, os.path.basename(xtc_path)))
            del trj

        # load and save demux trjs
        demux_xtc_paths = [os.path.join(system["traj_dir_demux"], f'Demux_{i}.xtc') for i in range(1, nreps + 1)]
        demux_outdir = outdir = t_dir + system["name"] + "/" + "demux_trjs/"
        os.makedirs(demux_outdir, exist_ok=True)
        for xtc_path in demux_xtc_paths:
            trj = md.load(xtc_path, top=pdb_path)
            trj = trj.slice(np.arange(frames_to_skip+1, trj.n_frames))
            trj.save_xtc(os.path.join(demux_outdir, os.path.basename(xtc_path)))
            del trj

if __name__ == "__main__":
    main()