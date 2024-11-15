#!/usr/bin/env python
# coding: utf-8

"""
REST_20reps_analysis.py

Author: Robustelli Lab
Date: 11.14.24

Description:
This script analyzes data from multiple replicates of REST (Replica Exchange with Solute Tempering) simulations.
It includes data processing and JSON encoding for custom data types

Usage:
    python REST_20reps_analysis.py [arguments if any]

Requirements:
- Python 3.x
- Packages: numpy, json, packages in tools.py

Functions:
- NpEncoder: Custom JSON encoder for handling numpy data types.
- [Function1]: finds the datatype instance of the object 

Notes:
- Ensure that the required file paths are accessible before running the script.

"""
from __future__ import print_function, division
# add folder with tools.py to path
import os 
import sys
from dataclasses import dataclass, field, fields 
sys.path.append(os.path.abspath("../scripts"))
from typing import List, Dict, Any 
from tools import * 
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


#### data paths and variables ####
@dataclass
class RESTOBJ:
    ligand: bool = True
    ligand_rings: List[List[int]] = field(default_factory=lambda: [[560, 561, 562, 563, 564, 565], [553, 554, 555, 556, 557, 558], [544, 545, 546, 547, 548, 549]])
    lig_hbond_donors: List[List[int]] = field(default_factory=lambda: [[541, 577], [552, 583]])
    ligand_pos_charges: List[int] = field(default_factory=list)
    ligand_neg_charges: List[int] = field(default_factory=list)
    out_dir: str = "./outdir_lig_prot_analysis_2.82/"
    helix: str = "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb"
    pdb: str = "/Users/f006j60/git/IAPP/structures/prot_s20g_yxa1_20reps.gro"
    traj_dir_pbc: str = "/Users/f006j60/Robustelli_Group/IAPP/s20g_yxa1/pbc_trj_2.82us/"
    traj_dir_demux: str = "/Users/f006j60/Robustelli_Group/IAPP/s20g_yxa1/demux_trj_2.82us/"
    nreps: int = 20
    rep_list: List[int] = field(default_factory=lambda: list(range(20)))
    stride: int = 10
    sim_length: float = 2.82
    residues: int = 38
    ligand_residue_index: int = 38
    residue_offset: int = 1
    residue_number: List[int] = field(default_factory=lambda: list(range(38)))
    residue_number_offset: List[int] = field(default_factory=lambda: list(range(1, 39)))
    trj_types: List[str] = field(default_factory=lambda: ['rep', 'demux'])
    temps: List[float] = field(default_factory=lambda: [round(300 * math.exp(i * math.log(500/300) / 19), 2) for i in range(20)])

    def create_outdir(self):
        # Ensure output directory exists
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def update_fields(self, updates: Dict[str, Any]):
        """Update multiple fields in the config with a dictionary of values."""
        valid_fields = {field.name for field in fields(self)}
        for key, value in updates.items():
            if key in valid_fields:
                setattr(self, key, value)
                print(f"Updated {key} to {value}")
            else:
                print(f"Warning: {key} is not a valid field in Config")
    
    def load_trj_dict(self):
        """Load trajectory and topology data with the specified stride."""
        trj_dict = {}
        top_dict = {}

        for trj_type in self.trj_types:
            xtc = []
            if trj_type == "rep": 
                xtc = [os.path.join(self.traj_dir_pbc, f'pbc_{i}.xtc') for i in range(1, self.nreps + 1)]
            else: 
                xtc = [os.path.join(self.traj_dir_demux, f'Demux_{i}.xtc') for i in range(1, self.nreps + 1)]
            
            trj_dict[trj_type] = md.load(xtc, top=self.pdb, stride=self.stride)
            trj_dict[trj_type].center_coordinates()
            top_dict[trj_type] = trj_dict[trj_type].topology

        # Load frame starts and ends
        t = int(trj_dict['rep'].n_frames / self.nreps)
        trj_frames = {}
        a = 0
        b = t
        self.sim_length = (t * self.stride * 80) / (10**6)
        for i in range(self.nreps):
            trj_frames[str(i)] = [a, b]
            a = b
            b += t

        return trj_dict, top_dict, trj_frames
    
    def load_p_trj_dict(self): 
        p_trj_dict ={}
        for trj_type in self.trj_types:
            xtc=[]
            if trj_type == "rep": 
                xtc=[self.traj_dir_pbc+'pbc_'+str(i)+'.xtc' for i in range(1,self.nreps+1)]
            else: 
                xtc=[self.traj_dir_demux+'Demux_'+str(i)+'.xtc' for i in range(1,self.nreps+1)]
            p_trj_dict[trj_type] = md.load(xtc, top=self.pdb, stride=self.stride)
            prot=p_trj_dict[trj_type].topology.select("residue 1 to " + str(self.residues))
            p_trj_dict[trj_type].restrict_atoms(prot)
            p_trj_dict[trj_type].center_coordinates()
        return p_trj_dict
    
    def save_json_(self, file_name, data):
        with open(os.path.join(self.out_dir, file_name), 'w') as file_:
            json.dump(data, file_,cls=NpEncoder)
        del data 
        return 
###################### RUN CALCULATIONS ######################

def analyze_ligand_contacts(restobj:RESTOBJ): 
    if not restobj.ligand:
        return 

    trj_dict, top_dict, trj_frames = restobj.load_trj_dict()

    ####### Create a contact matrix for the ligand contacts 
    contact_matrix={}
    for trj_type in restobj.trj_types:
        contact_matrix[trj_type]=contact_matrix_(trj_dict[trj_type],top_dict[trj_type],restobj.residues,restobj.ligand_residue_index)

    ####### Calculate the bound fraction and the kd 
    K={}
    for trj_type in restobj.trj_types:
        box_len=trj_dict[trj_type].unitcell_lengths[0][0]
        kd_bf_data=bound_frac_kd(box_len, restobj.nreps, contact_matrix[trj_type],trj_frames)
        K[trj_type]=kd_bf_data.T
        del kd_bf_data
    restobj.save_json_("bf_kd.json", K)

    ####### Calculate the KD timecourse 
    K_time={}
    for trj_type in restobj.trj_types:
        box_len=trj_dict[trj_type].unitcell_lengths[0][0]
        kd_time_data=kd_timecourse(box_len,restobj.nreps,contact_matrix[trj_type],
                                trj_frames,sim_length=restobj.sim_length, stride=restobj.stride)
        K_time[trj_type]=kd_time_data
    restobj.save_json_('kd_timecourse.json', K_time)

    ####### Calculate the BF timecourse 
    bf_time={}
    for trj_type in restobj.trj_types:
        bf_time_data=bf_timecourse(restobj.nreps,contact_matrix[trj_type],trj_frames,
                                sim_length=restobj.sim_length,stride=restobj.stride)
        bf_time[trj_type]=bf_time_data
    restobj.save_json_('bf_timecourse.json', bf_time)

    ####### Calculate the protein ligand contacts 
    contact_data={}
    for trj_type in restobj.trj_types:
        contact_data[trj_type]=[]

    for trj_type in restobj.trj_types:
        for val in restobj.rep_list:    
            C=contact_matrix[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])]
            contact_rows = contact_rows = np.sum(C,axis=1)
            #contact_frames = np.where(contact_rows > 0)[0]
            #nocontact_frames = np.where(contact_rows == 0)[0]

            bound_frac = 1.0
            contact_ave, contact_pyb_be = get_blockerror_pyblock_nanskip_bf(C, bound_frac=bound_frac)
            contact_be = np.column_stack((restobj.residue_number_offset, contact_ave, contact_pyb_be))
            contact_data[trj_type].append(contact_be)
            del C
        del contact_be; del contact_ave; del contact_pyb_be; del contact_rows
    restobj.save_json_('contact_prob.json', contact_data)
    return 

def analyze_intermolecular_contacts(restobj:RESTOBJ): 
    if not restobj.ligand:
        return 
    
    trj_dict, top_dict, trj_frames = restobj.load_trj_dict()
    # load bound fraction and kd data 
    K=json.load(open(os.path.abspath(os.path.join(restobj.out_dir,'bf_kd.json')),'r'))

    ####### calculate the charge contacts 
    charge_contact={}

    for trj_type in restobj.trj_types:
        charge_contact[trj_type]=charge_contacts_rw_(trj_dict[trj_type],top_dict[trj_type],0,
                                            Ligand_Neg_Charges=restobj.ligand_neg_charges,Ligand_Pos_Charges=restobj.ligand_pos_charges)

    charge_data={}
    for trj_type in restobj.trj_types:
        charge_data[trj_type]={}
        for selection in ['all','bf'] :
            charge_data[trj_type][selection]=[]

    for trj_type in restobj.trj_types:
        for val in restobj.rep_list:
            CH=charge_contact[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])]
            
            bound_frac=1.0
            charge_contact_fraction, charge_contact_fraction_pyb_be = get_blockerror_pyblock_nanskip_bf(CH, bound_frac=bound_frac)
            charge_c=np.column_stack((restobj.residue_number_offset, charge_contact_fraction, charge_contact_fraction_pyb_be))

            charge_c_bf=np.column_stack((restobj.residue_number_offset[:restobj.residues:], charge_contact_fraction[:restobj.residues:]/K[trj_type][0][val], 
                                    charge_contact_fraction_pyb_be[:restobj.residues:]/K[trj_type][0][val]))

            C_P=charge_c[:restobj.residues:]
            
            charge_data[trj_type]['all'].append(C_P)
            charge_data[trj_type]['bf'].append(charge_c_bf)
            
            del C_P ; del CH
            
        del charge_c; del charge_contact_fraction; del charge_contact_fraction_pyb_be
    restobj.save_json_('charge_prob.json', charge_data)

    ####### calculate the hydrophobic contacts
    hphob_contact={}
    for trj_type in restobj.trj_types:
        hphob_contact[trj_type]=hphob_contacts_rw_(trj_dict[trj_type],top_dict[trj_type],38, 39)
    hphob_data={}
    for trj_type in restobj.trj_types:
        hphob_data[trj_type]={}
        for selection in ['all','bf'] :
            hphob_data[trj_type][selection]=[] 

    for trj_type in restobj.trj_types:
        for val in restobj.rep_list:
            H=hphob_contact[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])]

            bound_frac=1
            hphob_ave, hphob_pyb_be = get_blockerror_pyblock_nanskip_bf(H, bound_frac=bound_frac)
            hphob_by_res = np.column_stack((restobj.residue_number_offset, hphob_ave, hphob_pyb_be))
            hphob_by_res_bf = np.column_stack((restobj.residue_number_offset[:restobj.residues:], hphob_by_res[:, 1][:restobj.residues:]/K['rep'][0][val], 
                                            hphob_by_res[:, 2][:restobj.residues:]/K[trj_type][0][val]))

            H_P=hphob_by_res[:restobj.residues:]
            hphob_data[trj_type]['all'].append(H_P)
            hphob_data[trj_type]['bf'].append(hphob_by_res_bf)
            del H_P; del H
        del hphob_by_res_bf; del hphob_by_res; del hphob_ave; del hphob_pyb_be; 
    restobj.save_json_('hyphob_prob.json', hphob_data)

    ####### Calculate Aromatic Contacts 
    aro_contact={}
    for trj_type in restobj.trj_types:    
        aro_contact[trj_type]=aro_contacts_rw_(trj_dict[trj_type],top_dict[trj_type],1,residues=restobj.residues,
                                        ligand_rings=restobj.ligand_rings)

    aro_data={}
    for trj_type in restobj.trj_types:    
        aro_data[trj_type]={}
        for selection in ['all','bf'] :
            aro_data[trj_type][selection]=[] 

    for trj_type in restobj.trj_types:    
        for val in restobj.rep_list:    
            AR=aro_contact[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])]
            aro_r0_ave, aro_r0_pyb_be = get_blockerror_pyblock_nanskip_bf(AR, bound_frac=1.0)
            aro_r0_by_res = np.column_stack((restobj.residue_number_offset, aro_r0_ave, aro_r0_pyb_be))

            aro_r0_by_res_bf = np.column_stack((restobj.residue_number_offset[:restobj.residues], aro_r0_ave[:restobj.residues]/K[trj_type][0][val], 
                                                aro_r0_pyb_be[:restobj.residues]/K[trj_type][0][val]))

            AR_P=aro_r0_by_res[:restobj.residues]
            aro_data[trj_type]['all'].append(AR_P)
            aro_data[trj_type]['bf'].append(aro_r0_by_res_bf)
            
            del AR_P; del AR
        del aro_r0_by_res_bf; del aro_r0_by_res; del aro_r0_ave; del aro_r0_pyb_be;
    restobj.save_json_('aro_prob.json', aro_data)

    ####### Calculate the hydrogen bond propensities
    hbond={}
    for trj_type in restobj.trj_types:    
        hbond[trj_type]=hbond_rw_(trj_dict[trj_type],top_dict[trj_type],restobj.residues,restobj.ligand_residue_index,restobj.lig_hbond_donors)

    hbond_data={}
    for trj_type in restobj.trj_types:    
        hbond_data[trj_type]={}
        for selection in ['all','bf'] :      
            hbond_data[trj_type][selection]=[] 

    for trj_type in restobj.trj_types:    
        for val in restobj.rep_list:
            HB=hbond[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])]

            HBond_ave, HBond_pyb_be = get_blockerror_pyblock_nanskip_bf(HB, 1.0)
            Hbond_by_res = np.column_stack((restobj.residue_number_offset[:restobj.residues], HBond_ave[:restobj.residues:], HBond_pyb_be[:restobj.residues:]))

            Hbond_by_res_bf = np.column_stack((restobj.residue_number_offset[:restobj.residues], HBond_ave[:restobj.residues:]/K[trj_type][0][val], 
                                            HBond_pyb_be[:restobj.residues:]/K[trj_type][0][val]))

            HB_P=Hbond_by_res[:restobj.residues]
            hbond_data[trj_type]['all'].append(HB_P)
            hbond_data[trj_type]['bf'].append(Hbond_by_res_bf)
            
            del HB_P ; del HB
        del Hbond_by_res_bf; del Hbond_by_res; del HBond_ave; del HBond_pyb_be;
    restobj.save_json_('hbond_prob.json', hbond_data)
    return 

def create_contact_maps(restobj:RESTOBJ): 
    trj_dict, top_dict, trj_frames = restobj.load_trj_dict()
    ####### Calculate the protein contact map 
    p_contact_map={}
    for trj_type in restobj.trj_types:    
        p_contact_map[trj_type]=[] 
        for val in restobj.rep_list:
            p_contact_map[trj_type].append(contact_map_avg(trj_dict[trj_type][int(trj_frames[str(val)][0]):
                                                                    int(trj_frames[str(val)][1])], restobj.residues)[0])
    restobj.save_json_('p_contact_map.json', p_contact_map)

    if restobj.ligand:
        ####### Calculate the dual residue contact map 
        l_contact_map={}
        for trj_type in restobj.trj_types:    
            l_contact_map[trj_type]=[]
            for val in restobj.rep_list:
                l_contact_map[trj_type].append(contact_map_ligand_(trj_dict[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])],
                                                                   restobj.residues, restobj.ligand_residue_index))
        restobj.save_json_('l_contact_map.json', l_contact_map)
        return 

def ss(trj):
    dssp=md.compute_dssp(trj, simplified=True)
    H1_H,H1_E=dssp_convert(dssp)
    return H1_H, H1_E

def calc_phipsi_(trj, residues=38):
    # Compute Phi and Psi
    indices_phi, phis = md.compute_phi(trj)
    indices_psi, psis = md.compute_psi(trj)
    
    psis=psis.T[:residues-1].T
    
    phi_label = []
    for i_phi in range(0, indices_phi.shape[0]):
        resindex = trj.topology.atom(indices_phi[i_phi][3]).residue.resSeq
        phi_label.append(resindex)
    phi_label = np.array(phi_label)
    psi_label = []
    for i_psi in range(0, indices_psi.shape[0]):
        resindex = trj.topology.atom(indices_psi[i_psi][3]).residue.resSeq
        psi_label.append(resindex)
    #psi_label = np.array(psi_label)
    psi_label = np.array(psi_label[:residues-1])
    
    phipsi = []
    for i in range(0, len(psi_label)-1):
        current_phipsi = np.column_stack((phis[:, i+1], psis[:, i]))
        phipsi.append(current_phipsi)
    phipsi_array = np.array(phipsi)

    def alphabeta_rmsd(phi, psi, phi_ref, psi_ref):
        alphabetarmsd = np.sum(0.5*(1+np.cos(psi-psi_ref)),
                               axis=1)+np.sum(0.5*(1+np.cos(phi-phi_ref)), axis=1)
        return alphabetarmsd

    Phi_all = phis
    Psi_all = psis
    alphabeta_alpharight = alphabeta_rmsd(Phi_all, Psi_all, -1.05, -0.79)
    alphabeta_betasheet = alphabeta_rmsd(Phi_all, Psi_all, 2.36, -2.36)
    alphabeta_ppII = alphabeta_rmsd(Phi_all, Psi_all, -1.31, 2.71)
    
    return alphabeta_alpharight, alphabeta_betasheet, alphabeta_ppII 

def calculate_other_metrics(restobj:RESTOBJ): 
    """
    SS, Rg, Sa, alpha-beta-rmsd 
    """
    # load protein and ligand trajectory
    trj_dict, top_dict, trj_frames = restobj.load_trj_dict()

    # load only the protein trajectory
    p_trj_dict = restobj.load_p_trj_dict()
    
    ####### Calculate the secondary structure propensity 
    ss_data={}
    for trj_type in restobj.trj_types:
        ss_data[trj_type]={}
        for sruct in ['helix','sheet'] :
            ss_data[trj_type][sruct]=[] 
    for trj_type in restobj.trj_types:
        for val in restobj.rep_list:
            H_H, H_E = ss(p_trj_dict[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])])
            ss_data[trj_type]['helix'].append(H_H)
            ss_data[trj_type]['sheet'].append(H_E)
    restobj.save_json_('ss_fraction.json', ss_data)

    ####### Calculate Rg
    rg_data={}
    for trj_type in restobj.trj_types:
        rg_data[trj_type]=[]
        for val in restobj.rep_list:
            rg_data[trj_type].append(calc_Rg(trj_dict[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])]))
    restobj.save_json_('rg_timeseries.json',rg_data)

    ####### Calculate SA
    helix=md.load(restobj.helix, top=restobj.helix)
    p_trj_dict = restobj.load_p_trj_dict()
    
    sa_data={}
    for trj_type in restobj.trj_types:
        sa_data[trj_type]=[]
        for val in restobj.rep_list:
            sa_data[trj_type].append(calc_SA(p_trj_dict[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])],
                                    helix,1,restobj.residues-6-1)) # no nh2 
    restobj.save_json_('sa_timeseries.json', sa_data)

    phipsi_data={}
    for trj_type in restobj.trj_types:
        phipsi_data[trj_type]=[]
        for val in restobj.rep_list:
            phipsi_data[trj_type].append(calc_phipsi_(trj_dict[trj_type][int(trj_frames[str(val)][0]):int(trj_frames[str(val)][1])], 
                                                      restobj.residues-1))
    restobj.save_json_('phipsi.json', phipsi_data)

def main():
    """
    analyze the total number of replicas that exist 
    """ 
    s20g_hiapp_apo = {"ligand": False,
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_s20g_apo.gro", 
                   "out_dir" : "s20g_hiapp_apo_20rep_analyses",
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/s20g/pbc_trj_3.8us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/s20g/demux_trj_3.8us/"}
    
    s20g_hiapp_yxa1 = {"ligand_rings" : [[560,561,562,563,564,565], [544,545,546,547,548,549], [553,554,555,556,557,558]], 
                   "lig_hbond_donors" : [[541,577],[552,583]],
                   "out_dir" : "s20g_hiapp_yxa1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_s20g_yxa1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/s20g_yxa1/pbc_trj_2.82us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/s20g_yxa1/demux_trj_2.82us/"}

    s20g_hiapp_yxi1={"ligand_rings" : [[533,534,535,536,537,538,539,540,541], [546,547,548,549,550,551]], 
                   "lig_hbond_donors" : [[545,571],[561,590]],
                   "out_dir" : "s20g_hiapp_yxi1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_s20g.pdb", 
                   "pdb" : "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_s20g_yxi1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/s20g_714f9/2.8us/pbc_trj/", 
                   "traj_dir_demux" : '/Users/f006j60/Robustelli_Group/IAPP/s20g_714f9/2.8us/demux_trj/'}

    wt_hiapp_apo={"ligand": False,
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_wt.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_wt_apo.gro",
                   "out_dir" : "wt_hiapp_apo_20rep_analyses",
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/wt/pbc_trj_4us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/wt/demux_trj_4us/"}
    
    wt_hiapp_yxi1={"ligand_rings" : [[537,538,539,540,541,542,543,544,545], [550,551,552,553,554,555]], 
                   "lig_hbond_donors" : [[549,575], [565,594]],
                   "out_dir" : "wt_hiapp_yxi1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_wt.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_wt_yxi1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/wt_714f9/pbc_trj/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/wt_714f9/demux_trj/"}

    wt_hiapp_yxa1={"ligand_rings" : [[564,565,566,567,568,569], [548,549,550,551,552,553], [557,558,559,560,561,562]], 
                   "lig_hbond_donors" : [[545,581],[556,587]],
                   "out_dir" : "wt_hiapp_yxa1_20rep_analyses",
                   "helix" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/helix_wt.pdb", 
                   "pdb" :  "/Users/f006j60/git/hIAPP_monomer_simulations/structure_files/hiapp_wt_yxa1.gro", 
                   "traj_dir_pbc" : "/Users/f006j60/Robustelli_Group/IAPP/wt_yxa1/pbc_trj_2.69us/", 
                   "traj_dir_demux" : "/Users/f006j60/Robustelli_Group/IAPP/wt_yxa1/demux_trj_2.69us/"}
    

    system_params_list = [s20g_hiapp_apo, s20g_hiapp_yxi1, s20g_hiapp_yxa1, wt_hiapp_apo, wt_hiapp_yxi1, wt_hiapp_yxa1]
    for system in system_params_list: 
        # instantiate and update fields according to dictionary values 
        restobj = RESTOBJ() 
        restobj.update_fields(system)
        restobj.create_outdir()

        analyze_ligand_contacts(restobj)
        analyze_intermolecular_contacts(restobj)
        create_contact_maps(restobj)
        calculate_other_metrics(restobj)
    return 

if __name__ == "__main__":
    main()