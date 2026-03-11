# %%
import os
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json



def sit2sphere(radius, digit):
    size = int((radius*2) * int(1/digit) + 1)
    cube = np.zeros([size, size, size])
    center = np.array([radius * int(1/digit), radius * int(1/digit), radius * int(1/digit)])

    for x in range(size):
        for y in range(size):
            for z in range(size):
                if np.linalg.norm(np.array([x, y, z]) - center) <= (radius * int(1/digit)):
                    cube[x, y, z] = 1
    
    return cube




def construct_space_sitesrdinate_system(atom_site, edge, digit, asym1, asym2):

    x_max = atom_site['_atom_site.Cartn_x'][(atom_site['_atom_site.label_asym_id']==asym1)|(atom_site['_atom_site.label_asym_id']==asym2)].astype(float).max()
    x_min = atom_site['_atom_site.Cartn_x'][(atom_site['_atom_site.label_asym_id']==asym1)|(atom_site['_atom_site.label_asym_id']==asym2)].astype(float).min()
    y_max = atom_site['_atom_site.Cartn_y'][(atom_site['_atom_site.label_asym_id']==asym1)|(atom_site['_atom_site.label_asym_id']==asym2)].astype(float).max()
    y_min = atom_site['_atom_site.Cartn_y'][(atom_site['_atom_site.label_asym_id']==asym1)|(atom_site['_atom_site.label_asym_id']==asym2)].astype(float).min()
    z_max = atom_site['_atom_site.Cartn_z'][(atom_site['_atom_site.label_asym_id']==asym1)|(atom_site['_atom_site.label_asym_id']==asym2)].astype(float).max()
    z_min = atom_site['_atom_site.Cartn_z'][(atom_site['_atom_site.label_asym_id']==asym1)|(atom_site['_atom_site.label_asym_id']==asym2)].astype(float).min()

    scale_x = np.round((x_max - x_min + edge*2) * int(1/digit)).astype(int) + 1
    scale_y = np.round((y_max - y_min + edge*2) * int(1/digit)).astype(int) + 1
    scale_z = np.round((z_max - z_min + edge*2) * int(1/digit)).astype(int) + 1 

    scs = np.zeros([scale_x, scale_y, scale_z])

    return scs, x_min, y_min, z_min


def atom_filter(asym1, asym2, atom_site, edge, digit, x_min, y_min, z_min):
    
    pep = atom_site[(atom_site['_atom_site.label_asym_id'] == asym1)]
    pro = atom_site[(atom_site['_atom_site.label_asym_id'] == asym2)]

    pep = pep.drop_duplicates(subset=['_atom_site.label_atom_id', '_atom_site.label_seq_id'])
    pro = pro.drop_duplicates(subset=['_atom_site.label_atom_id', '_atom_site.label_seq_id'])

    pep.reset_index(drop=True, inplace=True)
    pro.reset_index(drop=True, inplace=True)


    sites_pep = np.round((np.array(pep.iloc[:, -3:].astype(float)) - np.array([x_min, y_min, z_min]) + edge) * int(1/digit)).astype(int)
    sites_pro = np.round((np.array(pro.iloc[:, -3:].astype(float)) - np.array([x_min, y_min, z_min]) + edge) * int(1/digit)).astype(int)

    ca_pep = sites_pep[pep['_atom_site.label_atom_id'] == 'CA']

    return sites_pep, sites_pro, ca_pep 


def conformation(scs, r, sites, cube, label):
    scs_new = scs.copy()
    for x,y,z  in sites:
        if x-r >= 0 and x+r+1 <= scs_new.shape[0] and y-r >= 0 and y+r+1 <= scs_new.shape[1] and z-r >= 0 and z+r+1 <= scs_new.shape[2]:
            scs_new[x-r:x+r+1, y-r:y+r+1, z-r:z+r+1] += cube
    scs_new[scs_new > 0] = label

    return scs_new


def space_sitesrdinate_system(scs, radius, digit, sites_pep, sites_pro, cube):

    r = int(radius * int(1/digit))
    
    scs_pep = conformation(scs, r, sites_pep, cube, label=1)
    scs_pro = conformation(scs, r, sites_pro, cube, label=2)

    scs_complex = scs_pep + scs_pro

    return scs_complex, scs_pep, scs_pro


def sit2surface(radius_outer, radius_inner, digit):
    size = int((radius_outer*2) * int(1/digit) + 1)
    cube_s = np.zeros([size, size, size])
    center = np.array([radius_outer * int(1/digit), radius_outer * int(1/digit), radius_outer * int(1/digit)])

    surf = 0 
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if np.linalg.norm(np.array([x, y, z]) - center) <= (radius_outer * int(1/digit)) and np.linalg.norm(np.array([x, y, z]) - center) >= (radius_inner * int(1/digit)):
                    cube_s[x, y, z] = 1
                    surf += 1

    return cube_s, surf


def wrap_degree(scs_pro, radius_outer, digit, cube_s, surf, ca_pep):
    r = int(radius_outer * int(1/digit))
    pattern_surf = []

    for x_ca, y_ca, z_ca in ca_pep:

        cube_surf = scs_pro[x_ca-r:x_ca+r+1, y_ca-r:y_ca+r+1, z_ca-r:z_ca+r+1] + cube_s
        pattern_surf.append(((cube_surf == 3) * 1).sum() / surf)

    return pattern_surf




def metric(info='./datasets/InteractionMode_example_list', path_pdb='./datasets/example_PDB/', path_save='./output/InteractionModes_results/'):
    path_list = os.listdir(path_pdb)
    df_info = pd.read_table(info)

    edge = 6
    radius = 3
    digit = .1
    radius_outer, radius_inner = 6, 5.8

    cube = sit2sphere(radius, digit)
    cube_s, surf = sit2surface(radius_outer, radius_inner, digit)

    data = {}
    for i in range(len(path_list)):
        
        pdb = str(df_info['pdb'][i])
        path = f'{path_pdb}{pdb}.cif'
        asym1 = str(df_info['asym1'][i])
        asym2 = str(df_info['asym2'][i])

        print(f'{i}\t{pdb}', end='')


        if pdb.casefold() != path[-8:-4].casefold():
            print(f'Matching error ^_^*, index:{i}')
            break
        
        
        pdbx = MMCIF2Dict(path)

        atom_site = pd.DataFrame(pdbx, columns=['_atom_site.group_PDB', 
                                                '_atom_site.label_atom_id', 
                                                '_atom_site.label_comp_id', 
                                                '_atom_site.label_asym_id', 
                                                '_atom_site.label_seq_id', 
                                                '_atom_site.Cartn_x', 
                                                '_atom_site.Cartn_y', 
                                                '_atom_site.Cartn_z'])

        scs, x_min, y_min, z_min = construct_space_sitesrdinate_system(atom_site, edge, digit, asym1, asym2)
        sites_pep, sites_pro, ca_pep = atom_filter(asym1, asym2, atom_site, edge, digit, x_min, y_min, z_min)

        if len(ca_pep) <= 2:
            print(f'\tpeptide too short (<=2)')
            continue
        
        scs_complex, scs_pep, scs_pro = space_sitesrdinate_system(scs, radius, digit, sites_pep, sites_pro, cube)
        

        pattern_surf = wrap_degree(scs_pro, radius_outer, digit, cube_s, surf, ca_pep)
        
        data[pdb+'_'+asym1+'_'+asym2] = pattern_surf


        print(f'\tEnd')


    with open(f"{path_save}Bdeg.json", "w") as file:
        json.dump(data, file)

    print(f'Save')



if __name__ == "__main__":
    metric(info='./datasets/InteractionMode_example_list', path_pdb='./datasets/example_PDB/', path_save='./output/InteractionModes_results/')