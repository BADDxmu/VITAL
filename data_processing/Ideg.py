import os
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
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


def angle(scs_pro, ca_i, ca_pep, ca_x, dx, dy, dz, max_dis, digit, unew):

    space = scs_pro
    space_size = np.array(scs_pro.shape)


    P = ca_pep[ca_i]
    idx = int((unew.size-1) * ca_i / (len(ca_x)-1))
    tangent = [dx[idx], dy[idx], dz[idx]]

    line_direction = np.array([dx[idx], dy[idx], dz[idx]])
    line_direction = line_direction / np.linalg.norm(line_direction)

    if not np.allclose(line_direction, [1, 0, 0]):
        reference_vector = np.array([1, 0, 0])
    else:
        reference_vector = np.array([0, 1, 0])

    x_axis = np.cross(reference_vector, line_direction)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(line_direction, x_axis)

    rotation_matrix = np.column_stack((x_axis, y_axis, line_direction))

    num_angles = 360
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    detected_angles = 0

    max_distance = max_dis * int(1/digit) 

    for theta in angles:
        direction_in_plane = np.array([np.cos(theta), np.sin(theta), 0]) 
        
        direction_world = rotation_matrix @ direction_in_plane
        
        current_point = P.copy()
        step_size = 1 
        while True:
            current_point += (direction_world * step_size).round().astype(int)
            
            if np.any(current_point < 0) or np.any(current_point >= space_size):
                break
            
            if space[current_point[0], current_point[1], current_point[2]] == 2:
                detected_angles += 1
                break

            if np.linalg.norm(current_point - P) > max_distance:
                break

    angle_coverage = detected_angles / num_angles

    return angle_coverage


def deflection(ca_i, dx, dy, dz, unew):

    idx1 = int((unew.size-1) * (ca_i-1) / (len(ca_x)-1))
    idx2 = int((unew.size-1) * ca_i / (len(ca_x)-1))
    
    A = np.array([dx[idx1], dy[idx1], dz[idx1]])
    B = np.array([dx[idx2], dy[idx2], dz[idx2]])

    dot_product = np.dot(A, B)

    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    cos_theta = dot_product / (norm_A * norm_B)

    theta = np.arccos(cos_theta)

    theta_degrees = np.degrees(theta)

    return theta_degrees



# %%


def metric(info='./datasets/InteractionMode_example_list', path_pdb='./datasets/example_PDB/', path_save='./output/InteractionModes_results/'):
    path_list = os.listdir(path_pdb)
    df_info = pd.read_table(info)

    radius = 2.7
    digit = .1
    sss = 200
    kkk = 2
    ca_num = 20
    max_dis = 9.5
    edge = max(radius, max_dis)
    cube = sit2sphere(radius, digit)

    data_angle = {}
    data_deflection = {}
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

        ca_x,ca_y,ca_z = ca_pep.T
        tck, u = splprep([ca_x,ca_y,ca_z], s=sss, k=kkk)
        unew = np.linspace(0, 1, (len(ca_x)-1)*ca_num+1) 
        out = splev(unew, tck)


        dx, dy, dz = splev(unew, tck, der=1)

        pattern_angle = []
        for ca_i in range(len(ca_x)):
            pattern_angle.append(angle(scs_pro, ca_i, ca_pep, ca_x, dx, dy, dz, max_dis, digit, unew))

        data_angle[pdb+'_'+asym1+'_'+asym2] = pattern_angle 

        print(f'\tEnd')

    with open(f"{path_save}Ideg.json", "w") as file:
        json.dump(data_angle, file)


    print(f'Save')



if __name__ == "__main__":
    metric(info='./datasets/InteractionMode_example_list', path_pdb='./datasets/example_PDB/', path_save='./output/InteractionModes_results/')