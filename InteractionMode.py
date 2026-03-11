import json
import numpy as np
import pandas as pd
from data_processing.utils2 import parser
from data_processing import Bdeg, Ideg


args = parser()
Bdeg.metric(info=args.load_list, path_pdb=args.load_PDB, path_save=args.save_path)
Ideg.metric(info=args.load_list, path_pdb=args.load_PDB, path_save=args.save_path)

# 包裹程度：用于参考包裹程度
with open(f'{args.save_path}Bdeg.json', 'r') as file:
    data_surf = json.load(file)

# 横截面覆盖角度：主要参考指标
with open(f'{args.save_path}Ideg.json', 'r') as file:
    data_angle = json.load(file)


def CheckWrap(angle, surf):

    arr_a = angle >= 0.99
    arr_s = surf >= 0.8

    arr_a2 = (angle >= 0.89) & (angle < 0.99)
    arr_s2 = surf >= 0.7

    angle0 = np.convolve(a=angle, v=[0.4, 0.2, 0.4], mode='valid')

    if np.sum(arr_a) / len(arr_a) >= 0.3:
        if np.all(arr_s[~arr_a]): 
            return True
        elif min(angle) >= 0.89:
            if np.all(arr_s2[arr_a2]):
                return True
        elif np.all(angle0 >= 0.95):
            return True
        else:
            return False
    else:
        return False



def CheckPocket(surf, p=0.3, l=2):
    arr = (surf>=0.63)

    diff = np.diff(arr.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if arr[0]:
        starts = np.insert(starts, 0, 0)
    if arr[-1]:
        ends = np.append(ends, len(arr))

    lengths = ends - starts

    if len(lengths) == 0:
        return False

    if np.max(lengths) / len(arr) >= p or np.max(lengths) >= l:
        return True
    else:
        return False




def main():
    wrap = []
    plugin = []
    pocket = []
    surface = []

    for pdb in data_angle:
        angle = np.array(data_angle[pdb])
        surf = np.array(data_surf[pdb])


        wrap_type = angle >= 0.98

        if any(wrap_type[i] and wrap_type[i + 1] for i in range(len(wrap_type) - 1)):

            if CheckWrap(angle, surf):
                wrap.append(pdb)

            else:
                plugin.append(pdb)


        elif sum(wrap_type*1) == 1:
            pocket.append(pdb)

        else:
            if CheckPocket(surf, 0.3, 2):
                pocket.append(pdb)
            else:
                surface.append(pdb)



    print(f'wrap:\t\t{len(wrap)/len(data_angle)*100:.4f}%')
    print(f'plugin:\t\t{len(plugin)/len(data_angle)*100:.4f}%')
    print(f'pocket:\t\t{len(pocket)/len(data_angle)*100:.4f}%')
    print(f'surface:\t{len(surface)/len(data_angle)*100:.4f}%')

    dic = {}
    dic['wrap'] = wrap
    dic['plugin'] = plugin
    dic['pocket'] = pocket
    dic['surface'] = surface

    with open(f"{args.save_path}InteractionMode_results.json", "w") as file:
        json.dump(dic, file)


if __name__ == "__main__":
    main()

