import os
import pandas as pd
import torch
import glob
import json
from data_processing.utils2 import parser
from data_processing import padding_segment, ESM2, iFeature, SPOT1DSingle


# load_fasta='./datasets/example_data/example_list'
# path_load='./datasets/example_data/fasta/'
# path_save='./datasets/example_data/'


def sort_list(list_pt):
    return sorted(list_pt, key=lambda x : int(x.split("/")[-1].split("_")[-2]))
        

def feature_dic():
    args = parser()

    # path_load = args.load_fasta
    path_save = args.save_path.strip() 
    
    padding_segment.padding_segment(load_list=args.load_list, path_load=args.load_fasta)
    print('ESM2 start')
    ESM2.esm()
    print('ESM2 end')
    print('iFeature start')
    iFeature.ifeature()
    print('iFeature end')
    print('SPOT-1D-Single start')
    SPOT1DSingle.spot1d()
    print('SPOT-1D-Single end')

    names = pd.read_csv(args.load_list, sep='\t', header=None)

    os.makedirs(f'{path_save}feature_dic/', exist_ok=True)

    path_load = './data_processing/tmp/feature/'
    path_fasta = './data_processing/tmp/fasta/fragment/'

    nn = []
    pp = []
    for i in range(len(names)):

        id1 = names[0][i]
        id2 = names[1][i]
        nn.append(f'{id1}_{id2}')
        pp.append(f'{path_save}feature_dic/{id1}_{id2}')


        ### id-1 ###
        with open(f'{path_fasta}{id1}.fasta', 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        segments1 = lines[::2]
        chain1 = lines[1::2]

        spot1 = pd.read_table(f'{path_load}SPOT-1D-Single-binary/{id1}.tsv', keep_default_na=False)
        spot1.drop(spot1.columns[0], axis=1, inplace=True)
        f11 = spot1.values.tolist()

        aac1 = pd.read_table(f'{path_load}iFeature/{id1}_AAC.tsv', keep_default_na=False)
        aac1.drop(aac1.columns[0], axis=1, inplace=True)
        f12 = aac1.values.tolist()
        aaindex1 = pd.read_table(f'{path_load}iFeature/{id1}_AAINDEX.tsv', keep_default_na=False)
        aaindex1.drop(aaindex1.columns[0], axis=1, inplace=True)
        f13 = aaindex1.values.tolist()
        blosum1 = pd.read_table(f'{path_load}iFeature/{id1}_BLOSUM62.tsv', keep_default_na=False)
        blosum1.drop(blosum1.columns[0], axis=1, inplace=True)
        f14 = blosum1.values.tolist()
        
        pattern = os.path.join(f'{path_load}esm2_fragment/', f"{id1}*")
        list_pt = sort_list(glob.glob(pattern))
        print(list_pt)
        f15 = []
        for pt in list_pt:
            f15.append(torch.load(pt).tolist())
        
        feature1 = [sum(rows, []) for rows in zip(f11, f12, f13, f14, f15)]

        esm2_feature1 = torch.load(f'{path_load}esm2_sequence/{id1}.pt').tolist()


        ### id-2 ###
        with open(f'{path_fasta}{id2}.fasta', 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        segments2 = lines[::2]
        chain2 = lines[1::2]

        spot2 = pd.read_table(f'{path_load}SPOT-1D-Single-binary/{id2}.tsv', keep_default_na=False)
        spot2.drop(spot2.columns[0], axis=1, inplace=True)
        f21 = spot2.values.tolist()

        aac2 = pd.read_table(f'{path_load}iFeature/{id2}_AAC.tsv', keep_default_na=False)
        aac2.drop(aac2.columns[0], axis=1, inplace=True)
        f22 = aac2.values.tolist()
        aaindex2 = pd.read_table(f'{path_load}iFeature/{id2}_AAINDEX.tsv', keep_default_na=False)
        aaindex2.drop(aaindex2.columns[0], axis=1, inplace=True)
        f23 = aaindex2.values.tolist()
        blosum2 = pd.read_table(f'{path_load}iFeature/{id2}_BLOSUM62.tsv', keep_default_na=False)
        blosum2.drop(blosum2.columns[0], axis=1, inplace=True)
        f24 = blosum2.values.tolist()
        
        pattern = os.path.join(f'{path_load}esm2_fragment/', f"{id2}*")
        list_pt = sort_list(glob.glob(pattern))
        print(list_pt)
        f25 = []
        for pt in list_pt:
            f25.append(torch.load(pt).tolist())
        
        feature2 = [sum(rows, []) for rows in zip(f21, f22, f23, f24, f25)]

        esm2_feature2 = torch.load(f'{path_load}esm2_sequence/{id2}.pt').tolist()


        ### dic ###
        data = {
        "segments_x": segments1,
        "chain_x": chain1,
        "features_x": feature1,
        "esm2_feature_x": esm2_feature1,
        "segments_y": segments2,
        "chain_y": chain2,
        "features_y": feature2,
        "esm2_feature_y": esm2_feature2,
        "seg_labels": [1]*len(segments1)*len(segments2)
        }

        ### save ###
        with open(f'{path_save}feature_dic/{id1}_{id2}', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=None)


    data_csv = {
        'pair_id': nn,
        'feature_dict_path': pp,
    }

    pd.DataFrame(data_csv).to_csv(f'{path_save}feature_path.csv', index=False, na_rep=',')


if __name__ == "__main__":
    feature_dic()
