import os
import pandas as pd
from tqdm import tqdm
from .SPOT_1D_Single.spot1d_single2 import spot1d_single, binary_segment


def spot1d():
    CURRENT_DIR = os.path.dirname(__file__)

    path_load = os.path.join(CURRENT_DIR, 'tmp/fasta/sequence/')
    path_save1 = os.path.join(CURRENT_DIR, 'tmp/feature/SPOT-1D-Single/')
    path_save_list = os.path.join(CURRENT_DIR, 'tmp/feature/')

    path_tool = os.path.join(CURRENT_DIR, 'SPOT_1D_Single/spot1d_single2.py')

    batch=200
    device='cuda:0'


    files = os.listdir(path_load)
    fasta_path = [f'{path_load}{file}' for file in files]

    df_fasta = pd.DataFrame(fasta_path)

    os.makedirs(f'{path_save1}', exist_ok=True)

    ### batch handling ###
    # slice
    sli = list(range(0, len(df_fasta), batch))
    sli.append(len(df_fasta))
    i=0
    for i in range(len(sli)-1):
        # save
        df_fasta[sli[i]:sli[i+1]].to_csv(f'{path_save_list}batch_fasta_list', sep='\t', header=False, index=False)

        # spot1d
        spot1d_single(file_list=f'{path_save_list}batch_fasta_list', device=device, save_path=path_save1)
        # os.system(f'python {path_tool} --file_list {path_save_list}batch_fasta_list --save_path {path_save1} --device {device}')

        os.system(f'rm {path_save_list}batch_fasta_list')


    path_load = os.path.join(CURRENT_DIR, 'tmp/feature/SPOT-1D-Single/')
    path_seg = os.path.join(CURRENT_DIR, 'tmp/fasta/fragment/')
    path_save2 = os.path.join(CURRENT_DIR, 'tmp/feature/SPOT-1D-Single-binary/')

    os.makedirs(f'{path_save2}', exist_ok=True)
    binary_segment(path_load=path_load, path_seg=path_seg, path_save=path_save2, window=10)


if __name__ == "__main__":
    spot1d()