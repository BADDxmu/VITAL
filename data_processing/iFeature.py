import os
import pandas as pd
from tqdm import tqdm


def ifeature():
    CURRENT_DIR = os.path.dirname(__file__)
    path_load = os.path.join(CURRENT_DIR, 'tmp/fasta/fragment/')
    path_save = os.path.join(CURRENT_DIR, 'tmp/feature/iFeature/')
    path_tool = os.path.join(CURRENT_DIR, 'iFeature/iFeature.py')

    os.makedirs(f'{path_save}', exist_ok=True)

    files = os.listdir(path_load)

    for file in files:
        id = file[:-6]
        
        os.system(f'python {path_tool} --file {path_load}{id}.fasta --type AAC --out {path_save}{id}_AAC.tsv')
        os.system(f'python {path_tool} --file {path_load}{id}.fasta --type AAINDEX --out {path_save}{id}_AAINDEX.tsv')
        os.system(f'python {path_tool} --file {path_load}{id}.fasta --type BLOSUM62 --out {path_save}{id}_BLOSUM62.tsv')

if __name__ == "__main__":
    ifeature()